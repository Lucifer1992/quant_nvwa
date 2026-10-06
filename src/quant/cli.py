from __future__ import annotations

import json
from pathlib import Path

import typer

from .core.timeframe import interval_ms, parse_time, to_dt, to_ms
from .data.backfill import Backfiller, assert_complete
from .data.quality import htf_consistency
from .data.service import DataNotReady, MarketDataService
from .data.store import Store
from .data.wiring import build_rest_client
from .infra.log import setup
from .infra.settings import Settings, get_settings, validate_config
from .notify.cards import build_card, sample_signal_decision
from .notify.feishu import FeishuConfig, FeishuNotifier
from .risk.exchange_meta import parse_symbol_spec
from .strategy import get_strategy, list_strategies as registered_strategies

app = typer.Typer(add_completion=False, help="Quant LLM MVP CLI")


def _boot(level: str | None) -> Settings:
    s = get_settings()
    setup(level or s.log_level, s.log_file)
    # 跨文件引用（周期/杠杆/策略 htf）一旦漂移，回填会静少拉一个周期、风控会按错的
    # ATR 尺度算止损，所以这里当错误处理而不是警告。
    problems = validate_config(s)
    if problems:
        typer.echo("配置校验失败：", err=True)
        for p in problems:
            typer.echo(f"  - {p}", err=True)
        raise typer.Exit(code=2)
    return s


def _targets(s: Settings, symbol: str | None) -> dict[str, dict]:
    """解析命令作用品种：显式指定优先，否则取全部 enabled 品种。"""
    all_symbols = s.symbols.get("symbols", {})
    if symbol:
        cfg = all_symbols.get(symbol)
        if cfg is None:
            raise typer.Exit(f"symbols.yaml 中没有 {symbol}")
        if not cfg.get("enabled"):
            typer.echo(f"警告：symbols.{symbol}.enabled=false，仍按请求处理", err=True)
        return {symbol: cfg}
    enabled = s.enabled_symbols()
    if not enabled:
        raise typer.Exit("symbols.yaml 中没有 enabled 的交易对")
    return enabled


def _start_ms(s: Settings, cfg: dict, override: str | None) -> int:
    return to_ms(parse_time(override or cfg.get("since") or s.backtest_default_start))


@app.command()
def backfill(
    symbol: str | None = typer.Option(None, help="默认回填 symbols.yaml 中所有 enabled 品种"),
    start: str | None = typer.Option(None, help="默认取品种 since，再默认 backtest.default_start"),
    intervals: str | None = typer.Option(None, help="逗号分隔，如 5m,1h；默认取品种 intervals"),
    funding: bool = typer.Option(True, "--funding/--no-funding"),
    force_spec: bool = typer.Option(False, help="忽略 24h 缓存，重新拉取 exchangeInfo"),
    level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """从 Binance USDT-M REST 分页回填 K 线与资金费率到本地 SQLite（按水位线断点续传）。"""
    s = _boot(level)
    with Store(s.db_path) as store, build_rest_client(s) as client:
        bf = Backfiller(client, store)
        for name, cfg in _targets(s, symbol).items():
            ivs = [v.strip() for v in intervals.split(",")] if intervals else list(cfg["intervals"])
            since = _start_ms(s, cfg, start)
            spec = bf.refresh_spec(name, force=force_spec)
            typer.echo(
                f"[{name}] tick={spec.tick_size} step={spec.step_size} min_notional={spec.min_notional} "
                f"maint={spec.maint_margin_rate} max_lev={spec.max_leverage:.0f}x"
            )
            for iv in ivs:
                report = bf.bars(name, iv, since)
                typer.echo(report.summary())
                for gap_start, gap_end in report.gaps[:10]:
                    typer.echo(f"  ! 缺口 {to_dt(gap_start)} ~ {to_dt(gap_end)}", err=True)
            if funding:
                bf.funding(name, since)


@app.command("check-data")
def check_data(
    symbol: str | None = typer.Option(None),
    start: str | None = typer.Option(None),
    level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """校验本地数据：网格覆盖率、缺口、交易规则缓存与数据集指纹。回测前必跑。"""
    s = _boot(level)
    with Store(s.db_path) as store:
        svc = MarketDataService(store)
        for name, cfg in _targets(s, symbol).items():
            typer.echo(f"== {name} ==")
            try:
                spec = svc.spec(name)
                typer.echo(
                    f"  spec: tick={spec.tick_size} step={spec.step_size} maint={spec.maint_margin_rate} "
                    f"fetched_at={spec.fetched_at:%Y-%m-%d %H:%M}"
                )
            except DataNotReady as e:
                typer.echo(f"  {e}", err=True)
            since = _start_ms(s, cfg, start)
            for iv in cfg["intervals"]:
                watermark = store.watermark_ms(name, iv)
                if watermark is None:
                    typer.echo(f"  {iv}: 无数据，先执行 quant backfill --symbol {name}", err=True)
                    continue
                chk = assert_complete(store, name, iv, since, watermark + interval_ms(iv))
                mark = "OK  " if chk["coverage"] == 1.0 else "MISS"
                line = (
                    f"  {iv}: {mark} bars={chk['present']}/{chk['expected']} "
                    f"coverage={chk['coverage']:.4%} last={to_dt(watermark):%Y-%m-%d %H:%M}"
                )
                if chk["missing"]:
                    line += f" 首个缺失={chk['missing_head'][0] if chk['missing_head'] else '?'}"
                typer.echo(line)
            for rep in htf_consistency(store, name, cfg["base_interval"], cfg["intervals"], since):
                mark = "OK  " if rep["mismatched"] == 0 else "DIFF"
                line = (
                    f"  {rep['htf_interval']} 交易所值 vs {rep['base_interval']} 聚合值: {mark} "
                    f"{rep['compared'] - rep['mismatched']}/{rep['compared']} "
                    f"最大偏差 close={rep['max_close_dev']:.4f} open={rep['max_open_dev']:.4f}"
                )
                if rep["mismatched"]:
                    line += f"（异常 bar: {rep['worst_at']}，多周期特征直接读交易所值，需知悉此偏差）"
                typer.echo(line)
            typer.echo(f"  dataset_version={store.dataset_version(name, cfg['intervals'])}")


@app.command("exchange-info")
def exchange_info(
    symbol: str | None = typer.Option(None, help="默认全部 enabled 品种"),
    raw: bool = typer.Option(False, help="输出交易所原始 symbol 对象"),
    level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """查看交易规则。USDT-M 的 exchangeInfo 忽略 symbol 参数并返回全量品种，故在本地筛选。"""
    s = _boot(level)
    with build_rest_client(s) as client:
        for name in _targets(s, symbol):
            payload = client.symbol_info(name)
            if raw:
                typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
                continue
            spec = parse_symbol_spec(payload)
            typer.echo(
                f"{name} status={spec.status} tick={spec.tick_size} step={spec.step_size} "
                f"min_qty={spec.min_qty} min_notional={spec.min_notional} maint={spec.maint_margin_rate} "
                f"liq_fee={spec.liquidation_fee_rate} max_lev={spec.max_leverage:.0f}x "
                f"prec={spec.price_precision}/{spec.qty_precision}"
            )


@app.command()
def test_card(
    approved: bool = typer.Option(True, help="false 时发送被否决的卡片"),
    level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """发送一张样例入场卡片到飞书，验证 webhook 与卡片格式是否可用。"""
    s = _boot(level)
    sig, dec = sample_signal_decision(approved=approved)
    payload = build_card(sig, dec)
    if "feishu" not in s.notify.get("channels", []):
        typer.echo("notify.yaml channels 未包含 feishu，仅打印 payload：")
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    cfg = FeishuConfig.from_settings(s)
    if cfg is None:
        raise typer.Exit(code=2)
    # 飞书域名境内直连即可，走 Clash 出口反而可能被路由到境外节点，增加失败率
    with FeishuNotifier(cfg) as n:
        resp = n.send_payload(payload)
    typer.echo(f"sent approved={approved} resp={resp}")


@app.command()
def dump_card(approved: bool = True) -> None:
    """只打印卡片 payload，不发送。"""
    sig, dec = sample_signal_decision(approved=approved)
    typer.echo(json.dumps(build_card(sig, dec), ensure_ascii=False, indent=2))


@app.command("backtest")
def run_backtest(
    symbol: str = typer.Option("ETHUSDT"),
    start: str = typer.Option("2024-01-01", help="回测开始日期"),
    end: str = typer.Option("2026-10-01", help="回测结束日期（不包含当日数据）"),
    equity: float = typer.Option(100.0, help="初始权益 USDT（实盘 100U）"),
    risk_pct: float = typer.Option(0.01, help="单笔风险比例"),
    sl_atr: float | None = typer.Option(None, help="SL ATR 倍数（缺省按策略：5m 策略 1.5 / rule_v4 2.0）"),
    tp_atr: float | None = typer.Option(None, help="TP ATR 倍数（缺省按策略：5m 策略 3.0 / rule_v4 3.5）"),
    lev: float = typer.Option(10.0, help="全局最大杠杆（固定全程）"),
    short: bool = typer.Option(True, "--short/--no-short"),
    strategy: str = typer.Option(
        "rule_v1",
        help="策略名（取自注册表）: " + " / ".join(registered_strategies()),
    ),
    # rule_v2 专属（None = 用策略自己的默认阈值；各策略的同名字段默认值并不相同，
    # 在 CLI 里写死默认值会把它串给别的策略）
    srsi_oversold: float | None = typer.Option(None, help="rule_v2 StochRSI 超卖阈值（默认 15）"),
    srsi_overbought: float | None = typer.Option(None, help="rule_v2 StochRSI 超买阈值（默认 65）"),
    srsi_cross_margin: float | None = typer.Option(None, help="rule_v2 拐头幅度（默认 1）"),
    # 金字塔加仓参数
    entry_frac: float = typer.Option(0.10, help="试仓比例（满仓的 %）"),
    add_frac: float = typer.Option(0.20, help="每次加仓比例"),
    add_max: int = typer.Option(2, help="最大加仓次数"),
    add_trigger_r: float | None = typer.Option(None, help="加仓盈利 R 门槛（缺省按策略：0.3 / rule_v4 1.0）"),
    trail_trigger_r: float | None = typer.Option(None, help="移动止损激活 R 门槛（缺省按策略：1.0）"),
    no_pyramid: bool = typer.Option(False, "--no-pyramid", help="关闭金字塔加仓"),
    # 入场确认（连续 N 根 5m bar 满足条件才开仓，过滤假信号）
    confirm_bars: int | None = typer.Option(None, "--confirm-bars", help="入场确认 bar 数（默认按策略，1=不要求）"),
    # rule_v3 专属
    min_signals: int | None = typer.Option(None, "--min-signals", help="rule_v3 至少需要几个指标同向（默认 2）"),
    # rule_v4 专属（不传则用 RuleV4Config 的实测默认值）
    adx_min: float | None = typer.Option(None, "--adx-min", help="rule_v4 1h ADX 开仓门槛（默认 32）"),
    pullback_atr: float | None = typer.Option(None, "--pullback-atr", help="rule_v4 回调深度门槛，×1h ATR（默认 1.5）"),
    # 出场节奏（不传则按策略的 engine_defaults）
    time_stop_bars: int | None = typer.Option(None, "--time-stop-bars", help="时间止损（5m bar 数；5m 策略默认 96=8h，rule_v4 默认 864=3天）"),
    trail_atr_mult: float | None = typer.Option(None, "--trail-atr-mult", help="移动止损跟随距离（× entry_atr，默认 1.0，rule_v4 1.5）"),
    # lgbm_v1 专属（不传则用 LGBMv1Config 默认值）
    tau: float | None = typer.Option(None, "--tau", help="lgbm_v1 下注门槛：|P(up)-P(down)| 超过它才开仓（默认 0.15）"),
    model_dir: str | None = typer.Option(None, "--model-dir", help="lgbm_v1 折模型目录（默认 models/lgbm_v1）"),
    allow_gaps: bool = typer.Option(False, "--allow-gaps", help="数据有缺柱时仍然继续回测（默认中止）"),
    level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """运行事件驱动回测，产出 reports/{run_id}/ 含 equity/trades/metrics/report。"""
    s = _boot(level)

    from .backtest.engine import BacktestConfig, BacktestEngine
    from .backtest.reports import save_report

    # 策略分发（用注册表自动发现）
    try:
        StrategyCls = get_strategy(strategy)
    except ValueError as e:
        raise typer.Exit(str(e))

    # 策略参数由策略自己认领（BaseStrategy.build 按 config_cls 字段挑选），
    # CLI 不判断"这个选项属于哪个策略"
    rule_strategy = StrategyCls.build({
        "symbol": symbol,
        "allow_short": short,
        "confirm_bars": confirm_bars,
        "srsi_oversold": srsi_oversold,
        "srsi_overbought": srsi_overbought,
        "srsi_cross_margin": srsi_cross_margin,
        "min_signals": min_signals,
        "adx_min": adx_min,
        "pullback_atr": pullback_atr,
        "tau": tau,
        "model_dir": Path(model_dir) if model_dir else s.models_dir / "lgbm_v1",
    })

    # 出场/节奏参数：用户显式传参优先，否则采用策略声明的 engine_defaults
    # （1h 决策周期的策略需要 1h ATR 量级的止损与以天计的时间止损）
    ed = rule_strategy.engine_defaults()
    sl_atr = sl_atr if sl_atr is not None else ed.get("sl_atr_mult", 1.5)
    tp_atr = tp_atr if tp_atr is not None else ed.get("tp_atr_mult", 3.0)
    time_stop_bars = time_stop_bars if time_stop_bars is not None else ed.get("time_stop_bars", 96)
    trail_atr_mult = trail_atr_mult if trail_atr_mult is not None else ed.get("trail_atr_mult", 1.0)
    trail_trigger_r = trail_trigger_r if trail_trigger_r is not None else ed.get("trail_trigger_r", 1.0)
    add_trigger_r = add_trigger_r if add_trigger_r is not None else ed.get("add_trigger_r", 0.3)

    # 根据 required_intervals 决定加载哪些时间周期
    intervals_to_load = list(rule_strategy.required_intervals)

    with Store(s.db_path) as store:
        svc = MarketDataService(store)
        sym_cfg = s.symbols.get("symbols", {}).get(symbol)
        if sym_cfg is None:
            raise typer.Exit(f"symbols.yaml 中没有 {symbol}")

        try:
            spec = svc.spec(symbol)
        except DataNotReady as e:
            typer.echo(f"错误：{e}", err=True)
            raise typer.Exit(code=2)

        bars: dict[str] = {}
        # ML 策略的高周期指标需要窗口之前的历史预热，否则起点附近全是 NaN
        lookback_ms = rule_strategy.history_lookback()
        load_start_ms = to_ms(parse_time(start)) - lookback_ms
        load_start = to_dt(load_start_ms)
        for iv in intervals_to_load:
            df = svc.bars(symbol, iv, load_start, end)
            if df.empty:
                typer.echo(f"警告：{symbol} {iv} 在 [{load_start}, {end}) 无数据，跳过", err=True)
                continue
            bars[iv] = df
            typer.echo(f"  加载 {iv}: {len(df)} 根 {df.ts_open.min()} ~ {df.ts_open.max()}")

        funding = None
        if lookback_ms or any(iv in bars for iv in ("1h", "4h")):
            funding = svc.funding(symbol, load_start, end)

        # 结算节奏取自 symbols.yaml，但当前所有已注册策略的触发周期都写死在 5m 上，
        # 换 base 必须同时换策略实现，所以这里只允许 5m。
        base_interval = sym_cfg.get("base_interval", "5m")
        if base_interval != "5m":
            typer.echo(
                f"警告：symbols.{symbol}.base_interval={base_interval}，"
                "但已注册策略均按 5m 触发，仍按 5m 回测",
                err=True,
            )
            base_interval = "5m"
        if base_interval not in bars:
            raise typer.Exit(f"{base_interval} 数据为空，无法回测")

        # 数据闸门：只校验用户请求的 [start, end)（预热前缀允许缺），缺柱会让
        # 止损触发顺序与指标预热错位，产出的数字看着正常却不可复现。
        gate_start_ms, gate_end_ms = to_ms(parse_time(start)), to_ms(parse_time(end))
        incomplete: list[str] = []
        for iv in sorted(bars):
            watermark = store.watermark_ms(symbol, iv)
            if watermark is None:
                incomplete.append(f"{iv}: 无任何数据")
                continue
            chk = assert_complete(store, symbol, iv, gate_start_ms, min(gate_end_ms, watermark + interval_ms(iv)))
            if chk["coverage"] < 1.0:
                head = chk["missing_head"][0] if chk["missing_head"] else "?"
                incomplete.append(
                    f"{iv}: 缺 {chk['missing']}/{chk['expected']} 根 "
                    f"(coverage={chk['coverage']:.4%}, 首个缺失={head})"
                )
        if incomplete:
            typer.echo("数据完整性：", err=True)
            for line in incomplete:
                typer.echo(f"  ! {line}", err=True)
            if not allow_gaps:
                typer.echo("已中止回测；确认可以接受缺柱时加 --allow-gaps", err=True)
                raise typer.Exit(code=2)
        else:
            typer.echo(f"  数据完整性：{len(bars)} 个周期在 [{start}, {end}) 无缺柱")

        cfg = BacktestConfig(
            symbol=symbol,
            base_interval=base_interval,
            start=parse_time(start),
            end=parse_time(end),
            initial_equity=equity,
            risk_pct=risk_pct,
            sl_atr_mult=sl_atr,
            tp_atr_mult=tp_atr,
            allow_short=short,
            global_max_leverage=lev,
            time_stop_bars=time_stop_bars,
            strategy_name=strategy,
            pyramid_enabled=not no_pyramid,
            entry_frac=entry_frac,
            add_frac=add_frac,
            add_max_count=add_max,
            add_trigger_r=add_trigger_r,
            trail_trigger_r=trail_trigger_r,
            trail_atr_mult=trail_atr_mult,
        )

        def signal_fn(bar, row, htf, dt):
            return rule_strategy.on_bar(bar, row, htf, dt)

        engine = BacktestEngine(cfg, bars, spec, funding=funding)
        typer.echo(f"\n开始回测 {symbol} {start} ~ {end} (策略: {strategy}) ...")
        result = engine.run(signal_fn, strategy=rule_strategy)

        run_id = f"{symbol}_{strategy}_{start}_{end}"
        report_dir = s.reports_dir / run_id
        save_report(
            report_dir, result, cfg,
            store.dataset_version(symbol, list(bars.keys())),
            bars=bars,
        )
        typer.echo(f"\n报告已生成: {report_dir}")
        typer.echo(f"  初始权益: {equity:,.2f} → 最终权益: {result.final_equity:,.2f}")
        typer.echo(f"  总收益率: {result.total_return:+.2%}  CAGR: {result.cagr:+.2%}")
        typer.echo(f"  Sharpe: {result.sharpe:.2f}  Sortino: {result.sortino:.2f}")
        typer.echo(f"  最大回撤: {result.max_drawdown:,.2f} USD ({result.max_drawdown_pct:.2f}%)")
        typer.echo(f"  交易次数: {result.total_trades}  胜率: {result.win_rate:.1%}  盈亏比: {result.payoff_ratio:.2f}")
        typer.echo(f"  Profit Factor: {result.profit_factor:.2f}")
        typer.echo(f"  平均持仓: {result.avg_bars_held:.1f} 根 ({base_interval})")
        typer.echo(f"  金字塔加仓: {engine._add_count_total} 次成功 / {engine._add_skip_count} 次跳过")
        typer.echo(f"\n完整报告: {report_dir / 'report.html'}")
        typer.echo(f"指标: {report_dir / 'metrics.json'}")
        typer.echo(f"净值: {report_dir / 'equity.csv'}")
        typer.echo(f"交易: {report_dir / 'trades.csv'}")


@app.command("train")
def train_model(
    symbol: str = typer.Option("ETHUSDT"),
    data_start: str = typer.Option("2024-01-01", help="训练数据起点"),
    eval_start: str = typer.Option("2026-04-01", help="样本外评估区间起点（第一折的预测起点）"),
    eval_end: str = typer.Option("2026-10-01", help="样本外评估区间终点"),
    window_months: int = typer.Option(12, "--window-months", help="滚动训练窗（月）"),
    step_months: int = typer.Option(1, "--step-months", help="重训步长（月）"),
    horizon: int = typer.Option(24, help="标签时间障碍（1h bar 数）"),
    k_up: float = typer.Option(3.0, "--k-up", help="上障碍 = k_up × 1h ATR"),
    k_dn: float = typer.Option(2.0, "--k-dn", help="下障碍 = k_dn × 1h ATR"),
    no_balance: bool = typer.Option(False, "--no-balance", help="关闭类别均衡（对照用）"),
    out_dir: str | None = typer.Option(None, "--out-dir", help="模型输出目录（默认 models/lgbm_v1）"),
    level: str | None = typer.Option(None, "--log-level"),
) -> None:
    """walk-forward 训练 lgbm_v1：每折落盘 model.txt + train_meta.json，并输出折间稳定性报告。"""
    s = _boot(level)

    from .features.builder import feature_spec_hash
    from .train import TrainConfig, build_dataset, plan_folds, train_folds, write_fold_report

    cfg = TrainConfig(
        symbol=symbol, data_start=data_start, eval_start=eval_start, eval_end=eval_end,
        window_months=window_months, step_months=step_months, horizon=horizon,
        k_up=k_up, k_dn=k_dn, balance_classes=not no_balance,
        out_dir=str(Path(out_dir) if out_dir else s.root / "models" / "lgbm_v1"),
    )
    folds = plan_folds(cfg)
    typer.echo(f"共 {len(folds)} 折：{folds[0].tag} ~ {folds[-1].tag}（训练窗 {window_months} 个月滚动）")

    with Store(s.db_path) as store:
        frame, cols, version = build_dataset(store, cfg)
    labeled = frame["label"].notna().sum()
    dist = frame["label"].value_counts(normalize=True).round(4).to_dict()
    typer.echo(f"样本 {len(frame)} 根 1h，可打标 {labeled} 根，标签分布 {dist}")
    typer.echo(f"特征 {len(cols)} 维，spec hash {feature_spec_hash(cols)}，dataset_version {version}")

    summaries = train_folds(frame, cols, version, cfg)
    report = write_fold_report(cfg, summaries, Path(cfg.out_dir) / cfg.symbol / "folds.json")
    typer.echo(
        f"\n折间稳定性：logloss 均值 {report['logloss_mean']} "
        f"[{report['logloss_min']} ~ {report['logloss_max']}]，"
        f"accuracy 均值 {report['accuracy_mean']}"
    )
    typer.echo(f"模型目录: {Path(cfg.out_dir) / cfg.symbol}")
    typer.echo(f"折报告:   {Path(cfg.out_dir) / cfg.symbol / 'folds.json'}")


if __name__ == "__main__":
    app()
