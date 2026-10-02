from __future__ import annotations

import json

import typer

from .core.timeframe import interval_ms, parse_time, to_dt, to_ms
from .data.backfill import Backfiller, assert_complete
from .data.quality import htf_consistency
from .data.service import DataNotReady, MarketDataService
from .data.store import Store
from .data.wiring import build_rest_client
from .infra.log import setup
from .infra.settings import Settings, get_settings
from .notify.cards import build_card, sample_signal_decision
from .notify.feishu import FeishuConfig, FeishuNotifier
from .risk.exchange_meta import parse_symbol_spec

app = typer.Typer(add_completion=False, help="Quant LLM MVP CLI")


def _boot(level: str | None) -> Settings:
    s = get_settings()
    setup(level or s.log_level, s.log_file)
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
    intervals: str | None = typer.Option(None, help="逗号分隔，如 15m,1h；默认取品种 intervals"),
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


if __name__ == "__main__":
    app()
