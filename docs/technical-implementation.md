# Quant_LLM 技术实现文档（最终版）

> 本文是**当前代码实现状态**的唯一权威说明：以实际代码与实测数据为准，不是设计意图的复述。
> 设计决策的论证过程见 `docs/` 下其余设计文档；与本文冲突时，以本文为准（本文对齐 2026-10-07 的代码）。

- 代码基线：`master`。paper 服务 + TP1/保本/强平 + 剔除资金费这批改动已于 2026-10-07 入库
  （`feat:` 代码提交 + `docs:` 文档提交各一条）；工作区仅剩 `models/`（9.0 MB 折模型）未跟踪，
  "提交 vs gitignore"待用户决定。**push 仍需用户显式授权**
- 校验状态：`uv run pytest -q` → **213 passed**；`uv run ruff check --select F,E9 src tests` → clean
- 数据基线：`data/quant.db`，ETHUSDT 5m/15m/1h/4h/1d + funding，2024-01-01 起
- 可比性声明：回测口径改过**两轮**，跨轮次的数字一律不可比较，只能在同一轮内做相对排序
  - 第 1 轮（2026-10-04，前视/保证金/滑点修复）：第 13.3 节表格里"修复前"那一列全部作废
  - 第 2 轮（2026-10-07，**成本模型剔除资金费** + **TP1 分批止盈/保本移动/强平触发落地**）：13.3 节"修复后"那一列也随之作废，现行数字见 §16 基线表

---

## 1. 系统概览

### 1.1 目标与交易原则（决定所有技术口径）

| 原则 | 落地位置 | 说明 |
|---|---|---|
| 本金 100 USDT | `config/risk.yaml account.initial_equity_usdt`、CLI `--equity` 默认 100 | 小资金，成本占比敏感 |
| 固定 10× 全程杠杆 | `caps.global_max_leverage=10` → `BacktestConfig.global_max_leverage` | 定仓公式 `margin = notional / 10`，不是动态杠杆 |
| 1 成试仓 + 最多 2 次 ×2 成加仓 | `entry_frac=0.10 / add_frac=0.20 / add_max_count=2` | 名义敞口上限 = `equity × 10` |
| 全 taker 成交 | `costs.taker_fee_rate=0.0005`，入场/出场双向计费 | 不假设 maker 返佣 |
| 保守口径优先 | bar 极值止损、SL/TP 冲突取 SL、每笔成交取不利滑点 | 宁可低估收益 |
| 具体下单由人完成 | 系统只产信号（飞书卡片），不自动下单 | paper trading 只做模拟账本 |

### 1.2 分层与模块清单

```
core/        contracts.py（数据契约：Bar/FundingBar/SymbolSpec/Signal/Action/FillReason/Position/…）
             timeframe.py（interval 解析、floor_ms、grid、to_ms/to_dt、source_rank）
data/        binance_rest.py（REST 客户端 + 权重预算 + 封禁语义）
             store.py（SQLite DDL / upsert / 水位线 / dataset_version 指纹）
             backfill.py（分页回填 + assert_complete）
             quality.py（htf_consistency 跨周期一致性）
             service.py（MarketDataService：策略与回测的唯一只读入口）
             wiring.py（按 settings 构造 client）
risk/        exchange_meta.py（exchangeInfo → SymbolSpec 解析与序列化）
             sizing.py（compute_risk_size：唯一定仓实现 + 强平价近似）
features/    indicators.py（ema/atr/macd/rsi/stoch_rsi/bollinger/adx）
             builder.py（38 维 ML 特征帧 + triple-barrier 标签，训练/推理共用）
backtest/    position.py（**回测与 paper 共用的执行层**：纯触发函数 + PositionLedger + BarPipeline + HtfCursor）
             precompute.py（5m / 高周期指标预计算 + row_to_bar + warmup_bars）
             engine.py（只做数据准备与驱动，撮合全部委托给 BarPipeline）
             reports.py（equity.csv / trades.csv / metrics.json / report.html + 多周期 K 线）
strategy/    base.py（BaseStrategy 契约）  registry.py（@register 自动发现）
             rules/rule_v1..v4.py
             ml/base.py（MLBaseStrategy：折模型加载、特征帧、edge 推理）  ml/lgbm_v1.py
train/       walkforward.py（plan_folds / fold_masks / 训练 / edge 扫描 / 折报告）
paper/       config.py（PaperConfig ≡ BacktestConfig + GuardThresholds + PaperRuntime）
             state.py（PaperState：账本↔SQLite 双向还原 + 确定性主键）
             guards.py（EntryGuards：`[paper]` 事前闸门，只拦新开仓）
             runner.py（一个 (symbol,strategy) 的一轮：观测→实撮→落库→推送）
             service.py（PaperService：单进程多账户调度 + paper_status）
notify/      feishu.py（webhook + 签名 + 限流退避）  cards.py（入场/平仓/告警卡片 + exit_label）
infra/       settings.py（YAML 分层加载 + validate_config）  log.py
cli.py       typer 应用：backfill / check-data / exchange-info / test-card / dump-card / backtest / train
             + `paper run|once|status` 子命令组
scripts/paper.sh            start|stop|restart|status|once|logs（PID 文件 + 预检 + TERM 优雅退出）
deploy/paper-trading.service  systemd 模板（Restart=always；无 WatchdogSec，进程不发 sd_notify）
```

### 1.3 实现状态

**已实现并实测**：REST 数据层与断点续传、SQLite 落库与质量闸门、5m/1h/4h 多周期事件驱动回测、固定风险定仓、金字塔加仓、移动止损、**TP1 分批止盈 + 保本移动 + 强平价触发平仓**、飞书卡片、4 个规则策略 + LightGBM walk-forward 训练与推理、HTML 多周期 K 线报告、**模拟交易/信号推送服务（§15）与 213 项离线单测**、`scripts/paper.sh` + systemd 模板（本机实跑验证过 start/once/status/stop，全程 `--no-push`）。

**未实现（明确不做或延后）**：WebSocket/1m 数据、真实下单（由人手动执行）、多策略仓位共存与跨策略合并保证金（用户明确"先不管"）、多品种并行回测（BTCUSDT 未回填）、`backtest.intra_bar_path=exact_1m`（需要 1m 数据）、Web UI。

---

## 2. 运行环境与依赖

- Python 由 **uv** 管理，复用仓库根 `.venv`（禁止系统 pip、禁止新建环境）；实测解释器 3.13.1，`requires-python >=3.11`
- 运行时依赖：`httpx`、`pandas 3.0.6`、`numpy 2.5.3`、`pyyaml`、`python-dotenv`、`typer`、`plotly 7.1`、`pyecharts 2.1`、`lightgbm 4.7.0`、`scikit-learn 1.9`
- 开发依赖组：`pytest`、`hypothesis`、`ruff`
- 入口：`[project.scripts] quant = "quant.cli:app"` → `uv run quant <cmd>`
- 目录约定：`data/`（SQLite，gitignore）、`reports/`（回测产物，gitignore）、`logs/`（`quant.log` + `paper.out` + 临时校验脚本，gitignore）、`run/`（`paper.pid`，gitignore）、`models/`（折模型，**当前未被 gitignore 覆盖**，9.0 MB）

### 2.1 网络出网规则（实测硬约束）

| 目标 | 出网方式 | 原因 |
|---|---|---|
| `fapi.binance.com` | 必须走代理 `http://127.0.0.1:7897`（`QUANT_PROXY_URL` 环境变量优先于 `settings.yaml`） | 境内直连境外域名不通；`httpx` 显式传 `proxies` |
| 飞书 webhook | **直连**，不走代理 | 飞书域名境内可达；走 Clash 反而可能被路由到境外节点增加失败率 |

---

## 3. 配置体系与启动闸门

配置分四层，任何 CLI 命令都先 `_boot()` 加载并校验，**校验不通过直接退出码 2**（不是警告）：

| 文件 | 作用 | 关键点 |
|---|---|---|
| `config/settings.yaml` | 网络 / 路径 / 日志 / 回测全局默认 | `weight_budget_per_min=1700`（交易所 2400/min，留余量）、`max_retries=5`、`backtest.default_start=2024-01-01` |
| `config/symbols.yaml` | 品种、周期、起始日、默认策略 | `ETHUSDT enabled`，`base_interval=5m`，`intervals=[5m,1h,4h]`；`BTCUSDT enabled=false` |
| `config/risk.yaml` | 风控与成本参数 | 见 §3.2 |
| `config/strategies/<name>.yaml` | 策略声明文件 | **回测不加载其参数**，只作为 `validate_config` 的交叉校验输入与口径记录；运行时参数来自策略 `config_cls` 默认值 + CLI 选项 |
| `config/notify.yaml` | 通知渠道开关与 env 名 | `feishu.enabled`、`webhook_url_env`、`secret_env`、限流参数 |
| `.env` | 密钥 | `FEISHU_WEBHOOK_URL`、`FEISHU_SECRET`（本仓库未启用签名）、`QUANT_PROXY_URL`；`.gitignore` 已忽略 |

### 3.1 `validate_config` 校验项（`infra/settings.py:127`）

- `risk.yaml` 必须有 `margin` 段与 `caps.global_max_leverage`
- 每个 enabled 品种：`intervals` 非空；`base_interval ∈ intervals`（否则该周期根本不会被回填）；`stops.atr_interval ∈ intervals`（否则风控静默按错误 ATR 尺度算止损）
- `symbols.leverage_default` 不得超过 `caps.global_max_leverage`，也不得超过 `margin.max_leverage_default`（exchangeInfo 不可用时的兜底杠杆）
- `strategies` 引用的声明文件必须存在，且其 `htf ⊆ symbols.intervals`、`base_interval` 与品种一致
- `sizing.method=fixed_risk` 必须有 `risk_pct`
- `stops.tp_atr_mult > sl_atr_mult`（否则盈亏比结构性为负）

### 3.2 键的生效范围（回测与 paper 读的不是同一套）

| 键 | 回测 | paper |
|---|---|---|
| `caps.global_max_leverage` | 读（`--lev` 默认值） | 读 |
| `caps.max_notional_usdt`、`caps.max_margin_per_symbol_pct`、`caps.max_new_entries_per_day` | **不生效**（回测要的是策略裸表现） | **拦截新开仓**（`paper/guards.py`），不强平已有持仓 |
| `guard.*`（日亏停、回撤暂停、连亏冷却、数据新鲜度） | **不生效** | 同上；冷却到期时刻落库 `paper_accounts.pause_until_ms` |
| `stops.tp1_r` / `tp1_close_frac` / `breakeven_after_r` | 生效（`<=0` 关闭） | 生效（同一执行层） |
| `costs.include_funding` | 生效（**默认 false**） | 生效（默认 false） |
| `network.resync_grace_ms` | 仅文档（REST 覆盖 WS 尾 bar 的等待窗口，WS 未实现）；paper 的 `_pull` 用它判断"最新收盘 bar 该不该已落库" |
| `backtest.intra_bar_path` | 仅文档：`bar_hl`（当前口径）/ `exact_1m`（1m 数据回填后才可切换） | 同 |
| `paper.replay_days` / `paper.poll_sec` | 回测不读 | paper 调度参数 |
| `feishu.resend_cooldown_min` | 不读 | 补发退避（`_resend_unsent`） |

`PaperConfig` 的字段名与默认值必须和 `BacktestConfig` 完全一致（`start`/`end` 除外：纸面账户只有断点，没有回测窗口）。这条约束由 `tests/test_paper_runner.py::TestConfigParity` 逐字段断言守住 —— 两份配置只要漂移一个名字，paper 就会静默用错止损倍数。

配套的 `_warn_yaml_drift()`：`risk.yaml` 声称的口径与实际生效值不一致时 WARNING（回测 CLI 从不读 `costs.*`，改 YAML 会让两边静默分叉，而可比性是纸面账户唯一的存在意义）。

---

## 4. 数据层

### 4.1 REST 客户端（`data/binance_rest.py`）

端点：`/fapi/v1/time`（`sync_time`，偏差 >1800s 才提示）、`/fapi/v1/exchangeInfo`、`/fapi/v1/symbol` 变体（本地筛选，USDT-M 会忽略 `symbol` 参数返回全量）、`/fapi/v1/klines`（`limit` 默认 1500）、`/fapi/v1/fundingRate`（`limit` 1000）。

**权重表（实测交易所文档口径，分档下界含）**：

```python
KLINE_WEIGHT_TIERS = ((1000, 10), (500, 5), (100, 2), (1, 1))
# limit>=1000 → 10；>=500 → 5；>=100 → 2；否则 1
```

原实现把 1500 记成 5、1000 记成 4、`limit=100` 只算 1，会系统性低估权重，长历史回填容易把 IP 推进 429/418（封禁可达数小时）。

**预算与错误语义**：

- 分钟滚动权重预算 `_WeightBudget(budget_per_min)`：`acquire(w)` 超限即 sleep 到窗口滚动；每次响应后用 `x-mbx-used-weight-1m` **校正**本地估计（`observe`），避免自算漂移
- `HTTP >= 400` **一律失败**，即使响应体不是 JSON（旧实现按 body 解析失败当作成功）；错误消息保留 `-1100` 类业务码，`BinanceRestError.status/.code` 可供上层判断
- `403 / 418 / 451` → `AccessDenied`，**不重试**（多为代理出口节点被 Cloudflare 拦截或 IP 封禁，换节点而非重试）
- `429` → `RateLimited(retry_after=响应头 retry-after)`，按服务端指示休眠
- `5xx` → 指数退避重试，`max_retries` 用尽后 `raise last`（不静默返回部分数据）
- 未收盘 bar 判定：交易所 `closeTime = openTime + interval - 1ms`，必须 `closeTime < now_ms` 才算真正收盘，否则末毫秒仍在成形的 bar 会被写成已收盘

### 4.2 落库（`data/store.py`）

```sql
bars(symbol, time_interval, ts_open_ms, ts_close_ms, open, high, low, close,
     volume, quote_volume, taker_buy_volume, trades, is_closed, src, prio,
     PRIMARY KEY(symbol, time_interval, ts_open_ms))
funding(symbol, funded_at_ms, funding_rate, mark_price, rate_type, interval_h, src,
     PRIMARY KEY(symbol, funded_at_ms))
meta(key PRIMARY KEY, value)     -- 交易规则缓存：key = "spec:{symbol}"
```

另有 5 张 paper 表（`paper_accounts / paper_positions / paper_signals / paper_trades / paper_checkpoints`），列清单见 §15.4。建表用 `CREATE TABLE IF NOT EXISTS`，老库补列走 `PAPER_MIGRATIONS`（`paper_accounts` 的 `peak_equity`、`pause_until_ms`）+ `ALTER TABLE`，重复打开同一个库不报错。

- WAL 模式；连接级复用；`upsert_bars` 的守卫是 `ON CONFLICT ... WHERE excluded.prio >= bars.prio`
- `prio = source_rank(is_closed, src) = 2*is_closed + (src=="rest")`，因此**已收盘 REST 覆盖未收盘 WS**，同优先级允许重写（幂等重放），更低优先级永远不能覆盖更高优先级
- 写入条数用 `total_changes()` 差值统计，不是 `len(bars)`：被 prio 守卫拒绝的行不计入（回归测试覆盖：3 新→3；同 prio 重写→3；未收盘 WS→0；混合→1）
- `spec:{symbol}` 的 JSON 是**双层编码**（`meta_set` 对值 `json.dumps`，而 `spec_put` 传入的已是 JSON 字符串），`spec_raw` 经 `meta_get` 解外层后返回内层文本；读写成对，不是缺陷
- `dataset_version(symbol, intervals, include_funding)`：对各周期的行数与范围取 sha1 前 12 位，作为数据集指纹（`train_meta.json` 里也记录它）

### 4.3 回填（`data/backfill.py`）

- 页大小 `KLINE_PAGE=1500`、`FUNDING_PAGE=1000`；`SPEC_TTL_SEC=24h`
- 水位线续传：仅当库内最早 bar `<= 请求起点` 时才从 `watermark+step` 续拉，否则说明在补更早的历史，必须从头拉并靠 prio 守卫去重
- **右边界裁剪**：`klines` 按 `startTime+limit` 返回，可能越过本次 `--end`；越界行必须丢弃，否则同一个 `--end` 会得到不同数据集，破坏可复现性
- 未收盘行丢弃并计入 `unclosed_dropped`
- `report.gaps` 把全库口径的 `find_gaps` **裁到本次请求范围**再统计：混用会让 coverage 被范围外的历史缺口吃成负数
- `refresh_spec(force=)`：24h 内用缓存，`--force-spec` 忽略缓存重取

### 4.4 质量闸门（`backfill.assert_complete` + `data/quality.htf_consistency`）

- `assert_complete(store, symbol, interval, start_ms, end_ms)`：按网格 `grid()` 枚举期望的 `ts_open_ms` 集合，与库中已收盘 bar 比对，返回 `expected/present/missing/coverage/missing_head[:5]`
- `htf_consistency(store, symbol, base_interval, intervals, start_ms)`：用细周期聚合出高周期 OHLCV，与交易所返回的高周期 bar 比对。**首尾不完整桶被剔除**（`compared` 只统计完整桶），返回 `compared/mismatched/max_close_dev/max_open_dev/worst_at`
  - 实测偏差来源：多周期特征直接读**交易所值**，若与本地聚合值不一致，说明该周期数据本身有修复需要，报表会标注 `⚠ 异常 bar`
- 回测入口的数据闸门只校验用户请求的 `[start, end)`（预热前缀允许缺），缺柱默认**中止**并提示 `--allow-gaps`

### 4.5 只读入口（`data/service.py`）

`MarketDataService` 是策略与回测唯一的数据入口，**只读本地 SQLite，永不触网**：回测要能在无网络下复现同一数据集，因此连交易规则都只读回填时的缓存；`spec()` 在缓存超过 24h 时告警（实测 `ETHUSDT 已过期 4.3 天`）但仍按缓存口径计算，以保证可复现。缺数据抛 `DataNotReady`，提示执行 `quant backfill`。

### 4.6 交易规则解析（`risk/exchange_meta.py`）

- `tickSize/stepSize/minQty/MIN_NOTIONAL` 来自 `filters`；**`MIN_NOTIONAL` 才是成交额门槛**（`MARKET_LOT_SIZE` 只有 max/minQty，用它兜底等于没有兜底）
- 交易所对不适用字段返回空串（如 `liquidationFee=""`），`_dec_or` 用默认值兜底，不能直接把 `""` 喂给 `Decimal`
- 精度：`pricePrecision` 是"最多几位"的上限声明，`tickSize` 位数可以更少（BTCUSDT：`pricePrecision=2` 而 `tickSize=0.1`），因此**只校验下限**；小于 filter 步长直接报错
- 杠杆上限来自 `requiredMarginPercent`（ETHUSDT 实测 5% → 20×）；维持保证金率 `maintMarginPercent=2.5%` → 0.025
- 逐档 bracket 在 `/fapi/v1/leverageBracket`（需签名，MVP 无 Key），所以维持保证金率取的是**最高档口径**，比实际第 1 档更大 → 强平价更近、仓位更小，属安全侧偏差

### 4.7 实测库存（2026-10-07，`data/quant.db` 55 MB）

| 品种 | 周期 | 行数 | 已收盘 | 范围（UTC） |
|---|---|---|---|---|
| ETHUSDT | 5m | 291,015 | 291,015 | 2024-01-01 00:00 → 2026-10-07 11:10 |
| ETHUSDT | 15m | 96,517 | 96,517 | 2024-01-01 00:00 → 2026-10-02 09:00 |
| ETHUSDT | 1h | 24,251 | 24,251 | 2024-01-01 00:00 → 2026-10-07 10:00 |
| ETHUSDT | 4h | 6,062 | 6,062 | 2024-01-01 00:00 → 2026-10-07 04:00 |
| ETHUSDT | 1d | 1,005 | 1,005 | 2024-01-01 00:00 → 2026-10-01 00:00 |
| ETHUSDT | funding | 3,032 | — | 2024-01-01 00:00 → 2026-10-07 08:00 |

paper 侧（同一库）：`paper_accounts` 1 行、`paper_checkpoints` 1 行、`paper_signals` 4 行（全是演练期告警）、`paper_positions` 与 `paper_trades` 各 0 行 —— 本机只跑到"首轮只观察"，还没有真实成交。

`src` 全为 `rest`（WS 未实现）。缓存规则：`tick=0.01 step=0.001 min_qty=0.001 min_notional=20.0 maint=0.025 max_lev=20 liq_fee=0.0125 fetched_at=2026-10-02T09:23:57Z`。

---

## 5. 特征层

### 5.1 预计算列（`backtest/precompute.py`，回测与 paper 共用）

- 5m（`precompute(df, swing_window=12)`，一次性 O(n)）：`ema_9/12/20/50/60`、`atr`、`macd/macd_signal/macd_hist/prev_macd_hist`、`prev_close/prev_ema_20`、`rsi_14`、`srsi_k/srsi_d/prev_srsi_k`、`bb_mid/upper/lower`、`adx_14`、`atr_median`、`vol_median`、`swing_high/swing_low`（`rolling(swing_window).max().shift(1)`，**不含当前 bar**）
- 高周期（`precompute_htf(df)`）：`ema_12/20/60`、`macd_hist/prev_macd_hist`、`srsi_k/d/prev_srsi_k`、`atr`、`rsi_14`、`adx_14/prev_adx_14/adx_prev_3`、`ema_20_prev_6`（rule_v4 算斜率）、`atr_median_100`（波动扩张）、`donchian_high_24/low_24`
- `row_to_bar(row, symbol, interval)` 把指标帧的一行还原成 `Bar` 契约对象；`warmup_bars()` 返回 3120（见 §7.2），paper 用它换算加载窗口的前置长度

所有滚动量右移一位或按可见时刻对齐，无前视。两个函数**原地改传入的 DataFrame**，调用方需先 `sort_values("ts_open_ms").reset_index(drop=True)`（paper 的 `_load_frames` 返回的帧顺序不保证，见 §13.4 第 1 条修复）。

### 5.2 ML 特征（`features/builder.py`，固定 38 维，顺序即模型输入顺序）

| 组 | 列 |
|---|---|
| 收益 | `r_1 r_3 r_6 r_12 r_24`（对数收益） |
| 均线结构 | `ema20_dist ema_span ema20_slope` |
| 动量 | `macd_hist_n macd_hist_d1` |
| 趋势强度 | `adx_14 adx_d1 adx_d3` |
| 通道位置 | `don_pos don_break`（`shift(1)`，不含当前 bar） |
| 摆动 | `rsi_14 srsi_k srsi_d srsi_diff` |
| 波动 | `atr_pct atr_ratio rv_24 bb_width` |
| 量能 | `vol_ratio taker_ratio qvol_z` |
| 日历 | `hour_sin hour_cos dayofweek` |
| 4h 背景 | `h4_ema20_dist h4_macd_hist_n h4_adx_14 h4_atr_pct h4_basis` |
| 资金费 | `funding_rate funding_annual mark_basis hrs_to_funding` |

**可见性规则**（全部按"已收盘才可见"）：
- 4h：`merge_asof` 到 `ts_visible_ms = h4.ts_open + 4h`，`direction=backward, allow_exact_matches=True` → 4h bar 只有整根收盘后才对 1h 可见
- funding：按 `ts_close_ms` 向前取**已发生的最近一次结算**；`hrs_to_funding = interval - mod(elapsed, interval)`（数据集末尾与 paper 实时场景最后一次结算总在过去，直接用 `next_funding_ms` 会算出负数）
- 所有列除以 `atr` 归一，预热期保留 NaN 不填充（LightGBM 原生处理缺失）
- `feature_columns()` 缺列直接 `KeyError`；`feature_spec_hash = sha1("|".join(cols))[:12]`，推理端与折模型 hash 不一致则**抛错要求重训**（防 train/serve skew）

### 5.3 标签（`build_labels`，triple-barrier 三分类）

- 上障碍 `close + k_up*ATR`（`k_up=3.0`）、下障碍 `close - k_dn*ATR`（`k_dn=2.0`）、时间障碍 `horizon=24` 根 1h
- 循环内**先查下障碍** → 同一根 bar 双触发时判 `-1`，与引擎 `sl_first_on_conflict=True` 一致
- 到期未触障碍：位移需同时超过 `flat_atr_mult*ATR`（1.0）**和**往返成本 `ROUND_TRIP_COST_FRAC = 2*0.0005 + 2*0.0002 = 0.0014`，否则记 `0`。只按成本判平会让 flat 类几乎为空（实测 2%），模型学不到"不该开仓"
- 末尾 `horizon` 根无法构造完整标签 → NaN（训练时 `dropna`）
- 类别映射：`-1→0, 0→1, 1→2`（`LABEL_TO_CLASS`），`CLASS_DOWN=0 / CLASS_UP=2`

---

## 6. 风控与定仓（`risk/sizing.py`）

`compute_risk_size` 是唯一实现，两条 qty 来源：

1. 调用方传 `qty`（金字塔路径：`target_qty = equity × frac × size_mult × lev / price_ref`）→ 直接采用，**风险额在 qty 定稿后回算**（不照抄 `risk_pct × equity`）
2. 未传 → 固定风险比例：`qty = equity × risk_pct / (sl_atr_mult × ATR)`

处理后顺序：`_floor_to(qty, step_size)` → `notional > equity × lev` 则按上限裁 qty → `qty >= min_qty` → `notional >= min_notional` → `margin = notional / global_max_leverage`。

**拒绝条件（写入 `RiskDecision.reject_reasons`，逐条对应一次真实失败）**：

| 条件 | 消息 |
|---|---|
| `equity <= 0` | 权益已归零，无法开仓 |
| `global_max_leverage <= 0` | 配置错误 |
| `atr <= 0` 且走风险预算路径 | ATR 不可用，无法按风险定仓 |
| `qty < min_qty`（按 step 向下取整后） | 低于最小下单量 |
| `notional < min_notional` | 低于最小成交额（ETHUSDT 实测 20 USDT） |

保证金口径注释是硬约束：不能写成 `margin = notional / (notional / equity)`（恒等于 equity），那会让每笔成交把全部权益记成保证金、`ret_pct` 退化成对权益的收益率、现金账户被抽干。

**强平价近似**（逐仓，不计资金费与持仓期收益）：
`long: entry × (1 − 1/lev + maint + taker_fee)`，`short: entry × (1 + 1/lev − maint − taker_fee)`。10× / maint 2.5% → 多头约在 −7.45% 价格处强平。**强平触发已实现**（`FillReason.LIQ`，见 §7.5b）。

`decide_position(...)` 是回测与 paper 唯一的 `RiskDecision` 构造入口（卡片上的数量与账本里的数量必须同源），`est_liq_price` 随决策一起返回、试仓时写进 `Position.liq_price`。**加仓后不重算 `liq_price`** —— 加权均价变化会让真实强平价移动，这条近似记在 §14 第 2 条。

---

## 7. 执行层与回测引擎（`backtest/position.py` + `backtest/engine.py`）

### 7.1 配置（`BacktestConfig` 默认值）

```
symbol, base_interval=5m, start, end, initial_equity,
risk_pct=0.01, sl_atr_mult=1.5, tp_atr_mult=3.0, time_stop_bars=96,
allow_short=True, global_max_leverage=10.0, taker_fee=0.0005, slippage_bps=2.0,
include_funding=False, sl_first_on_conflict=True, strategy_name="rule_v1",
pyramid_enabled=True, entry_frac=0.10, add_frac=0.20, add_max_count=2,
add_trigger_r=0.3, trail_trigger_r=1.0, trail_atr_mult=1.0, swing_lookback=12,
tp1_r=1.0, tp1_close_frac=0.5, breakeven_after_r=1.0
```

策略可通过 `engine_defaults()` 覆盖出场/节奏默认值（CLI 显式传参优先于策略声明）。`PaperConfig` 与它同名同默认值，只有 `start`/`end` 换成断点，所以同一套 `position.py` 撮合两条链路（§15.2）。

### 7.2 时间线（每根 5m bar 一次迭代，`BarPipeline.on_bar`）

```
i.   HtfCursor.advance(可见时刻 = bar.ts_close + 1) → htf dict（指针只前进不回退）
ii.  equity_open = equity                       ← 本根开盘前已知的净值，定仓只用它
iii. snapshot(bar)：MTM 净值快照，记入 equity_curve
iv.  accrue_funding(可见时刻, close)             ← 指针每根都推进，含预热期与空仓期
v.   mode != "live" → 到此为止（warmup 不撮合、不调策略）
vi.  执行上一根 bar 收盘产生的信号：在本根 bar 开盘成交（pending）
vii. update_trailing(pos, cfg, bar)
viii.settle_exits(bar) = settle_price_exits + advance_time_stop：
        SL（bar 极值） → 若未触发则 LIQ（强平价） → TP1 部分止盈 + 保本移动
        → 保本后新止损在同根 bar 立即参与判定 → TP → 时间止损
ix.  signal = signal_fn(bar, row, htf, ts_close) → pending
```

`mode` 三档：`live`（完整序列）、`warmup`（只做净值快照与资金费记账，回测窗口最前面那段）、`observe`（只调策略、丢弃返回值、账本一行不写 —— paper 重启用它把策略内部状态推平到断点，绝不能出现在回测里）。

回测里 `warmup` 的判定是 `i < warmup_bars()`；`warmup = max(4h/5m × 65, 1h/5m × 35, 30) = max(48×65=3120, 12×35=420, 30) = 3120` 根 5m ≈ 10.8 天：规则策略的加载窗口不带前置历史，起点处 4h EMA60 / ADX14 仍是 NaN，必须留预热段。ML 策略 `warmup()=0`，改用 `history_lookback()`（90 天）让引擎多加载前缀。

循环结束若仍持仓 → `FillReason.FLAT` 平仓（`pipeline.finalize(bar)`），并**回写最后一个净值点**（循环内快照是持仓 MTM，不含这笔的滑点/平仓费/资金费，否则尾部差一个量导致恒等式破）。

### 7.3 成交与成本口径（全部取不利方向）

| 场景 | 成交价 | 时间戳 |
|---|---|---|
| 入场（试仓/加仓） | `bar.open × (1 ± slippage_bps)` | 该 5m bar 的 `ts_close` |
| 信号平仓 / 反手 | `bar.open × (1 ∓ slip)` | 同上 |
| **止损（B 模式）** | `bar.low`（多）或 `bar.high`（空）再减/加 slip | **bar 中点** `bar_mid` |
| 止盈 | `pos.tp_price ∓ slip` | bar 中点 |
| **TP1 部分止盈** | `avg_entry ± sl_atr_mult × entry_atr × tp1_r`，再取不利滑点 | bar 中点 |
| **强平** | `pos.liq_price ∓ slip`（逐仓被打穿，没有讨价还价余地） | bar 中点 |
| 时间止损 / 期末平仓 | `bar.close ∓ slip` | bar 收盘 |

- 止损按 bar 极值成交是"最坏情况"假设：一根 bar 内无法证明价格先到哪侧，按 TP 成交会把本该止损的交易记成盈利，属系统性乐观偏差
- 同一根 bar 双触发：默认（`sl_first_on_conflict=True`）保守取 SL，与标签口径一致；配置为 False 时取 TP
- 止损/止盈价基于**实际成交价**（`bar.open ± slip`）而非 `signal.price_ref`（=bar.close），避免 open-close gap 立刻击穿 SL
- 策略可自带决策周期 ATR（`signal.meta["atr"]`，如 rule_v4 / lgbm_v1 用 1h ATR），否则回落 5m ATR；5m ATR 约 0.17%，1h 趋势仓用它会因噪音被扫掉
- `size_mult` 钳在 `[0.2, 1.0]`，高周期背景只允许减仓不允许超额

### 7.4 仓位与加仓

- 试仓：新建 `Position`，`cash -= margin + 开仓腿手续费`，手续费记在 `add_levels[0].fee_usdt`
- 加仓（`should_add`）：`add_count < add_max_count` ∧ `unreal_r >= add_trigger_r` ∧ 结构突破（多头 `close > swing_high`，空头 `close < swing_low`，swing 已 `shift(1)`）；成功后加权均价、qty、margin、`add_levels` 累加，`sync_stop_after_add` 把 SL 上移到 `avg_entry ± sl_atr_mult × entry_atr`（只朝有利方向）
- 移动止损（`update_trailing`）：极值 `trailing_peak_price` 每根更新，`unreal_r >= trail_trigger_r` 激活，激活后 `stop_loss = max/min(prev, peak ∓ trail_atr_mult × entry_atr)`
- 反手：在同一根 bar 开盘先平对向仓（`eq = cash`），再开新仓
- paper 专属的一处差别：开/加仓在撮合前多一道 `ledger.approve_fn`（`EntryGuards.vetoes`）追加否决理由；回测把它留 `None`，所以**逐笔成交序列与同参数回测逐位相同**（`tests/test_paper_runner.py::TestBacktestParity` 锁住）

### 7.5 资金费（机制保留，默认**不计入净值**）

`include_funding` 默认 `False`：实测 ETHUSDT 2026-04~10 的资金费在 10× 下只占保证金约 0.24%/3 天，而往返 taker 手续费是保证金的 1%（名义万 5 × 2 / 杠杆 10）。量级太小、又要额外维护一根结算指针，用户决定**回测与 paper 都把它从成本模型里去掉**，避免引入无关变量。资金费数据照常入库、`_load_frames` 照旧加载（ML 特征 `funding_rate/funding_annual/mark_basis/hrs_to_funding` 仍要用）。

开关打开时的计提实现（三条不变量，回归测试覆盖）：

- 指针 `_funding_ptr` 每根 bar 推进（**包括空仓与预热期**，否则空仓期间的结算会被算进之后才开的仓位）；`seek_funding(None)` 表示首轮没有断点，指针归 0（**不能**把 `None` 喂给 `searchsorted`，见 §13.4 第 3 条）
- `cost = qty × mark_price(优先) 或 close × rate`，多头为正（付费）、空头为负（收费），费率非有限或为 0 跳过
- 只累计到 `funding_since_entry` 与 `pos.funding_paid`，**现金在平仓时经 `net_cash` 一次性扣除**，避免持仓期与平仓时重复扣款

### 7.5b 分批止盈、保本移动与强平触发

同一个 `settle_price_exits(bar)` 返回**事件列表**：一根 bar 里可以既吃到 1R 的部分止盈、又吃到 3.5R 的全平。判定顺序（保守优先，逐条有测试）：

1. `exit_reason`：SL 与 TP 双触发时按 `sl_first_on_conflict` 定序，默认取 SL → 全平，**后续环节不再评估**
2. 未触发任何价位时 `liq_hit`：`bar.low ≤ liq_price`（多）或 `bar.high ≥ liq_price`（空）→ `FillReason.LIQ` 全平
3. `settle_tp1`：`tp1_r > 0` 且未 `tp1_done` 且价格越过 `avg_entry ± sl_atr_mult × entry_atr × tp1_r` → 先置 `tp1_done`、再 `apply_breakeven` 把止损抬到加权开仓价，然后按 `step_size` 向下取整平掉 `tp1_close_frac`，记 `FillReason.TP1`
   - 余量或平仓量低于 `min_qty` 时**只标记 `tp1_done` 不重试**（100U 小本金的试仓阶段经常出现），保本移动照样生效
4. 保本后的新止损在**同一根 bar** 立即参与判定（与移动止损收紧后按 bar 极值判触发是同一套保守约定）
5. TP 命中 → 全平

`tp1_r<=0` 关闭分批止盈、`breakeven_after_r<=0` 关闭保本移动，两者回测与 paper 都读同一个字段名。

**成本对照实测**（ETHUSDT 2026-04→10，100U，同一策略 rule_v4，只改 TP1 开关）：

| 口径 | 收益 | CAGR | maxDD | 笔数 | 胜率 | 盈亏比 | PF |
|---|---|---|---|---|---|---|---|
| TP1 开（默认 1R 平一半 + 保本） | **+4.79%** | +9.77% | −12.44% | 64 | 64.1% | 0.62 | 1.10 |
| TP1 关（`--tp1-r 0 --breakeven-r 0`，已复跑验证） | **+12.42%** | +26.31% | −12.35% | 45 | 51.1% | 1.20 | 1.25 |

TP1 把胜率从 51.1% 抬到 64.1%，代价是盈亏比 1.20→0.62 —— 赢的那一半仓位被提前兑现，趋势策略的大 R 尾部（3.5×ATR 的 TP）被系统性削掉。**rule_v4 这类趋势策略在半年窗口上是"让利润奔跑"更划算**；`config/strategies/rule_v4.yaml` 仍保持默认 1.0（未改成 `tp1_r=0`），因为单窗口结论不足以定政策，等 paper 观察期跑完再决定。

### 7.6 净值与账本恒等式

```
MTM   equity = cash + unreal(相对 avg_entry_price) + pos.margin_usdt − funding_since_entry
平仓  net_cash = pnl − fee_exit − funding_since_entry ;  cash += margin_usdt + net_cash
逐笔  TradeRecord.pnl = net = pnl − fee_entry_paid − fee_exit − funding_since_entry   （往返全成本）
      ret_pct = net / margin_usdt × 100      （分母是保证金，不是名义仓位）
```

不变量（每个 run 都必须成立，回归已验证）：

```
sum(trades.pnl) == final_equity − initial_equity      （容差 ≤1e-3，纯 CSV 4 位小数舍入）
```

浮盈基准必须是 `avg_entry_price`（加仓后用 `entry_price` 会虚增/虚减浮盈）。

- `include_funding=False` 时公式里的 `funding_since_entry` 恒为 0，等式形式不变
- TP1 部分止盈会把**一个仓位拆成多笔流水**（`FillReason.TP1` 一条 + 最终全平一条），所以 `total_trades` 大于"开仓次数"；两条流水各自带自己的 `margin_part`，`ret_pct` 的分母是分摊到这一腿的那部分保证金，恒等式照样成立

### 7.7 绩效指标定义（`_metrics`）

- `total_return = eq[-1]/eq[0] − 1`；`cagr = (eq[-1]/eq[0])^(1/years) − 1`，`years = n_bars × base_ms / (1d × 365)`
- Sharpe/Sortino 用对数收益按 bar 年化：`mean(log)/std(log) × √(bars_per_year)`；Sortino 用下行标准差
- **最大回撤**：`cummax = np.maximum.accumulate(eq)`，`dd_pct = min((eq−cummax)/cummax)`，**`dd_usdt = min(eq − cummax)`**（按峰值到谷值；旧实现用 `eq[0]` 折算，任何"先涨后跌"的曲线都会低估真实亏损额）
- `win_rate`、`payoff_ratio = avg_win/avg_loss`（**无亏损交易时为 `inf`**，与 `profit_factor` 同口径；旧实现用 1.0 兜底会把"平均盈利额（USDT）"当比值显示）、`profit_factor = Σwin/|Σloss|`（无亏损时 `inf`）、`avg_bars_held`
- `BacktestResult` 不再携带 `dataset_version`（数据集指纹由 CLI 直接取 `store.dataset_version()` 传给报表，避免两个来源）

---

## 8. 策略框架与已实现策略

### 8.1 契约（`strategy/base.py` + `registry.py`）

```python
class BaseStrategy(ABC):
    name: str; required_intervals: tuple[str,...]; config_cls: type|None
    @classmethod build(cls, opts: dict) -> Self      # 按 config_cls 字段挑选认领参数
    def on_bar(bar, row, htf, dt) -> Signal|None      # 引擎逐根驱动
    def engine_defaults() -> dict[str,float]          # 出场/节奏默认值（CLI 可覆盖）
    def warmup() -> int ; history_lookback() -> int   # 两种预热机制，规则策略用前者
    def reset()/finalize() ; allow_short -> bool
```

- `@register` 装饰即入表，重复名报错；`get_strategy(name)` 未知时列出可用名
- **CLI 不做策略分发判断**：`--tau/--adx-min/...` 全部塞进 `build(opts)`，由策略自己认领；不引入独立于基类的私有协议
- ML 基类额外要求引擎调用 `bind_history(bars, funding)` + `reset()`（`run()` 内 E7/E8 钩子）

### 8.2 策略一览

| 名称 | 周期 | 入场逻辑（摘要） | 关键默认参数 | `engine_defaults` |
|---|---|---|---|---|
| `rule_v1` | 5m 触发 / 1h 确认 / 4h 过滤 | 4h MACD hist>0 ∧ EMA20>EMA60 → 多头趋势；1h StochRSI_k>20 且上升 → 确认；5m MACD 柱转正或上穿 EMA20 ∧ ATR>median(ATR,20) → 入场 | `confirm_bars=1`，srsi 20/80 | 无（用全局 1.5/3.0/96） |
| `rule_v2` | 5m / 1h（无 4h） | StochRSI 极值反转：1h 方向过滤 + 5m 同向共振 | `srsi 15/65`，`cross_margin=1`，`require_macd_agree` | 无 |
| `rule_v3` | 5m | 4 指标投票（MACD 交叉 / EMA 突破 / StochRSI 反转 / 量能放大），同向数 ≥ `min_signals` 入场 | `min_signals=2`，`vol_mult=1.5` | 无 |
| `rule_v4` | 5m 择时 / 1h 状态机 / 4h 软对齐 | 1h 趋势状态 `RANGING→BUILDING→CONFIRMED→MATURE→BROKEN`，强度门 `ADX14 ≥ adx_min`；4h 只调仓位 `{1.0,0.85,0.70,0}` 从不否决方向；入场口 A=回调顺势（回撤 ≥`pullback_atr×ATR1h` + 5m 三选二），B=结构突破（仅 EXPANDING）；出场用 1h ATR | `adx_min=32`，`pullback_atr=1.5` | `sl=2.0 tp=3.5 time_stop=864 trail=1.5/1.0 add_r=1.0` |
| `lgbm_v1` | 5m 计数 / 1h 决策 | `edge = P(up) − P(down)`，`|edge|>tau` 按符号开仓，`size_mult = clip((|edge|−tau)/(edge_full−tau), 0.2, 1.0)`；出场完全交给引擎（与 rule_v4 同口径以保证可比） | `tau=0.15 edge_full=0.40 cooldown_bars=12 history_days=90` | `sl=2.0 tp=3.0 time_stop=300 trail=1.5/1.0 add_r=1.0` |

`lgbm_v1` 的 TP 取 `3.0×ATR`（标签上障碍的 ~90%）：TP 设在障碍内侧，才不会系统性等到"已越过障碍才平仓"的下沿结果；`time_stop_bars=300`（25h ≈ 24h 时间障碍 + 1 根执行延迟）。

---

## 9. ML 训练与推理

### 9.1 折规划（`train/walkforward.py`）

```
TrainConfig 默认：data_start=2024-01-01, eval_start=2026-04-01, eval_end=2026-10-01,
window_months=12, step_months=1, valid_months=1, horizon=24, embargo_bars=26,
k_up=3.0, k_dn=2.0, flat_atr_mult=1.0, balance_classes=True, num_rounds=2000,
early_stopping_rounds=100, params={multiclass num_class=3, lr=0.03, num_leaves=31,
max_depth=6, min_child_samples=200, feature_fraction=0.7, bagging_fraction=0.8,
lambda_l2=5.0, seed=7}
```

- `plan_folds`：每折 `[train_start, train_end)` → `[valid_start, valid_end)` → `[test_start, test_end)`，**必须无缝衔接**（`prev.test_end == cur.test_start`），末折右端钳到 `eval_end`；窗口不足以容纳 `window+valid+step` 时抛 `ValueError("撑不起")`
- `fold_masks(ts, f, embargo)`：**两端都 purge**

```python
train = (ts >= f.train_start) & (ts < f.train_end - embargo)
valid = (ts >= f.valid_start) & (ts < f.valid_end - embargo)
```

  训练窗最后 `horizon` 根的标签右边界落在验证窗内，不 purge 等于让早停的验证指标被训练样本"预习"过。`embargo_bars=26 > horizon=24` 保证最后一根训练样本的标签结束时刻不越过验证窗起点。
- 类别编码 `−1→0, 0→1, 1→2`；`balance_classes` 用类权重
- 产物：`models/<name>/<symbol>/<tag>/{model.txt, train_meta.json}`（含 `features`、`feature_spec_hash`、`dataset_version`、`label_params`、`n_train/n_valid`、`valid_metrics`）+ 同级 `folds.json`（折间稳定性与 `edge_sweep`）

### 9.2 实测训练结果（ETHUSDT，6 折，样本外 2026-04~09）

| 模型集（同一 `lgbm_v1` 策略；训练 `--out-dir`，回测 `--model-dir` 指向该目录） | logloss 均值（区间） | accuracy 均值（区间） |
|---|---|---|
| `models/lgbm_v1` | 1.0366（0.9890–1.0653） | 0.4179（0.3621–0.4805） |
| `models/lgbm_v2sym` | 0.9929（0.9083–1.0493） | 0.4305（0.3689–0.5139） |

单折样例（`lgbm_v1` 202604 折）：`n_train=8016 n_valid=718 trees=384`，验证 logloss 1.0266 / acc 0.422，逐类 recall：down 0.535 / flat 0.492 / **up 0.218**（模型明显偏空）。`edge_sweep` 在该折 tau=0.15 时 282 个信号（35 多 / 247 空）命中率 0.578，tau=0.40 时 45 信号命中 0.733 —— **高门槛命中率好但信号太少**，与三分类里 up 类 recall 低一致。

### 9.3 推理路径（`strategy/ml/base.py`）

- `bind_history` 后惰性构建特征帧（`_ensure_frame`），1 行 1 个 1h bar，`_row_of` 按 `ts_open_ms` 定位
- `_check_features`：折模型 `feature_spec_hash` 与当前代码不一致 → 抛错要求 `quant train` 或回退代码版本（**不会静默用错特征**）
- `edge_at`：`_fold_for(ts)` 选折（**只在 `[test_start, test_end)` 内生效**），折外 `on_bar` 返回 None 并**按方向各告警一次**（早于首折起点 / 超出末折），避免"以为策略在跑"
- `history_lookback()=90 天`：EWM 类特征的理论值依赖整段前缀，只给最低滚动窗口会让推理端种子短于训练端（实测 span=60 时 200 根种子残留 ~1e-3）；实测 90 天前缀与训练全长前缀差 `< 3e-10`
- 冷却以引擎基础周期（5m 根）计：`n_bars − _last_entry_bar >= cooldown_bars`

### 9.4 样本外结论（诚实记录）

2026-10-07 口径（剔除资金费 + TP1/保本默认开）下 ETHUSDT、100U、`lgbm_v1` 默认参数：

| 窗口 | 收益 | 笔数 | 胜率 | PF | maxDD |
|---|---|---|---|---|---|
| 2026-04-01 → 10-01（样本外半年） | **−20.59%** | 189 | 58.7% | 0.54 | −21.28% |
| 2024-01-01 → 2026-10-01（全历史，首折起点前跳过） | **−22.03%** | 207 | 58.9% | 0.55 | −22.69% |

（旧口径 2026-04→10 是 −23.75% / 136 笔 / 胜率 44.1%，笔数与胜率都受 TP1 拆分影响、收益额受资金费剔除影响，**两列数字不可比**，只能各自做窗口内的相对排序。）

`folds.json` 的 `edge_sweep` 与回测一致表明：**当前 38 维特征在 24h 障碍上没有可交易的样本外 edge**。胜率 58.7% 却仍亏 20%，是因为盈亏比 0.38 —— 赢的小、亏的大，方向命中不等于有正期望。保留该策略在 paper 订阅列表里做**前瞻观察**，不作为主策略。同期 `rule_v4` 在同一口径下 +4.79%（TP1 默认开）/ +12.42%（TP1 关），买入持有约 +27%。

---

## 10. 报告与可视化（`backtest/reports.py`）

产物目录 `reports/{symbol}_{strategy}_{start}_{end}/`：

| 文件 | 内容 |
|---|---|
| `metrics.json` | 全部绩效指标 + 配置快照 + `dataset_version` + `run_at` |
| `equity.csv` | 每根 bar 的净值与回撤（北京时间） |
| `trades.csv` | 逐笔（北京时间、方向、均价、qty、pnl、margin_usdt、ret_pct、fee、reason、r_achieved、bars_held、add_count、initial_qty） |
| `report.html` | plotly 主图（净值/回撤/逐笔 PnL/月度收益 4 traces）+ 核心指标卡 + **指标口径说明表** + 逐笔流水 + 多周期 K 线区块 |
| `kline_15m/1h/4h/1d.html` | pyecharts 蜡烛图 + `markPoint` 买卖点（悬停显示交易详情） |

关键实现约定：

- **北京时间（UTC+8）** 贯穿所有展示与月度分桶；日线桶起点即北京 08:00（=UTC 00:00）；月度分桶首月用窗口起点权益做基准（否则首月盈亏整月消失，见 §13.1 第 34 条）
- K 线源周期按 `_FINER_FIRST = ("1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d")` **从细到粗**取第一个可用序列（源只能向粗降采样，判定写反会让 5m 源分支变死代码）；当前 `base_interval=5m`，故 15m/1h/4h/1d 全部由 5m 聚合（`_agg_bars` 用 pandas `resample`），文件名与标题标注"聚合自 5m"
- 目标周期列表会过滤掉比源周期更细的项（`interval_ms(target) >= src_ms`）
- 单个周期渲染失败只 `_LOG.warning` 并跳过，不让整个 K 线区块凭空消失
- 报表内不写死数值：指标口径表按 `leverage / taker / slippage / base_interval` 参数化注入
- 同一根 bar 内事件多于 `MAX_TIP_EVENTS` 笔时 tooltip 折叠，完整明细在逐笔流水

**浏览器实测结论（2026-10-07，`reports/ETHUSDT_rule_v4_2026-04-01_2026-10-01`）**：
- `kline_4h.html`：1098 根 bar（区间 2026-04-01~09-30），`markPoint` 90 个 = 45 开仓 + 45 平仓（与 `trades.csv` 一致），x 轴标签为北京时间且首 `04-01 08:00`、末 `10-01 04:00`；`tooltip.confine=true`
- `kline_1d.html`：183 根（正好 183 天），90 个标记点
- 像素级校验：隐藏页里 rAF 不跑，需 `setOption({animation:false}) + zr.refreshImmediately()` 才能取到最终帧；强制刷新后阳线 `#27ae60` 5797 px、阴线 `#e74c3c` 6033 px 均正常（未强刷时会误判"红色缺失"）
- `report.html`：plotly 1 图 4 traces（3000/3000/45/6 点）、4 张表、周期切换按钮 `.tf-tab` 4 个可点，点击后 `#kline-frame` 的 `src` 正确切换且 active 类跟随

上述标记点数量对应 **TP1 关闭口径（45 笔）** 的那次产物；默认口径（TP1 开，64 笔 → 一个仓位可能两条流水）会多出 `tp1` 原因的买卖点，未重复做浏览器校验。`REASON_CN`/`REASON_COLOR` 已覆盖 `FillReason` 的全部取值（含 `tp1`/`liq`，`tests/test_report_metrics.py` 逐枚举断言），所以不会出现"图例里少一项"。

---

## 11. CLI 手册（`quant`：7 个顶层命令 + `paper` 子命令组 3 个）

| 命令 | 作用 | 关键参数 |
|---|---|---|
| `backfill` | REST 分页回填 K 线 + funding 到 SQLite（水位线断点续传） | `--symbol --start --intervals 5m,1h --funding/--no-funding --force-spec` |
| `check-data` | 本地数据校验：网格覆盖率、缺口、规则缓存、数据集指纹（**回测前必跑**） | `--symbol --start` |
| `exchange-info` | 查看交易规则（USDT-M 忽略 symbol 参数，本地筛选） | `--symbol --raw` |
| `backtest` | 事件驱动回测并产出报告 | 见下 |
| `train` | walk-forward 训练 `lgbm_v1` 折模型 + 稳定性报告 | `--symbol --data-start --eval-start --eval-end --window-months --step-months --horizon --k-up --k-dn --no-balance --out-dir` |
| `test-card` | 发送样例入场卡片验证飞书通路 | `--approved/--no-approved` |
| `dump-card` | 只打印卡片 payload，不发送 | `--approved` |
| `paper run` | 常驻服务：增量补数据 → 多策略并行推理 → 纸面撮合 → 推卡 | `--symbol --strategy --push/--no-push --pull/--no-pull --log-level` |
| `paper once` | 只跑一轮就退出（部署后手工验链路） | 同上 |
| `paper status` | 只读本地库打印每个纸面账户 | `--json` |

`backtest` 选项：`--symbol --start --end --equity --risk-pct --lev --short/--no-short --strategy`；出场 `--sl-atr --tp-atr --time-stop-bars --trail-atr-mult --trail-trigger-r`；分批止盈与保本 `--tp1-r --tp1-frac --breakeven-r`（`<=0` 关闭）；`include_funding` **没有 CLI 选项**，只能改 `BacktestConfig` 字段默认值（现为 `False`，即不计资金费；`funding` 帧照旧加载给 ML 特征用）；金字塔 `--entry-frac --add-frac --add-max --add-trigger-r --no-pyramid`；数据 `--allow-gaps`；策略专属（缺省一律用策略自身默认值，**不在 CLI 写死以免串给别的策略**）`--confirm-bars --srsi-oversold/--srsi-overbought/--srsi-cross-margin --min-signals --adx-min --pullback-atr --tau --model-dir`。

执行顺序固定为：`_boot`（配置校验）→ 注册表取策略类 → `build(opts)` → `engine_defaults()` 合并 CLI 显式值 → `required_intervals` 决定加载周期（含 `history_lookback` 前缀）→ 数据闸门 → 引擎 → 报表。

**退出码**：`0` 成功；`2` 配置校验失败 / 交易规则未缓存 / 数据缺柱且未 `--allow-gaps` / 未知策略名；其余异常按 Python 默认非零退出。

标准用法（新品种首跑或换窗口）：

```bash
uv run quant backfill --symbol ETHUSDT --intervals 5m,1h,4h   # 需代理可达
uv run quant check-data --symbol ETHUSDT
uv run quant backtest --symbol ETHUSDT --strategy rule_v4 --start 2026-04-01 --end 2026-10-01
```

---

## 12. 测试体系（`uv run pytest -q` → **213 passed**，全部离线）

| 文件 | 数量 | 覆盖点 |
|---|---|---|
| `test_data_layer.py` | 16 | upsert prio 守卫与真实写入条数、回填缺口裁剪、`htf_consistency` 边缘桶剔除、REST `HTTP≥400` 非 dict body、403 不重试、429 按 `retry-after` 休眠、5xx 退避后抛错、`x-mbx-used-weight-1m` 校正预算、`_kline_weight` 分档 |
| `test_engine_ledger.py` | 25 | SL/TP 同 bar 定序（可配置）、止损 bar 极值 + 中点时间、**TP1 分批止盈（含 min_qty 拆不出来只标记）、保本移动、强平 LIQ 触发与成交价**、资金费一次性扣款/空头收/空仓推进/可关闭、`margin=notional/lev`、定仓用 `equity_open`、市价平仓不利滑点、账本恒等式、USD 回撤峰值→谷值、payoff 无亏损时为 inf |
| `test_engine_pyramid.py` | 15 | 试仓/加仓路径、`add_max_count`、结构突破判定、加仓后 SL 上移、移动止损 |
| `test_engine_signal_execution.py` | 13 | pending_signal 下一根开盘成交（防前视）、反手、信号平仓、warmup 边界 |
| `test_features_builder.py` | 13 | 4h/funding 可见性、`hrs_to_funding` 非负、`donchian shift`、缺列报错、hash 稳定、标签 flat 双条件、`sl_first_on_conflict` 一致 |
| `test_paper_guards.py` | 26 | 持续型否决（行情滞后、日亏 3%、北京日界、每日试仓上限只数当天）、冷却型闸门（回撤 15%、连亏 5 笔、盈利打断连亏、冷却期不重复评估、到期重锚峰值、**冷却跨进程重启存活**）、否决缝合点（闸门拦 open/add 但**永不拦出场**、回测账本没有 veto 钩子、保证金/名义上限、保留原有拒绝理由）、老库 `ALTER` 迁移幂等与 `PAPER_MIGRATIONS` 与 SCHEMA 一致 |
| `test_paper_runner.py` | 31 | `PaperConfig` 与 `BacktestConfig` 逐字段对齐、`execution_params()` 等价于"裸 backtest"并叠加策略 `engine_defaults`、`risk.yaml` 漂移告警与"策略覆盖不算漂移"、整轮跳过（无本地数据/5m 缺柱/高周期缺柱/无折模型/折加载失败/样本外窗口外）、首轮只观察、实撮轮写库与出卡、演练模式落库即视为已投递、一个仓位多条平仓流水主键不撞、`add_skipped` 不出卡、投递失败同轮重试与 `resend_cooldown_min` 退避、行情滞后只否决不跳轮、断点那根的 pending 信号不被吃掉、更早的 observe 信号不重复撮合、重跑同一轮零新增、**分段跑 == 连续跑**、**paper 与回测逐笔一致（`warmup_bars→0` 对齐起点 + 账本恒等式）**、`seek_funding(None)` 首轮不炸、`_account_due` 只在有新 bar 或数据滞后时进轮、单账户异常不拖垮其余账户、`paper_status` 汇报断点与推送统计 |
| `test_strategy_build.py` | 5 | `build(opts)` 只认领自家字段、注册表重复名/未知名 |
| `test_strategy_ml.py` | 16 | 折选择、折外告警且不产信号、`feature_spec_hash` drift guard、冷却、`size_mult` 钳位、`bind_history` 缺失时安全返回 None |
| `test_strategy_rule_v4.py` | 31 | 状态机转移、4h 仓位倍数、A/B 两个入场口、BROKEN 平仓信号 |
| `test_train_walkforward.py` | 8 | `plan_folds` 衔接与末折钳位、窗口不足抛错、`fold_masks` 两端 purge 边界（含 `embargo=0` 的反证）、标签编码往返、`edge_from_proba` |
| `test_report_metrics.py` | 14 | 月度收益含首月、逐月复利等于总收益、月末分桶按北京时间、每个 `FillReason` 都有中文标签与颜色（含 `tp1`/`liq`） |

测试全部用 `httpx.MockTransport`、内存 SQLite（`tmp_path` 下的真库文件）与正弦+漂移合成长 K 线，不触网、不依赖 `data/quant.db`、不发飞书（`notifier=None` 走演练分支）。

---

## 13. 本轮整体 Review 的修复清单与回归证据

### 13.1 修复项（按模块）

**回测引擎 / 账本**
1. **前视偏差**：信号在 bar 收盘产生，旧实现在同一根 bar 的 `open` 成交 → 改为 `pending_signal` 在**下一根 bar 开盘**成交
2. **止损成交**：改为 B 模式（多头 `bar.low`、空头 `bar.high`，时间记 bar 中点），同 bar 双触发默认取 SL（`sl_first_on_conflict`），与标签口径一致
3. **保证金语义**：`margin = notional / global_max_leverage`，非金字塔路径不再把整份权益记成保证金；`ret_pct` 因此才是"对保证金的收益率"
4. **定仓权益**：使用 `equity_open`（本根开盘前已知净值），不用成交后的 MTM `equity`
5. **资金费**：只累计到 `funding_since_entry`、平仓时一次性扣现金（修掉重复扣款）；指针每根 bar 推进（修掉空仓期结算被算进后续仓位）；MTM 公式减去未结算负债
6. **账本恒等式**：`TradeRecord.pnl` 统一为往返净成本，期末强制平仓后**回写最后一个净值点**，`sum(pnl) == Δequity` 成立
7. **时间止损/期末平仓计不利滑点**（全 taker 口径统一）
8. **USD 最大回撤**：由 `eq[0]` 折算改为 `min(eq − cummax)`（峰值→谷值）
9. **payoff_ratio**：无亏损交易时返回 `inf`（原 `avg_loss=1.0` 兜底会把平均盈利额当比值显示）
10. 死代码清理：删除 `BacktestResult.dataset_version`（与 CLI 传入的真实指纹重复）、`_precompute_htf` 未使用的 `interval` 参数

**数据层**
11. `KLINE_WEIGHT_TIERS` 修正（1500→10 而非 5、100→2 而非 1），避免系统性低估权重触发 429/418
12. `HTTP≥400` 即使 body 非 JSON 也必须失败；`403/418/451 → AccessDenied` 不重试；`429` 尊重 `retry-after`；`5xx` 退避用尽后抛原始错误
13. 未收盘判定：`closeTime < now_ms` 严格小于（原写法会把末毫秒正在成形的 bar 当已收盘写库）
14. `upsert` 写入条数改用 `total_changes()` 差值（原 `len(bars)` 把被 prio 拒绝的行也算成写入）
15. 回填**右边界裁剪**：越过本次 `--end` 的行丢弃，保证同一窗口可复现
16. `report.gaps` 从全库口径裁到本次请求范围（原混用导致 coverage 可能为负）
17. `htf_consistency` 剔除首尾不完整桶，`compared` 口径明确
18. 权重预算用 `x-mbx-used-weight-1m` 回读校正
19. `exchange_meta`：`MIN_NOTIONAL`（非 `MARKET_LOT_SIZE`）作成交额门槛、空串 `liquidationFee` 走 `_dec_or`、`pricePrecision` 只校验下限
20. `MarketDataService` 严格只读 + `DataNotReady` 明确提示 + 规则过期告警保留缓存口径

**特征 / 训练**
21. 4h 与 funding 的 `merge_asof` 可见性规则修正（整根 4h 收盘才可见；funding 取"已发生的最近一次结算"）
22. `hrs_to_funding` 改为按周期外推，消除末段负值
23. 标签 flat 判定需同时超过 `flat_atr_mult×ATR` 与往返成本（原只按成本 → flat 类仅 2%）
24. `fold_masks` 抽出为可测函数并**两端 purge**（`embargo_bars=26 > horizon=24`）
25. `feature_columns` 缺列抛错 + `feature_spec_hash` drift guard

**策略 / CLI / 报表**
26. 策略参数由 `BaseStrategy.build(opts)` 认领，CLI 不再按策略名写 if/elif；专属选项默认 `None`（写死默认值会把某策略的阈值串给别的策略）
27. 出场参数改由 `engine_defaults()` 提供、CLI 显式值优先
28. ML 策略接入 `bind_history/reset/finalize` 生命周期（`warmup()=0` + `history_lookback()=90d`）
29. K 线源周期选择 `_FINER_FIRST` 从细到粗、目标周期过滤比源更细的项、渲染失败告警而非静默跳过
30. 报表指标口径表参数化（不再写死杠杆/手续费/周期数值）、统一北京时间、月度分桶按北京
31. 数据闸门：`backtest` 默认在缺柱时中止（`--allow-gaps` 可放行），`check-data` 输出覆盖率/缺口/一致性/指纹
32. `_boot` 把 `validate_config` 当错误处理（退出码 2）；`backfill --intervals` 帮助文本示例改为与现行周期一致（`5m,1h`）
33. 飞书通路：webhook 直连不挂代理（走 Clash 反被路由到境外节点）；限流 `FeishuBusy` 退避；`sample_signal_decision` 与 10× 政策对齐（`margin=notional/10`、1h 决策周期、真实 tick/step 量级的 qty 与价位）
34. 月度收益永久漏掉首月：`resample("ME").pct_change().dropna()` 让第一个月恒为 NaN 被丢弃，报表看不到首月盈亏（lgbm_v1 半年报里 2026-04 的 23 笔 / −8.28% 整月消失）。现用窗口起点权益作首月基准，逐月复利回到总收益
35. 平仓原因的中文/颜色在 kline tooltip、图例、逐笔流水三处各写一份且都漏 `signal`：流水表显示英文 "signal"、图例只有写死的三项。改为 `REASON_CN`/`REASON_COLOR` 一份共享，图例按本次真实出现的原因生成

### 13.2 回归证据：账本恒等式

对每个 run 校验 `sum(trades.pnl) − (final − initial)`（阈值 1e-3）：

| run | 修复前 err | 修复后 err |
|---|---|---|
| rule_v1 2024-01→2026-10 | −0.0020 | −0.0005 |
| rule_v1 2026-04→2026-10 | −0.0003 | −0.0004 |
| rule_v1 2026-07→2026-10 | **+406.7915** | −0.0007 |
| rule_v2 2024-01→2026-10 | **+6.1517** | +0.0001 |
| rule_v2 2026-04→2026-10 | −0.0000 | −0.0000 |
| rule_v2 2026-07→2026-10 | **+9.9812** | +0.0001 |
| rule_v3 2024-01→2026-10 | +0.0001 | +0.0007 |
| rule_v3 2026-04→2026-10 | +0.0011 | +0.0004 |
| rule_v3 2026-07→2026-10 | （无基线） | +0.0004 |
| rule_v4 2026-04→2026-10 | −0.0001 | −0.0001 |
| lgbm_v1 2026-04→2026-10 | （无基线） | −0.0004 |

修复前 **5/9 违反**（最大 406.79 USDT，且那一 run 的 `--equity` 默认还是 10000），修复后 **0/11 违反**，残差仅为 CSV 保留 4 位小数带来的舍入。

### 13.3 回归证据：修复前后指标（**不可比**，且"修复后"列已被第 2 轮口径取代，仅作历史留档 → 现行数字见 §16）

| run（100U 起，除注明） | 修复前 收益 / 笔数 / TP 笔数 | 修复后 收益 / 笔数 / TP 笔数 |
|---|---|---|
| rule_v1 2024-01→2026-10 | −95.58% / 1181 / — | −80.04% / 423 / — |
| rule_v1 2026-04→2026-10 | −38.77% / 191 | −46.76% / 191 |
| rule_v1 2026-07（旧 10000U） | −30.11% / 67 | −27.57% / 83（100U） |
| rule_v2 2024-01→2026-10 | −29.16% / 148 | −13.58% / 32 |
| rule_v2 2026-04→2026-10 | −1.65% / 4 | −1.65% / 4 |
| rule_v3 2024-01→2026-10 | **+8.78% / 74 / 23** | **−80.12% / 512 / 2** |
| rule_v3 2026-04→2026-10 | −85.72% / 760 | −80.04% / 524 |
| rule_v4 2026-04→2026-10 | **+18.98% / 45**（DD −12.39%） | **+12.08% / 45**（DD −12.38%，dd_usdt 12.39→14.71） |
| lgbm_v1 2026-04→2026-10 | — | −23.75% / 136 |

差异的可解释来源（不是"策略突然变差"）：
1. **成交通过率**：旧保证金口径让 `notional > equity×lev` 频繁触发裁剪与最小成交额拒绝，信号被大量吞掉；修好后同一策略的真实信号数才全部落地（rule_v3 2024：74 → 512 笔）
2. **SL 优先定序**：5m 策略的 TP（≈0.5%）与 SL（≈0.25%）常落在同一根 bar 区间内，保守定序把这类交易全部记为止损（rule_v3 的 TP 笔数 23 → 2）
3. **成本完整性**：资金费一次性计入逐笔净盈亏、双向 taker、时间止损也加不利滑点
4. **展示口径**：USD 回撤改为峰值→谷值（rule_v4 的 `dd_pct` 不变而 `dd_usdt` 12.39 → 14.71，正是原公式低估了 2.3 USDT 真实回撤额）

结论：**当前报表数字是保守口径下的可信结果；历史文档（含 `docs/rule_v4-trend-design.md`、`config/strategies/lgbm_v1.yaml` 旧注释）里的绝对数字均属修复前口径，只能用于相对排序，不能与新数字直接对比。**

### 13.4 本轮（paper 服务 + TP1/保本/强平/闸门 + 资金费剔除）新增的改动与缺陷

**功能落地**（原本只存在于设计文档）：
- TP1 分批止盈 + 保本移动 + 强平 `FillReason.LIQ` 触发，全部写进共用执行层 `backtest/position.py`，回测与 paper 同序（§7.5b）
- `caps.*` / `guard.*` 的 `[paper]` 闸门从"只告警"改为**拦截新开仓**（用户明确要求），实现为 `ledger.approve_fn` 缝合点（§15.5）。回测端该钩子恒为 `None`，所以回测成交序列与加闸门之前**逐位相同**
- 成本模型剔除资金费（`include_funding` 默认 `False`），数据仍加载供 ML 特征使用（§7.5）

**新发现的 3 个真实缺陷**（都不是测试写错，是实现 bug；每个都配了回归用例）：

| # | 症状 | 根因与修法 | 暴露途径 |
|---|---|---|---|
| 1 | `TypeError: precompute() got an unexpected keyword argument 'swing_lookback'` | `paper/runner.py` 调用 `precompute` 时把配置字段名当参数名传了 → 改为 `swing_window=max(2, int(cfg.swing_lookback))` | 首轮真跑 paper（**有数据就必炸**，之前的单测没覆盖到真实调用） |
| 2 | 重启后可能把上一轮已成交的信号晚一根 bar 重复开仓 | `BarPipeline.on_bar(mode="observe")` 原先只在策略返回非 None 时覆盖 `pending`，历史信号会残留在游标里 → 改为**无条件覆盖**（含 None）：只有断点那根 bar 的信号还没被撮合过，留着它才能在重启后第一根 live bar 成交 | 设计重启等价用例时发现 |
| 3 | `TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'` | `seek_funding(None)`（首轮没有断点）把 `None` 喂给 `np.searchsorted` → 加 `t_ms is None` 短路，指针归 0 | 本机真跑 `paper once`（`include_funding=False` 后 funding 帧仍非空，所以照样触发） |

缺陷 2 的两条方向都有测试钉住（`test_pending_signal_at_breakpoint_is_not_eaten` / `test_signal_from_an_earlier_observed_bar_is_not_re_executed`）—— 只写其中一条都会让这个改动"看起来对"。

**测试侧的一处口径纠正**：`tests/test_paper_guards.py` 的迁移用例原先手工拼 `INSERT` 行，漏了 `paper_accounts` 的 NOT NULL 列而报 `IntegrityError`；改为走 `PaperState.commit()` 生产路径写入，顺带证明了 `ALTER` 迁移的幂等性。

**本轮改动后 `ruff check --select F,E9 src tests` clean，`pytest -q` 213 passed。**

---

## 14. 已知限制（明确记录，不掩饰）

1. **定仓价与成交价近似**：`margin`/`notional` 用 `signal.price_ref`（bar.close）计算，实际成交在下一根 bar.open ± 滑点，误差约 ±2bps 量级；`stop_loss`/`tp_price` 已改用真实成交价重算，故只有仓位量带此近似
2. **强平价只在试仓时估算，加仓不重算**：`FillReason.LIQ` 已会触发平仓，但 `Position.liq_price` 来自首腿的 `est_liq_price`，加仓后加权均价抬高、真实强平价会更靠近均价而账面仍用旧值 → 对多头加仓单偏乐观（更晚才判强平）。10× 下 SL 设在 1.5~2.0×ATR（远更近），常规路径不会走到强平，所以影响面小；`Position.realized_pnl`、`extreme_price` 仍是预留字段
3. **TP1 的默认值未经多窗口验证**：默认 `tp1_r=1.0 / tp1_close_frac=0.5 / breakeven_after_r=1.0`，而 rule_v4 半年窗口实测 TP1 关闭更好（§7.5b）。当前只在单一窗口上对比过，政策未定，`config/strategies/rule_v4.yaml` 仍保持默认
4. **成交时间戳**：入场/信号平仓记在 bar 收盘，止损/止盈/TP1/强平记 bar 中点 —— 逐笔时长与真实 tick 路径有差异（1m/秒级路径数据未回填，`backtest.intra_bar_path=exact_1m` 是未来开关）
5. **只有 REST、只有 ETHUSDT**：无 WebSocket、无 1m；BTCUSDT 在 `symbols.yaml` 中 `enabled=false` 且未回填。`quant backtest` 一次仍只跑一个 `--strategy`（多策略并行只在 paper 服务里）
6. **维持保证金率取最高档**：缺签名接口 `leverageBracket`，属安全侧偏差（仓位偏小）
7. **K 线由 5m 聚合**：15m/1h/4h/1d 图与指标读的是交易所 1h/4h 值；`check-data` 的 `htf_consistency` 会显式报出偏差（当前实测偏差为 0），两者不一致时以交易所值为准并在报表标注
8. **paper 闸门与回测刻意不一致**：`caps.*`/`guard.*` 只拦 paper 的新开仓，回测不启用。所以**纸面账户的收益率不会等于同参数回测** —— 一致的是"给定同一串信号与同一份配置，撮合序列逐位相同"（`TestBacktestParity`），不是"闸门效果"
9. **paper 不做跨策略仓位合并**：每个 `(symbol, strategy)` 各读各的 100U 纸面账户，同品种两策略同时持仓时真实账户会共用保证金，纸面不会（用户明确"先不管"）
10. **`models/` 未纳入 gitignore**：9.0 MB 折模型会被 `git add` 带进仓库，提交前需用户确认（commit / ignore 二选一，尚未决定）
11. **ML 策略需要续训才能进 paper**：`_model_readiness` 要求当前时刻落在某折 `[test_start, test_end)` 内，本机实测 `lgbm_v1` 已超出末折（2026-10 之后）→ 每轮只推"需 quant train 续训"告警，不产信号

---

## 15. 模拟交易 / 信号推送服务（`src/quant/paper/`，已实现）

需求（用户原话口径）：服务器上以 shell 脚本启动，多策略并行运行策略执行与飞书信号推送；**真实下单由用户手动完成**（系统只产信号，不接下单 API）；暂不处理多策略仓位共存。

### 15.1 进程形态与节拍

单进程多账户（`quant paper run`）。理由：REST 权重预算多进程会重复扣、SQLite 单写者更安全、飞书 40/min 限流只有单进程才真正生效。

```
每 poll_sec=20s 一轮（PaperService.run_cycle）
  1. now_ms() 用交易所时间（client.now_ms()），本地时钟只做日志
  2. _pull：期望的最新已收盘 bar = floor(now − resync_grace_ms − step)
     水位线没追到才增量回填 spec/5m/1h/4h + funding（追到了不占配额）
  3. 逐账户判 _account_due，只对"有新 bar 或数据滞后"的账户跑 run_round
  4. 任一账户抛异常 → 该轮记 `skipped=执行异常：…`，主循环继续（服务停摆比少推一条信号贵得多）
  5. SIGTERM/SIGINT → 置 `_stop`，当前轮跑完再退；`_sleep` 分段睡，TERM 后最多 1s 醒
```

`_account_due` 的两条准入：`watermark > checkpoint`（有新 bar），或 `now − (watermark+step) > stale_data_max_gap_min×60_000 + step`（数据滞后）。第二条是必须的 —— 不然行情断了的时候 stale 闸门永远得不到评估、也就永远不会有"行情滞后"告警。5m 周期的阈值 = 5 min + 5 min = **10 分钟**（测试里用 +20 min 才进轮，留的是余量）。首轮没有断点则无条件进轮。

### 15.2 一轮的口径（`PaperRunner.run_round`）

```
observe_start = checkpoint − replay_days(7d)      # 首轮没有断点就用最新收盘 bar 为基准
load_start    = observe_start − warmup_bars()×step − strategy.history_lookback()
① 数据闸门：对 strategy.required_intervals 逐个 assert_complete(grid, floor(start), floor(watermark+step))
   任一周期 coverage < 1.0 → 整轮跳过 + 告警（残缺窗口里的指标与 bar 间距都不可信）
② 模型闸门（ML 策略）：折能加载 且 ref_ms 落在某折 [test_start, test_end) 内，否则跳过并告警
③ precompute / precompute_htf → HtfCursor；bind_history + strategy.reset()
④ ledger.seek_funding(checkpoint + step)          # 窗口每轮滑动，落库存数组下标没有意义
⑤ guards.refresh(now, data_gap)                   # 本轮否决理由 + 冷却到期 + 告警
⑥ 逐根 bar：ts_open < observe_start 只做指标预热
             to_ms(bar.ts_close) > now → break     # 交易所未收盘的 bar 绝不能出信号
             ts_open <= checkpoint → mode="observe"（只调策略，账本一行不写）
             否则            → mode="live"（与回测同一个 BarPipeline）
⑦ 事件 → paper_signals / paper_trades 行，state.commit() 一次事务落库
⑧ 先落库再推送（_deliver），然后补发历史失败（_resend_unsent）
```

- **observe 重放**取代了设计里的"只喂新 bar"：策略的隐藏状态（rule_v4 的 1h 趋势状态机、冷却计数、ML 特征帧）必须与断点对齐，而每轮都从 `reset()` 开始，所以"跑第 1 轮"与"重启第 100 轮"完全等价。observe 区间 `replay_days=7` 天。
- **落库先于推送**：崩溃只会留下 `sent_ok=0` 的行等下一轮补发，不会出现"卡片发了、账本没记"的幽灵成交。
- 模型可用性**不用 `dataset_version` 做闸门**：paper 每轮都在增量拉数据，版本串必然变化，拿它当闸门等于永远拒绝服务。特征漂移由 `feature_spec_hash` 断言拦住，"模型过期"由折窗口拦住。

### 15.3 纸面撮合与回测同源（关键设计约束）

`PaperConfig` 字段名与默认值必须和 `BacktestConfig` 完全一致（只有 `start`/`end` 换成断点），`execution_params()` = 字段默认值 + 策略 `engine_defaults()` + `risk.account.initial_equity_usdt`，口径等于"不带任何开关的 `quant backtest`"。执行层是同一个 `BarPipeline`/`PositionLedger`，`precompute` 也是同一份。

由 `tests/test_paper_runner.py` 双重钉住：字段逐一对齐断言 + **把 `warmup_bars` 打到 0 让两条链路从同一根 bar 开始 live，逐笔比对 `trade_id` 序列、净值点，并校验 `equity − 100 == sum(pnl)`**。

唯一的差异是 paper 多挂了 `ledger.approve_fn`（回测端为 `None`）。

### 15.4 持久化（`data/store.py`，5 张表）

| 表 | 主键 | 作用 |
|---|---|---|
| `paper_accounts` | `(symbol, strategy)` | 现金/净值/`equity_open`/`peak_equity`（回撤闸门的高水位）/`pause_until_ms`（冷却到期）/`bars_since_entry`/`funding_ptr`/加仓计数 |
| `paper_positions` | `position_id` | 持仓全量还原字段：`avg_entry_price`、`stop_loss`、`base_stop_price`、`tp_price`、`liq_price`、`add_levels_json`、`trailing_*`、`tp1_done`、`breakeven_done`、`entry_fee_outstanding`、`signal_id` |
| `paper_signals` | `signal_id` | 卡片审计 + 待投递队列（`sent_ok`/`sent_at_ms`/`payload_json`）+ 闸门告警 + 日计数 |
| `paper_trades` | `trade_id` | 逐笔平仓流水，`paper_status` 的已实现盈亏与连亏笔数都从这里算 |
| `paper_checkpoints` | `(symbol, strategy)` | 断点 `last_processed_ts_ms` + `dataset_version` |

**确定性主键**（重启幂等全靠它，`paper/state.py`）：

| 行 | id 形式 | 幂等效果 |
|---|---|---|
| 信号 / 事件 | `{symbol}:{strategy}:{signal_id}:{kind}` | 同一笔信号重启后 `INSERT OR IGNORE` 撞主键，不推第二次 |
| 平仓卡片 | `{symbol}:{strategy}:{position_id}:{reason}:{exit_time_ms}` | 一个仓位的 TP1 部分止盈 + 全平是**两条**流水（少了 `reason:exit_time` 就会互相撞掉、少推一张卡） |
| 告警 | `{symbol}:{strategy}:alert:{tag}`，tag 带 `day_start_ms` 或触发时刻 | 同一闸门同一自然日只推一张 |
| 交易 | `{position_id}:{reason}:{exit_time_ms}` | 同上 |

`paper_accounts` 的两列（`peak_equity`、`pause_until_ms`）通过 `PAPER_MIGRATIONS` + `ALTER TABLE` 给老库补上，重复打开同一库不报错（测试用真实生产写入路径验证幂等）。

### 15.5 `[paper]` 风控闸门：只拦新开仓，绝不强平已有持仓

`EntryGuards.refresh()` 每轮评估一次，`vetoes()` 在撮合前给 `RiskDecision` 追加拒绝理由（`veto()` 在 frozen dataclass 上重建，保留原有 `reject_reasons`）。被拦的动作只有 `OPEN_LONG/OPEN_SHORT/ADD_TO_LONG/ADD_TO_SHORT`；出场（SL/TP/TP1/LIQ/时间止损）一律由执行层决定，与回测逐位一致 —— **否则纸面数字失去参考价值**。

| 闸门 | 阈值（`risk.yaml`） | 类型 | 触发后 |
|---|---|---|---|
| 行情滞后 | `guard.stale_data_max_gap_min=5` | 持续型 | 本轮不再试仓 + 告警（tag 带北京日界，每日一张） |
| 日亏停 | `daily_loss_stop_pct=0.03` | 持续型 | 当日已实现 ≤ −3%×当日期初净值 → 停止试仓（期初净值由 `equity − 当日已实现` 回推） |
| 每日试仓上限 | `max_new_entries_per_day=4` | 持续型 | 只数**当天** `kind=open` 的信号（北京日界 `day_start_ms`） |
| 回撤暂停 | `max_drawdown_pause_pct=0.15` | 冷却型 | 相对 `peak_equity` 回撤 15% → 写 `pause_until_ms = now + cooldown_hours(24h)` + 告警 |
| 连亏冷却 | `max_consecutive_losses=5` | 冷却型 | 最近 5 笔全亏 → 同样 24h 冷却；一笔盈利即清零 |
| 单品种保证金占比 | `max_margin_per_symbol_pct=0.30` | 逐笔 | `已占用 margin + 本笔 margin > 30%×净值` → 否决本笔 |
| 单笔名义上限 | `max_notional_usdt=5000` | 逐笔 | 超限否决 |

三条实现细节（都有测试）：
- **冷却中不再评估其余闸门**：连亏笔数在下一笔盈利之前会一直成立，每轮重触发会让冷却期永远走不完。
- **冷却到期把 `peak_equity` 重锚到当前净值**，否则恢复的第一轮会被回撤闸门再触发一次（死循环）。
- 闸门状态一律从 SQLite 读，进程重启不会把"今天已经亏了 3%"忘掉。

被拦下的信号仍会出卡片，但 `kind=reject`、`note` 里逐条写明拒绝理由 —— 用户要看到的是"策略想开但被风控拦了"，而不是静默消失。

### 15.6 卡片矩阵（`notify/cards.py`）

| `kind` | 卡片 | 触发 | 备注 |
|---|---|---|---|
| `open` | `build_card(signal, decision)` | 试仓 | 方向/价格/qty/SL/TP/杠杆/保证金/置信度/理由，北京时间（UTC+8） |
| `add` | 同上 | 金字塔加仓 | |
| `close` | `build_close_card(trade, ...)` | 全平或 TP1 部分止盈（`partial` 事件也映射成 `close`） | 含 `pnl` / `ret_pct` / 平仓原因中文标签 `exit_label` |
| `reject` | `build_card`（`approved=False`） | 定仓被拒或被闸门否决 | 卡片正文列出拒绝理由 |
| `alert` | `build_alert_card(title, text, ...)` | 数据缺口 / 模型不可用 / 行情滞后 / 回撤或连亏冷却 | 主键去重，同 tag 同一天只一张 |

`add_skipped` 事件**不出卡**（只计数）：金字塔条件不满足时每根 bar 都会产生一条，推出去就是把用户的消息流变成日志流。

飞书失败：`FeishuBusy/FeishuError` 与其它异常都被吞成 `sent_ok=0` 并 WARNING/ERROR，绝不让卡片投递拖垮主循环；补发按 `notify.feishu.resend_cooldown_min=10` 退避（`paper_signals_unsent(..., attempt_before_ms=now−cooldown)`），设为 0 时同一轮就会重试。演练模式（`--no-push`，`notifier=None`）下 `_send` 直接返回 True，落库即视为已处理，不进补发队列。

### 15.7 部署与实跑验证

`scripts/paper.sh start|stop|restart|status|once|logs`：预检 `.env` 存在、`FEISHU_WEBHOOK_URL` **是否定义**（绝不打印值）、代理配置；优先用 `.venv/bin/quant`，否则 `uv run quant`；PID 写 `run/paper.pid`（`running_pid()` 用 `ps -o command=` 校验，防 PID 复用）；`stop` = TERM → 最多等 30s → 才 KILL。环境变量开关 `PAPER_SYMBOL/PAPER_STRATEGY/PAPER_NO_PUSH/PAPER_NO_PULL`。

`deploy/paper-trading.service`：`KillSignal=SIGTERM` + `TimeoutStopSec=60`、`Restart=always/RestartSec=10`、`PYTHONUNBUFFERED=1`、`EnvironmentFile=-…/.env`、`ReadWritePaths=data logs`、`NoNewPrivileges/PrivateTmp/ProtectSystem=full`。**故意没有 `WatchdogSec`**：进程不发 `sd_notify`，加了会被 systemd 定期误杀。

本机实测（**全程 `--no-push`，没有向飞书群发过任何一条真实消息**）：
- `paper once --no-pull`：1h/4h 数据只到 2026-10-02 而 5m 到 14:10 → 两个账户都整轮跳过并各写一条 gap 告警（`1h 缺 5/168 根（首个缺失 2026-10-02T09:00）；4h 缺 1/42 根`）。这是**设计行为**：高周期缺柱会让 `HtfCursor` 错位。
- `backfill` 补齐后 `paper once --no-pull`（rule_v4）：`观察 2016 根 / 实撮 0 根，事件 0 条，净值 100.00，空仓，闸门 放行`，checkpoint 落后 0 根。首轮只 observe 是预期的（断点为空 → 全部进 observe 区间）。
- `lgbm_v1`：`模型不可用，本轮跳过 —— 当前时刻超出最后一折的样本外窗口，需 quant train 续训`（折模型 eval 到 2026-10-01 为止）。
- `start` → `status` → `stop`：PID 正常，TERM 后日志给出"收到信号 15，当前轮结束后退出"，优雅退出。
- 库里现存 1 行 `paper_accounts`（`rule_v4`，净值 100）、1 行 `paper_checkpoints`、4 行 `paper_signals`（全是 alert）、0 行 `paper_trades`。

### 15.8 成本口径复述（用户 2026-10-07 决定）

10× 杠杆下往返 taker = 名义 × 0.1% = **保证金的 1%**（单边万 5 × 2，除以杠杆 10），加双边 2bp 不利滑点后约 **1.4%**；资金费按实测均值约 **0.24%/3 天**。因此成本模型只保留手续费 + 滑点，`include_funding=False`（§7.5）。这条口径同时决定了回测与 paper 的净值曲线，卡片上的保证金/数量也都是这个口径。

---

## 16. 现行回归基线（2026-10-07 第 2 轮口径）

统一条件：ETHUSDT，`--equity 100`，10×，全 taker 0.05% + 2bp 不利滑点，**不计资金费**，TP1/保本默认开（`tp1_r=1.0 / frac=0.5 / breakeven=1.0`），保守口径（SL 用 bar 极值 + 中点时间、双触发取 SL）。原始输出留在 `logs/rebaseline/*.log`（gitignore）。

| 策略 | 窗口 | 收益 | CAGR | maxDD | 笔数 | 胜率 | 盈亏比 | PF |
|---|---|---|---|---|---|---|---|---|
| rule_v1 | 2024-01→2026-10 | −80.10% | −44.39% | −80.15% | 664 | 41.9% | 0.31 | 0.22 |
| rule_v1 | 2026-04→2026-10 | −42.28% | −66.59% | −42.74% | 249 | 35.3% | 0.34 | 0.18 |
| rule_v2 | 2024-01→2026-10 | −13.53% | −5.15% | −13.70% | 36 | 16.7% | 0.22 | 0.04 |
| rule_v2 | 2026-04→2026-10 | −1.65% | −3.26% | −1.81% | 4 | 0.0% | 0.00 | 0.00 |
| rule_v3 | 2024-01→2026-10 | −80.05% | −44.34% | −80.07% | 750 | 39.5% | 0.30 | 0.20 |
| rule_v3 | 2026-04→2026-10 | −80.13% | −96.02% | −80.27% | 763 | 35.1% | 0.27 | 0.14 |
| **rule_v4** | 2024-01→2026-10 | **−53.55%** | −24.33% | −60.71% | 511 | 62.6% | 0.48 | 0.80 |
| **rule_v4** | 2026-04→2026-10 | **+4.79%** | +9.77% | −12.44% | 64 | 64.1% | 0.62 | **1.10** |
| rule_v4（TP1 关） | 2026-04→2026-10 | +12.42% | +26.31% | −12.35% | 45 | 51.1% | 1.20 | 1.25 |
| lgbm_v1 | 2024-01→2026-10 | −22.03% | −8.65% | −22.69% | 207 | 58.9% | 0.38 | 0.55 |
| lgbm_v1 | 2026-04→2026-10 | −20.59% | −36.86% | −21.28% | 189 | 58.7% | 0.38 | 0.54 |

对照：同一半年窗口买入持有约 **+27%**。

账本恒等式 `sum(trades.pnl) == final − initial` 在以上每个 run 都成立（残差 ≤1e-3，纯 CSV 4 位小数舍入）；`tests/test_engine_ledger.py` 与 `test_paper_runner.py` 各有一份断言。

诚实结论：
1. **rule_v4 的半年正收益不稳健**。全历史 −53.55%、逐笔期望为负（PF 0.80），半年 +4.79% 只说明它恰好适应这段上行趋势；"周 10%"的原始目标在任何参数组合下都不可达（见记忆 `project-rule-v4-result`，本轮数字已按第 2 轮口径更正）。
2. **TP1 是净值的减分项而不是加分项**（趋势策略）：胜率从 51.1% 抬到 64.1% 看着好看，盈亏比 1.20→0.62 把大 R 尾部削掉了。默认值暂未调整（§14 第 3 条）。
3. **资金费剔除的实际影响很小**：rule_v4 半年、同样关闭 TP1/保本的口径下，计资金费 +12.08% → 不计 +12.42%，**笔数都是 45**（说明这个窗口里新增的保本/强平触发没有改变成交序列），差额约 0.34pp 就是持仓期资金费 —— 与"量级太小、只会引入无关变量"的判断一致。
4. `lgbm_v1` 全样本外负收益，维持"只留 paper 观察"的处置。
5. **本节是唯一现行基线**。其余文档里的头条数字（`rule_v4-trend-design.md` 的 +18.98%、
   `ml-regime-gate-design.md` / `multimodal-kline-strategy-design.md` 的 −63.93%、
   `lgbm-strategy-and-paper-trading-design.md` §11.4 的 τ 扫描）属第 0/1 轮口径，各文档开头已加
   口径注记并把本节链过去；那些实验是在旧口径下跑的，重跑会让当时的表格与自身结论脱节，
   所以保留原值 + 标注，不做覆盖式改写。

---

## 附录 A：数据字典要点

- `bars.prio`：`2*is_closed + (src=="rest")`，数值越大越权威
- `bars.time_interval`（不是 `interval`）、`funding.funded_at_ms`、`funding.interval_h`（`features.builder` 同时兼容 `interval_hours`）
- `meta.key`：`spec:{symbol}`（双层 JSON，见 §4.2）
- 品种规则实测：`tick=0.01 step=0.001 min_qty=0.001 min_notional=20 maint=0.025 liq_fee=0.0125 max_lev=20x prec=2/3`

## 附录 B：命令速查

```bash
uv sync                                     # 复用 .venv
uv run quant backfill --symbol ETHUSDT      # 断点续传，需代理
uv run quant check-data --symbol ETHUSDT    # 回测前必跑
uv run quant backtest --strategy rule_v4 --start 2026-04-01 --end 2026-10-01
uv run quant backtest --strategy rule_v4 --tp1-r 0 --breakeven-r 0     # 关闭分批止盈/保本
uv run quant train --symbol ETHUSDT --eval-start 2026-04-01 --eval-end 2026-10-01
uv run quant dump-card                      # 只看卡片
uv run quant test-card                      # 真发飞书（需 .env）
uv run pytest -q && uv run ruff check --select F,E9 src tests

# paper 服务（先 --no-push 演练，确认链路后再真发）
uv run quant paper once  --strategy rule_v4 --no-push --no-pull
uv run quant paper status [--json]
./scripts/paper.sh start|status|stop|logs   # PID 在 run/paper.pid，日志 logs/paper.out
```

## 附录 C：指标口径速查

| 指标 | 公式 |
|---|---|
| 单笔保证金收益率 | `pnl(净) / margin_usdt × 100`，10× 下约为价格变动幅度 ×10 |
| 保证金 | `notional / global_max_leverage`，加仓按每次占用累计 |
| 净盈亏（逐笔） | `价格盈亏 − 开仓费 − 平仓费 − 持仓期资金费` |
| R 倍数 | `实现价距 / (sl_atr_mult × entry_atr)`，符号跟随盈亏方向 |
| 账本不变量 | `sum(trades.pnl) == final_equity − initial_equity` |
