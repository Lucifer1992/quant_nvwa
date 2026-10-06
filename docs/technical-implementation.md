# Quant_LLM 技术实现文档（最终版）

> 本文是**当前代码实现状态**的唯一权威说明：以实际代码与实测数据为准，不是设计意图的复述。
> 设计决策的论证过程见 `docs/` 下其余设计文档；与本文冲突时，以本文为准（本文对齐 2026-10-07 的代码）。

- 代码基线：`master`，`git ls-files` 之上另有 40 个未提交文件（用户要求：本轮不提交 git）
- 校验状态：`uv run pytest -q` → **147 passed**；`uv run ruff check --select F,E9 src tests` → clean
- 数据基线：`data/quant.db`，ETHUSDT 5m/15m/1h/4h/1d + funding，2024-01-01 起
- 可比性声明：**本文第 13 节列出的修复之前的所有回测数字都不可与现在的数字比较**（口径变了，不是策略变差了）

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
backtest/    engine.py（事件驱动撮合 + 账本 + 指标）
             reports.py（equity.csv / trades.csv / metrics.json / report.html + 多周期 K 线）
strategy/    base.py（BaseStrategy 契约）  registry.py（@register 自动发现）
             rules/rule_v1..v4.py
             ml/base.py（MLBaseStrategy：折模型加载、特征帧、edge 推理）  ml/lgbm_v1.py
train/       walkforward.py（plan_folds / fold_masks / 训练 / edge 扫描 / 折报告）
notify/      feishu.py（webhook + 签名 + 限流退避）  cards.py（入场卡片构造与样例）
infra/       settings.py（YAML 分层加载 + validate_config）  log.py
cli.py       typer 应用：backfill / check-data / exchange-info / test-card / dump-card / backtest / train
```

### 1.3 实现状态

**已实现并实测**：REST 数据层与断点续传、SQLite 落库与质量闸门、5m/1h/4h 多周期事件驱动回测、固定风险定仓、金字塔加仓、移动止损、资金费计提、飞书卡片、4 个规则策略 + LightGBM walk-forward 训练与推理、HTML 多周期 K 线报告、147 项离线单测。

**未实现（明确不做或延后）**：WebSocket/1m 数据、模拟交易服务与 shell 启动脚本（§15）、TP1 分批止盈与保本移动、`caps`/`guard` 中的 `[paper]` 风控闸门、强平价触发平仓、多品种并行回测（BTCUSDT 未回填）。

---

## 2. 运行环境与依赖

- Python 由 **uv** 管理，复用仓库根 `.venv`（禁止系统 pip、禁止新建环境）；实测解释器 3.13.1，`requires-python >=3.11`
- 运行时依赖：`httpx`、`pandas 3.0.6`、`numpy 2.5.3`、`pyyaml`、`python-dotenv`、`typer`、`plotly 7.1`、`pyecharts 2.1`、`lightgbm 4.7.0`、`scikit-learn 1.9`
- 开发依赖组：`pytest`、`hypothesis`、`ruff`
- 入口：`[project.scripts] quant = "quant.cli:app"` → `uv run quant <cmd>`
- 目录约定：`data/`（SQLite，gitignore）、`reports/`（回测产物，gitignore）、`logs/quant.log`、`models/`（折模型，**当前未被 gitignore 覆盖**，9.0 MB）

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

### 3.2 纯文档键（当前代码不读取，不要误以为生效）

| 键 | 状态 |
|---|---|
| `network.resync_grace_ms` | 仅文档：REST 覆盖 WS 尾 bar 的等待窗口，WS 未实现 |
| `backtest.intra_bar_path` | 仅文档：`bar_hl`（当前口径）/ `exact_1m`（1m 数据回填后才可切换） |
| `caps.max_notional_usdt`、`caps.max_margin_per_symbol_pct`、`caps.max_new_entries_per_day` | `[paper]`，回测未强制 |
| `guard.*`（日亏停、回撤暂停、连亏次数、冷却、数据新鲜度） | `[paper]`，回测不生效 |
| `stops` 中注释掉的 `tp1_r` / `breakeven_after_r` / `trail_atr_mult` | 未实现，只记录设计意图 |

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
| ETHUSDT | 5m | 289,611 | 289,611 | 2024-01-01 00:00 → 2026-10-02 14:10 |
| ETHUSDT | 15m | 96,517 | 96,517 | 2024-01-01 00:00 → 2026-10-02 09:00 |
| ETHUSDT | 1h | 24,129 | 24,129 | 2024-01-01 00:00 → 2026-10-02 08:00 |
| ETHUSDT | 4h | 6,032 | 6,032 | 2024-01-01 00:00 → 2026-10-02 04:00 |
| ETHUSDT | 1d | 1,005 | 1,005 | 2024-01-01 00:00 → 2026-10-01 00:00 |
| ETHUSDT | funding | 3,017 | — | 2024-01-01 00:00 → 2026-10-02 08:00 |

`src` 全为 `rest`（WS 未实现）。缓存规则：`tick=0.01 step=0.001 min_qty=0.001 min_notional=20.0 maint=0.025 max_lev=20 liq_fee=0.0125 fetched_at=2026-10-02T09:23:57Z`。

---

## 5. 特征层

### 5.1 回测引擎预计算列

- 5m（`engine._precompute`，一次性 O(n)）：`ema_9/12/20/50/60`、`atr`、`macd/macd_signal/macd_hist/prev_macd_hist`、`prev_close/prev_ema_20`、`rsi_14`、`srsi_k/srsi_d/prev_srsi_k`、`bb_mid/upper/lower`、`adx_14`、`atr_median`、`vol_median`、`swing_high/swing_low`（`rolling(swing_window).max().shift(1)`，**不含当前 bar**）
- 高周期（`engine._precompute_htf`）：`ema_12/20/60`、`macd_hist/prev_macd_hist`、`srsi_k/d/prev_srsi_k`、`atr`、`rsi_14`、`adx_14/prev_adx_14/adx_prev_3`、`ema_20_prev_6`（rule_v4 算斜率）、`atr_median_100`（波动扩张）、`donchian_high_24/low_24`

所有滚动量右移一位或按可见时刻对齐，无前视。

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
`long: entry × (1 − 1/lev + maint + taker_fee)`，`short: entry × (1 + 1/lev − maint − taker_fee)`。10× / maint 2.5% → 多头约在 −7.45% 价格处强平。回测**不模拟强平触发**（见 §14）。

---

## 7. 回测引擎（`backtest/engine.py`）

### 7.1 配置（`BacktestConfig` 默认值）

```
symbol, base_interval=5m, start, end, initial_equity=100, mode,
risk_pct=0.01, sl_atr_mult=1.5, tp_atr_mult=3.0, time_stop_bars=96,
allow_short=True, global_max_leverage=10.0, taker_fee=0.0005, slippage_bps=2.0,
include_funding=True, sl_first_on_conflict=True, strategy_name="rule_v1",
pyramid_enabled=True, entry_frac=0.10, add_frac=0.20, add_max_count=2,
add_trigger_r=0.3, trail_trigger_r=1.0, trail_atr_mult=1.0, swing_lookback=12
```

策略可通过 `engine_defaults()` 覆盖出场/节奏默认值（CLI 显式传参优先于策略声明）。

### 7.2 时间线（每根 5m bar 一次迭代，`run()`）

```
i.   弹出所有"可见时刻 <= 本根收盘"的 1h / 4h bar → htf dict（指针只前进不回退）
ii.  equity_open = equity                       ← 本根开盘前已知的净值，定仓只用它
iii. MTM 快照（有持仓则 _update_mtm，无持仓 equity=cash），记入 equity_curve
iv.  _accrue_funding(可见时刻, close)            ← 指针每根都推进，含预热期与空仓期
v.   i < warmup → continue                       （warmup = 3120 根 5m ≈ 10.8 天）
vi.  执行上一根 bar 收盘产生的信号：在本根 bar 开盘成交（pending_signal）
vii. _update_trailing_stop(bar)
viii._check_exits(bar)：SL / TP（同根双触发按 sl_first_on_conflict 定序）
ix.  持仓根数 >= time_stop_bars → 市价平仓（按 close ± 不利滑点）
x.   signal = strategy.on_bar(bar, row, htf, dt) → pending_signal
```

循环结束若仍持仓 → `FillReason.FLAT` 平仓，并**回写最后一个净值点**（循环内快照是持仓 MTM，不含这笔的滑点/平仓费/资金费，否则尾部差一个量导致恒等式破）。

`warmup = max(4h/5m × 65, 1h/5m × 35, 30) = max(48×65=3120, 12×35=420, 30) = 3120`：规则策略的加载窗口不带前置历史，起点处 4h EMA60 / ADX14 仍是 NaN，必须留预热段。ML 策略 `warmup()=0`，改用 `history_lookback()`（90 天）让引擎多加载前缀。

### 7.3 成交与成本口径（全部取不利方向）

| 场景 | 成交价 | 时间戳 |
|---|---|---|
| 入场（试仓/加仓） | `bar.open × (1 ± slippage_bps)` | 该 5m bar 的 `ts_close` |
| 信号平仓 / 反手 | `bar.open × (1 ∓ slip)` | 同上 |
| **止损（B 模式）** | `bar.low`（多）或 `bar.high`（空）再减/加 slip | **bar 中点** `_bar_mid` |
| 止盈 | `pos.tp_price ∓ slip` | bar 中点 |
| 时间止损 / 期末平仓 | `bar.close ∓ slip` | bar 收盘 |

- 止损按 bar 极值成交是"最坏情况"假设：一根 bar 内无法证明价格先到哪侧，按 TP 成交会把本该止损的交易记成盈利，属系统性乐观偏差
- 同一根 bar 双触发：默认（`sl_first_on_conflict=True`）保守取 SL，与标签口径一致；配置为 False 时取 TP
- 止损/止盈价基于**实际成交价**（`bar.open ± slip`）而非 `signal.price_ref`（=bar.close），避免 open-close gap 立刻击穿 SL
- 策略可自带决策周期 ATR（`signal.meta["atr"]`，如 rule_v4 / lgbm_v1 用 1h ATR），否则回落 5m ATR；5m ATR 约 0.17%，1h 趋势仓用它会因噪音被扫掉
- `size_mult` 钳在 `[0.2, 1.0]`，高周期背景只允许减仓不允许超额

### 7.4 仓位与加仓

- 试仓：新建 `Position`，`cash -= margin + 开仓腿手续费`，手续费记在 `add_levels[0].fee_usdt`
- 加仓（`_should_add`）：`add_count < add_max_count` ∧ `unreal_r >= add_trigger_r` ∧ 结构突破（多头 `close > swing_high`，空头 `close < swing_low`，swing 已 `shift(1)`）；成功后加权均价、qty、margin、`add_levels` 累加，并把 SL 上移到 `avg_entry ± sl_atr_mult × entry_atr`（只朝有利方向）
- 移动止损：极值 `trailing_peak_price` 每根更新，`unreal_r >= trail_trigger_r` 激活，激活后 `stop_loss = max/min(prev, peak ∓ trail_atr_mult × entry_atr)`
- 反手：在同一根 bar 开盘先平对向仓（`eq = cash`），再开新仓

### 7.5 资金费计提

- 指针 `_funding_ptr` 每根 bar 推进（**包括空仓与预热期**，否则空仓期间的结算会被算进之后才开的仓位）
- `cost = qty × mark_price(优先) 或 close × rate`，多头为正（付费）、空头为负（收费），费率非有限或为 0 跳过
- 只累计到 `funding_since_entry` 与 `pos.funding_paid`，**现金在平仓时经 `net_cash` 一次性扣除**，避免持仓期与平仓时重复扣款
- `include_funding=False` 可关闭

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

`lgbm_v1` 在 2026-04-01~10-01 实测 **−23.75%**（136 笔，胜率 44.1%，PF 0.47，DD −24.41%）；`folds.json` 的 `edge_sweep` 与回测一致表明：**当前 38 维特征在 24h 障碍上没有可交易的样本外 edge**。保留该策略用于 paper 观察，不作为主策略。同期 `rule_v4` 在同一口径下 +12.08%，买入持有约 +27%。

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

---

## 11. CLI 手册（`quant`，共 7 个命令）

| 命令 | 作用 | 关键参数 |
|---|---|---|
| `backfill` | REST 分页回填 K 线 + funding 到 SQLite（水位线断点续传） | `--symbol --start --intervals 5m,1h --funding/--no-funding --force-spec` |
| `check-data` | 本地数据校验：网格覆盖率、缺口、规则缓存、数据集指纹（**回测前必跑**） | `--symbol --start` |
| `exchange-info` | 查看交易规则（USDT-M 忽略 symbol 参数，本地筛选） | `--symbol --raw` |
| `backtest` | 事件驱动回测并产出报告 | 见下 |
| `train` | walk-forward 训练 `lgbm_v1` 折模型 + 稳定性报告 | `--symbol --data-start --eval-start --eval-end --window-months --step-months --horizon --k-up --k-dn --no-balance --out-dir` |
| `test-card` | 发送样例入场卡片验证飞书通路 | `--approved/--no-approved` |
| `dump-card` | 只打印卡片 payload，不发送 | `--approved` |

`backtest` 选项：`--symbol --start --end --equity --risk-pct --lev --short/--no-short --strategy`；出场 `--sl-atr --tp-atr --time-stop-bars --trail-atr-mult --trail-trigger-r`；金字塔 `--entry-frac --add-frac --add-max --add-trigger-r --no-pyramid`；数据 `--allow-gaps`；策略专属（缺省一律用策略自身默认值，**不在 CLI 写死以免串给别的策略**）`--confirm-bars --srsi-oversold/--srsi-overbought/--srsi-cross-margin --min-signals --adx-min --pullback-atr --tau --model-dir`。

执行顺序固定为：`_boot`（配置校验）→ 注册表取策略类 → `build(opts)` → `engine_defaults()` 合并 CLI 显式值 → `required_intervals` 决定加载周期（含 `history_lookback` 前缀）→ 数据闸门 → 引擎 → 报表。

**退出码**：`0` 成功；`2` 配置校验失败 / 交易规则未缓存 / 数据缺柱且未 `--allow-gaps` / 未知策略名；其余异常按 Python 默认非零退出。

标准用法（新品种首跑或换窗口）：

```bash
uv run quant backfill --symbol ETHUSDT --intervals 5m,1h,4h   # 需代理可达
uv run quant check-data --symbol ETHUSDT
uv run quant backtest --symbol ETHUSDT --strategy rule_v4 --start 2026-04-01 --end 2026-10-01
```

---

## 12. 测试体系（`uv run pytest -q` → 147 passed，全部离线）

| 文件 | 数量 | 覆盖点 |
|---|---|---|
| `test_data_layer.py` | 16 | upsert prio 守卫与真实写入条数、回填缺口裁剪、`htf_consistency` 边缘桶剔除、REST `HTTP≥400` 非 dict body、403 不重试、429 按 `retry-after` 休眠、5xx 退避后抛错、`x-mbx-used-weight-1m` 校正预算、`_kline_weight` 分档 |
| `test_engine_ledger.py` | 16 | SL/TP 同 bar 定序（可配置）、止损 bar 极值 + 中点时间、资金费一次性扣款/空头收/空仓推进/可关闭、`margin=notional/lev`、定仓用 `equity_open`、市价平仓不利滑点、账本恒等式、USD 回撤峰值→谷值、payoff 无亏损时为 inf |
| `test_engine_pyramid.py` | 15 | 试仓/加仓路径、`add_max_count`、结构突破判定、加仓后 SL 上移、移动止损 |
| `test_engine_signal_execution.py` | 13 | pending_signal 下一根开盘成交（防前视）、反手、信号平仓、warmup 边界 |
| `test_features_builder.py` | 13 | 4h/funding 可见性、`hrs_to_funding` 非负、`donchian shift`、缺列报错、hash 稳定、标签 flat 双条件、`sl_first_on_conflict` 一致 |
| `test_strategy_build.py` | 5 | `build(opts)` 只认领自家字段、注册表重复名/未知名 |
| `test_strategy_ml.py` | 16 | 折选择、折外告警且不产信号、`feature_spec_hash` drift guard、冷却、`size_mult` 钳位、`bind_history` 缺失时安全返回 None |
| `test_strategy_rule_v4.py` | 31 | 状态机转移、4h 仓位倍数、A/B 两个入场口、BROKEN 平仓信号 |
| `test_train_walkforward.py` | 8 | `plan_folds` 衔接与末折钳位、窗口不足抛错、`fold_masks` 两端 purge 边界（含 `embargo=0` 的反证）、标签编码往返、`edge_from_proba` |
| `test_report_metrics.py` | 14 | 月度收益含首月、逐月复利等于总收益、月末分桶按北京时间、每个 `FillReason` 都有中文标签与颜色 |

测试全部用 `httpx.MockTransport` 与内存 SQLite，不触网、不依赖 `data/quant.db`。

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

### 13.3 回归证据：修复前后指标（**不可比**）

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

---

## 14. 已知限制（明确记录，不掩饰）

1. **定仓价与成交价近似**：`margin`/`notional` 用 `signal.price_ref`（bar.close）计算，实际成交在下一根 bar.open ± 滑点，误差约 ±2bps 量级；`stop_loss` 已改用真实成交价重算，故只有仓位量带此近似
2. **不模拟强平**：`FillReason.LIQ`、`Position.tp1_done/breakeven_done/extreme_price/realized_pnl` 等为预留字段，引擎不触发强平；10× 下强平约在 −7.45% 价格，而 SL 设在 1.5~2.0×ATR（远更近），常规路径不会走到
3. **无 TP1 分批止盈 / 保本移动**：TP 命中即全平；`risk.yaml` 里对应键已注释说明未实现
4. **成交时间戳**：入场/信号平仓记在 bar 收盘，止损/止盈记 bar 中点 —— 逐笔时长与真实 tick 路径有差异（1m/秒级路径数据未回填，`backtest.intra_bar_path=exact_1m` 是未来开关）
5. **只有 REST、只有 ETHUSDT**：无 WebSocket、无 1m；BTCUSDT 在 `symbols.yaml` 中 `enabled=false` 且未回填
6. **维持保证金率取最高档**：缺签名接口 `leverageBracket`，属安全侧偏差（仓位偏小）
7. **单策略单品种回测**：`backtest` 一次只跑一个 `--strategy`；多策略并行是 paper 服务的需求（§15）
8. **`models/` 未纳入 gitignore**：9.0 MB 折模型会被 `git add` 带进仓库，提交前需确认
9. **K 线由 5m 聚合**：15m/1h/4h/1d 图与指标读的是交易所 1h/4h 值；`check-data` 的 `htf_consistency` 会显式报出偏差（当前实测偏差为 0），两者不一致时以交易所值为准并在报表标注
10. **`guard`/`caps` 的风控闸门只在设计上存在**：回测不启用，paper 服务必须自己实现

---

## 15. 下一步：模拟交易 / 信号推送服务（未实现，接口已预留）

需求（用户原话口径）：服务器上以 shell 脚本启动，多策略并行运行策略执行与飞书信号推送；真实下单由用户手动完成；暂不处理多策略仓位共存。

已具备的地基：`contracts.Signal/FillReason/Position`、`BaseStrategy`（`Mode.LIVE` 已在契约里）、`MarketDataService`（只读、无副作用）、`FeishuNotifier + build_card`、`sizing.compute_risk_size`（拒绝原因可直接进卡片）、`risk.yaml` 的 `[paper]` 键（`caps.*`、`guard.*`）。

待落地（设计已在 `docs/lgbm-strategy-and-paper-trading-design.md`）：
1. `src/quant/paper/`：复用 `backtest/position.py` 的持仓对象，模拟账本 + 检查点；Store 新增表 `paper_accounts / paper_positions / paper_signals / paper_checkpoints`，`signal_id` 幂等去重
2. CLI `quant paper run|once|status`（多策略 = 多个独立账户并行，各自 `strategy + symbol`）
3. `scripts/paper.sh`：`start/stop/restart/status/once/logs`，进程内 Binance 走代理、飞书直连；配套 `deploy/paper-trading.service`
4. 实跑验证：至少一次 `once` 出卡 + 一次 `run` 跨 1h 收盘周期

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
uv run quant train --symbol ETHUSDT --eval-start 2026-04-01 --eval-end 2026-10-01
uv run quant dump-card                      # 只看卡片
uv run quant test-card                      # 真发飞书（需 .env）
uv run pytest -q && uv run ruff check --select F,E9 src tests
```

## 附录 C：指标口径速查

| 指标 | 公式 |
|---|---|
| 单笔保证金收益率 | `pnl(净) / margin_usdt × 100`，10× 下约为价格变动幅度 ×10 |
| 保证金 | `notional / global_max_leverage`，加仓按每次占用累计 |
| 净盈亏（逐笔） | `价格盈亏 − 开仓费 − 平仓费 − 持仓期资金费` |
| R 倍数 | `实现价距 / (sl_atr_mult × entry_atr)`，符号跟随盈亏方向 |
| 账本不变量 | `sum(trades.pnl) == final_equity − initial_equity` |
