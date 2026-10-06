# LightGBM 策略（lgbm_v1）+ 多策略模拟交易信号服务 · 技术设计

状态：**待确认决策点，未开始编码**（见 §10）
关联：`docs/rule_v4-trend-design.md`（同区间同口径的对照组）、`docs/strategy-base-design.md`、`docs/quant_platform_design.md`

---

## 0. 需求与背景

用户原话拆成两件事：

1. **rule_v4 达不到周 10% → 换 LightGBM 再试一次**，"尽可能靠近该策略目标即可"（目标已从硬约束降级为方向性目标）。
2. **做一个 shell 脚本，在服务器上启动模拟交易服务**：执行策略 + 飞书推送信号，**支持多策略并行**；仓位共存问题暂不处理；**真实下单由用户手动完成**。

rule_v4 的实测基线（同区间、同会计口径，2026-10-04 复算确认）：

| 指标 | rule_v4 |
|---|---|
| 区间 | 2026-04-01 ~ 2026-10-01（ETHUSDT，100U，10×） |
| 总收益 / 最大回撤 | **+18.98% / −12.39%** |
| 交易数 / 胜率 / PF / Sharpe | 45 / 51.1% / 1.37 / 1.35 |
| 周收益 ≥10%（对保证金）的周数 | 3 / 16 个有仓周 |
| 瓶颈 | **信号频率**（ADX≥32 门槛 → 10 个空仓周），不是单笔赔率 |

ML 的假设价值正好打在这个瓶颈上：**在保持单笔期望为正的前提下把开仓频率提上来**。
这句话是本设计的立论，也是它必须被样本外数据检验的地方。

---

## 1. 数据事实（实测，不是假设）

`data/quant.db` 当前覆盖（2026-10-06 查询）：

| 表 / 周期 | 根数 | 范围（UTC） |
|---|---|---|
| bars 5m | 289,611 | 2024-01-01 00:00 ~ 2026-10-02 14:10 |
| bars 15m | 96,517 | 2024-01-01 ~ 2026-10-02 09:00 |
| bars 1h | 24,129 | 2024-01-01 ~ 2026-10-02 08:00 |
| bars 4h | 6,032 | 2024-01-01 ~ 2026-10-02 04:00 |
| funding | 3,017 | 2024-01-01 ~ 2026-10-02（含 `funding_rate`、`mark_price`、`interval_h`） |

推论：

* 训练集可用 **2024-01-01 ~ 2026-03-31（约 19.5k 根 1h）**，评估区间保持 **2026-04-01 ~ 2026-10-01（4,392 根 1h / 26 周）**，与 rule_v4 **完全同区间、同口径**，数字可直接对比。
* 资金费率与标记价基差可以做特征（表里已有），不需要新数据源。
* **没有**订单簿、持仓量、清算数据；`ws_base_usdm` 配置悬空无代码使用 → 本设计不依赖这些。
* 26 周样本外的统计功效很低：约 60~120 笔交易时，PF 的 95% 置信区间宽到 ±0.4。
  所以**样本外唯一可信的结论方向是"能不能明显超过 rule_v4"**，而不是"能不能精确达到周 10%"。

---

## 2. 为什么选 LightGBM（选型对比）

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| **LightGBM（GBDT）** | 小样本表格数据上通常最优；原生处理 NaN（预热期指标不必丢样本）；特征重要值/SHAP 可解释；训练秒级；CPU 即可 | 时序分布漂移敏感；超参易过拟合 | **选它** |
| LSTM / Transformer | 理论上能学时序依赖 | 4k~20k 样本远不够，金融时序上对表格特征通常打不过 GBDT；调参成本高；服务器推理要 GPU 或慢 | 不选（与既有"ML 基类仍待做"的规划不冲突，基类留接口） |
| 线性 / Logistic | 极稳健、可解释 | 表达不了 ADX×波动×资金费率的交互，天花板低 | 作为 baseline 一并跑（`lgbm_v1` 的对照组） |
| 规则继续调参 | 已做过 | rule_v4 的 `sl/tp` 已在窄脊上，`adx_min` 是唯一稳的轴 | 已到边界 |

依赖现状：`pyproject.toml` 里**没有** lightgbm / scikit-learn / scipy；本机 numpy 2.5.3 + pandas 3.0.6（较新），
libomp 已安装（`/opt/homebrew/opt/libomp`）。落地第一步是 `uv add lightgbm scikit-learn` 并跑通 smoke test，
若 pandas 3 不兼容则锁版本，**这一条在写任何策略代码之前先验证**。

---

## 3. 特征工程

### 3.1 硬约束

1. **训练与推理同一份代码**：`features/builder.py::build_feature_frame(bars_1h, bars_4h, funding) -> DataFrame`。
   训练取全部行，推理取最后一行。绝不允许两处各写一遍公式（train/serve skew 是这类系统最常见的死法）。
2. **只用已收盘 bar**。第 t 行的所有特征只能索引 ≤t 的数据；跨周期对齐一律用"4h bar 已收盘"的口径
   （沿用引擎现有 `htf` 指针语义：4h bar 只在其时间戳 ≤ 当前 1h 收盘时刻时才可见）。
3. 价格类特征全部用**比值/差 ÷ ATR**，不用绝对价，保证跨期可比。
4. 缺失用 NaN 保留（LightGBM 原生支持），不做 fillna(0)（0 会被模型当成真实读数）。

### 3.2 特征清单（约 34 维，全部来自现有 `features/indicators.py`）

| 组 | 特征 | 来源 |
|---|---|---|
| 收益 | log return 1/3/6/12/24 根 1h | `close.pct_change()` |
| 趋势结构 | `(close-ema20)/atr`、`(ema20-ema60)/atr`、`macd_hist/atr`、`macd_hist` 一阶差分 | ema/macd |
| 趋势强度 | `adx_14`、`adx_14` 差分、`adx_prev_3` | adx |
| 位置 | Donchian 位置 `(close-don_low24)/(don_high24-don_low24)`、`(close-don_high24)/atr` | 引擎 `_precompute_htf` 已有 |
| 摆动 | `rsi_14`、`srsi_k`、`srsi_d`、`srsi_k - srsi_d` | rsi/stoch_rsi |
| 波动 | `atr/close`、`atr/atr_median_100`、24 根实现波动率、`(bb_upper-bb_lower)/atr` | atr/bollinger |
| 量能 | `volume/vol_median`、`taker_buy_volume/volume`、`quote_volume` 的 24 根 z 分数 | bars |
| 4h 背景 | 4h `(close-ema20)/atr`、4h `macd_hist/atr`、4h `adx_14`、4h/1h 收盘价基差 | `_precompute_htf` |
| 资金 | `funding_rate`（8h 年化）、`mark_price/close - 1`、距下次结算的小时数 | funding 表 |
| 日历 | `sin/cos(hour_utc)`、`dayofweek`（加密市场有周内效应） | ts |

刻意**不放**的东西：原始价格、原始成交量绝对值、任何未来信息、任何由标签衍生出的列。

---

## 4. 标签设计

Triple-Barrier（Lopez de Prado），障碍用 **1h ATR** 表达，与引擎止损口径同源：

```
对每根 1h bar t：
  entry   = close[t]
  upper   = entry + k_up * atr[t]      k_up 默认 3.0
  lower   = entry - k_dn * atr[t]      k_dn 默认 2.0
  horizon = H 根 1h                    H   默认 24
  逐根扫描 t+1 .. t+H：
    先触 lower → y = -1
    先触 upper → y = +1
    同时触（同一根 bar 内）→ 保守取 -1（与回测引擎 sl_first_on_conflict 一致）
    到期未触 → r = (close[t+H] - entry) / atr[t]
              |r| <= theta → y = 0（flat）；否则按符号取 ±1
  theta = 往返成本 / atr，默认 (0.05%*2 + 2bp)*10 / atr_median  ≈ 0.45% 价格位移
        （10× 杠杆下 0.45% 价格 ≈ 4.5% 保证金，低于这个幅度扣掉手续费对本金没意义）
```

**模型只负责"开不开、往哪个方向开"，平仓继续交给引擎**（SL/TP/移动止损/时间止损，全部沿用
rule_v4 已验证的那套）。理由：平仓规则是风控而不是预测问题，交给模型会让回测口径与 rule_v4 不可比，
也会让"保守成交假设"这条既定原则失效。

### 4.1 决策规则

`multi:softprob` 输出 `(p_up, p_dn, p_flat)`，令 `edge = p_up - p_dn`：

* `edge > τ` → `OPEN_LONG`；`edge < -τ` → `OPEN_SHORT`（`allow_short=False` 时只保留多头）
* 已有持仓时不因反向信号自动平仓（出场仍由引擎规则决定），避免与 rule_v4 行为不一致
* 冷却：同方向开仓后至少间隔 `cooldown_bars`（默认 6 根 1h），控制换手
* `Signal.meta["atr"] = atr_1h`（决策周期 ATR，E1 已实现的通道）
* `Signal.meta["size_mult"] = clip((edge - τ) / (0.5 - τ), 0.2, 1.0)` —— 置信度越高仓位越大，
  与 rule_v4 的 `w4` 共用同一条钳制通道，仓位原则不变（1 成试仓 + 最多 2 次 ×2 成加仓）

τ 不是拍脑袋定的：在**验证集**（不是测试集）上取"每笔平均毛利 > 往返成本"的最小 τ，
并把 τ 从 0.05 扫到 0.35 的完整曲线写进报告（§7）。

---

## 5. 训练协议（防泄漏是第一位）

```
起点 2024-01-01，评估区间 [2026-04-01, 2026-10-01)
每 step = 1 个月重训一折（walk-forward / 时间外推）：
  fold_i:
    train  = [T_i - W, T_i)          W = 12 个月（滚动，决策点 Q2）
    valid  = train 的最后 1 个月      早停用（early_stopping_rounds=100）
    embargo = (H + 2) 根 1h 直接从 train/valid 尾部剔除   ← 标签右边界会溢出到测试窗
    预测区间 = [T_i, T_i + 1month)
共约 21 折覆盖评估区间。
```

* **禁止随机 K-fold**（时序泄漏会让指标虚高到不可信）。
* 每折模型落盘：`models/lgbm_v1/{fold_tag}/model.txt` + `feature_spec.json`（特征列顺序 + hash）
  + `train_meta.json`（训练区间、dataset_version、LightGBM 版本、参数）。
  推理时按 `ts_signal` 选"训练截止 < 当前 bar"的**最新一折**模型 —— 这条选择逻辑必须有单元测试，
  选错折 = 前视 = 整份报告作废。
* 超参（写进 `config/strategies/lgbm_v1.yaml`，不再硬编码）：
  `num_leaves=31, learning_rate=0.03, n_estimators=2000(早停截断), min_child_samples=200,
  feature_fraction=0.7, bagging_fraction=0.8, lambda_l2=5.0, max_depth=6`
  —— 对 2 万样本这些值偏保守是故意的，`min_child_samples=200` 是第一道防过拟合闸。
* 随机种子固定；`dataset_version` 记进 metrics.json（数据变了要重训，不能混用旧模型）。

---

## 6. 代码接入点（改动清单）

| # | 位置 | 改动 | 为什么必须 |
|---|---|---|---|
| M1 | `features/builder.py`（新） | `build_feature_frame()`、`build_labels()` | 训练/推理同源 |
| M2 | `strategy/ml/base.py`（新） | `MLBaseStrategy(BaseStrategy)`：历史窗口持有、按 ts 选模型、`predict→Signal` | 现在只有规则策略，无 ML 基类 |
| M3 | `strategy/ml/lgbm_v1.py`（新） | 具体策略 + `LGBMv1Config` + `engine_defaults()`（sl/tp 与 rule_v4 同：2.0/3.5、time_stop 864） | — |
| M4 | `strategy/__init__.py` | import 新策略完成注册（现有注册靠 import 副作用） | 否则 `get_strategy("lgbm_v1")` 找不到 |
| M5 | `backtest/engine.py` | **E7**：`run()` 开头调用 `strategy.reset()`；**E8**：`bind_history(bars)` 只读注入，让策略能取滚动窗口 | 引擎现在完全不调 `reset()`（rule 策略无状态所以没暴露），ML 有 buffer 必须重置；`on_bar` 只给当前 row，无法算窗口特征 |
| M6 | `train/`（新）+ CLI `quant train` | 特征/标签构建、walk-forward、模型落盘、训练报告 | 仓库现在零训练代码 |
| M7 | `config/strategies/lgbm_v1.yaml` | 特征列清单、超参、τ、embargo、模型目录 | 现有 v2/v3/v4 参数硬编码，ML 参数必须可配置可复现 |
| M8 | `pyproject.toml` | `uv add lightgbm scikit-learn`（+ 版本锁定） | 依赖缺失 |

E7/E8 是对既有引擎的**行为增强**，改完必须回归 rule_v1~v4（数字必须一字不差，因为它们无状态）。

---

## 7. 评估与验收协议（先写死，再跑，避免挑数据）

**主表（样本外 2026-04-01 ~ 2026-10-01，与 rule_v4 同口径）**

| 指标 | rule_v4 基线 | lgbm_v1 |
|---|---|---|
| 总收益 / 最大回撤 | +18.98% / −12.39% | 待测 |
| 交易数 / 胜率 / PF | 45 / 51.1% / 1.37 | 待测 |
| 周收益 ≥10%（对保证金）周数 | 3/16 有仓周 | 待测 |
| 空仓周数（频率瓶颈的直接证据） | 10/26 | 待测 |
| 会计恒等式 `sum(pnl) == Δequity` | 必须成立 | 必须成立 |

**必须一并输出的稳健性检查**（缺任何一项，结论不算数）：

1. τ 扫描曲线（0.05→0.35）：收益/回撤/交易数三条线，看是否存在宽平台还是尖峰；
2. 成本 ×2、×4 后的收益（rule_v4 在 ×4 时从 +19% 掉到 −13%，这是它的致命项）；
3. 分月收益 + 多空拆分（利润是否集中在单一月份/单一方向）；
4. 折间方差：21 折各自验证集指标分布，方差过大说明机制不稳；
5. 特征重要值 top10 + PSI（训练窗 vs 评估窗的分布漂移）；
6. 对照组：同特征的 logistic 回归 + "永远做多" + rule_v4。

**过拟合红旗**（出现即如实报告为"不可用"，不做参数挽救）：样本外 PF<1；
最佳 τ 与次佳 τ 的结果差 >2 倍；单月贡献 >50% 总收益；PSI >0.25 的特征超过 1/3。

---

## 8. 模拟交易信号服务（paper trading）

### 8.1 范围边界

* **做**：定时拉数据 → 多策略并行推理 → 纸面持仓管理 → 飞书信号卡片 → 状态落库。
* **不做**：真实下单/撤单（用户手动执行）、跨策略合并保证金与仓位共存（用户明确"先不管"）、
  Web UI、历史回测重放。

### 8.2 进程形态

推荐**单进程多策略**（`quant paper run`）：主循环按最细决策周期的 bar 收盘唤醒，
依次跑 `symbols.yaml` 里每个 symbol × 该 symbol 启用的策略列表
（`strategies: [rule_v1]` 这个字段**已存在但从未被读取**，本服务是第一个消费者）。

理由：数据只拉一次（Binance REST 有分钟级 weight 预算，多进程会重复扣配额）、
SQLite 单写者更安全、飞书限流（已实现 40/min）在单进程内才真正生效。

### 8.3 运行链路

```
bar 收盘对齐唤醒（决策周期 closeTime < 交易所 server_now，再等 resync_grace_ms）
  → Backfiller(resume=True) 增量补 5m/1h/4h + funding
  → 每个 (symbol, strategy)：
      读 checkpoint(last_processed_ts)
      加载窗口数据 → strategy.on_bar(...)（新 bar 逐根喂，不重放历史）
      纸面持仓管理（SL/TP/移动止损/时间止损，复用引擎同一套函数）
      有信号 → 写 paper_signals（signal_id 唯一键，幂等）→ 飞书卡片
  → 更新 checkpoint
```

* **收盘对齐**：绝不在 bar 未收盘时评估信号（否则推的是会变的信号）。
* **时钟**：启动时 `sync_time()`，之后每 6h 复检；偏差 >2s 记 WARNING 并按 server_time 校正。
* **幂等重启**：`signal_id = sha1(symbol|strategy|action|ts_open_ms|seq)` 落库唯一键，
  重启后从 checkpoint 续跑，**不会重发历史信号**（现在 `dedup_ttl_min` 只存在于 YAML，全仓库无人读取）。

### 8.4 新增持久化（`data/store.py` SCHEMA 追加，`CREATE TABLE IF NOT EXISTS`）

| 表 | 关键字段 | 用途 |
|---|---|---|
| `paper_accounts` | symbol, strategy, equity, initial_equity, updated_at | 每策略独立 100U 纸面账户（Q4） |
| `paper_positions` | side, entry_price, qty, margin_usdt, entry_atr, sl, tp, add_count, opened_at, peak_favorable | 持仓与加仓/移动止损状态 |
| `paper_signals` | signal_id PK, ts_signal, symbol, strategy, action, price_ref, reason, sent_ok, sent_at, payload_json | 去重 + 审计 + 事后统计 |
| `paper_checkpoints` | symbol, strategy, last_processed_ts_ms | 断点续跑 |

### 8.5 纸面撮合必须与回测同一套代码

把 `BacktestEngine` 里的持仓管理（SL/TP 冲突取先、bar 极值成交、中点时间、金字塔加仓触发、
移动止损）抽到 `backtest/position.py`，回测与 paper 共用。
**不允许复制第二份实现** —— 两套口径迟早漂移，漂移后 paper 结果就失去意义。
（保守口径是既定原则：止损按 bar 极值成交 + 中点时间。）

### 8.6 卡片与风控

* 复用 `notify/cards.py::build_card(signal, decision)`；
  需要把 `RiskDecision` 的构造从引擎里解耦成 `risk/sizing.py::decide(...)`（现在 `compute_risk_size` 只在引擎内部被调用）。
* **卡片时间当前是 UTC**（`cards.py:75`），与"展示一律北京时间"的既定要求冲突 → 一并修成 UTC+8。
* 卡片内容：方向/价格/数量/止损/止盈/杠杆/保证金占用/置信度（模型概率或规则评分）/策略名/触发原因；
  标题色按 Action 映射（现有 green/red/blue）。
* 发送失败：重试已在 `FeishuNotifier.send_payload` 实现（网络/429/5xx 退避）；
  业务错不重试 → `sent_ok=0` 落库，下一轮补发一次，再失败则记 ERROR 不阻塞主循环。

### 8.7 shell 脚本 `scripts/paper.sh`

```
./scripts/paper.sh start|stop|restart|status|once|logs|add-strategy
```

* `start`：预检（`.venv` 存在、`uv` 可用、`.env` 里 `FEISHU_WEBHOOK_URL` 非空、DB 可读、
  模型工件存在且 `dataset_version` 匹配）→ `nohup uv run quant paper run >> logs/paper.out 2>&1 &`
  → PID 写 `run/paper.pid`。
* `once`：只跑一轮就退出（部署后先手工验证一次，不常驻）。
* `status`：PID 存活 + 每个 (symbol,strategy) 的 checkpoint 落后多少根 bar + 最近一次成功推送时间。
* `stop`：`TERM` → 最多等 30s 优雅退出（等当前轮跑完，避免写坏状态）→ 仍存活才 `KILL`。
* `logs`：`tail -f logs/paper.out`。
* 环境变量：`QUANT_PROXY_URL` 驱动出网代理（本机必须 `127.0.0.1:7897`；服务器按实际情况设或留空直连）。
* 另附 `deploy/paper-trading.service`（systemd 模板，`Restart=always` + `EnvironmentFile=.env`）作为可选托管方式。

### 8.8 失败模式与处置

| 情况 | 处置 |
|---|---|
| 数据缺口（`find_gaps`） | 该轮跳过该策略并推一条告警卡片，不用残缺窗口出信号 |
| 模型缺失/`dataset_version` 不匹配 | 拒绝启动该策略（宁可不推，不推错） |
| 评估区间外无新 bar（交易所延迟） | 等 3 次 ×10s，仍无则跳过 |
| 飞书 429 | 已有退避；持续失败则 `sent_ok=0` 落库，恢复后补发 |
| 进程崩溃 | 重启从 checkpoint 续，`signal_id` 唯一键保证不重发 |
| 时钟漂移 | 以 `server_time_ms` 为准；本地时钟只做日志 |

---

## 9. 里程碑与顺序（每步都有可验证产物）

1. **依赖预检**：`uv add lightgbm scikit-learn` + smoke test（pandas 3.0.6 / numpy 2.5.3 是否兼容）→ 不通就先锁版本。
2. M1 特征/标签构建 + 单测（防前视用"截断数据重算必须逐位相同"的断言来证明，不靠人肉 review）。
3. M6 `quant train` + walk-forward → 产出 21 折模型 + 折间指标报告。
4. M2/M3/M4/M5 ML 基类 + 策略 + 引擎 E7/E8 → **回归 rule_v1~v4 数字必须完全不变**。
5. 样本外回测 + §7 全套稳健性 → 如实报告（达不到就写达不到）。
6. `backtest/position.py` 抽取 + paper 服务 + 新表 + 卡片北京时间修正 + 单测。
7. `scripts/paper.sh` + systemd 模板 + 服务器 `once` → `start` 实跑，验证真收到卡片。

第 5 步是**决策门**：如果样本外明显不如 rule_v4，就停在报告结论，不硬推 paper 服务去跑一个更差的策略。

---

## 10. 需要确认的决策点

* **Q1 决策周期**：1h（与 rule_v4 同口径，样本 19.5k 根，可直接对比）／15m（样本 ×4、频率更高、成本与噪音也翻倍）。
* **Q2 训练窗口**：滚动 12 个月（适应近期机制）／expanding 全历史（样本多但 2024 老机制可能误导）。
* **Q3 服务形态**：单进程多策略（推荐）／每策略一进程（shell 并行）。
* **Q4 纸面账户口径**：每策略独立 100U（推荐，符合"先不管仓位共存"）／共享 100U 但仓位独立记账。
* **Q5 模型更新**：服务器每周自动重训（依赖全、能吃到最新数据）／本地训练 + 上传工件（服务器轻依赖）。

（Q1~Q5 的答案会直接改写 §4/§5/§8 的默认参数，所以先问再动手。）

## 11. 决策结果与实测结论（2026-10-06 回填）

### 11.1 决策点

| 决策 | 选择 | 落地位置 |
|---|---|---|
| Q1 决策周期 | **1h** | `MLBaseStrategy.decision_interval = "1h"`，`required_intervals=("5m","1h","4h")` |
| Q2 训练窗口 | **滚动 12 个月** | `TrainConfig.window_months=12`，`plan_folds()` 每折 train 12M / valid 1M / test 1M |
| Q3 服务形态 | **单进程多策略** | `src/quant/paper/`，见 §12 |
| Q4 纸面账户 | **每策略独立 100U** | `paper_accounts` 表按 `(strategy, symbol)` 记账 |
| Q5 模型更新 | **本地训练 + 上传工件** | 服务器只读 `models/{name}/{symbol}/{tag}/`，不装训练依赖 |

### 11.2 §4 标签口径修正

原文写「θ = 往返成本」，实测 θ 只按成本时 flat 类仅占 2%（任何微小波动都算方向信号），
模型学到的其实是噪音。改为 **θ = max(1.0×ATR, 成本)** 后 flat 占 10~15%。
`flat_atr_mult` 取 0.5 / 1.0 / 1.5 时 flat 占比实测为 8.3% / 14.8% / 19.3%，默认 1.0。

### 11.3 训练与推理一致性（实测）

| 校验项 | 方法 | 结果 |
|---|---|---|
| 无前视 | 输入截断到第 N 根重算，比较前 N 根特征 | 逐位相同（`tests/test_features_builder.py`） |
| train/serve skew | 推理端 90 天前缀 vs 训练端 2024 起全长前缀，38 维逐列取最大归一化差 | **2.3e-10**，最大在 `bb_width`，其余 ≤1e-11 |
| 特征集漂移 | `feature_spec_hash` 不符即抛 `ValueError` | 折加载时强制校验 |
| 账本恒等式 | `sum(trades.pnl)` vs 净值增长 | 误差 0.000（全部 τ × 方向组合） |

### 11.4 样本外结果：当前特征集没有可交易的 edge

评估区间 2026-04-01 ~ 2026-10-01（ETHUSDT，100U，10×，taker + 2bp 滑点，保守口径）：

| τ | 允许做空 收益/回撤/笔数/PF | 仅做多 收益/回撤/笔数/PF |
|---|---|---|
| 0.10 | -16.6% / -18.4% / 180 / 0.61 | -3.4% / -4.9% / 98 / 0.84 |
| 0.15 | -13.9% / -14.8% / 145 / 0.62 | -4.0% / -5.4% / 85 / 0.79 |
| 0.20 | -10.7% / -11.5% / 103 / 0.58 | -2.1% / -3.8% / 61 / 0.82 |
| 0.25 | -9.8% / -10.8% / 75 / 0.54 | -1.0% / -3.6% / 49 / 0.89 |
| 0.30 | -10.2% / -10.7% / 48 / 0.39 | -0.8% / -3.6% / 34 / 0.88 |

对照：**rule_v4 +18.98% / -12.39%（45 笔，PF 1.37）**，**买入持有 +27.0%**。

原因定位（不是接线问题）：

1. **方向命中率 45~48%，等于抛硬币**。验证月（紧邻训练窗）57.8%，测试月 47.8% —— 典型的
   窗口外机制漂移，模型拟合的是训练窗内的 regime。
2. **按障碍口径算，平均单笔在成本之前就已经是负的**（对称障碍对照：-0.09 ~ -0.17 ATR，
   成本约 0.2 ATR）。所以不存在「只是成本吃掉」的情况。
3. 做空腿显著更差（胜率 32.8% vs 做多 47.6%），而区间是 +27% 的上行 —— `k_dn < k_up`
   把「下跌」学成多数类，上行情势下系统性做空。
4. 移动止损不是原因：关掉移动止损（`trail_trigger_r→∞`）各组合仍为负。
5. 折模型 top gain 特征里 `dayofweek` 排第一，属于过拟合气味，事后看确实成立。

**结论与处置**：`lgbm_v1` 实现保留、默认参数不改（靠调 τ 或关闭做空把它「调成正收益」
属于挑数据）。真正的瓶颈不是规则 vs 模型，而是 24h 障碍上这 38 维特征本身不含
可提取的方向信息。下一步接入 paper trading 做**前瞻性**观察：回测已证明它不行，
模拟盘的意义是验证这个否定结论在实时链路上是否成立，而不是再挖一遍同一份数据。

### 11.5 LightGBM API 实测注意

* `lgb.train(..., valid_sets=...)` 才有效；sklearn 接口的 `eval_set` 触发
  `LGBMDeprecationWarning`，sklearn 1.9 + lightgbm 4.7 用 `eval_X` / `eval_y`。
* `LGBMClassifier` 没有 `save_model`，要用 `booster_.save_model()`，加载用
  `lgb.Booster(model_file=...)`。
* macOS 需要 `libomp`（本机 `/opt/homebrew/opt/libomp`），CI/服务器要注意。
