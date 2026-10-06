# 金字塔加仓策略 · 技术设计（已确认）

## 用户确认的决策

| 决策 | 结论 |
|------|------|
| 止损收紧频率 | 加仓后切换为**移动止损**（Trailing Stop） |
| 加仓 qty 来源 | 固定 `target_qty × add_frac`（不重走风险计算） |
| 加仓失败处理 | **跳过**，不降低 qty 重试 |
| 最大杠杆 | **固定 10×**，全程不调整 |

---

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_leverage` | 10 | 固定杠杆，全程不调整 |
| `entry_frac` | 0.30 | 初始试仓：30% 目标 qty |
| `add_frac` | 0.20 | 每次加仓：20% 目标 qty |
| `add_max_count` | 3 | 最大加仓次数（不含初始） |
| `add_profit_threshold` | 0.5R | 加仓盈利门槛（≥0.5R 才允许） |
| `add_atr_filter` | 1.0 | 加仓 ATR 过滤 |
| `stop_mode` | "atr" | 止损模式：atr / pct / manual |
| `stop_atr_mult` | 1.5 | ATR 止损倍数（stop_mode=atr 时） |
| `stop_pct` | 2.0 | 百分比止损（stop_mode=pct 时，% of entry） |
| `stop_manual` | 0 | 手动止损价（stop_mode=manual 时） |
| `trail_mode` | "pct" | 移动止损模式：pct / atr |
| `trail_pct` | 1.0 | 移动止损 %（低于最近高点） |
| `trail_atr_mult` | 0.8 | 移动止损 ATR 倍数（trail_mode=atr 时） |
| `trail_trigger` | 1.0R | 激活移动止损的盈利门槛（R） |

**最终配比**：初始 30% + 3×20% = 90% 目标 qty（预留 10% buffer）

---

## 止损模式详解

### 模式一：ATR 倍数（默认，trend following）
```
初始 SL 距离 = stop_atr_mult × ATR
SL = entry_price ± SL 距离（多头减，空头加）
```

### 模式二：百分比（mean reversion / scalp）
```
SL = entry_price × (1 - stop_pct%)   # 多头
SL = entry_price × (1 + stop_pct%)   # 空头
```

### 模式三：手动固定价
```
SL = stop_manual（固定价格，不随行情变动）
```

### 移动止损（加仓后激活）

激活条件：浮盈 ≥ trail_trigger（如 1.0R）且有加仓

**Pct 模式**：
```
stop = 最近 N 周期高点 × (1 - trail_pct%)
多头：stop = min(当前 stop, high_since_entry × (1 - trail_pct%))
```
stop 只上移，不下移。

**ATR 模式**：
```
stop = 最近 N 周期高点 - trail_atr_mult × ATR
```

---

## 仓位结构

```
target_qty = equity × risk_pct / (stop_dist)  # 基于配置的 stop_dist 计算
initial_qty = target_qty × entry_frac          # 30%
add_qty_n   = target_qty × add_frac             # 20% × 3次
final_qty   ≈ target_qty (90%)
notional    = final_qty × price_ref
leverage    = notional / equity                 # ≈ 9×（< 10× 因为 entry_frac+add_frac<1）
```

注意：`target_qty` 基于 `stop_dist = stop_atr_mult × ATR`（或 pct 模式对应值）计算，
保证无论几批次入场，总风险 = equity × risk_pct。

---

## 数据结构改动

### `Action` 新增枚举

```python
class Action(Enum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    ADD_TO_LONG = "add_to_long"   # 新增
    ADD_TO_SHORT = "add_to_short" # 新增
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
```

### `RiskDecision` 新增字段

```python
@dataclass
class RiskDecision:
    # 现有字段保留...
    target_qty: float = 0.0    # 目标全量 qty
    entry_frac: float = 1.0    # 本次入场比例
    is_add: bool = False       # 是否为加仓
    add_index: int = 0         # 第几次加仓（0=初始）
    stop_price: float = 0.0    # 本次入场的止损价
    trail_stop: bool = False   # 是否已激活移动止损
```

### `Position` 新增字段

```python
@dataclass
class Position:
    # 现有字段保留...
    target_qty: float = 0.0          # 目标全量 qty
    add_count: int = 0               # 已加仓次数
    avg_entry_price: float = 0.0     # 加权平均入场价
    base_stop_price: float = 0.0     # 初始止损价
    trail_stop_price: float = 0.0    # 当前移动止损价
    trail_active: bool = False       # 移动止损是否激活
    trail_peak_price: float = 0.0    # 持仓期间最高/最低价（用于 trail）
```

---

## 回测引擎改动

### 1. `compute_risk_size()` 新接口

```python
def compute_risk_size(
    signal, decision, equity, risk_pct, atr,
    sl_atr_mult: float, global_max_leverage: float,
    symbol_spec,
    entry_frac: float = 1.0,     # 试仓比例
    is_add: bool = False,
    add_index: int = 0,
) -> RiskDecision:
    # 核心逻辑：
    # 1. stop_dist = sl_atr_mult × atr（或 pct 模式）
    # 2. target_qty = equity × risk_pct / stop_dist   # 理论全量
    # 3. actual_qty = target_qty × entry_frac
    # 4. notional = actual_qty × price_ref
    # 5. leverage = notional / equity（cap 到 10×）
    # 6. if leverage > 10: margin = notional / 10
    # 7. margin = notional / leverage（= notional/10）
    # 8. stop_price = entry_price ± stop_dist
```

### 2. `_scale_in()` 方法

```python
def _scale_in(self, signal, decision, bar):
    """初始开仓或加仓到已有仓位。"""
    entry = _slip(bar.open)
    pos = self.position

    if pos is None:
        # 初始开仓
        pos = Position(
            position_id=uuid.uuid4().hex[:8],
            symbol=signal.symbol,
            side="long"/"short",
            qty=decision.qty,
            target_qty=decision.target_qty,
            entry_price=entry,
            avg_entry_price=entry,
            entry_ts=bar.ts_close,
            margin_usdt=decision.margin_usdt,
            stop_loss=decision.stop_price,
            base_stop_price=decision.stop_price,
            trail_stop_price=decision.stop_price,
            add_count=0,
            add_levels=[AddLevel(entry, decision.qty, bar.ts_close)],
        )
        self.position = pos
        self.cash -= decision.margin_usdt
    else:
        # 加仓：更新加权均价
        prev_qty = pos.qty
        pos.avg_entry_price = (
            pos.avg_entry_price * prev_qty + entry * decision.qty
        ) / (prev_qty + decision.qty)
        pos.qty = prev_qty + decision.qty
        pos.add_count += 1
        pos.add_levels.append(AddLevel(entry, decision.qty, bar.ts_close))
        self.cash -= decision.margin_usdt

    fee = _fee_qty(decision.qty, entry, self.cfg.taker_fee)
    self.cash -= fee
    self.fills.append(Fill(...))
```

### 3. 移动止损更新（每根 bar）

```python
def _update_trail_stop(self, bar):
    """每根 bar 更新移动止损。"""
    pos = self.position
    if pos is None:
        return

    cur = bar.high if pos.side == "long" else bar.low

    # 更新 peak
    if pos.side == "long":
        pos.trail_peak_price = max(pos.trail_peak_price, cur)
    else:
        pos.trail_peak_price = min(pos.trail_peak_price, cur)

    # 计算当前浮盈（R）
    if pos.side == "long":
        unreal = (cur - pos.avg_entry_price) / (self.cfg.sl_atr_mult * pos.entry_atr)
    else:
        unreal = (pos.avg_entry_price - cur) / (self.cfg.sl_atr_mult * pos.entry_atr)

    # 激活条件
    if not pos.trail_active and unreal >= self.cfg.trail_trigger:
        pos.trail_active = True

    if pos.trail_active:
        if self.cfg.trail_mode == "pct":
            if pos.side == "long":
                new_stop = pos.trail_peak_price * (1 - self.cfg.trail_pct / 100)
            else:
                new_stop = pos.trail_peak_price * (1 + self.cfg.trail_pct / 100)
        else:  # atr
            if pos.side == "long":
                new_stop = pos.trail_peak_price - self.cfg.trail_atr_mult * pos.entry_atr
            else:
                new_stop = pos.trail_peak_price + self.cfg.trail_atr_mult * pos.entry_atr

        # trail stop 只向有利于持仓方向移动
        if pos.side == "long":
            pos.trail_stop_price = max(pos.trail_stop_price, new_stop)
        else:
            pos.trail_stop_price = min(pos.trail_stop_price, new_stop)

        pos.stop_loss = pos.trail_stop_price
```

### 4. `_check_stops()` 使用当前 stop_loss

```python
def _check_stops(self, bar):
    pos = self.position
    if pos is None:
        return
    sl = pos.stop_loss  # base_stop 或 trail_stop
    hit = False
    exit_px = bar.close
    if pos.side == "long" and bar.low <= sl:
        hit, exit_px = True, float(sl)
    elif pos.side == "short" and bar.high >= sl:
        hit, exit_px = True, float(sl)
    if hit:
        self._close_trade(bar, FillReason.SL, _slip(exit_px, self.cfg.slippage_bps))
```

---

## 策略层改动（rule_v1）

`on_bar` 返回值增加 `Action.ADD_TO_LONG / ADD_TO_SHORT`：

```python
# 检测加仓条件
if self._should_add(bar, row, htf, dt, pos):
    return Signal(
        action=Action.ADD_TO_LONG,  # 或 ADD_TO_SHORT
        ...
    )
```

engine 识别 `is_add=True` 的 signal，走 `_scale_in` 而非清仓重建。

---

## CLI 参数扩展

```python
@app.command("backtest")
def run_backtest(
    ...
    # 新增
    lev: float = typer.Option(10.0, "--lev"),
    entry_frac: float = typer.Option(0.30, help="初始试仓比例"),
    add_frac: float = typer.Option(0.20, help="每次加仓比例"),
    add_max: int = typer.Option(3, help="最大加仓次数"),
    stop_mode: str = typer.Option("atr", help="止损模式: atr/pct/manual"),
    stop_atr: float = typer.Option(1.5, help="SL ATR 倍数"),
    stop_pct: float = typer.Option(2.0, help="SL 百分比"),
    trail_mode: str = typer.Option("pct", help="移动止损: pct/atr"),
    trail_pct: float = typer.Option(1.0, help="移动止损 %"),
    trail_trigger: float = typer.Option(1.0, help="激活移动止损的 R 门槛"),
):
```

---

## 风险控制要点

1. **杠杆固定 10×**：target_qty 基于 stop_dist 计算，notional 不超 10× equity
2. **总风险上限**：target_qty × stop_dist × pyramid_mult ≤ equity × risk_pct
3. **移动止损激活**：必须盈利 ≥ trail_trigger R 才激活，防止震荡被打
4. **加仓上限**：最多 3 次，失败则跳过
5. **止损可个性化**：ATR / 百分比 / 手动，差异化于市场通用策略
