# 100U 实盘金字塔加仓 · 技术设计

## 用户目标

- 实盘资金 100U，10x 杠杆
- 每笔先用 1 成（10U 名义）试仓，验证方向后最多加仓 2 次
- 每次加仓 ≥ 2 成（≥ 20U 名义）
- 加仓后必须重设移动止损
- 真实模拟 100U 本金下的手续费压力

## 设计要点

### 1. 金字塔仓位结构

| 阶段 | 仓位占比 | 数量（相对满仓） | 累计风险 |
|------|---------|-----------------|----------|
| 试仓 | 10% | 10% × 满仓 | 1 次止损额 |
| 第 1 次加仓 | ≥ 20% | 30% × 满仓 | 同上 |
| 第 2 次加仓 | ≥ 20% | ≥ 50% × 满仓 | 同上 |
| 加仓后止损 | — | — | 重设为加权均价 ± ATR |

**目标总仓位**：试仓 10% + 加仓 20% + 加仓 20% = 50% 满仓（半仓），留 50% buffer 应对极端波动。

### 2. 加仓条件（动态，需确认）

"方向正确"如何判定？下面三种规则需用户选择：

**A. 浮盈 R 倍数阈值**（最简单）
- 浮盈 ≥ 0.5R（如 0.5×ATR），即价格朝有利方向移动 ≥ 0.5 个止损距离
- 加仓次数 ≤ 2
- 优点：硬性阈值，避免主观
- 缺点：可能错过大波段（如果价格快速突破 0.5R 后回调到 0.3R）

**B. 结构突破 + 浮盈**（推荐）
- 浮盈 ≥ 0.3R **且** 5m bar 收盘价突破近期 swing high/low
- 优点：顺应趋势，过滤假突破
- 缺点：需额外指标

**C. 时间 + 浮盈**（最宽松）
- 开仓后每过 N 根 5m bar 检查一次，浮盈 ≥ 0.3R 即可加仓
- 优点：捕获震荡中段
- 缺点：震荡市容易被骗

### 3. 加仓后止损重设

每次加仓后：
1. 重新计算**加权均价** = Σ(qty_i × price_i) / Σ(qty_i)
2. SL = 加权均价 ± ATR × sl_atr_mult
3. 加仓后**激活移动止损**：盈利 ≥ trail_trigger 后，止损沿趋势方向跟随

### 4. 100U 实盘资金模拟模式

新增 `--live-equity 100` 参数：
- 实际风控/仓位计算仍用初始 `--equity`（保持回测可重复性）
- **手续费占比统计**按 live-equity 模拟
- 报告新增"小资金压力测试"卡片：显示手续费占比 × 本金放大效应

或者更彻底：**所有计算都用 live-equity**，避免误导。

### 5. 引擎改动清单

```python
@dataclass
class BacktestConfig:
    # 已有字段...
    live_equity: float | None = None   # 100U 实盘模拟（None 则用 initial_equity）

@dataclass
class RiskDecision:
    # 已有字段...
    is_pyramid_add: bool = False
    add_index: int = 0

@dataclass
class Position:
    # 已有字段...
    target_qty: float
    filled_qty: float
    avg_entry_price: float
    add_count: int = 0
    add_levels: list[AddLevel] = field(default_factory=list)
    trailing_active: bool = False
    trailing_peak_price: float = 0.0
```

新增引擎方法：
- `_scale_in(signal, decision, bar)` — 试仓或加仓
- `_check_add_condition(bar, row, htf)` — 加仓触发
- `_update_trailing_stop(bar)` — 移动止损
- `_recalc_sl_after_add()` — 加仓后重设 SL

### 6. 策略接口调整

`on_bar` 返回新增 Action：
```python
class Action(Enum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    ADD_TO_LONG = "add_to_long"     # 新
    ADD_TO_SHORT = "add_to_short"   # 新
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
```

### 7. CLI 参数

```python
@app.command("backtest")
def run_backtest(
    ...
    # 已有
    equity: float = typer.Option(100.0, help="实际资金（含 live 模拟）"),
    lev: float = typer.Option(10.0),
    # 新增
    entry_pct: float = typer.Option(0.10, help="试仓比例"),
    add_pct: float = typer.Option(0.20, help="每次加仓比例下限"),
    add_max: int = typer.Option(2, help="最大加仓次数"),
    add_trigger_r: float = typer.Option(0.5, help="加仓盈利 R 门槛"),
    trail_trigger_r: float = typer.Option(1.0, help="移动止损激活 R 门槛"),
    add_mode: str = typer.Option("r_threshold", help="加仓模式: r_threshold / structure / time"),
):
```

### 8. 报告新增卡片

在 HTML 报告的"策略参数"区域：
- 试仓 / 加仓 / 最大加仓次数
- 加仓盈利门槛（R）
- 移动止损激活门槛（R）
- 加仓模式说明

## 决策点（待用户确认）

1. **方向确认规则**：选 A / B / C 中的哪一个？推荐 B（结构突破 + 浮盈），但用户可能觉得 A 简单直接
2. **加仓后是否重设 SL 到加权均价 - ATR**？还是更紧（如 -0.5 × ATR）？
3. **100U 实盘模拟**：是仅用于手续费占比统计，还是所有金额都用 100U 计算（更贴近实盘）？
4. **加仓次数上限**：用户已确认 ≤ 2，OK
5. **是否保留"纯一次性入场"模式**：作为开关，默认开启金字塔
