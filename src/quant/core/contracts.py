from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

BarSource = Literal["rest", "ws"]


class Action(StrEnum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    ADD_TO_LONG = "add_to_long"
    ADD_TO_SHORT = "add_to_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class Mode(StrEnum):
    BACKTEST = "backtest"
    LIVE = "live"
    TEST = "test"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class FillReason(StrEnum):
    SIGNAL = "signal"
    SL = "sl"
    TP = "tp"
    TP1 = "tp1"
    TP2 = "tp2"
    TRAIL = "trail"
    TIME = "time"
    LIQ = "liq"
    FLIP = "flip"
    FLAT = "flat"


@dataclass(frozen=True)
class Bar:
    symbol: str
    interval: str
    ts_open: datetime
    ts_close: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    taker_buy_volume: float
    is_closed: bool
    trades: int = 0
    src: BarSource = "rest"


@dataclass(frozen=True)
class FundingBar:
    """资金费率是结算时刻的标量，无 OHLC，且结算网格独立于 K 线网格，故不并入 Bar。"""

    symbol: str
    funding_time: datetime
    funding_rate: float
    mark_price: float | None = None
    rate_type: str = "Regular"
    interval_hours: float = 8.0
    src: BarSource = "rest"


@dataclass(frozen=True)
class SymbolSpec:
    """来自 exchangeInfo 的交易规则。size/price 一律用 Decimal 承载，避免浮点裁剪错手数。"""

    symbol: str
    status: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: float
    maint_margin_rate: float
    max_leverage: float
    liquidation_fee_rate: float
    funding_interval_hours: float
    price_precision: int
    qty_precision: int
    fetched_at: datetime

    @property
    def is_tradable(self) -> bool:
        return self.status == "TRADING"


@dataclass(frozen=True)
class Signal:
    """策略只表达意愿，不带数量也不带价位；价位与数量由风控层决定。"""

    signal_id: str
    ts_signal: datetime
    symbol: str
    timeframe: str
    strategy: str
    action: Action
    price_ref: float
    strength: float
    reason: str
    meta: dict[str, Any] = field(default_factory=dict)
    mode: Mode = Mode.LIVE


@dataclass(frozen=True)
class RiskDecision:
    signal: Signal
    approved: bool
    qty: float = 0.0
    notional_usdt: float = 0.0
    leverage: float = 0.0
    margin_usdt: float = 0.0
    stop_loss: float | None = None
    take_profits: tuple[float, ...] = ()
    est_liq_price: float | None = None
    equity_usdt: float = 0.0
    risk_usdt: float = 0.0
    reject_reasons: tuple[str, ...] = ()

    @property
    def is_open(self) -> bool:
        return self.signal.action in (Action.OPEN_LONG, Action.OPEN_SHORT)


@dataclass
class AddLevel:
    """单次加仓记录（含试仓）。"""
    price: float
    qty: float
    ts: datetime
    fee_usdt: float = 0.0


@dataclass
class Position:
    position_id: str
    symbol: str
    side: Literal["long", "short"]
    qty: float
    entry_price: float
    entry_ts: datetime
    margin_usdt: float
    stop_loss: float
    take_profits: tuple[float, ...]
    liq_price: float | None = None
    opened_by: str = ""
    signal_id: str = ""
    realized_pnl: float = 0.0
    funding_paid: float = 0.0
    entry_atr: float | None = None
    bars_held: int = 0
    tp1_done: bool = False
    breakeven_done: bool = False
    extreme_price: float | None = None
    status: Literal["open", "closed"] = "open"
    # ---- 金字塔加仓扩展字段 ----
    target_qty: float = 0.0          # 目标满仓 qty
    avg_entry_price: float = 0.0     # 加权平均入场价（加仓后变化）
    add_count: int = 0               # 已加仓次数（不含试仓）
    add_levels: list[AddLevel] = field(default_factory=list)  # 各批次入场记录
    trailing_active: bool = False    # 移动止损是否激活
    trailing_peak_price: float = 0.0 # 移动止损追踪的极值（多头最高/空头最低）
    base_stop_price: float = 0.0     # 原始止损价（用于统计）
    tp_price: float = 0.0           # 止盈价（0 表示未设）


@dataclass
class Fill:
    ts: datetime
    symbol: str
    side: Side
    qty: float
    price: float
    fee_usdt: float
    reason: FillReason
    position_id: str
    signal_id: str = ""
    pnl_usdt: float = 0.0
