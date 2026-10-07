"""固定风险比例仓位管理：单笔最大亏损 = equity × risk_pct。"""
from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import Any, Literal

from ..core.contracts import RiskDecision, Signal, SymbolSpec


def floor_to(value: float, step: float) -> float:
    if step <= 0:
        return value
    return float(Decimal(str(value)).quantize(Decimal(str(step)), rounding=ROUND_DOWN))


def _rejected(decision: RiskDecision, reason: str) -> RiskDecision:
    return RiskDecision(
        signal=decision.signal,
        approved=False,
        reject_reasons=(*decision.reject_reasons, reason),
    )


def compute_risk_size(
    signal: Signal,
    decision: RiskDecision,
    equity: float,
    risk_pct: float,
    atr: float,
    sl_atr_mult: float,
    global_max_leverage: float,
    symbol_spec: SymbolSpec,
    qty: float = 0.0,
    taker_fee_rate: float = 0.0005,
) -> RiskDecision:
    """把一次开/加仓意向换算成 qty / notional / margin / stop_loss。

    qty 来源二选一：
      1. 调用方传入 qty（金字塔路径：由 entry_frac / add_frac 算出）→ 直接采用；
      2. 未传入 → 按固定风险比例：qty = equity × risk_pct / (sl_atr_mult × ATR)。

    保证金口径固定为 `margin = notional / global_max_leverage`（设计原则：全程 10×）。
    这一步不能写成 `margin = notional / (notional / equity)`，那样恒等于 equity，
    任何一笔成交都会把全部权益记成保证金，`ret_pct` 退化成对权益的收益率、现金账户被抽干。
    """
    if equity <= 0:
        return _rejected(decision, "权益已归零，无法开仓")
    if global_max_leverage <= 0:
        return _rejected(decision, "global_max_leverage 必须为正")

    # 止损距离决定实际风险敞口；金字塔路径的 qty 来自仓位比例而非风险预算，
    # 所以风险额在 qty 定稿后统一回算，而不是照抄 risk_pct × equity。
    if qty <= 0:
        stop_dist = sl_atr_mult * atr
        if stop_dist <= 0 or atr <= 0:
            return _rejected(decision, "ATR 不可用，无法按风险定仓")
        qty = (equity * risk_pct) / stop_dist

    step = float(symbol_spec.step_size)
    qty = floor_to(qty, step)

    # 名义敞口不得超过固定杠杆下的可用本金
    max_notional = equity * global_max_leverage
    notional = qty * signal.price_ref
    if notional > max_notional:
        qty = floor_to(max_notional / signal.price_ref, step)
        notional = qty * signal.price_ref

    if qty < float(symbol_spec.min_qty):
        return _rejected(decision, f"qty={qty} 低于最小下单量 {symbol_spec.min_qty}")
    if notional < symbol_spec.min_notional:
        return _rejected(decision, f"名义 {notional:.2f} 低于最小成交额 {symbol_spec.min_notional}")

    margin = notional / global_max_leverage
    risk_amount = qty * sl_atr_mult * atr if atr > 0 else 0.0
    side: Literal["long", "short"] = (
        "long" if signal.action.value.startswith(("open_long", "add_to_long")) else "short"
    )
    stop_loss = (
        signal.price_ref - sl_atr_mult * atr if side == "long"
        else signal.price_ref + sl_atr_mult * atr
    )

    return RiskDecision(
        signal=signal,
        approved=True,
        qty=qty,
        notional_usdt=notional,
        leverage=global_max_leverage,
        margin_usdt=margin,
        stop_loss=stop_loss if atr > 0 else None,
        est_liq_price=estimate_liq_price(
            signal.price_ref, side, global_max_leverage,
            symbol_spec.maint_margin_rate, taker_fee_rate,
        ),
        equity_usdt=equity,
        risk_usdt=risk_amount,
    )


def estimate_liq_price(
    entry_price: float,
    side: Literal["long", "short"],
    leverage: float,
    maint_margin_rate: float,
    taker_fee_rate: float,
) -> float:
    """逐仓强平价近似（不计资金费率与持仓期间收益）。"""
    if side == "long":
        return entry_price * (1 - 1.0 / leverage + maint_margin_rate + taker_fee_rate)
    else:
        return entry_price * (1 + 1.0 / leverage - maint_margin_rate - taker_fee_rate)


def decide_position(
    *,
    signal: Signal,
    equity: float,
    atr: float,
    cfg: Any,
    spec: SymbolSpec,
    size_mult: float = 1.0,
    is_add: bool = False,
) -> RiskDecision:
    """把一条开/加仓信号换算成 RiskDecision，仓位比例取自 cfg 的金字塔政策。

    回测与 paper 都只走这里，避免"卡片上的数量"与"账本里的数量"来自两份代码。
    cfg 需具备：risk_pct / sl_atr_mult / global_max_leverage / taker_fee /
    pyramid_enabled / entry_frac / add_frac。

    `size_mult` 钳在 [0.2, 1.0]：策略可按高周期背景减仓，但不允许超额下注。
    """
    size_mult = min(max(float(signal.meta.get("size_mult", size_mult)), 0.2), 1.0)
    want_add = is_add and bool(cfg.pyramid_enabled)
    frac = cfg.add_frac if want_add else cfg.entry_frac
    target_qty = 0.0
    if cfg.pyramid_enabled:
        target_qty = equity * frac * size_mult * cfg.global_max_leverage / signal.price_ref
    return compute_risk_size(
        signal=signal,
        decision=RiskDecision(signal=signal, approved=False),
        equity=equity,
        risk_pct=cfg.risk_pct,
        atr=atr,
        sl_atr_mult=cfg.sl_atr_mult,
        global_max_leverage=cfg.global_max_leverage,
        symbol_spec=spec,
        taker_fee_rate=cfg.taker_fee,
        qty=target_qty,
    )
