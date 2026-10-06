from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from ..core.contracts import SymbolSpec


def _dec(value: Any) -> Decimal:
    return Decimal(str(value))


def _dec_or(value: Any, default: float) -> Decimal:
    """交易所会把不适用的字段返回成空串（如 liquidationFee=""），不能直接喂给 Decimal。"""
    if value is None or value == "":
        return Decimal(str(default))
    return _dec(value)


def _precision(step: Decimal) -> int:
    if step == step.to_integral_value():
        return 0
    return -step.normalize().as_tuple().exponent


def _pct(value: Any, default: float | None) -> float | None:
    """exchangeInfo 的百分比字段（如 "2.5000" 表示 2.5%）换算成小数比率。"""
    if value is None or value == "":
        return default
    return float(_dec(value)) / 100.0


def parse_symbol_spec(
    payload: dict[str, Any],
    *,
    default_max_leverage: float = 3.0,
    default_maint_margin_rate: float = 0.025,
    default_liquidation_fee_rate: float = 0.0125,
    funding_interval_hours: float = 8.0,
    fetched_at: datetime | None = None,
) -> SymbolSpec:
    """从 exchangeInfo 的单个 symbol 对象解析交易规则。

    USDT-M 的 exchangeInfo 公开携带 maintMarginPercent / liquidationFee /
    requiredMarginPercent，可直接用于强平近似与杠杆上限；逐档 bracket 在
    /fapi/v1/leverageBracket（需签名，MVP 无 Key），因此这里的维持保证金率取的是
    最高档口径——比实际第 1 档更大，强平价更近、仓位更小，属于安全侧偏差。
    """
    filters: dict[str, dict[str, Any]] = {f["filterType"]: f for f in payload.get("filters", [])}
    price_filter = filters.get("PRICE_FILTER", {})
    lot_size = filters.get("LOT_SIZE", {})
    # MIN_NOTIONAL 才是成交额门槛；MARKET_LOT_SIZE 只有 maxQty/minQty，拿它兜底等于没有兜底
    min_notional = filters.get("MIN_NOTIONAL", {})

    tick = _dec_or(price_filter.get("tickSize"), 0.01)
    step = _dec_or(lot_size.get("stepSize"), 0.001)

    price_precision = int(payload.get("pricePrecision", _precision(tick)))
    qty_precision = int(payload.get("quantityPrecision", _precision(step)))
    # pricePrecision 是交易所对"最多几位小数"的上限声明，tickSize 的位数可以更少
    # （如 BTCUSDT：pricePrecision=2 而 tickSize=0.1 只有 1 位），因此只校验下限
    if price_precision < _precision(tick) or qty_precision < _precision(step):
        raise ValueError(
            f"{payload.get('symbol')}: 精度字段小于 filter 步长 "
            f"(price {price_precision} vs tick {_precision(tick)}, qty {qty_precision} vs step {_precision(step)})"
        )

    maint_rate = _pct(payload.get("maintMarginPercent"), default_maint_margin_rate)
    liq_fee = _dec_or(payload.get("liquidationFee"), default_liquidation_fee_rate)
    req_margin = _pct(payload.get("requiredMarginPercent"), None)
    max_leverage = float(1.0 / req_margin) if req_margin else default_max_leverage

    notional_raw = min_notional.get("notional") or min_notional.get("minNotional")

    return SymbolSpec(
        symbol=payload["symbol"],
        status=payload.get("status", "UNKNOWN"),
        tick_size=tick,
        step_size=step,
        min_qty=_dec_or(lot_size.get("minQty"), 0.001),
        min_notional=float(notional_raw) if notional_raw not in (None, "") else 20.0,
        maint_margin_rate=maint_rate,
        max_leverage=max_leverage,
        liquidation_fee_rate=float(liq_fee),
        funding_interval_hours=funding_interval_hours,
        price_precision=price_precision,
        qty_precision=qty_precision,
        fetched_at=fetched_at or datetime.now(UTC),
    )


def spec_to_json(spec: SymbolSpec) -> str:
    d = {k: (str(v) if isinstance(v, Decimal) else v.isoformat() if isinstance(v, datetime) else v) for k, v in vars(spec).items()}
    return json.dumps(d)


def spec_from_json(raw: str) -> SymbolSpec:
    d = json.loads(raw)
    return SymbolSpec(
        symbol=d["symbol"],
        status=d["status"],
        tick_size=Decimal(d["tick_size"]),
        step_size=Decimal(d["step_size"]),
        min_qty=Decimal(d["min_qty"]),
        min_notional=float(d["min_notional"]),
        maint_margin_rate=float(d["maint_margin_rate"]),
        max_leverage=float(d["max_leverage"]),
        liquidation_fee_rate=float(d["liquidation_fee_rate"]),
        funding_interval_hours=float(d["funding_interval_hours"]),
        price_precision=int(d["price_precision"]),
        qty_precision=int(d["qty_precision"]),
        fetched_at=datetime.fromisoformat(d["fetched_at"]),
    )
