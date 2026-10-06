from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from ..core.contracts import Action, RiskDecision, Signal

BJ_TZ = timezone(timedelta(hours=8))  # 北京时间 UTC+8

# 金字塔加仓是独立动作，漏了会在卡片标题/方向处 KeyError 或显示原始枚举名
_TEMPLATE_BY_ACTION: dict[Action, tuple[str, str]] = {
    Action.OPEN_LONG: ("green", "做多信号"),
    Action.OPEN_SHORT: ("red", "做空信号"),
    Action.ADD_TO_LONG: ("light_green", "加仓做多"),
    Action.ADD_TO_SHORT: ("light_red", "加仓做空"),
    Action.CLOSE_LONG: ("blue", "平多"),
    Action.CLOSE_SHORT: ("blue", "平空"),
}

_ICON = {
    Action.OPEN_LONG: "🟢",
    Action.OPEN_SHORT: "🔴",
    Action.ADD_TO_LONG: "➕🟢",
    Action.ADD_TO_SHORT: "➕🔴",
    Action.CLOSE_LONG: "🔵",
    Action.CLOSE_SHORT: "🔵",
}

_DIRECTION = {
    Action.OPEN_LONG: "做多",
    Action.OPEN_SHORT: "做空",
    Action.ADD_TO_LONG: "加多",
    Action.ADD_TO_SHORT: "加空",
    Action.CLOSE_LONG: "平多",
    Action.CLOSE_SHORT: "平空",
}


def _px(v: float | None) -> str:
    return "—" if v is None else f"{v:,.2f}"


def _title(sig: Signal, dec: RiskDecision) -> str:
    side = _TEMPLATE_BY_ACTION.get(sig.action, ("grey", sig.action.value))[1]
    prefix = "⚠️ 信号被否决" if not dec.approved else side
    return f"{_ICON.get(sig.action, 'ℹ️')} {prefix} · {sig.symbol} {sig.timeframe} · {sig.strategy}"


def _overview(sig: Signal, dec: RiskDecision) -> str:
    direction = _DIRECTION.get(sig.action, str(sig.action))
    return (
        f"**方向** {direction} ｜ **杠杆** {dec.leverage:g}x ｜ **信号强度** {sig.strength:.2f}\n"
        f"**参考价** {_px(sig.price_ref)}（{sig.timeframe} 收盘价）｜ **模式** {sig.mode.value}"
    )


def _risk_block(dec: RiskDecision) -> str:
    tps = "\n".join(f"TP{i + 1} {_px(p)}" for i, p in enumerate(dec.take_profits)) or "TP —"
    return (
        "📊 **风控**\n"
        f"SL {_px(dec.stop_loss)}\n{tps}\n"
        f"预估强平 {_px(dec.est_liq_price)}"
    )


def _sizing_block(dec: RiskDecision) -> str:
    return (
        "💰 **仓位**\n"
        f"权益 {dec.equity_usdt:,.0f} USDT\n"
        f"风险 {dec.risk_usdt:,.0f} USDT\n"
        f"数量 {dec.qty:g}\n"
        f"名义 {dec.notional_usdt:,.0f} USDT ｜ 保证金 {dec.margin_usdt:,.0f}"
    )


def _reject_block(dec: RiskDecision) -> str:
    return "🚫 **否决原因**\n" + "\n".join(f"- {r}" for r in dec.reject_reasons)


def build_elements(sig: Signal, dec: RiskDecision) -> list[dict[str, Any]]:
    md = lambda t: {"tag": "markdown", "content": t}
    elements: list[dict[str, Any]] = [md(_overview(sig, dec)), {"tag": "hr"}]

    if dec.approved:
        cols = [
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [md(_risk_block(dec))]},
            {"tag": "column", "width": "weighted", "weight": 1, "elements": [md(_sizing_block(dec))]},
        ]
        elements.append({"tag": "column_set", "columns": cols})
    else:
        elements.append(md(_reject_block(dec)))

    elements.append({"tag": "hr"})
    elements.append(md(f"**理由** {sig.reason}"))
    note = (
        f"bar close {sig.ts_signal.astimezone(BJ_TZ):%Y-%m-%d %H:%M}（北京时间） ｜ "
        "模拟交易信号，非投资建议"
    )
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": note}]})
    return elements


def build_card(sig: Signal, dec: RiskDecision) -> dict[str, Any]:
    """飞书 webhook 自定义机器人只接受卡片 1.0 结构（实测 schema 2.0 会返回 code 11246）。"""
    template, _ = _TEMPLATE_BY_ACTION.get(sig.action, ("grey", ""))
    header = {
        "title": {"tag": "plain_text", "content": _title(sig, dec)},
        "template": template,
    }
    return {
        "msg_type": "interactive",
        "card": {"config": {"wide_screen_mode": True, "update_multi": True}, "header": header, "elements": build_elements(sig, dec)},
    }


def sample_signal_decision(*, approved: bool = True) -> tuple[Signal, RiskDecision]:
    """用于通知链路的冒烟测试与卡片排版检查，数值与固定 10× 杠杆的风控口径一致。"""
    price, atr, qty = 2615.42, 32.0, 0.05
    sig = Signal(
        signal_id="test-ethusdt-1h-long",
        ts_signal=datetime.now(UTC).replace(second=0, microsecond=0),
        symbol="ETHUSDT",
        timeframe="1h",
        strategy="rule_v1",
        action=Action.OPEN_LONG,
        price_ref=price,
        strength=0.72,
        reason="4h 趋势向上(MACD>0, EMA20>EMA60)；1h StochRSI 自超卖区上穿；5m MACD 柱转正且收盘上穿 EMA20，ATR 处于近 20 根中位以上",
    )
    dec = RiskDecision(
        signal=sig,
        approved=approved,
        qty=qty,
        notional_usdt=round(qty * price, 2),
        leverage=10.0,
        margin_usdt=round(qty * price / 10.0, 2),
        stop_loss=round(price - 2.0 * atr, 2),
        take_profits=(round(price + 2.0 * atr, 2), round(price + 3.5 * atr, 2)),
        est_liq_price=round(price * (1 - 1 / 10.0 + 0.025), 2),
        equity_usdt=100.0,
        risk_usdt=round(qty * 2.0 * atr, 2),
        reject_reasons=() if approved else ("净值不足以覆盖保证金",),
    )
    return sig, dec
