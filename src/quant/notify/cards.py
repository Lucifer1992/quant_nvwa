from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..core.contracts import Action, RiskDecision, Signal

_TEMPLATE_BY_ACTION = {
    Action.OPEN_LONG: ("green", "做多信号"),
    Action.OPEN_SHORT: ("red", "做空信号"),
    Action.CLOSE_LONG: ("blue", "平多"),
    Action.CLOSE_SHORT: ("blue", "平空"),
}

_ICON = {Action.OPEN_LONG: "🟢", Action.OPEN_SHORT: "🔴", Action.CLOSE_LONG: "🔵", Action.CLOSE_SHORT: "🔵"}


def _px(v: float | None) -> str:
    return "—" if v is None else f"{v:,.2f}"


def _title(sig: Signal, dec: RiskDecision) -> str:
    side = _TEMPLATE_BY_ACTION.get(sig.action, ("grey", sig.action.value))[1]
    prefix = "⚠️ 信号被否决" if not dec.approved else side
    return f"{_ICON.get(sig.action, 'ℹ️')} {prefix} · {sig.symbol} {sig.timeframe} · {sig.strategy}"


def _overview(sig: Signal, dec: RiskDecision) -> str:
    direction = {"open_long": "Long", "open_short": "Short", "close_long": "Close Long", "close_short": "Close Short"}[sig.action.value]
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
        f"bar close {sig.ts_signal.astimezone(UTC).strftime('%Y-%m-%d %H:%M')} UTC ｜ "
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
    """用于通知链路的冒烟测试与卡片排版检查。"""
    sig = Signal(
        signal_id="test-ethusdt-15m-long",
        ts_signal=datetime.now(UTC).replace(second=0, microsecond=0),
        symbol="ETHUSDT",
        timeframe="15m",
        strategy="rule_v1",
        action=Action.OPEN_LONG,
        price_ref=2615.42,
        strength=0.72,
        reason="4h 趋势向上(MACD>0, EMA20>EMA60)；1h StochRSI 自超卖区上穿；15m MACD 柱转正且收盘上穿 EMA20，ATR 处于近 20 根中位以上",
    )
    dec = RiskDecision(
        signal=sig,
        approved=approved,
        qty=0.38,
        notional_usdt=994.0,
        leverage=3.0,
        margin_usdt=331.0,
        stop_loss=2576.10,
        take_profits=(2654.74, 2694.06),
        est_liq_price=1832.5,
        equity_usdt=10000.0,
        risk_usdt=100.0,
        reject_reasons=() if approved else ("单品种保证金占比超限 30%", "名义敞口超上限"),
    )
    return sig, dec
