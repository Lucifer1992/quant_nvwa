"""rule_v2: StochRSI 极值反转策略。

1h 方向过滤（无 4h 过滤，专注反转信号）：
  - 多头：StochRSI_k < srsi_oversold（如 15）且拐头向上 + MACD hist > 0
  - 空头：StochRSI_k > srsi_overbought（如 65）且拐头向下 + MACD hist < 0

5m 入场触发（1h 方向确认后）：
  - 多头：5m 出现同样反转信号（srsi_k < 15 且拐头向上 + macd > 0）
  - 空头：5m 出现同样反转信号（srsi_k > 65 且拐头向下 + macd < 0）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ...core.contracts import Action, Bar, Mode, Signal
from ..base import BaseStrategy
from ..registry import register


@dataclass
class RuleV2Config:
    srsi_oversold: float = 15.0
    srsi_overbought: float = 65.0
    srsi_cross_margin: float = 1.0
    require_macd_agree: bool = True
    allow_short: bool = True
    confirm_bars: int = 1


@dataclass
class RuleV2State:
    bars: list[Bar] = field(default_factory=list)
    condition_streak: int = 0


@register
class RuleV2Strategy(BaseStrategy):
    """StochRSI 极值反转策略（1h 方向 + 5m 入场共振）。"""

    name = "rule_v2"
    required_intervals = ("5m", "1h")
    config_cls = RuleV2Config


    def __init__(self, config: RuleV2Config | None = None):
        super().__init__(config or RuleV2Config())
        self._state = RuleV2State()

    def warmup(self) -> int:
        return 50

    def reset(self) -> None:
        self._state = RuleV2State()

    def _srsi_turn_up(self, cur: float, prev: float | None, margin: float) -> bool:
        return prev is not None and cur > prev + margin

    def _srsi_turn_down(self, cur: float, prev: float | None, margin: float) -> bool:
        return prev is not None and cur < prev - margin

    def on_bar(
        self,
        bar: Bar,
        row: pd.Series,
        htf: dict[str, pd.Series],
        dt: datetime,
    ) -> Signal | None:
        s = self._state
        cfg = self.cfg
        s.bars.append(bar)
        if len(s.bars) > 200:
            s.bars = s.bars[-200:]

        # ---- 1h 方向过滤 ----
        h1 = htf.get("1h")
        if h1 is None:
            return None

        cur_1h_srsi = float(h1["srsi_k"]) if not pd.isna(h1["srsi_k"]) else 50.0
        prev_1h_srsi = (
            float(h1["prev_srsi_k"]) if not pd.isna(h1["prev_srsi_k"]) else None
        )
        cur_1h_hist = (
            float(h1["macd_hist"]) if not pd.isna(h1["macd_hist"]) else 0.0
        )

        if prev_1h_srsi is None:
            return None

        long_1h_confirmed = (
            cur_1h_srsi < cfg.srsi_oversold
            and self._srsi_turn_up(cur_1h_srsi, prev_1h_srsi, cfg.srsi_cross_margin)
            and (cur_1h_hist > 0 if cfg.require_macd_agree else True)
        )
        short_1h_confirmed = (
            cfg.allow_short
            and cur_1h_srsi > cfg.srsi_overbought
            and self._srsi_turn_down(cur_1h_srsi, prev_1h_srsi, cfg.srsi_cross_margin)
            and (cur_1h_hist < 0 if cfg.require_macd_agree else True)
        )

        if not long_1h_confirmed and not short_1h_confirmed:
            s.condition_streak = 0
            return None

        # ---- 5m 入场触发：同样规则 ----
        cur_5m_srsi = (
            float(row["srsi_k"]) if "srsi_k" in row and not pd.isna(row["srsi_k"]) else 50.0
        )
        prev_5m_srsi = (
            float(row["prev_srsi_k"])
            if "prev_srsi_k" in row and not pd.isna(row["prev_srsi_k"])
            else None
        )
        cur_5m_hist = (
            float(row["macd_hist"])
            if "macd_hist" in row and not pd.isna(row["macd_hist"])
            else 0.0
        )

        if prev_5m_srsi is None:
            return None

        long_5m_trigger = (
            cur_5m_srsi < cfg.srsi_oversold
            and self._srsi_turn_up(cur_5m_srsi, prev_5m_srsi, cfg.srsi_cross_margin)
            and (cur_5m_hist > 0 if cfg.require_macd_agree else True)
        )
        short_5m_trigger = (
            cfg.allow_short
            and cur_5m_srsi > cfg.srsi_overbought
            and self._srsi_turn_down(cur_5m_srsi, prev_5m_srsi, cfg.srsi_cross_margin)
            and (cur_5m_hist < 0 if cfg.require_macd_agree else True)
        )

        cur_close = float(row["close"])

        long_condition = long_1h_confirmed and long_5m_trigger
        short_condition = cfg.allow_short and short_1h_confirmed and short_5m_trigger

        if not long_condition and not short_condition:
            s.condition_streak = 0
            return None

        # 入场确认
        s.condition_streak += 1
        if s.condition_streak < cfg.confirm_bars:
            return None
        s.condition_streak = 0

        signal_id = f"rule_v2_{dt:%Y%m%d%H%M%S}"

        if long_condition:
            return Signal(
                signal_id=signal_id,
                ts_signal=dt,
                symbol=bar.symbol,
                timeframe="5m",
                strategy=self.name,
                action=Action.OPEN_LONG,
                price_ref=cur_close,
                strength=min(1.0, 0.5 + 0.5 * (1 - cur_5m_srsi / cfg.srsi_oversold)),
                reason="1h_stochrsi_oversold_up+macd_pos|5m_confirm",
                mode=Mode.BACKTEST,
            )

        if short_condition:
            return Signal(
                signal_id=signal_id,
                ts_signal=dt,
                symbol=bar.symbol,
                timeframe="5m",
                strategy=self.name,
                action=Action.OPEN_SHORT,
                price_ref=cur_close,
                strength=min(1.0, 0.5 + 0.5 * (cur_5m_srsi - cfg.srsi_overbought) / (100 - cfg.srsi_overbought)),
                reason="1h_stochrsi_overbought_down+macd_neg|5m_confirm",
                mode=Mode.BACKTEST,
            )

        return None
