"""rule_v1: 多周期规则策略（5m 触发 / 1h 确认 / 4h 趋势过滤），支持多空。

信号生成（无前视）：
  1. 4h: macd_hist > 0 且 EMA(20) > EMA(60) → 多头趋势；反之 → 空头趋势
  2. 1h: StochRSI_k > 20 且相对前一根上升 → 多头确认；< 80 且相对前一根下降 → 空头确认
  3. 5m: MACD 柱由负转正 或 收盘上穿 EMA(20)，且 ATR > median(ATR,20) → 多头入场
         MACD 柱由正转负 或 收盘下穿 EMA(20)，且 ATR > median(ATR,20) → 空头入场

confirm_bars 参数：连续 N 根 5m bar 满足所有条件才开仓，过滤假信号。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ...core.contracts import Action, Bar, Mode, Signal
from ...core.timeframe import interval_ms
from ..base import BaseStrategy
from ..registry import register


@dataclass
class RuleV1Config:
    # 趋势/触发用的 EMA、MACD、ATR 列由 features.indicators 以固定参数预计算，
    # 这里只暴露策略真正读取的阈值。
    confirm_srsi_oversold: float = 20.0
    confirm_srsi_overbought: float = 80.0
    trigger_atr_median_ratio: float = 1.0
    allow_short: bool = True
    confirm_bars: int = 1  # 连续 N 根 bar 满足条件才开仓


@dataclass
class RuleV1State:
    bars: list[Bar] = field(default_factory=list)
    condition_streak: int = 0


@register
class RuleV1Strategy(BaseStrategy):
    """多周期均线 + MACD + StochRSI 规则策略。"""

    name = "rule_v1"
    required_intervals = ("5m", "1h", "4h")
    config_cls = RuleV1Config


    def __init__(self, config: RuleV1Config | None = None):
        super().__init__(config or RuleV1Config())
        self._state = RuleV1State()
        self._warmup = max(
            interval_ms("4h") // interval_ms("5m") * 60,
            interval_ms("1h") // interval_ms("5m") * 30,
            26,
        )

    def warmup(self) -> int:
        return self._warmup

    def reset(self) -> None:
        self._state = RuleV1State()

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
        if len(s.bars) > self._warmup + 100:
            s.bars = s.bars[-(self._warmup + 100):]

        if len(s.bars) < 26:
            return None

        # ---- 4h 趋势过滤 ----
        h4 = htf.get("4h")
        if h4 is None:
            return None
        cur4h_hist = float(h4["macd_hist"]) if not pd.isna(h4["macd_hist"]) else 0
        is_uptrend = cur4h_hist > 0 and float(h4["ema_20"]) > float(h4["ema_60"])
        is_downtrend = cur4h_hist < 0 and float(h4["ema_20"]) < float(h4["ema_60"])
        if not is_uptrend and not is_downtrend:
            s.condition_streak = 0
            return None

        # ---- 1h 中期确认 ----
        h1 = htf.get("1h")
        if h1 is None:
            return None
        cur_1h_srsi = float(h1["srsi_k"]) if not pd.isna(h1["srsi_k"]) else 50.0
        prev_1h_srsi = float(h1["prev_srsi_k"]) if not pd.isna(h1["prev_srsi_k"]) else None
        if prev_1h_srsi is None:
            return None

        long_confirmed = cur_1h_srsi > cfg.confirm_srsi_oversold and cur_1h_srsi > prev_1h_srsi
        short_confirmed = (
            cfg.allow_short
            and cur_1h_srsi < cfg.confirm_srsi_overbought
            and cur_1h_srsi < prev_1h_srsi
        )
        if not long_confirmed and not short_confirmed:
            s.condition_streak = 0
            return None

        # ---- 5m 入场触发 ----
        atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
        atr_med = float(row["atr_median"]) if not pd.isna(row["atr_median"]) else atr_val
        if atr_val <= atr_med * cfg.trigger_atr_median_ratio:
            s.condition_streak = 0
            return None

        cur_close = float(row["close"])
        prev_close = float(row["prev_close"]) if not pd.isna(row["prev_close"]) else cur_close
        ema20 = float(row["ema_20"]) if not pd.isna(row["ema_20"]) else cur_close
        prev_ema20 = float(row["prev_ema_20"]) if not pd.isna(row["prev_ema_20"]) else ema20
        cur_hist = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0.0
        prev_hist = float(row["prev_macd_hist"]) if not pd.isna(row["prev_macd_hist"]) else 0.0

        macd_turn_up = prev_hist < 0 < cur_hist
        close_cross_up = prev_close <= prev_ema20 and cur_close > ema20
        macd_turn_down = prev_hist > 0 > cur_hist
        close_cross_down = prev_close >= prev_ema20 and cur_close < ema20

        long_condition = is_uptrend and long_confirmed and (macd_turn_up or close_cross_up)
        short_condition = (
            cfg.allow_short
            and is_downtrend
            and short_confirmed
            and (macd_turn_down or close_cross_down)
        )
        if not long_condition and not short_condition:
            s.condition_streak = 0
            return None

        # ---- 入场确认：连续 N 根 bar 满足条件 ----
        s.condition_streak += 1
        if s.condition_streak < cfg.confirm_bars:
            return None
        s.condition_streak = 0

        signal_id = f"rule_v1_{dt:%Y%m%d%H%M%S}"

        if long_condition:
            return Signal(
                signal_id=signal_id, ts_signal=dt, symbol=bar.symbol,
                timeframe="5m", strategy=self.name, action=Action.OPEN_LONG,
                price_ref=cur_close,
                strength=0.5 + 0.5 * min(cur_1h_srsi / 50.0, 1.0),
                reason="trend_up+1h_srsi_confirm+5m_macd_cross+atr",
                mode=Mode.BACKTEST,
            )

        if short_condition:
            return Signal(
                signal_id=signal_id, ts_signal=dt, symbol=bar.symbol,
                timeframe="5m", strategy=self.name, action=Action.OPEN_SHORT,
                price_ref=cur_close,
                strength=0.5 + 0.5 * min((100.0 - cur_1h_srsi) / 50.0, 1.0),
                reason="trend_down+1h_srsi_confirm+5m_macd_cross+atr",
                mode=Mode.BACKTEST,
            )

        return None
