"""rule_v3: 多指标投票策略（4 个指标 / ≥2 个同向即入场）。

**信号源（每个独立返回 +1=多 / -1=空 / 0=无信号）**：
  1. MACD 交叉   hist 由负转正 → +1，由正转负 → -1，否则 0
  2. EMA 突破    close 上穿 EMA(20) → +1，下穿 → -1，否则 0
  3. StochRSI 反转  k < 30 且拐头向上 → +1，k > 70 且拐头向下 → -1，否则 0
  4. 量能放大    volume > median(20) × vol_mult 且价变同向 → ±1，否则 0

**入场规则**：加权求和 score，score ≥ min_signals → 开仓（方向由 score 符号决定）。
   例如 4 指标各 1 分，min_signals=2 → score ≥ 2 才入场；
   min_signals=3 → 更严格，需要 3 个指标共振。

**适用场景**：震荡市（多指标共振过滤假信号），避免单指标误判。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd

from ...core.contracts import Action, Bar, Mode, Signal
from ..base import BaseStrategy
from ..registry import register


@dataclass
class RuleV3Config:
    # 投票门槛
    min_signals: int = 2  # 至少几个指标同向才入场（1-4）
    # MACD
    macd_weight: float = 1.0
    # EMA
    ema_period: int = 20
    ema_weight: float = 1.0
    # StochRSI 反转
    srsi_oversold: float = 30.0
    srsi_overbought: float = 70.0
    srsi_cross_margin: float = 1.0
    srsi_weight: float = 1.0
    # 量能放大
    vol_lookback: int = 20
    vol_mult: float = 1.5  # 当前量 > median × 1.5 才算放量
    vol_weight: float = 1.0
    # 多空
    allow_short: bool = True
    confirm_bars: int = 1  # 连续 N 根 bar 同向才开仓


SignalVal = Literal[-1, 0, 1]


@dataclass
class RuleV3State:
    bars: list[Bar] = field(default_factory=list)
    condition_streak: int = 0
    last_score: int = 0


@register
class RuleV3Strategy(BaseStrategy):
    """多指标投票策略（4 个独立指标，任意 ≥ min_signals 个同向即入场）。"""

    name = "rule_v3"
    required_intervals = ("5m",)  # 仅需 5m
    config_cls = RuleV3Config


    def __init__(self, config: RuleV3Config | None = None):
        super().__init__(config or RuleV3Config())
        self._state = RuleV3State()
        # 预热期：EMA20 + 成交量 20-bar 中位数 + StochRSI 14 周期 → 至少 30 根 5m
        self._warmup = 50

    def warmup(self) -> int:
        return self._warmup

    def reset(self) -> None:
        self._state = RuleV3State()

    def _signal_macd(self, row: pd.Series) -> SignalVal:
        cur = float(row.get("macd_hist", 0))
        prev = float(row.get("prev_macd_hist", 0))
        if pd.isna(cur) or pd.isna(prev):
            return 0
        if prev < 0 < cur:
            return 1
        if prev > 0 > cur:
            return -1
        return 0

    def _signal_ema(self, row: pd.Series) -> SignalVal:
        cur = float(row.get("close", 0))
        prev = float(row.get("prev_close", 0))
        ema = float(row.get("ema_20", 0))
        prev_ema = float(row.get("prev_ema_20", 0))
        if any(pd.isna(x) for x in (cur, prev, ema, prev_ema)):
            return 0
        if prev <= prev_ema and cur > ema:
            return 1
        if prev >= prev_ema and cur < ema:
            return -1
        return 0

    def _signal_srsi(self, row: pd.Series) -> SignalVal:
        cur = float(row.get("srsi_k", 50))
        prev = float(row.get("prev_srsi_k", 50))
        cfg = self.cfg
        if pd.isna(cur) or pd.isna(prev):
            return 0
        cfg = self.cfg
        if cur < cfg.srsi_oversold and cur > prev + cfg.srsi_cross_margin:
            return 1
        if cur > cfg.srsi_overbought and cur < prev - cfg.srsi_cross_margin:
            return -1
        return 0

    def _signal_volume(self, row: pd.Series) -> SignalVal:
        cfg = self.cfg
        vol = float(row.get("volume", 0))
        vol_med = float(row.get("vol_median", 0))
        if pd.isna(vol) or pd.isna(vol_med) or vol_med <= 0:
            return 0
        if vol < vol_med * cfg.vol_mult:
            return 0  # 量能未放大
        cur = float(row.get("close", 0))
        prev = float(row.get("prev_close", 0))
        if pd.isna(cur) or pd.isna(prev):
            return 0
        if cur > prev:
            return 1
        if cur < prev:
            return -1
        return 0

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

        if len(s.bars) < self._warmup:
            return None

        # 计算各指标信号
        sigs = [
            (self._signal_macd(row), cfg.macd_weight),
            (self._signal_ema(row), cfg.ema_weight),
            (self._signal_srsi(row), cfg.srsi_weight),
            (self._signal_volume(row), cfg.vol_weight),
        ]

        # 加权求和（每个指标只算同向贡献，符号不同的忽略）
        long_score = sum(w for sig, w in sigs if sig > 0)
        short_score = sum(w for sig, w in sigs if sig < 0)

        # 入场条件：多头 ≥ min_signals 或 空头 ≥ min_signals
        cur_close = float(row.get("close", 0))
        if pd.isna(cur_close):
            return None

        direction = 0
        score = 0
        if long_score >= cfg.min_signals and long_score > short_score:
            direction = 1
            score = int(long_score)
        elif short_score >= cfg.min_signals and short_score > long_score:
            direction = -1
            score = int(short_score)

        if direction == 0:
            s.condition_streak = 0
            s.last_score = 0
            return None

        s.last_score = score

        # 入场确认
        if s.condition_streak < cfg.confirm_bars:
            s.condition_streak += 1
            return None
        s.condition_streak = 0

        signal_id = f"rule_v3_{dt:%Y%m%d%H%M%S}"
        reason = (
            f"multi_vote_{'long' if direction > 0 else 'short'}_score{score}"
            f"/macd:{sigs[0][0]}*ema:{sigs[1][0]}*srsi:{sigs[2][0]}*vol:{sigs[3][0]}"
        )

        return Signal(
            signal_id=signal_id, ts_signal=dt, symbol=bar.symbol,
            timeframe="5m", strategy=self.name,
            action=Action.OPEN_LONG if direction > 0 else Action.OPEN_SHORT,
            price_ref=cur_close,
            strength=min(1.0, score / 4.0),
            reason=reason,
            mode=Mode.BACKTEST,
        )
