"""lgbm_v1：用 walk-forward LightGBM 概率替代 rule_v4 的 1h 趋势状态机。

决策节奏仍是 1h bar 收盘，模型只决定**是否入场**与**仓位倍数**，出场完全交给引擎
（1h ATR 止损/止盈/移动止损/时间止损）—— 与 rule_v4 同一套出场口径，
样本外数字才可比。

下单规则（docs/lgbm-strategy-and-paper-trading-design.md §5）：
    edge = P(涨) - P(跌)
    |edge| > tau  →  按 sign(edge) 开仓
    size_mult = clip((|edge| - tau) / (edge_full - tau), 0.2, 1.0)

`tau` 是唯一的下注门槛，实测扫描见 folds.json 的 edge_sweep。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ...core.contracts import Action, Bar, Mode, Signal
from .base import MLBaseStrategy
from ..registry import register


@dataclass
class LGBMv1Config:
    symbol: str = "ETHUSDT"
    model_dir: Path | str = "models/lgbm_v1"
    tau: float = 0.15              # 下注门槛：|P(up)-P(down)| 超过它才开仓
    edge_full: float = 0.40        # |edge| 达到它就给满仓倍数 1.0
    min_size_mult: float = 0.2     # 门槛附近的最小试仓倍数
    cooldown_bars: int = 12        # 两次入场信号的最小间隔（5m 根）
    history_days: int = 90         # 推理需要回溯的历史天数
    allow_short: bool = True


@register
class LGBMv1Strategy(MLBaseStrategy):
    """概率驱动的 1h 入场策略。"""

    name = "lgbm_v1"
    required_intervals = ("5m", "1h", "4h")
    config_cls = LGBMv1Config

    def __init__(self, config: LGBMv1Config | None = None):
        super().__init__(config or LGBMv1Config())
        self._last_entry_bar = -10_000

    def reset(self) -> None:
        super().reset()
        self._last_entry_bar = -10_000

    def engine_defaults(self) -> dict[str, float]:
        """与 rule_v4 同量级：1h 决策周期必须配 1h ATR 的止损和以天计的时间止损。

        止盈取 3.0×ATR（≈ 标签上障碍 3.0×ATR 的 90%）：障碍用 k_up×ATR 定价，
        TP 设在障碍内侧才不会系统性等到「已越过障碍才平仓」的下沿结果。
        """
        return {
            "sl_atr_mult": 2.0,      # = 标签下障碍 k_dn
            "tp_atr_mult": 3.0,      # ≈ 标签上障碍 k_up
            "time_stop_bars": 300,   # 25h ≈ 标签 24h 时间障碍 + 1 根执行延迟
            "trail_atr_mult": 1.5,
            "trail_trigger_r": 1.0,
            "add_trigger_r": 1.0,
        }

    def _decide(
        self,
        bar: Bar,
        row: pd.Series,
        r1: pd.Series,
        ts_open_ms: int,
        dt: datetime,
    ) -> Signal | None:
        cfg = self.cfg
        # 冷却以引擎的基础周期（5m 根）计，与 rule_v4 的 min_bars_between_entries 同口径
        if self.n_bars - self._last_entry_bar < cfg.cooldown_bars:
            return None

        out = self.edge_at(ts_open_ms)
        if out is None:
            return None
        edge, fold, frow = out

        direction = 1 if edge > 0 else -1
        if abs(edge) < cfg.tau:
            return None
        if direction < 0 and not cfg.allow_short:
            return None

        span = max(cfg.edge_full - cfg.tau, 1e-9)
        size_mult = (abs(edge) - cfg.tau) / span
        size_mult = min(1.0, max(cfg.min_size_mult, size_mult))

        atr_1h = float(frow["atr"])
        close = float(row["close"])
        if math.isnan(atr_1h) or atr_1h <= 0 or math.isnan(close):
            return None

        self._last_entry_bar = self.n_bars
        side = "long" if direction > 0 else "short"
        reason = (
            f"lgbm_{side}|edge{edge:+.3f}|tau{cfg.tau:.2f}"
            f"|atr{atr_1h:.2f}|fold{fold.tag}"
        )
        return Signal(
            signal_id=f"lgbm_v1_{dt:%Y%m%d%H%M%S}", ts_signal=dt, symbol=bar.symbol,
            timeframe=self.decision_interval, strategy=self.name,
            action=Action.OPEN_LONG if direction > 0 else Action.OPEN_SHORT,
            price_ref=close,
            strength=min(1.0, abs(edge) / max(cfg.edge_full, 1e-9)),
            reason=reason,
            mode=Mode.BACKTEST,
            meta={"atr": atr_1h, "size_mult": round(size_mult, 4)},
        )
