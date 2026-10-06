"""rule_v4: 1h 粒度趋势跟随 + 4h 软对齐 + 5m 择时触发。

**三层结构**（详见 docs/rule_v4-trend-design.md）：

  1. **1h 趋势状态机**：只在新的已收盘 1h bar 上更新。
     方向分 = 均线结构 + 均线斜率 + MACD 动量（−3..+3）；
     强度门 = ADX14(1h) ≥ adx_min，ADX 相对 3 根前上升记为 EXPANDING。
     状态：RANGING → BUILDING → CONFIRMED → MATURE → BROKEN。
  2. **4h 软对齐**：产出仓位倍数 w4 ∈ {1.00, 0.85, 0.70, 0}。
     同向满额；4h 震荡 0.85；4h 反向但平缓 0.70（**照常做多，只减仓**）；
     只有「4h 强趋势逆向且已破 4h EMA60」才置 0 拦下本次入场。
     4h 从不否决方向，只调节仓位大小。
  3. **5m 择时触发**：两个入场口
     A 回调顺势（主入口）：1h 腿自极值回撤 ≥ pullback_atr×ATR1h 且触及 1h EMA20 带，
        再由 5m 三选二确认（动量拐上 / 更高低点止跌 / 放量承接）；
     B 结构突破（仅 EXPANDING）：5m 收盘上穿最近 24 根 1h 高点，且 1h ATR 高于其 100 根中位数。

**风控价位**：止损/止盈/移动止损全部基于 **1h ATR**（通过 `signal.meta["atr"]` 交给引擎），
而非引擎默认的 5m ATR —— 5m ATR 只有约 0.17%，1h 趋势仓会被噪音扫掉。

**出场**：1h 状态转 BROKEN（破 EMA60 或趋势熄火）→ 发 `CLOSE_*`；
1h 方向反转 → 直接发反向 `OPEN_*`，由引擎在同一根 bar 开盘完成反手。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ...core.contracts import Action, Bar, Mode, Signal
from ..base import BaseStrategy
from ..registry import register

NAN = float("nan")


def _f(row: pd.Series, key: str) -> float:
    """安全取浮点：列缺失或 NaN 都返回 nan，由调用方判断。"""
    v = row.get(key)
    if v is None:
        return NAN
    try:
        x = float(v)
    except (TypeError, ValueError):
        return NAN
    return x if not math.isnan(x) else NAN


@dataclass
class RuleV4Config:
    # ---- 1h 趋势状态机 ----
    # adx_min 是实测中最强的轴：20→-20%，25→-9%，30→+18%，32→+22%，40→+17%（近半年 ETHUSDT）。
    # 30~40 都是正收益平台区，默认取中段 32。
    trend_dir_min: int = 2          # 方向分绝对值达到它才算 CONFIRMED
    adx_min: float = 32.0           # 开仓强度门（1h ADX14）
    reversal_dir_score: int = 2     # 反向方向分达到它才算一次反转
    reversal_bars: int = 2          # 连续 N 根反转才算破位（回调不算）
    adx_ranging_exit: float = 15.0  # 持仓中 ADX 掉到这里以下视为趋势熄火
    slope_atr_ratio: float = 0.15   # EMA20 的 6 小时斜率需超过 0.15×ATR 才算推进
    # ---- 回调 / 突破入场 ----
    pullback_atr: float = 1.5       # 自 1h 腿极值回撤门槛（×ATR1h）
    # 0.2 时回调入口半年只触发 2 次（回撤 1.5ATR 时价格通常已远离 1h EMA20），0.6 才有 7 次
    ema_band_atr_ratio: float = 0.60  # 触及 1h EMA20 带的容差
    confirm_requires: int = 2       # 5m 三选二确认
    vol_mult: float = 1.2           # 放量确认倍数
    # ---- 4h 软对齐权重 ----
    w4_aligned: float = 1.00
    w4_neutral: float = 0.85        # 4h 无方向或只是震荡
    w4_weak_against: float = 0.70   # 4h 反向有趋势，但未构成自由落体
    w4_blocked: float = 0.00        # 4h 强趋势逆向且价格已在错误一侧
    adx_4h_trending: float = 20.0
    adx_4h_strong: float = 25.0
    # ---- 入场节奏 ----
    min_bars_between_entries: int = 12  # 两次入场信号至少间隔 N 根 5m（=1 小时）
    allow_short: bool = True


@dataclass
class RuleV4State:
    last_1h_ts: int | None = None
    bias: int = 0            # 1h 趋势方向 +1/-1/0
    status: str = "RANGING"  # RANGING / BUILDING / CONFIRMED / MATURE / BROKEN
    expanding: bool = False
    dir_score: int = 0
    adx: float = NAN
    reversal_streak: int = 0
    leg_high: float = 0.0    # 本趋势腿的极值（回调基准）
    leg_low: float = 0.0
    pulled_back: bool = False
    bars_since_entry: int = 10_000


@register
class RuleV4Strategy(BaseStrategy):
    """1h 趋势跟随（4h 软对齐 / 5m 择时）。"""

    name = "rule_v4"
    required_intervals = ("5m", "1h", "4h")
    config_cls = RuleV4Config


    def __init__(self, config: RuleV4Config | None = None):
        super().__init__(config or RuleV4Config())
        self._s = RuleV4State()
        self._warmup = 65  # 需要 1h EMA60 + ADX14，预热以引擎的 HTF 覆盖长度为准

    def warmup(self) -> int:
        return self._warmup

    def engine_defaults(self) -> dict[str, float]:
        """1h 趋势仓的出场参数（实测于近半年 ETHUSDT，见 docs/rule_v4-trend-design.md）。

        止损 2.0×ATR1h：1.8 会退化成噪音止损（半年 +18% → 0%）。
        时间止损 864 根 5m = 3 天：引擎默认 96 根（8 小时）会在趋势腿中段强平。
        """
        return {
            "sl_atr_mult": 2.0,
            "tp_atr_mult": 3.5,
            "time_stop_bars": 864,
            "trail_atr_mult": 1.5,
            "trail_trigger_r": 1.0,
            "add_trigger_r": 1.0,
        }

    def reset(self) -> None:
        self._s = RuleV4State()

    # ---- 第一层：1h 方向分与状态机 ----

    def _dir_score(self, r: pd.Series) -> int:
        """1h/4h 通用方向分：均线结构 + 斜率 + MACD 动量，返回 −3..+3。"""
        close, e20, e60 = _f(r, "close"), _f(r, "ema_20"), _f(r, "ema_60")
        atr, prev_e20 = _f(r, "atr"), _f(r, "ema_20_prev_6")
        hist, prev_hist = _f(r, "macd_hist"), _f(r, "prev_macd_hist")
        if any(math.isnan(x) for x in (close, e20, e60, atr)):
            return 0

        score = 0
        if close > e20 > e60:
            score += 1
        elif close < e20 < e60:
            score -= 1

        if not math.isnan(prev_e20):
            need = self.cfg.slope_atr_ratio * atr
            if e20 - prev_e20 > need and close > e20:
                score += 1
            elif prev_e20 - e20 > need and close < e20:
                score -= 1

        if not math.isnan(hist) and not math.isnan(prev_hist):
            if hist > 0 and hist > prev_hist:
                score += 1
            elif hist < 0 and hist < prev_hist:
                score -= 1
        return score

    def _update_1h(self, r1: pd.Series) -> None:
        cfg, s = self.cfg, self._s
        d = self._dir_score(r1)
        adx = _f(r1, "adx_14")
        prev_adx = _f(r1, "adx_prev_3")
        close, e60 = _f(r1, "close"), _f(r1, "ema_60")
        if math.isnan(adx):
            s.status = "RANGING"
            s.bias = 0
            s.expanding = False
            s.dir_score = d
            return

        s.dir_score = d
        s.adx = adx
        strong = adx >= cfg.adx_min

        # 已有方向的腿：维护极值，供回调判定
        if s.bias > 0:
            s.leg_high = max(s.leg_high, _f(r1, "high"))
            retr = s.leg_high - close
            if s.leg_high > 0 and retr >= cfg.pullback_atr * _f(r1, "atr"):
                s.pulled_back = True
        elif s.bias < 0:
            lo = _f(r1, "low")
            s.leg_low = min(s.leg_low, lo) if s.leg_low > 0 else lo
            retr = close - s.leg_low
            if s.leg_low > 0 and retr >= cfg.pullback_atr * _f(r1, "atr"):
                s.pulled_back = True

        if s.bias != 0:
            # 回调中的 1h bar 方向分转负是正常的，单根反向不构成破位；
            # 只有「结构破 EMA60」或「连续 2 根强反向分」或「ADX 熄火」才算趋势结束。
            structural = (s.bias > 0 and close < e60) or (s.bias < 0 and close > e60)
            reversal = (s.bias > 0 and d <= -cfg.reversal_dir_score) or \
                       (s.bias < 0 and d >= cfg.reversal_dir_score)
            s.reversal_streak = s.reversal_streak + 1 if reversal else 0
            exhausted = adx < cfg.adx_ranging_exit
            if structural or s.reversal_streak >= cfg.reversal_bars or exhausted:
                s.status = "BROKEN"
                s.expanding = False
                return
            if strong and abs(d) >= cfg.trend_dir_min and not math.isnan(prev_adx) and adx > prev_adx:
                s.status = "CONFIRMED"
                s.expanding = True
            else:
                s.status = "MATURE" if strong else "BUILDING"
                s.expanding = False
            return

        # 无方向：重新起势
        if strong and abs(d) >= cfg.trend_dir_min:
            s.bias = 1 if d > 0 else -1
            s.status = "CONFIRMED"
            s.expanding = not math.isnan(prev_adx) and adx > prev_adx
            s.leg_high = _f(r1, "high")
            s.leg_low = _f(r1, "low")
            s.pulled_back = False
            s.reversal_streak = 0
        elif abs(d) >= 1:
            s.status = "BUILDING"
            s.expanding = False
        else:
            s.status = "RANGING"
            s.expanding = False

    # ---- 第二层：4h 软对齐（仓位权重，不否决方向）----

    def _w4(self, r4: pd.Series | None, direction: int) -> float:
        """4h 背景 → 本次入场仓位倍数。任何情况都只调大小，不改 1h 的方向判断。"""
        cfg = self.cfg
        if r4 is None or direction == 0:
            return cfg.w4_neutral
        d4 = self._dir_score(r4)
        adx4 = _f(r4, "adx_14")
        e60 = _f(r4, "ema_60")
        c4 = _f(r4, "close")

        if d4 == 0:
            return cfg.w4_neutral            # 4h 没有方向，不帮忙也不碍事
        if (d4 > 0) == (direction > 0):
            return cfg.w4_aligned            # 4h 同向共振，满额试仓

        # 反向：只有「4h 强趋势逆向且价格已在错误一侧」才拦下（自由落体不接刀）
        falling_knife = (
            not math.isnan(adx4) and adx4 >= cfg.adx_4h_strong
            and not math.isnan(e60) and not math.isnan(c4)
            and ((direction > 0 and c4 < e60) or (direction < 0 and c4 > e60))
        )
        if falling_knife:
            return cfg.w4_blocked
        if not math.isnan(adx4) and adx4 < cfg.adx_4h_trending:
            return cfg.w4_neutral            # 4h 只是震荡 → 轻度减仓
        return cfg.w4_weak_against           # 4h 反向有趋势，但未构成自由落体 → 减 30% 照做

    # ---- 第三层：5m 择时确认 ----

    def _confirm_5m(self, row: pd.Series, direction: int) -> int:
        """返回满足的确认项数（0..3）。"""
        cfg = self.cfg
        close, prev_close = _f(row, "close"), _f(row, "prev_close")
        ema9 = _f(row, "ema_9")
        hist, prev_hist = _f(row, "macd_hist"), _f(row, "prev_macd_hist")
        n = 0
        if direction > 0:
            if not any(map(math.isnan, (close, ema9, hist, prev_hist))) and close > ema9 and hist > prev_hist:
                n += 1  # 5m 动量拐上
            swing_low = _f(row, "swing_low")
            if not math.isnan(swing_low) and close > prev_close and _f(row, "low") >= swing_low:
                n += 1  # 止跌回升（未创新低）
        else:
            if not any(map(math.isnan, (close, ema9, hist, prev_hist))) and close < ema9 and hist < prev_hist:
                n += 1
            swing_high = _f(row, "swing_high")
            if not math.isnan(swing_high) and close < prev_close and _f(row, "high") <= swing_high:
                n += 1  # 止升回落（未创新高）
        vol, vol_med = _f(row, "volume"), _f(row, "vol_median")
        if not any(map(math.isnan, (vol, vol_med, close, prev_close))) and vol_med > 0:
            if vol > vol_med * cfg.vol_mult and (close - prev_close) * direction > 0:
                n += 1  # 顺向放量
        return n

    def _touch_ema20_band(self, row: pd.Series, r1: pd.Series, direction: int) -> bool:
        e20_1h = _f(r1, "ema_20")
        atr1h = _f(r1, "atr")
        if math.isnan(e20_1h) or math.isnan(atr1h) or e20_1h <= 0:
            return False
        tol = self.cfg.ema_band_atr_ratio * atr1h
        if direction > 0:
            return _f(row, "low") <= e20_1h + tol
        return _f(row, "high") >= e20_1h - tol

    def _breakout(self, row: pd.Series, r1: pd.Series, direction: int) -> bool:
        dc_hi, dc_lo = _f(r1, "donchian_high_24"), _f(r1, "donchian_low_24")
        atr1h, atr_med = _f(r1, "atr"), _f(r1, "atr_median_100")
        if math.isnan(atr1h) or (not math.isnan(atr_med) and atr1h <= atr_med):
            return False  # 波动未扩张
        if direction > 0:
            return not math.isnan(dc_hi) and _f(row, "close") > dc_hi
        return not math.isnan(dc_lo) and _f(row, "close") < dc_lo

    # ---- 主循环（每根 5m bar 调用）----

    def on_bar(
        self,
        bar: Bar,
        row: pd.Series,
        htf: dict[str, pd.Series],
        dt: datetime,
    ) -> Signal | None:
        cfg, s = self.cfg, self._s
        s.bars_since_entry += 1

        r1 = htf.get("1h")
        if r1 is None:
            return None
        r4 = htf.get("4h")

        new_1h = s.last_1h_ts != int(r1["ts_open_ms"])
        if new_1h:
            s.last_1h_ts = int(r1["ts_open_ms"])
            prev_bias, prev_status = s.bias, s.status
            self._update_1h(r1)
            # 1h 收盘后先处理退出/反手，避免带着旧方向继续加仓
            exit_sig = self._maybe_exit(bar, dt, prev_bias, prev_status)
            if exit_sig is not None:
                return exit_sig

        direction = s.bias
        if direction == 0 or (direction < 0 and not cfg.allow_short):
            return None
        if s.status != "CONFIRMED":
            return None  # MATURE 只管理已有持仓，不开新仓

        w4 = self._w4(r4, direction)
        if w4 <= 0:
            return None  # 4h 强趋势逆向：本次不入场（不改变方向判断）

        atr1h = _f(r1, "atr")
        if math.isnan(atr1h) or atr1h <= 0:
            return None

        close = _f(row, "close")
        if math.isnan(close):
            return None

        entry_mode: str | None = None
        if s.pulled_back and self._touch_ema20_band(row, r1, direction):
            if self._confirm_5m(row, direction) >= cfg.confirm_requires:
                entry_mode = "pullback"
        if entry_mode is None and s.expanding and w4 >= cfg.w4_neutral:
            # 突破入口只在 4h 不逆风时启用：追高 + 逆大周期是最差的组合
            if self._breakout(row, r1, direction):
                entry_mode = "breakout"
        if entry_mode is None:
            return None
        if s.bars_since_entry < cfg.min_bars_between_entries:
            return None

        s.bars_since_entry = 0
        s.pulled_back = False  # 一次回调只允许试一次，等新的极值再重新武装

        action = Action.OPEN_LONG if direction > 0 else Action.OPEN_SHORT
        reason = (
            f"1h_trend_{entry_mode}_{'long' if direction > 0 else 'short'}"
            f"|dir{s.dir_score}|adx{s.adx:.1f}"
            f"|{'EXP' if s.expanding else 'MAT'}|w4{w4:.2f}"
        )
        return Signal(
            signal_id=f"rule_v4_{dt:%Y%m%d%H%M%S}", ts_signal=dt, symbol=bar.symbol,
            timeframe="1h", strategy=self.name,
            action=action, price_ref=close,
            strength=min(1.0, abs(s.dir_score) / 3.0),
            reason=reason,
            mode=Mode.BACKTEST,
            meta={"atr": atr1h, "size_mult": w4},
        )

    def _maybe_exit(self, bar: Bar, dt: datetime, prev_bias: int, prev_status: str) -> Signal | None:
        """1h 状态机转 BROKEN → 发对向持仓的平仓信号。"""
        s = self._s
        if prev_bias == 0 or s.status != "BROKEN":
            return None
        action = Action.CLOSE_LONG if prev_bias > 0 else Action.CLOSE_SHORT
        reason = (
            f"1h_trend_broken|dir{s.dir_score}|adx{s.adx:.1f}"
            f"|{'below_ema60' if prev_bias > 0 else 'above_ema60'}"
        )
        sig = Signal(
            signal_id=f"rule_v4_x_{dt:%Y%m%d%H%M%S}", ts_signal=dt, symbol=bar.symbol,
            timeframe="1h", strategy=self.name,
            action=action, price_ref=bar.close, strength=1.0,
            reason=reason, mode=Mode.BACKTEST,
        )
        # 复位腿状态，等待重新起势
        s.bias = 0
        s.status = "RANGING"
        s.expanding = False
        s.leg_high = 0.0
        s.leg_low = 0.0
        s.pulled_back = False
        return sig
