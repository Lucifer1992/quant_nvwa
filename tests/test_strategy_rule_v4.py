"""rule_v4 策略测试：1h 趋势状态机、4h 软对齐、5m 择时触发与信号 meta。

覆盖 docs/rule_v4-trend-design.md 的核心承诺：
  - 状态机只在新的已收盘 1h bar 上推进（不逐 5m 抖动）
  - 4h 只调仓位大小，不否决 1h 方向（4h 下跌时 1h 多头回调仍开仓，只是 0.70×）
  - 破位需要连续两根强反向 1h（或结构破 EMA60 / ADX 熄火），单根回调不算
  - 只有 CONFIRMED 开新仓；MATURE 仅管理已有持仓
  - signal.meta 携带 1h ATR 与 size_mult，交给引擎做价位与仓位
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.core.contracts import Action, Bar, Signal
from quant.strategy.registry import get_strategy, list_strategies
from quant.strategy.rules.rule_v4 import RuleV4Config, RuleV4Strategy

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
NAN = float("nan")
H_MS = 3_600_000


def row_1h(ts_ms: int = 0, **kw) -> pd.Series:
    d = dict(
        ts_open_ms=ts_ms, close=NAN, high=NAN, low=NAN,
        ema_20=NAN, ema_60=NAN, ema_20_prev_6=NAN,
        atr=NAN, atr_median_100=NAN, adx_14=NAN, adx_prev_3=NAN,
        macd_hist=NAN, prev_macd_hist=NAN,
        donchian_high_24=NAN, donchian_low_24=NAN,
    )
    d.update(kw)
    return pd.Series(d)


def row_5m(**kw) -> pd.Series:
    d = dict(
        close=NAN, prev_close=NAN, high=NAN, low=NAN, ema_9=NAN,
        macd_hist=NAN, prev_macd_hist=NAN, swing_low=NAN, swing_high=NAN,
        volume=NAN, vol_median=NAN,
    )
    d.update(kw)
    return pd.Series(d)


# ---- 一组「1h 多头趋势」输入：方向分 +3，ADX 35 且较 3 根前上升 ----
UP1H = dict(
    close=2000.0, high=2005.0, low=1975.0,
    ema_20=1980.0, ema_60=1950.0, ema_20_prev_6=1975.0,
    atr=20.0, atr_median_100=18.0, adx_14=35.0, adx_prev_3=30.0,
    macd_hist=5.0, prev_macd_hist=2.0, donchian_high_24=2002.0,
)
# 回调后的 1h：腿高 2005 → 收盘 1975，回撤 30 = 1.5×ATR；方向分仍为 +3
UP1H_PULLBACK = dict(
    close=1975.0, high=2000.0, low=1960.0,
    ema_20=1970.0, ema_60=1950.0, ema_20_prev_6=1965.0,
    atr=20.0, atr_median_100=18.0, adx_14=35.0, adx_prev_3=30.0,
    macd_hist=1.0, prev_macd_hist=0.5, donchian_high_24=2002.0,
)
# 5m 三项确认全中，且 low 触及 1h EMA20 带（1970 + 0.6×20 = 1982）
PB_5M = dict(
    close=1974.0, prev_close=1970.0, high=1980.0, low=1966.0, ema_9=1972.0,
    macd_hist=1.0, prev_macd_hist=0.5, swing_low=1965.0, swing_high=1990.0,
    volume=100.0, vol_median=50.0,
)
DOWN1H = dict(
    close=1900.0, high=1910.0, low=1895.0,
    ema_20=1930.0, ema_60=1960.0, ema_20_prev_6=1936.0,
    atr=20.0, atr_median_100=18.0, adx_14=35.0, adx_prev_3=30.0,
    macd_hist=-5.0, prev_macd_hist=-2.0, donchian_low_24=1898.0,
)
# 4h 空头但 ADX 只有 22（<25）：属于「逆风」而非「自由落体」
DOWN4H_MILD = dict(
    close=1900.0, high=1910.0, low=1890.0,
    ema_20=1930.0, ema_60=1960.0, ema_20_prev_6=1940.0,
    atr=40.0, adx_14=22.0, macd_hist=-8.0, prev_macd_hist=-3.0,
)
# 4h 空头且 ADX 30、价格已跌破 4h EMA60：自由落体，拦下
DOWN4H_KNIFE = dict(
    close=1900.0, high=1910.0, low=1890.0,
    ema_20=1930.0, ema_60=1950.0, ema_20_prev_6=1940.0,
    atr=40.0, adx_14=30.0, macd_hist=-8.0, prev_macd_hist=-3.0,
)


def make_bar(i: int = 0, close: float = 1974.0) -> Bar:
    t = BASE + timedelta(minutes=5 * i)
    return Bar(
        symbol="ETHUSDT", interval="5m", ts_open=t,
        ts_close=t + timedelta(minutes=5) - timedelta(milliseconds=1),
        open=close, high=close + 5, low=close - 5, close=close, volume=10.0,
        quote_volume=10.0 * close, taker_buy_volume=5.0, is_closed=True,
    )


def strat(config: RuleV4Config | None = None) -> RuleV4Strategy:
    return RuleV4Strategy(config or RuleV4Config())


def feed(st, r1, r4, r5, ts_ms=0, i=0):
    return st.on_bar(make_bar(i), r5, {"1h": r1, "4h": r4}, BASE + timedelta(milliseconds=ts_ms))


class TestRegistration:
    def test_registered_and_requires_three_intervals(self):
        assert "rule_v4" in list_strategies()
        assert get_strategy("rule_v4") is RuleV4Strategy
        st = strat()
        assert st.required_intervals == ("5m", "1h", "4h")
        assert st.warmup() > 0

    def test_engine_defaults_cover_exit_params(self):
        ed = strat().engine_defaults()
        # 1h 趋势仓必须放宽出场，否则引擎的 5m 口径会 8 小时强平
        assert ed["sl_atr_mult"] == 2.0
        assert ed["time_stop_bars"] > 500
        assert set(ed) >= {"sl_atr_mult", "tp_atr_mult", "time_stop_bars", "trail_atr_mult"}


class TestDirScore:
    def test_full_bullish_stack_scores_plus_3(self):
        assert strat()._dir_score(row_1h(**UP1H)) == 3

    def test_full_bearish_stack_scores_minus_3(self):
        assert strat()._dir_score(row_1h(**DOWN1H)) == -3

    def test_missing_indicators_score_zero(self):
        assert strat()._dir_score(row_1h(ts_open_ms=0)) == 0

    def test_slope_below_atr_threshold_not_counted(self):
        r = row_1h(**{**UP1H, "ema_20_prev_6": 1979.0})  # 斜率 1 < 0.15×20
        assert strat()._dir_score(r) == 2


class TestStateOnlyAdvancesOnNew1hBar:
    def test_same_1h_bar_does_not_reupdate(self):
        st = strat()
        # 起势这根既没回撤武装、也未破 24 根高（donchian 2002 > 5m 收 1974），不开仓
        assert feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m(**PB_5M)) is None
        assert st._s.bias == 1 and st._s.status == "CONFIRMED"
        # 同一根 1h 内后续 5m：即使 1h 指标已换成空头，状态机也不重算
        feed(st, row_1h(ts_ms=0, **DOWN1H), None, row_5m(**PB_5M), i=1)
        assert (st._s.bias, st._s.status, st._s.dir_score) == (1, "CONFIRMED", 3)
        assert st._s.adx == pytest.approx(35.0)
        # 下一根 1h 收盘才重算 → 反手成空头
        feed(st, row_1h(ts_ms=H_MS, **DOWN1H), None, row_5m(**PB_5M), i=2)
        assert st._s.dir_score == -3

    def test_no_1h_row_is_noop(self):
        st = strat()
        assert st.on_bar(make_bar(), row_5m(**PB_5M), {}, BASE) is None
        assert st._s.bias == 0


class TestFourHourSoftAlignment:
    """4h 只调仓位，不否决方向。"""

    def test_aligned_full_size(self):
        st = strat()
        r4 = row_1h(**{**UP1H, "atr": 40.0})
        assert st._w4(r4, 1) == pytest.approx(RuleV4Config().w4_aligned)

    def test_ranging_4h_is_neutral_weight(self):
        st = strat()
        r4 = row_1h(**{**UP1H, "adx_14": 12.0, "close": 1900.0, "ema_20": 1930.0})
        assert st._w4(r4, 1) == pytest.approx(RuleV4Config().w4_neutral)

    def test_missing_4h_row_is_neutral_weight(self):
        assert strat()._w4(None, 1) == pytest.approx(RuleV4Config().w4_neutral)

    def test_mild_counter_trend_reduces_but_does_not_block(self):
        st = strat()
        assert st._w4(row_1h(**DOWN4H_MILD), 1) == pytest.approx(RuleV4Config().w4_weak_against)

    def test_falling_knife_blocks(self):
        st = strat()
        assert st._w4(row_1h(**DOWN4H_KNIFE), 1) == pytest.approx(RuleV4Config().w4_blocked)

    def test_long_fires_in_4h_downtrend_at_reduced_size(self):
        """决策点 Q3 的落点：4h 下跌 + 1h 多头回调 → 仍然开多，仓位 0.70×。"""
        st = strat()
        assert feed(st, row_1h(ts_ms=0, **UP1H), row_1h(**DOWN4H_MILD), row_5m()) is None
        sig = feed(st, row_1h(ts_ms=H_MS, **UP1H_PULLBACK), row_1h(**DOWN4H_MILD),
                   row_5m(**PB_5M), ts_ms=H_MS, i=1)
        assert isinstance(sig, Signal)
        assert sig.action is Action.OPEN_LONG
        assert sig.meta["size_mult"] == pytest.approx(0.70)

    def test_falling_knife_suppresses_entry(self):
        st = strat()
        feed(st, row_1h(ts_ms=0, **UP1H), row_1h(**DOWN4H_KNIFE), row_5m())
        assert feed(st, row_1h(ts_ms=H_MS, **UP1H_PULLBACK), row_1h(**DOWN4H_KNIFE),
                    row_5m(**PB_5M), ts_ms=H_MS) is None


class TestEntries:
    def test_pullback_entry_carries_1h_atr(self):
        st = strat()
        feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m())
        sig = feed(st, row_1h(ts_ms=H_MS, **UP1H_PULLBACK), row_1h(**UP1H),
                   row_5m(**PB_5M), ts_ms=H_MS)
        assert sig is not None
        assert sig.reason.startswith("1h_trend_pullback_long")
        assert sig.timeframe == "1h"
        # 价位口径用 1h ATR（20），不是引擎的 5m ATR
        assert sig.meta["atr"] == pytest.approx(20.0)
        assert sig.meta["size_mult"] == pytest.approx(1.0)
        assert 0 < sig.strength <= 1.0

    def test_no_entry_before_pullback_armed(self):
        st = strat()
        # 起势那根没有回撤，突破口也因 donchian 未破而关闭
        assert feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m(**PB_5M)) is None
        assert st._s.pulled_back is False

    def test_breakout_entry_needs_expanding_and_no_headwind(self):
        st = strat()
        feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m())
        r5 = row_5m(**{**PB_5M, "close": 2010.0, "prev_close": 2000.0, "low": 2000.0})
        sig = feed(st, row_1h(ts_ms=H_MS, **UP1H), None, r5, ts_ms=H_MS)
        assert sig is not None and sig.reason.startswith("1h_trend_breakout_long")
        # 4h 逆风时突破口关闭（追高 + 逆大周期）
        st2 = strat()
        feed(st2, row_1h(ts_ms=0, **UP1H), row_1h(**DOWN4H_MILD), row_5m())
        assert feed(st2, row_1h(ts_ms=H_MS, **UP1H), row_1h(**DOWN4H_MILD), r5,
                    ts_ms=H_MS) is None

    def test_mature_state_opens_nothing(self):
        st = strat()
        feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m())
        st._s.status = "MATURE"
        st._s.pulled_back = True
        assert feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m(**PB_5M)) is None

    def test_entry_cooldown_blocks_a_second_signal(self):
        st = strat()
        feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m())
        pb = row_1h(ts_ms=H_MS, **UP1H_PULLBACK)
        assert feed(st, pb, row_1h(**UP1H), row_5m(**PB_5M), ts_ms=H_MS) is not None
        assert st._s.bars_since_entry == 0
        st._s.pulled_back = True
        assert feed(st, pb, row_1h(**UP1H), row_5m(**PB_5M), ts_ms=H_MS + 300_000) is None
        st._s.bars_since_entry = RuleV4Config().min_bars_between_entries
        assert feed(st, pb, row_1h(**UP1H), row_5m(**PB_5M), ts_ms=H_MS + 600_000) is not None

    def test_pullback_is_single_shot_until_new_extreme(self):
        st = strat()
        feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m())
        feed(st, row_1h(ts_ms=H_MS, **UP1H_PULLBACK), row_1h(**UP1H), row_5m(**PB_5M),
             ts_ms=H_MS)
        # 一次回调只试一次
        assert st._s.pulled_back is False


class TestShortSide:
    """做空镜像路径 + allow_short 开关。"""

    DOWN_PB_1H = dict(
        close=1975.0, high=1990.0, low=1970.0,
        ema_20=1980.0, ema_60=2000.0, ema_20_prev_6=1986.0,
        atr=20.0, atr_median_100=18.0, adx_14=35.0, adx_prev_3=30.0,
        macd_hist=-1.0, prev_macd_hist=0.5, donchian_low_24=1930.0,
    )
    DOWN_PB_5M = dict(
        close=1974.0, prev_close=1978.0, high=1969.0, low=1960.0, ema_9=1976.0,
        macd_hist=0.4, prev_macd_hist=1.0, swing_low=1940.0, swing_high=1980.0,
        volume=100.0, vol_median=50.0,
    )

    def _setup_short(self, st):
        first = row_1h(ts_ms=0, **{**DOWN1H, "low": 1940.0})
        feed(st, first, row_1h(**DOWN1H), row_5m())
        assert st._s.bias == -1

    def test_short_entry_when_allowed(self):
        st = strat()
        self._setup_short(st)
        sig = feed(st, row_1h(ts_ms=H_MS, **self.DOWN_PB_1H), row_1h(**DOWN1H),
                   row_5m(**self.DOWN_PB_5M), ts_ms=H_MS)
        assert sig is not None
        assert sig.action is Action.OPEN_SHORT
        assert sig.reason.startswith("1h_trend_pullback_short")
        assert sig.meta["atr"] == pytest.approx(20.0)

    def test_short_blocked_when_allow_short_false(self):
        st = strat(RuleV4Config(allow_short=False))
        self._setup_short(st)
        assert feed(st, row_1h(ts_ms=H_MS, **self.DOWN_PB_1H), row_1h(**DOWN1H),
                    row_5m(**self.DOWN_PB_5M), ts_ms=H_MS) is None
        assert st._s.bias == -1  # 只是不开仓，方向判断照旧


class TestBrokenExit:
    def _confirmed_long(self, st):
        feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m())
        return st._s.bias == 1 and st._s.status == "CONFIRMED"

    def _reversal_row(self, ts_ms: int) -> pd.Series:
        """结构未破 EMA60、但斜率与动量同时转负 → 方向分 -2。"""
        return row_1h(ts_ms=ts_ms, **{
            **UP1H, "close": 1965.0, "ema_20": 1970.0, "ema_20_prev_6": 1976.0,
            "macd_hist": -1.0, "prev_macd_hist": 1.0, "adx_prev_3": 36.0,
        })

    def test_single_reversal_bar_is_not_a_break(self):
        st = strat()
        assert self._confirmed_long(st)
        assert feed(st, self._reversal_row(H_MS), None, row_5m(**PB_5M), ts_ms=H_MS) is None
        assert st._s.status != "BROKEN"
        assert st._s.reversal_streak == 1

    def test_two_consecutive_reversal_bars_exit_long(self):
        st = strat()
        self._confirmed_long(st)
        feed(st, self._reversal_row(H_MS), None, row_5m(), ts_ms=H_MS)
        sig = feed(st, self._reversal_row(2 * H_MS), None, row_5m(), ts_ms=2 * H_MS)
        assert sig is not None
        assert sig.action is Action.CLOSE_LONG
        assert "broken" in sig.reason
        # 腿状态复位，等待重新起势
        assert st._s.bias == 0 and st._s.status == "RANGING"
        assert st._s.leg_high == 0.0 and st._s.pulled_back is False

    def test_structural_break_exits_immediately(self):
        st = strat()
        self._confirmed_long(st)
        r = row_1h(ts_ms=H_MS, **{**UP1H, "close": 1940.0})  # 跌破 EMA60=1950
        sig = feed(st, r, None, row_5m(), ts_ms=H_MS)
        assert sig is not None and sig.action is Action.CLOSE_LONG
        assert "below_ema60" in sig.reason

    def test_adx_exhaustion_exits(self):
        st = strat()
        self._confirmed_long(st)
        r = row_1h(ts_ms=H_MS, **{**UP1H, "adx_14": 10.0})
        sig = feed(st, r, None, row_5m(), ts_ms=H_MS)
        assert sig is not None and sig.action is Action.CLOSE_LONG

    def test_short_bias_emits_close_short(self):
        st = strat()
        feed(st, row_1h(ts_ms=0, **{**DOWN1H, "low": 1940.0}), None, row_5m())
        assert st._s.bias == -1
        r = row_1h(ts_ms=H_MS, **{**DOWN1H, "close": 1990.0})  # 收回 EMA60=1960 上方
        sig = feed(st, r, None, row_5m(), ts_ms=H_MS)
        assert sig is not None and sig.action is Action.CLOSE_SHORT

    def test_exit_signal_not_emitted_when_flat_bias(self):
        st = strat()
        feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m())
        st._s.bias = 0  # 从未持仓的方向：BROKEN 不该凭空发平仓
        st._s.status = "MATURE"
        sig = feed(st, row_1h(ts_ms=H_MS, **{**UP1H, "adx_14": 10.0}), None,
                   row_5m(), ts_ms=H_MS)
        assert sig is None or sig.action not in (Action.CLOSE_LONG, Action.CLOSE_SHORT)


class TestSignalSanity:
    def test_no_nan_leaks_into_signal(self):
        st = strat()
        feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m())
        sig = feed(st, row_1h(ts_ms=H_MS, **UP1H_PULLBACK), None, row_5m(**PB_5M), ts_ms=H_MS)
        assert sig.price_ref > 0 and not math.isnan(sig.meta["atr"])
        assert 0.2 <= sig.meta["size_mult"] <= 1.0

    def test_reset_clears_leg_state(self):
        st = strat()
        feed(st, row_1h(ts_ms=0, **UP1H), None, row_5m())
        st.reset()
        assert st._s.bias == 0 and st._s.last_1h_ts is None
        assert st._s.status == "RANGING"
