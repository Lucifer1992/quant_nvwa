"""ML 策略接线测试：折窗口边界、特征集守卫、下注规则。

不打 lightgbm（`booster()` 是懒加载），统一用 stub 注入 edge，
这样测试只覆盖「接线逻辑」，不依赖模型工件是否存在。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant.core.contracts import Action, Bar
from quant.core.timeframe import to_ms
from quant.strategy.ml.base import FoldModel, MLBaseStrategy
from quant.strategy.ml.lgbm_v1 import LGBMv1Config, LGBMv1Strategy
from quant.strategy.rules.rule_v1 import RuleV1Strategy

TS = to_ms(datetime(2026, 5, 1, 12, tzinfo=timezone.utc))


def _fold(tag: str, start_ms: int, end_ms: int, spec_hash: str = "deadbeefcafe") -> FoldModel:
    return FoldModel(
        tag=tag, path=Path("/unused/model.txt"), test_start_ms=start_ms, test_end_ms=end_ms,
        features=["f1"], feature_spec_hash=spec_hash, label_params={"horizon": 24},
    )


class _Stub(MLBaseStrategy):
    """只保留折选择/特征守卫，绕开真实数据帧。"""

    name = "_stub_ml"
    required_intervals = ("5m",)

    def __init__(self, fold, config=None):
        super().__init__(config)
        self._folds = [fold] if isinstance(fold, FoldModel) else fold

    def symbol(self) -> str:
        return "ETHUSDT"

    def model_dir(self) -> Path:
        return Path("/unused")

    def _decide(self, bar, row, r1, ts_open_ms, dt):  # pragma: no cover
        raise NotImplementedError


def _frame() -> pd.DataFrame:
    from quant.features.builder import FEATURE_COLUMNS

    df = pd.DataFrame({c: 0.5 for c in FEATURE_COLUMNS}, index=[0])
    df["ts_open_ms"] = TS
    df["atr"] = 10.0
    return df


class TestFoldSelection:
    def test_window_is_half_open(self):
        f = _Stub(_fold("a", TS, TS + 3_600_000))
        f._folds = [_fold("a", TS, TS + 3_600_000)]
        assert f._fold_for(TS).tag == "a"
        assert f._fold_for(TS + 3_599_999) is not None
        assert f._fold_for(TS + 3_600_000) is None   # 右开：跨月就换折，不会用旧模型

    def test_earliest_fold_wins_before_window(self):
        f = _Stub(None)
        f._folds = [_fold("a", TS, TS + 3_600_000)]
        assert f._fold_for(TS - 1) is None

    def test_warn_missing_does_not_crash(self, caplog):
        """回归用例：折窗口外的 bar 要跳过并告警一次，而不是抛异常。

        旧实现拿 `datetime` 和 `int` 比较（`self._first_test_start()` 返回 dt），
        任何超出折窗口的样本外区间都会 TypeError 崩溃。
        """
        import logging

        f = _Stub(_fold("a", TS, TS + 3_600_000))
        with caplog.at_level(logging.WARNING):
            assert f._fold_for(TS - 1) is None       # 早于首折
            assert f._fold_for(TS + 7_200_000) is None  # 晚于末折
            f._warn_missing(TS - 1)
            f._warn_missing(TS - 2)                  # 同一类只告警一次
            f._warn_missing(TS + 7_200_000)
        msgs = [r.getMessage() for r in caplog.records if "无可用折模型" in r.getMessage()]
        assert len(msgs) == 2

    def test_spec_hash_mismatch_raises(self):
        """特征集漂移必须炸，不能静默用旧模型预测新特征。"""
        f = _Stub(_fold("a", TS, TS + 3_600_000, spec_hash="000000000000"))
        with pytest.raises(ValueError, match="特征集"):
            f._check_features(_frame())

    def test_load_folds_missing_dir(self, tmp_path):
        from quant.strategy.ml.base import load_folds

        with pytest.raises(FileNotFoundError):
            load_folds(tmp_path, "NOPE")


class TestRuleStrategyLookback:
    def test_rule_strategies_declare_no_lookback(self):
        """回归护栏：任何规则策略一旦返回非 0，CLI 就会多取历史、改变指标预热长度。"""
        from quant.strategy.rules.rule_v2 import RuleV2Strategy
        from quant.strategy.rules.rule_v3 import RuleV3Strategy
        from quant.strategy.rules.rule_v4 import RuleV4Strategy

        for cls in (RuleV1Strategy, RuleV2Strategy, RuleV3Strategy, RuleV4Strategy):
            assert cls().history_lookback() == 0


def test_lgbm_lookback_is_90_days():
    """非 0 的 lookback 会改变高周期指标预热长度，所以只有 ML 策略允许声明。"""
    assert LGBMv1Strategy(LGBMv1Config()).history_lookback() == 90 * 24 * 3_600_000


class TestBettingRule:
    def _mk(self, tau=0.15, edge_full=0.40, allow_short=True, cooldown=12):
        strat = LGBMv1Strategy(LGBMv1Config(
            tau=tau, edge_full=edge_full, allow_short=allow_short, cooldown_bars=cooldown))
        strat._decide_ref = strat._decide
        return strat

    def _bar(self):
        dt = datetime(2026, 5, 1, 13, tzinfo=timezone.utc)
        return Bar(symbol="ETHUSDT", interval="5m", ts_open=dt, ts_close=dt,
                   open=1.0, high=1.0, low=1.0, close=2000.0, volume=1.0,
                   quote_volume=1.0, taker_buy_volume=1.0, is_closed=True)

    def _call(self, strat, edge, cooled=True):
        if cooled:
            # 冷却按引擎基础周期（5m 根）计数；单独测下注规则时把游标推过冷却窗口
            strat.n_bars += strat.cfg.cooldown_bars
        row = pd.Series({"close": 2000.0})
        r1 = pd.Series({"ts_open_ms": TS})
        fold = _fold("202605", TS - 1000, TS + 1000)
        strat.edge_at = lambda ts, _e=edge, _f=fold: (_e, _f, pd.Series({"atr": 25.0}))
        return strat._decide(self._bar(), row, r1, TS, datetime(2026, 5, 1, 13, tzinfo=timezone.utc))

    def test_below_tau_no_signal(self):
        s = self._mk()
        assert self._call(s, 0.14) is None
        assert self._call(s, -0.14) is None

    def test_size_mult_ramps_from_min_to_one(self):
        s = self._mk(tau=0.15, edge_full=0.40)
        lo = self._call(s, 0.16)
        hi = self._call(s, 0.60)
        assert lo.meta["size_mult"] == pytest.approx(0.2)         # 门槛附近只给最小试仓
        assert hi.meta["size_mult"] == 1.0                        # 上限封顶
        mid = self._call(s, 0.275)
        assert 0.2 < mid.meta["size_mult"] < 1.0

    def test_direction(self):
        s = self._mk()
        assert self._call(s, 0.30).action is Action.OPEN_LONG
        s2 = self._mk()
        assert self._call(s2, -0.30).action is Action.OPEN_SHORT

    def test_short_can_be_disabled(self):
        s = self._mk(allow_short=False)
        assert self._call(s, -0.5) is None

    def test_cooldown_blocks_consecutive_entries(self):
        """回归用例：冷却单位是 5m 根，不是 1h 决策根。

        旧实现把 12 当作 12 次 1h 决策（= 12 小时），与配置文档和 rule_v4 的
        `min_bars_between_entries` 口径都差 12 倍。
        """
        s = self._mk(cooldown=12)
        assert self._call(s, 0.30) is not None
        entry_bar = s.n_bars
        assert s._last_entry_bar == entry_bar
        assert self._call(s, 0.30, cooled=False) is None   # 下一根 5m 仍在冷却里
        s.n_bars = entry_bar + 11                          # 差 1 根，仍未解锁
        assert self._call(s, 0.30, cooled=False) is None
        s.n_bars = entry_bar + 12                          # 刚好满足 12 根 5m
        assert self._call(s, 0.30, cooled=False) is not None

    def test_atr_handed_to_engine(self):
        """出场必须用 1h ATR，否则引擎会退到 5m ATR（约 0.17%）被噪音扫掉。"""
        s = self._mk()
        sig = self._call(s, 0.30)
        assert sig.meta["atr"] == 25.0

    def test_missing_model_returns_none(self):
        s = self._mk()
        s.edge_at = lambda ts: None
        row = pd.Series({"close": 2000.0})
        assert s._decide(self._bar(), row, pd.Series({"ts_open_ms": TS}), TS,
                         datetime(2026, 5, 1, tzinfo=timezone.utc)) is None


def test_on_bar_fires_once_per_1h_bar():
    s = LGBMv1Strategy(LGBMv1Config())
    dt = datetime(2026, 5, 1, 13, tzinfo=timezone.utc)
    bar = Bar(symbol="ETHUSDT", interval="5m", ts_open=dt, ts_close=dt, open=1, high=1,
              low=1, close=2000.0, volume=1, quote_volume=1, taker_buy_volume=1,
              is_closed=True)
    row = pd.Series({"close": 2000.0})
    calls = []
    s._decide = lambda *a: (calls.append(a[3]) or None)
    htf = {"1h": pd.Series({"ts_open_ms": TS})}
    for _ in range(12):                     # 同一根 1h 内的 12 根 5m
        s.on_bar(bar, row, htf, dt)
    assert len(calls) == 1
    htf = {"1h": pd.Series({"ts_open_ms": TS + 3_600_000})}
    s.on_bar(bar, row, htf, dt)
    assert len(calls) == 2


def test_reset_clears_decision_cursor():
    s = LGBMv1Strategy(LGBMv1Config())
    s._last_ts = 123
    s.n_bars = 500
    s._last_entry_bar = 400
    s.reset()
    assert s._last_ts is None
    # 5m 游标和入场锚点必须一起归零，否则第二次回测的冷却会串台
    assert s.n_bars == 0
    assert s._last_entry_bar < -1000
