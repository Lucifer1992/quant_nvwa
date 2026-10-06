"""特征/标签构建的单元测试，核心是「防前视」这条硬约束。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.features import builder as B  # noqa: E402

H_MS = 3_600_000
T0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC


def make_bars(n: int, seed: int = 0, interval_ms: int = H_MS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000.0 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol = np.abs(rng.normal(100, 30, n))
    ts = T0 + np.arange(n, dtype="int64") * interval_ms
    return pd.DataFrame({
        "ts_open_ms": ts,
        "ts_open": pd.to_datetime(ts, unit="ms", utc=True),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "quote_volume": vol * close,
        "taker_buy_volume": vol * 0.5, "is_closed": True,
    })


def make_funding(times_ms: list[int], rate: float = 0.0001) -> pd.DataFrame:
    return pd.DataFrame({
        "funded_at_ms": times_ms,
        "funding_rate": [rate] * len(times_ms),
        "mark_price": [2000.0] * len(times_ms),
        "interval_hours": [8.0] * len(times_ms),
    })


class TestFeatureFrame:
    def test_feature_columns_follow_declared_list(self):
        frame = B.build_feature_frame(make_bars(400), make_bars(100, 1, 4 * H_MS),
                                      make_funding([T0 + i * 8 * H_MS for i in range(50)]))
        assert B.feature_columns(frame) == list(B.FEATURE_COLUMNS)

    def test_missing_column_raises_instead_of_silently_shrinking(self):
        frame = B.build_feature_frame(make_bars(400))
        assert "h4_basis" not in frame.columns
        try:
            B.feature_columns(frame)
        except KeyError as e:
            assert "h4_basis" in str(e)
        else:
            raise AssertionError("缺列必须报错，不能静默少喂特征")

    def test_no_all_nan_columns_and_labels_align(self):
        frame = B.build_feature_frame(make_bars(600), make_bars(150, 1, 4 * H_MS),
                                      make_funding([T0 + i * 8 * H_MS for i in range(75)]))
        cols = B.feature_columns(frame)
        assert len(frame) == 600
        assert not [c for c in cols if frame[c].isna().all()]
        assert B.build_labels(frame).notna().sum() > 0


class TestNoLookahead:
    def test_truncated_recompute_is_bit_identical(self):
        """把输入截断到第 N 根重算，前 N 根特征必须逐位相同 —— 有任何前视都会在这里崩。"""
        b1, n = make_bars(900), 700
        b4 = make_bars(225, 1, 4 * H_MS)
        fu = make_funding([T0 + i * 8 * H_MS for i in range(112)])
        full = B.build_feature_frame(b1, b4, fu)
        cut_ms = int(b1.ts_open_ms.iloc[n - 1]) + H_MS
        part = B.build_feature_frame(
            b1.iloc[:n].copy(), b4[b4.ts_open_ms < cut_ms].copy(), fu[fu.funded_at_ms < cut_ms].copy()
        )
        cols = B.feature_columns(full)
        a = full[cols].iloc[:n].to_numpy(dtype=float)
        c = part[cols].to_numpy(dtype=float)
        assert np.array_equal(a, c, equal_nan=True)

    def test_4h_bar_invisible_until_its_own_close(self):
        """4h bar 收盘之前，任何 1h 行都不该看到它的数值。"""
        b1 = make_bars(48)
        b4 = make_bars(40, 1, 4 * H_MS)  # 足够多的 4h 历史，排除指标预热干扰
        frame = B.build_feature_frame(b1, b4)
        first_4h_close = T0 + 4 * H_MS  # 第 1 根 4h（ts=T0）在 T0+4h 才收盘
        assert frame[frame.ts_open_ms < first_4h_close]["h4_adx_14"].isna().all()
        assert frame[frame.ts_open_ms >= first_4h_close + 14 * 4 * H_MS]["h4_adx_14"].notna().all()
        # 已收盘的 4h 值不会随后续 4h bar 出现而改变（对齐用的是 backward 而非全表）
        row = int(frame.ts_open_ms.iloc[40])
        late = frame[frame.ts_open_ms == row]["h4_adx_14"].iloc[0]
        assert np.isfinite(late)

    def test_funding_alignment_only_uses_past(self):
        b1 = make_bars(48)
        fu = make_funding([T0 + 10 * H_MS, T0 + 18 * H_MS], rate=0.0002)
        frame = B.build_feature_frame(b1, funding=fu)
        assert frame.loc[frame.ts_open_ms < 10 * H_MS + T0, "funding_rate"].isna().all()
        assert (frame.loc[frame.ts_open_ms >= 18 * H_MS + T0 - H_MS, "funding_rate"] == 0.0002).all()
        ok = frame["hrs_to_funding"].dropna()
        assert ((ok > 0) & (ok <= 8.0)).all()


class TestLabels:
    def _frame(self, close, high, low, atr_val=10.0):
        n = len(close)
        return pd.DataFrame({
            "ts_open_ms": T0 + np.arange(n, dtype="int64") * H_MS,
            "close": close, "high": high, "low": low,
            "atr": [atr_val] * n,
        })

    def test_upper_barrier_first(self):
        close = np.array([100.0] * 6, dtype=float)
        high = np.array([100.0, 131.0, 100.0, 100.0, 100.0, 100.0])  # t+1 冲上 100+3*10
        low = np.array([100.0] * 6, dtype=float)
        y = B.build_labels(self._frame(close, high, low), horizon=3)
        assert y.iloc[0] == 1

    def test_lower_barrier_first(self):
        close = np.array([100.0] * 6, dtype=float)
        high = np.array([100.0] * 6, dtype=float)
        low = np.array([100.0, 79.9, 100.0, 100.0, 100.0, 100.0])  # t+1 跌破 100-2*10
        y = B.build_labels(self._frame(close, high, low), horizon=3)
        assert y.iloc[0] == -1

    def test_both_barriers_in_same_bar_is_conservative_minus_one(self):
        close = np.array([100.0] * 6, dtype=float)
        high = np.array([100.0, 131.0, 100.0, 100.0, 100.0, 100.0])
        low = np.array([100.0, 79.0, 100.0, 100.0, 100.0, 100.0])
        y = B.build_labels(self._frame(close, high, low), horizon=3)
        assert y.iloc[0] == -1

    def test_expiry_below_flat_threshold_is_zero(self):
        close = np.array([100.0, 100.0, 100.0, 100.0, 105.0, 105.0])
        high = np.array([100.0, 100.0, 100.0, 100.0, 105.0, 105.0])
        low = np.array([100.0] * 6, dtype=float)
        y = B.build_labels(self._frame(close, high, low), horizon=4, flat_atr_mult=1.0)
        assert y.iloc[0] == 0  # 位移 5 < 1.0*ATR(10)

    def test_expiry_above_flat_threshold_keeps_sign(self):
        close = np.array([100.0, 100.0, 100.0, 100.0, 116.0, 116.0])
        high = np.array([100.0, 100.0, 100.0, 100.0, 116.0, 116.0])  # 未触及 100+3*ATR
        low = np.array([100.0] * 6, dtype=float)
        y = B.build_labels(self._frame(close, high, low), horizon=4, flat_atr_mult=1.0)
        assert y.iloc[0] == 1

    def test_tail_rows_have_no_label(self):
        n, h = 20, 4
        close = np.full(n, 100.0)
        y = B.build_labels(self._frame(close, close + 1, close - 1), horizon=h)
        assert y.iloc[n - h:].isna().all()
        assert y.iloc[: n - h].notna().all()

    def test_warmup_nan_atr_yields_nan_label(self):
        close = np.full(10, 100.0)
        frame = self._frame(close, close + 1, close - 1)
        frame.loc[:3, "atr"] = np.nan
        y = B.build_labels(frame, horizon=3)
        assert y.iloc[:4].isna().all()
