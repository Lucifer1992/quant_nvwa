"""walk-forward 折划分与 purge：折窗左闭右开、连续无缝，训练/验证两端都要留出 embargo。

泄漏口径见 docs/lgbm-strategy-and-paper-trading-design.md §5。
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.train.walkforward import (  # noqa: E402
    LABEL_TO_CLASS,
    TrainConfig,
    edge_from_proba,
    encode_labels,
    fold_masks,
    plan_folds,
)


@pytest.fixture()
def cfg():
    return TrainConfig()


class TestFoldPlanning:
    def test_windows_are_half_open_and_contiguous(self, cfg):
        folds = plan_folds(cfg)
        assert len(folds) >= 3
        assert folds[0].test_start == pd.Timestamp(cfg.eval_start, tz="UTC").to_pydatetime()
        assert folds[-1].test_end == pd.Timestamp(cfg.eval_end, tz="UTC").to_pydatetime()
        for f in folds:
            assert f.train_end == f.valid_start and f.valid_end == f.test_start
            assert f.test_start < f.test_end
            assert f.train_start < f.train_end
        for prev, cur in zip(folds, folds[1:], strict=False):
            assert prev.test_end == cur.test_start, "折之间必须无缝衔接，否则测试窗有盲区"

    def test_last_fold_clamped_to_eval_end(self, cfg):
        cfg.eval_end = "2026-04-15"
        assert plan_folds(cfg)[-1].test_end == pd.Timestamp("2026-04-15", tz="UTC").to_pydatetime()

    def test_raises_when_history_too_short(self, cfg):
        cfg.window_months = 60
        with pytest.raises(ValueError, match="撑不起"):
            plan_folds(cfg)


class TestFoldMasks:
    """purge 是防泄漏的唯一屏障：末根训练样本的标签右边界不得进入验证窗。"""

    def _ts(self, f, cfg):
        return pd.Series(
            pd.date_range(f.train_start, f.test_end + pd.Timedelta(days=3), freq="h", tz="UTC")
        )

    def test_both_ends_purged(self, cfg):
        f = plan_folds(cfg)[0]
        ts = self._ts(f, cfg)
        embargo = timedelta(hours=cfg.embargo_bars)
        tr_m, va_m = fold_masks(ts, f, embargo)
        tr, va = ts[tr_m], ts[va_m]
        step = pd.Timedelta(hours=1)
        assert tr.min() == f.train_start
        assert tr.max() == f.train_end - embargo - step
        assert va.min() == f.valid_start
        assert va.max() == f.valid_end - embargo - step
        assert not (tr_m & va_m).any()

    def test_train_label_horizon_cannot_reach_valid(self, cfg):
        f = plan_folds(cfg)[0]
        ts = self._ts(f, cfg)
        tr_m, _ = fold_masks(ts, f, timedelta(hours=cfg.embargo_bars))
        last_label_end = ts[tr_m].max() + cfg.horizon * pd.Timedelta(hours=1)
        assert last_label_end <= f.valid_start, "embargo 必须 > horizon，否则训练标签见过验证窗价格"
        assert cfg.embargo_bars > cfg.horizon

    def test_without_embargo_labels_would_overlap(self, cfg):
        """反证：embargo=0 时训练窗最后一根的标签落在验证窗内（这就是历史 bug）。"""
        f = plan_folds(cfg)[0]
        ts = self._ts(f, cfg)
        tr_m, _ = fold_masks(ts, f, timedelta(0))
        assert ts[tr_m].max() + cfg.horizon * pd.Timedelta(hours=1) > f.valid_start


class TestEncoding:
    def test_label_class_roundtrip(self):
        y = pd.Series([-1, 0, 1, 1, -1])
        encoded = encode_labels(y)
        assert encoded.tolist() == [LABEL_TO_CLASS[v] for v in y]
        assert encoded.min() == 0 and encoded.max() == 2  # LightGBM 要求 0..num_class-1

    def test_edge_is_long_minus_down(self):
        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.34, 0.33, 0.33]])
        np.testing.assert_allclose(edge_from_proba(proba), [-0.6, 0.6, -0.01])
        assert edge_from_proba(proba).min() >= -1.0 and edge_from_proba(proba).max() <= 1.0
