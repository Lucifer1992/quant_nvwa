"""ML 策略基类：把「已收盘 bar 的特征帧 + walk-forward 折模型」接入事件驱动回测。

三条硬约束：
  1. 训练与推理共用 `features.builder.build_feature_frame`，从根上消除 train/serve skew；
     无前视由 tests/test_features_builder.py 的「截断重算逐位相同」测试证明。
  2. 折模型只在自己的 test 窗口 [test_start, test_end) 内生效，窗口外一律不出信号。
     walk-forward 的样本外性质靠这条保证，而不是靠事后统计。
  3. `feature_spec_hash` 不匹配直接报错，禁止特征集静默漂移。

子类只需实现 `_decide()`：拿到 edge 与折元数据，决定是否开仓、方向与仓位倍数。
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...core.contracts import Bar, Signal
from ...core.timeframe import to_dt, to_ms
from ...features.builder import build_feature_frame, feature_columns, feature_spec_hash
from ..base import BaseStrategy

NAN = float("nan")
_H_MS = 3_600_000


def _iso_to_ms(s: str) -> int:
    return to_ms(datetime.fromisoformat(s))


@dataclass
class FoldModel:
    """一折 = 一个 model.txt + 它的样本外生效窗口。"""

    tag: str
    path: Path
    test_start_ms: int
    test_end_ms: int
    features: list[str]
    feature_spec_hash: str
    label_params: dict[str, Any]
    valid_metrics: dict[str, Any] = field(default_factory=dict)
    _booster: Any = None

    @property
    def horizon(self) -> int:
        return int(self.label_params.get("horizon", 24))

    def booster(self):
        if self._booster is None:
            import lightgbm as lgb

            self._booster = lgb.Booster(model_file=str(self.path))
        return self._booster


def load_folds(model_dir: Path, symbol: str) -> list[FoldModel]:
    """扫描 {model_dir}/{symbol}/*/train_meta.json，按 test_start 升序返回。"""
    root = Path(model_dir) / symbol
    if not root.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{root}（先执行 quant train）")

    folds: list[FoldModel] = []
    for sub in sorted(root.iterdir()):
        meta_p, model_p = sub / "train_meta.json", sub / "model.txt"
        if not (meta_p.is_file() and model_p.is_file()):
            continue
        meta = json.loads(meta_p.read_text())
        folds.append(FoldModel(
            tag=str(meta["tag"]),
            path=model_p,
            test_start_ms=_iso_to_ms(meta["test_start"]),
            test_end_ms=_iso_to_ms(meta["test_end"]),
            features=list(meta["features"]),
            feature_spec_hash=str(meta["feature_spec_hash"]),
            label_params=dict(meta.get("label_params", {})),
            valid_metrics=dict(meta.get("valid_metrics", {})),
        ))
    if not folds:
        raise FileNotFoundError(f"{root} 下没有任何完整折（需 model.txt + train_meta.json）")
    folds.sort(key=lambda f: f.test_start_ms)
    return folds


class MLBaseStrategy(BaseStrategy):
    """ML 策略基类：持有 1h 特征帧，按折模型输出概率，子类负责下注规则。

    引擎需要在 `run()` 里调用 `reset()` + `bind_history(bars, funding)`（见 E7/E8），
    否则没有历史可算特征，`on_bar` 会直接返回 None。
    """

    name = ""
    required_intervals = ("5m", "1h", "4h")
    decision_interval = "1h"

    def __init__(self, config: Any = None):
        super().__init__(config)
        self._bars: dict[str, pd.DataFrame] = {}
        self._funding: pd.DataFrame | None = None
        self._frame: pd.DataFrame | None = None
        self._x: np.ndarray | None = None
        self._row_of: dict[int, int] = {}
        self._folds: list[FoldModel] | None = None
        self._warned: set[str] = set()
        self._last_ts: int | None = None
        # 已见过的 5m bar 根数：冷却等以基础周期计的节律统一挂在引擎节奏上
        self.n_bars = 0

    # ---- 历史注入（E8）----

    def bind_history(
        self,
        bars: dict[str, pd.DataFrame],
        funding: pd.DataFrame | None = None,
    ) -> None:
        self._bars = dict(bars)
        self._funding = funding
        self._frame = None
        self._x = None
        self._row_of = {}

    def reset(self) -> None:
        self._last_ts = None
        self._warned = set()
        self.n_bars = 0

    def history_lookback(self) -> int:
        """需要的 pre-start 历史长度（ms）。

        EWM 类特征（ema/rsi/macd/adx）的理论值依赖整段前缀，只给最低滚动窗口会让
        推理端的种子比训练端短，产生 train/serve skew。实测 span=60 时 200 根的
        种子残留约 1e-3，故取 90 天把误差压到可忽略。
        """
        return getattr(self.cfg, "history_days", 90) * 24 * _H_MS

    def warmup(self) -> int:
        return 0

    # ---- 模型 ----

    def model_dir(self) -> Path:
        return Path(getattr(self.cfg, "model_dir"))

    def folds(self) -> list[FoldModel]:
        if self._folds is None:
            self._folds = load_folds(self.model_dir(), self.symbol())
        return self._folds

    def symbol(self) -> str:
        return getattr(self.cfg, "symbol", "") or ""

    def _fold_for(self, ts_open_ms: int) -> FoldModel | None:
        for f in self.folds():
            if f.test_start_ms <= ts_open_ms < f.test_end_ms:
                return f
        return None

    # ---- 特征帧 ----

    def _ensure_frame(self) -> bool:
        if self._frame is not None:
            return True
        bars_1h = self._bars.get("1h")
        if bars_1h is None or bars_1h.empty:
            return False
        frame = build_feature_frame(
            bars_1h,
            bars_4h=self._bars.get("4h"),
            funding=self._funding,
        )
        frame = frame.sort_values("ts_open_ms").reset_index(drop=True)
        cols = self._check_features(frame)
        if cols is None:
            return False
        self._frame = frame
        self._x = frame[cols].to_numpy(dtype="float64", copy=False)
        self._row_of = {int(t): i for i, t in enumerate(frame["ts_open_ms"].to_list())}
        return True

    def _check_features(self, frame: pd.DataFrame) -> list[str] | None:
        cols = feature_columns(frame)
        want = feature_spec_hash(cols)
        for f in self.folds():
            if f.feature_spec_hash != want:
                raise ValueError(
                    f"折 {f.tag} 的特征集与当前代码不一致："
                    f"模型 {f.feature_spec_hash} vs 推理 {want}；"
                    "请重新 quant train 或切回对应代码版本"
                )
        return cols

    def edge_at(self, ts_open_ms: int) -> tuple[float, FoldModel, pd.Series] | None:
        """返回该 1h bar 收盘时刻可见的 (edge, fold, 特征行)；无模型/无数据 → None。"""
        if not self._ensure_frame():
            return None
        fold = self._fold_for(ts_open_ms)
        if fold is None:
            self._warn_missing(ts_open_ms)
            return None
        i = self._row_of.get(ts_open_ms)
        if i is None:
            return None
        row = self._frame.iloc[i]
        if math.isnan(float(row["atr"])) or float(row["atr"]) <= 0:
            return None
        proba = fold.booster().predict(self._x[[i]])[0]
        up, dn = proba[2], proba[0]   # CLASS_UP=2 / CLASS_DOWN=0，与 walkforward.LABEL_TO_CLASS 一致
        return float(up - dn), fold, row

    def _warn_missing(self, ts_open_ms: int) -> None:
        """折窗口外的 bar 不出信号 —— 静默会让人误以为策略在跑。"""
        if ts_open_ms < self._first_test_start():
            key = "before"
            msg = "早于首折样本外起点，已跳过（回测 start 需 >= 训练 eval_start）"
        else:
            key = "after"
            msg = "超出最后一折窗口，已跳过（需 quant train 续训）"
        if key not in self._warned:
            self._warned.add(key)
            logging.getLogger(__name__).warning(
                "%s %s 无可用折模型", to_dt(ts_open_ms).strftime("%Y-%m-%d %H:%M"), msg
            )

    def _first_test_start(self) -> int:
        try:
            return self.folds()[0].test_start_ms
        except Exception:
            return 0

    # ---- 子类入口 ----

    def on_bar(
        self,
        bar: Bar,
        row: pd.Series,
        htf: dict[str, pd.Series],
        dt: datetime,
    ) -> Signal | None:
        r1 = htf.get(self.decision_interval)
        if r1 is None:
            return None
        self.n_bars += 1          # 引擎按基础周期（5m）逐根驱动，冷却计数以此为准
        ts_open_ms = int(r1["ts_open_ms"])
        if ts_open_ms == self._last_ts:
            return None          # 本根 1h 已决策过
        self._last_ts = ts_open_ms
        return self._decide(bar, row, r1, ts_open_ms, dt)

    def _decide(
        self,
        bar: Bar,
        row: pd.Series,
        r1: pd.Series,
        ts_open_ms: int,
        dt: datetime,
    ) -> Signal | None:
        raise NotImplementedError
