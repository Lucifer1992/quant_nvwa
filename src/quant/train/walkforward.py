"""walk-forward 训练：滚动窗口 + 时间外推验证 + embargo，禁止随机 CV。

设计口径见 docs/lgbm-strategy-and-paper-trading-design.md §5。
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features import builder as B

# 标签编码：LightGBM 要求 0..num_class-1
LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {v: k for k, v in LABEL_TO_CLASS.items()}
CLASS_DOWN, CLASS_FLAT, CLASS_UP = LABEL_TO_CLASS[-1], LABEL_TO_CLASS[0], LABEL_TO_CLASS[1]


def encode_labels(y: pd.Series) -> np.ndarray:
    return y.map(LABEL_TO_CLASS).to_numpy(dtype=int)


def edge_from_proba(proba: np.ndarray) -> np.ndarray:
    """多头倾向：P(+1) - P(-1)，落在 [-1, 1]。"""
    return proba[:, CLASS_UP] - proba[:, CLASS_DOWN]


def _month_add(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return (ts + pd.DateOffset(months=n)).normalize()


@dataclass
class FoldSpec:
    tag: str
    train_start: datetime
    train_end: datetime
    valid_start: datetime
    valid_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class TrainConfig:
    symbol: str = "ETHUSDT"
    data_start: str = "2024-01-01"
    eval_start: str = "2026-04-01"
    eval_end: str = "2026-10-01"
    window_months: int = 12
    step_months: int = 1
    valid_months: int = 1
    horizon: int = 24
    embargo_bars: int = 26  # > horizon，切掉标签右边界溢出到验证/测试窗
    k_up: float = 3.0
    k_dn: float = 2.0
    flat_atr_mult: float = 1.0
    balance_classes: bool = True
    num_rounds: int = 2000
    early_stopping_rounds: int = 100
    out_dir: str = "models/lgbm_v1"
    params: dict = field(default_factory=lambda: {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbose": -1,
        "seed": 7,
        "num_threads": 4,
    })

    def label_kwargs(self) -> dict:
        return {"k_up": self.k_up, "k_dn": self.k_dn,
                "horizon": self.horizon, "flat_atr_mult": self.flat_atr_mult}


def plan_folds(cfg: TrainConfig) -> list[FoldSpec]:
    """每 step_months 一折；valid 取训练窗最后 valid_months 个月，与 test 之间再留 embargo。"""
    data_start = pd.Timestamp(cfg.data_start, tz="UTC")
    t = pd.Timestamp(cfg.eval_start, tz="UTC")
    eval_end = pd.Timestamp(cfg.eval_end, tz="UTC")
    folds: list[FoldSpec] = []
    while t < eval_end:
        test_end = min(_month_add(t, cfg.step_months), eval_end)
        valid_start = _month_add(t, -(cfg.valid_months))
        train_start = _month_add(t, -(cfg.window_months))
        if train_start < data_start:
            raise ValueError(
                f"数据起点 {cfg.data_start} 撑不起 {cfg.window_months} 个月训练窗"
                f"（折 {t:%Y%m} 需要 {train_start:%Y-%m-%d} 起）"
            )
        folds.append(FoldSpec(
            tag=t.strftime("%Y%m"),
            train_start=train_start.to_pydatetime(),
            train_end=valid_start.to_pydatetime(),
            valid_start=valid_start.to_pydatetime(),
            valid_end=t.to_pydatetime(),
            test_start=t.to_pydatetime(),
            test_end=test_end.to_pydatetime(),
        ))
        t = test_end
    return folds


def build_dataset(store, cfg: TrainConfig) -> tuple[pd.DataFrame, list[str], str]:
    """一次算好全区间特征帧，折切片与推理共用同一份口径。"""
    from ..data.service import MarketDataService

    svc = MarketDataService(store)
    start = datetime.fromisoformat(cfg.data_start)
    end = datetime.fromisoformat(cfg.eval_end) + timedelta(days=2)
    b1 = svc.bars(cfg.symbol, "1h", start, end)
    b4 = svc.bars(cfg.symbol, "4h", start, end)
    fu = svc.funding(cfg.symbol, start, end)
    if b1.empty:
        raise SystemExit(f"{cfg.symbol} 1h 无数据")
    frame = B.build_feature_frame(b1, None if b4.empty else b4, None if fu.empty else fu)
    frame["label"] = B.build_labels(frame, **cfg.label_kwargs())
    version = store.dataset_version(cfg.symbol, ["1h", "4h"])
    return frame, B.feature_columns(frame), version


def _class_weights(y: np.ndarray, enabled: bool) -> np.ndarray | None:
    if not enabled:
        return None
    counts = np.bincount(y, minlength=3).astype(float)
    freq = counts / counts.sum()
    inv = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-9), 0.0)
    w = inv[y]
    return w / w.mean()


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    eps = 1e-12
    logloss = float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], eps, 1.0))))
    per_class = {}
    for c in (CLASS_DOWN, CLASS_FLAT, CLASS_UP):
        mask = y == c
        per_class[str(int(c))] = {
            "label": {v: k for k, v in LABEL_TO_CLASS.items()}[int(c)],
            "support": int(mask.sum()),
            "recall": round(float((pred[mask] == c).mean()), 4) if mask.any() else None,
        }
    return {
        "logloss": round(logloss, 5),
        "accuracy": round(float((pred == y).mean()), 4),
        "class_counts": {str(int(c)): int((y == c).sum()) for c in (0, 1, 2)},
        "per_class": per_class,
    }


def edge_sweep(y: np.ndarray, proba: np.ndarray, taus: list[float]) -> list[dict]:
    """验证集上扫 τ 看信号密度与方向命中率（命中率按标签口径，不含成本）。"""
    edge = edge_from_proba(proba)
    rows = []
    for tau in taus:
        lg, sh = edge > tau, edge < -tau
        n = int(lg.sum() + sh.sum())
        hit = int((y[lg] == CLASS_UP).sum() + (y[sh] == CLASS_DOWN).sum())
        rows.append({
            "tau": tau, "signals": n, "longs": int(lg.sum()), "shorts": int(sh.sum()),
            "hit_rate": round(hit / n, 4) if n else None,
            "flat_rate": round(float((y[lg | sh] == CLASS_FLAT).mean()), 4) if n else None,
        })
    return rows


def fold_masks(ts: pd.Series, f: FoldSpec, embargo: timedelta) -> tuple[pd.Series, pd.Series]:
    """训练/验证布尔掩码。

    两端都要 purge：训练窗最后 horizon 根的标签右边界落在验证窗内，
    留着会让早停看到的验证指标被训练样本"预习"过（valid 端同理，见下一行）。
    """
    train = (ts >= f.train_start) & (ts < f.train_end - embargo)
    valid = (ts >= f.valid_start) & (ts < f.valid_end - embargo)
    return train, valid


def _fit(tr: pd.DataFrame, y_tr: np.ndarray, va: pd.DataFrame, y_va: np.ndarray,
         cols: list[str], cfg: TrainConfig) -> lgb.Booster:
    ds_tr = lgb.Dataset(
        tr[cols].to_numpy(dtype=float), label=y_tr,
        weight=_class_weights(y_tr, cfg.balance_classes),
    )
    ds_va = lgb.Dataset(va[cols].to_numpy(dtype=float), label=y_va, reference=ds_tr)
    return lgb.train(
        dict(cfg.params), ds_tr,
        num_boost_round=cfg.num_rounds,
        valid_sets=[ds_va], valid_names=["valid"],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                   lgb.log_evaluation(0)],
    )


def train_folds(frame: pd.DataFrame, cols: list[str], dataset_version: str,
                cfg: TrainConfig, folds: list[FoldSpec] | None = None,
                taus: list[float] | None = None) -> list[dict]:
    out_root = Path(cfg.out_dir)
    folds = folds or plan_folds(cfg)
    taus = taus or [round(0.05 + 0.05 * i, 2) for i in range(8)]
    ts = pd.to_datetime(frame["ts_open_ms"], unit="ms", utc=True)
    embargo = timedelta(hours=cfg.embargo_bars)
    spec_hash = B.feature_spec_hash(cols)
    summaries: list[dict] = []

    for f in folds:
        tr_m, va_m = fold_masks(ts, f, embargo)
        tr = frame[tr_m].dropna(subset=["label"])
        va = frame[va_m].dropna(subset=["label"])
        if len(tr) < 500 or len(va) < 100:
            raise SystemExit(f"折 {f.tag} 样本不足：train={len(tr)} valid={len(va)}")

        y_tr, y_va = encode_labels(tr["label"]), encode_labels(va["label"])
        booster = _fit(tr, y_tr, va, y_va, cols, cfg)
        proba = np.asarray(booster.predict(va[cols].to_numpy(dtype=float)))

        fold_dir = out_root / cfg.symbol / f.tag
        fold_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(fold_dir / "model.txt"))
        meta = {
            "tag": f.tag, "symbol": cfg.symbol,
            "train_start": f.train_start.isoformat(), "train_end": f.train_end.isoformat(),
            "valid_start": f.valid_start.isoformat(), "valid_end": f.valid_end.isoformat(),
            "test_start": f.test_start.isoformat(), "test_end": f.test_end.isoformat(),
            "n_train": int(len(tr)), "n_valid": int(len(va)),
            "features": cols, "feature_spec_hash": spec_hash,
            "dataset_version": dataset_version,
            "label_params": cfg.label_kwargs(), "balance_classes": cfg.balance_classes,
            "params": cfg.params, "num_trees": int(booster.num_trees()),
            "lightgbm": lgb.__version__, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "valid_metrics": _metrics(y_va, proba),
            "edge_sweep": edge_sweep(y_va, proba, taus),
            "feature_importance_gain": dict(zip(cols, np.round(booster.feature_importance("gain"), 2).tolist(), strict=False)),
        }
        (fold_dir / "train_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries.append(meta)
        vm = meta["valid_metrics"]
        print(f"  折 {f.tag}: train={meta['n_train']} valid={meta['n_valid']} "
              f"trees={meta['num_trees']} logloss={vm['logloss']} acc={vm['accuracy']}")
    return summaries


def write_fold_report(cfg: TrainConfig, summaries: list[dict], path: Path) -> dict:
    """折间稳定性：logloss 的均值/极差直接写进 folds.json，作为过拟合红旗之一。"""
    lls = [s["valid_metrics"]["logloss"] for s in summaries]
    accs = [s["valid_metrics"]["accuracy"] for s in summaries]
    out = {
        "config": asdict(cfg),
        "n_folds": len(summaries),
        "logloss_mean": round(float(np.mean(lls)), 5) if lls else None,
        "logloss_min": min(lls, default=None), "logloss_max": max(lls, default=None),
        "accuracy_mean": round(float(np.mean(accs)), 4) if accs else None,
        "accuracy_min": min(accs, default=None), "accuracy_max": max(accs, default=None),
        "folds": [{"tag": s["tag"], "test_start": s["test_start"], "test_end": s["test_end"],
                   "num_trees": s["num_trees"], "valid_metrics": s["valid_metrics"],
                   "edge_sweep": s["edge_sweep"]} for s in summaries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
