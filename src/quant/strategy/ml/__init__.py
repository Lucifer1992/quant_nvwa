"""ML 策略子包。

lightgbm 采用懒加载（只在首次 `booster()` 时 import），因此导入本子包不会
强制要求 libomp —— 纯规则策略的环境可以照常运行。
"""
from __future__ import annotations

from .base import FoldModel, MLBaseStrategy, load_folds
from .lgbm_v1 import LGBMv1Config, LGBMv1Strategy

__all__ = [
    "MLBaseStrategy",
    "FoldModel",
    "load_folds",
    "LGBMv1Config",
    "LGBMv1Strategy",
]
