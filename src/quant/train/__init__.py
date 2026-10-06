"""模型训练入口（walk-forward）。"""
from .walkforward import (  # noqa: F401
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    FoldSpec,
    TrainConfig,
    build_dataset,
    edge_from_proba,
    encode_labels,
    plan_folds,
    train_folds,
    write_fold_report,
)
