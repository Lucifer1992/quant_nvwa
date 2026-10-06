"""策略模块入口：导出基类、注册表、规则策略。

新策略只需在 rules/ 下新建文件，用 @register 装饰即可被自动发现。
"""
from __future__ import annotations

from .base import BaseStrategy
from .registry import get_strategy, list_strategies, register

# 自动导入已实现的策略，触发 @register 装饰
from .rules import rule_v1, rule_v2, rule_v3, rule_v4  # noqa: F401
# ml/ 内部懒加载 lightgbm，导入本身不要求 libomp
from .ml import lgbm_v1  # noqa: F401

__all__ = [
    "BaseStrategy",
    "register",
    "get_strategy",
    "list_strategies",
]
