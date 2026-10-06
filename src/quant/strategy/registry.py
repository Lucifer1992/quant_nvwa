"""策略注册表：自动发现可用策略。

使用：
  - @register 装饰策略类，自动加入注册表
  - get_strategy(name) 按名字获取策略类
  - list_strategies() 列出所有可用策略
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseStrategy


_REGISTRY: dict[str, type["BaseStrategy"]] = {}


def register(cls: type["BaseStrategy"]) -> type["BaseStrategy"]:
    """类装饰器：自动注册策略类到注册表。

    用法：
        @register
        class MyStrategy(BaseStrategy):
            name = "my_strategy"
            ...
    """
    if not cls.name:
        raise ValueError(f"{cls.__name__}.name 必须设置非空字符串")
    if cls.name in _REGISTRY:
        raise ValueError(f"策略名 '{cls.name}' 已被注册（重复）")
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> type["BaseStrategy"]:
    """按名字获取策略类，未找到则报错列出可用策略。"""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"未知策略: {name}（可用: {available}）")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    """列出所有已注册策略名（按字母排序）。"""
    return sorted(_REGISTRY.keys())
