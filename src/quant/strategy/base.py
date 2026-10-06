"""策略抽象基类：所有策略的接口契约。

子类必须设置：
  - name: str                       唯一策略名（用于注册表）
  - required_intervals: tuple[str, ...]  需要的时间周期元组

子类必须实现：
  - on_bar(bar, row, htf, dt) -> Signal | None
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from ..core.contracts import Bar, Signal


class BaseStrategy(ABC):
    """策略抽象基类。

    使用示例：
        @register
        class MyStrategy(BaseStrategy):
            name = "my_strategy"
            required_intervals = ("5m", "1h")

            def __init__(self, config=None):
                super().__init__(config)

            def on_bar(self, bar, row, htf, dt):
                if htf.get("1h") is None:
                    return None
                ...
    """

    name: str = ""
    required_intervals: tuple[str, ...] = ("5m",)
    config_cls: type | None = None

    @classmethod
    def build(cls, opts: dict[str, object]) -> "BaseStrategy":
        """按 CLI 传下来的统一选项构造实例。

        CLI 只负责收集选项、不解释选项归属，否则每加一个策略都要在 CLI 里补
        if/elif。子类只要声明 `config_cls`，属于它的字段会自动挑出来。
        """
        if cls.config_cls is None:
            return cls()
        from dataclasses import fields

        kw = {
            f.name: opts[f.name]
            for f in fields(cls.config_cls)
            if f.name in opts and opts[f.name] is not None
        }
        return cls(cls.config_cls(**kw))

    def __init__(self, config: Any = None):
        self.cfg = config

    @abstractmethod
    def on_bar(
        self,
        bar: Bar,
        row: pd.Series,
        htf: dict[str, pd.Series],
        dt: datetime,
    ) -> Signal | None:
        """每根 5m bar 收盘时调用。

        参数：
          bar: 当前 5m bar
          row: 该 bar 的预计算指标行（含 macd_hist, ema_20, atr, swing_high 等）
          htf: 高周期最近一根已收盘 bar 的 Series 字典 {interval: Series}
          dt: 信号时间（bar.ts_close）
        """
        ...

    # ---- 可选生命周期钩子 ----

    def engine_defaults(self) -> dict[str, float]:
        """策略所需的回测引擎出场参数默认值。

        决策周期与引擎基础周期（5m）不一致时才有意义：例如 1h 趋势策略
        必须用 1h ATR 量级的止损和以天计的时间止损，否则会退化。
        返回 BacktestConfig 字段的子集，CLI 只在用户没有显式传参时采用。
        """
        return {}

    def warmup(self) -> int:
        """返回需要的预热 bar 数。"""
        return 0

    def history_lookback(self) -> int:
        """回测窗口之前还需要多少毫秒历史数据（ML 特征预热用）。

        规则策略从窗口起点就能算全部指标，返回 0；任何非 0 的值都会让 CLI 向前
        多取数据，从而改变高周期指标的预热长度 —— 所以默认必须为 0。
        """
        return 0

    def reset(self) -> None:
        """每个回测 session 开始前调用。ML 策略可清空 buffer。"""
        pass

    def finalize(self) -> None:
        """回测结束后调用（清理资源、保存指标）。"""
        pass

    @property
    def allow_short(self) -> bool:
        """默认从 cfg.allow_short 读取。"""
        return getattr(self.cfg, "allow_short", True)
