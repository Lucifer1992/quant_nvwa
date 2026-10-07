"""模拟交易（paper）信号服务：定时拉数据 → 多策略并行推理 → 纸面持仓 → 飞书卡片。

模块分工：
  * `config.py`  —— `PaperConfig`（与 `BacktestConfig` 字段一一对齐的执行参数）、
    `GuardThresholds`/`PaperRuntime`（闸门阈值与调度参数）。
  * `guards.py`  —— `[paper]` 事前闸门：只拦新增敞口，出场路径与回测同源。
  * `state.py`   —— 纸面账户状态 ↔ `paper_*` 表的映射与原子落库。
  * `runner.py`  —— 单个 (symbol, strategy) 账户的一轮：加载 → observe 追平 → live 撮合 → 推卡。
  * `service.py` —— 多账户调度循环与增量回填。

真实下单不在范围内：卡片只播报信号，用户手动执行。
"""
from .config import GuardThresholds, PaperConfig, PaperRuntime, execution_params
from .guards import EntryGuards, GateReport
from .runner import PaperRunner, RoundReport
from .service import PaperService
from .state import PaperState

__all__ = [
    "EntryGuards",
    "GateReport",
    "GuardThresholds",
    "PaperConfig",
    "PaperRunner",
    "PaperRuntime",
    "PaperService",
    "PaperState",
    "RoundReport",
    "execution_params",
]
