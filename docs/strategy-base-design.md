# 策略基类与注册表机制 · 技术设计

## 背景

当前问题：
- `rule_v1.py` 和 `rule_v2.py` 都是独立类，重复定义 `name`、`required_intervals`、`__init__`、`on_bar`
- CLI 里硬编码 `if strategy == "rule_v1"` 分发，新增策略需改 CLI
- 未来接入 ML/深度学习策略时，规则策略和 ML 策略的接口需求差异大：
  - 规则策略：单 bar 输入，输出 `Signal | None`
  - ML 策略：需要滚动窗口特征，需要加载模型，需要 warmup/finalize

---

## 设计目标

1. **统一接口**：所有策略继承同一个 `BaseStrategy`，强制实现核心方法
2. **自动注册**：用 `@register` 装饰器，CLI 自动发现可用策略
3. **可扩展性**：支持规则策略、ML 策略、深度学习策略等异构实现
4. **生命周期**：提供 `warmup/reset/finalize` 钩子，让 ML 策略能加载模型/清理状态

---

## 核心接口（`base.py`）

```python
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """所有策略的抽象基类。

    子类必须设置类属性：
      - name: 唯一策略名（用于注册和 CLI）
      - required_intervals: 需要的时间周期元组

    子类必须实现：
      - on_bar(bar, row, htf, dt) -> Signal | None
    """

    name: str = ""
    required_intervals: tuple[str, ...] = ("5m",)

    def __init__(self, config=None):
        self.cfg = config

    @abstractmethod
    def on_bar(
        self,
        bar: Bar,
        row: pd.Series,
        htf: dict[str, pd.Series],
        dt: datetime,
    ) -> Signal | None:
        """每根 bar 收盘时调用，返回信号或 None。"""
        ...

    # ---- 可选生命周期钩子 ----

    def warmup(self) -> int:
        """返回需要的预热 bar 数（默认 0，子类可重写）。"""
        return 0

    def reset(self) -> None:
        """每个回测开始前调用（ML 策略清空 buffer）。"""
        pass

    def finalize(self) -> None:
        """回测结束后调用（保存统计、释放资源）。"""
        pass

    # ---- 配置工具 ----

    @property
    def allow_short(self) -> bool:
        """默认从 cfg.allow_short 读取，子类可重写。"""
        return getattr(self.cfg, "allow_short", True)
```

---

## 注册表（`registry.py`）

```python
_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """类装饰器：自动注册到策略表。"""
    if not cls.name:
        raise ValueError(f"{cls.__name__}.name is required")
    if cls.name in _REGISTRY:
        raise ValueError(f"strategy name '{cls.name}' already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> type[BaseStrategy]:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"未知策略: {name}（可用: {available}）")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    return sorted(_REGISTRY.keys())
```

---

## 规则策略迁移（`rule_v1.py` / `rule_v2.py`）

只需改基类和加 `@register`：

```python
from .base import BaseStrategy
from .registry import register


@register
class RuleV1Strategy(BaseStrategy):
    name = "rule_v1"
    required_intervals = ("5m", "1h", "4h")

    def __init__(self, config: RuleV1Config | None = None):
        super().__init__(config or RuleV1Config())

    def on_bar(self, bar, row, htf, dt):
        # 原有逻辑保持不变
        ...
```

---

## 深度学习策略适配（`ml/base_ml.py`）

为未来 ML/深度学习策略提供专用基类，封装共性需求：

```python
class MLBaseStrategy(BaseStrategy, ABC):
    """ML 策略基类：模型加载、特征 buffer、窗口预测。

    子类必须实现：
      - _load_model(path) -> Any        加载模型（PyTorch / sklearn / ONNX）
      - _extract_features(bar, row, htf) -> np.ndarray  单 bar 特征向量
      - _signal_from_prob(prob, bar, row, htf, dt) -> Signal | None
    """

    model_path: str = ""
    prob_threshold: float = 0.6

    def __init__(self, config=None):
        super().__init__(config)
        self.model = self._load_model(config.model_path)
        self._buf: list[np.ndarray] = []
        self._window_size = self._get_window_size()

    @abstractmethod
    def _load_model(self, path) -> Any: ...

    @abstractmethod
    def _extract_features(self, bar, row, htf) -> np.ndarray: ...

    @abstractmethod
    def _get_window_size(self) -> int: ...

    @abstractmethod
    def _signal_from_prob(self, prob, bar, row, htf, dt) -> Signal | None: ...

    def warmup(self) -> int:
        return self._window_size

    def reset(self) -> None:
        self._buf.clear()

    def on_bar(self, bar, row, htf, dt):
        feat = self._extract_features(bar, row, htf)
        self._buf.append(feat)
        if len(self._buf) > self._window_size:
            self._buf.pop(0)
        if len(self._buf) < self._window_size:
            return None

        window = np.stack(self._buf)  # (window, features)
        prob = self.model.predict(window)  # 子类定义 predict 接口
        return self._signal_from_prob(prob, bar, row, htf, dt)
```

**示例（仅示意，不实现实际训练）**：

```python
@register
class LSTMStrategy(MLBaseStrategy):
    name = "lstm_v1"
    required_intervals = ("5m", "1h")

    def __init__(self, config):
        super().__init__(config)
        self.cfg: LSTMStrategyConfig

    def _load_model(self, path):
        return torch.jit.load(path)

    def _extract_features(self, bar, row, htf):
        return np.array([
            row["close"], row["atr"], row["macd_hist"],
            row["rsi_14"], row["srsi_k"], row["vol_ratio"],
            row["bb_upper"], row["bb_lower"],
            htf["1h"]["srsi_k"], htf["1h"]["macd_hist"],
        ])

    def _get_window_size(self):
        return 60  # 60 根 5m bar = 5 小时

    def _signal_from_prob(self, prob, bar, row, htf, dt):
        long_prob, short_prob = prob[0], prob[1]
        if long_prob > self.prob_threshold:
            return Signal(action=Action.OPEN_LONG, ...)
        if short_prob > self.prob_threshold:
            return Signal(action=Action.OPEN_SHORT, ...)
        return None
```

---

## CLI 改造（`cli.py`）

替换硬编码分发：

```python
# 旧
if strategy == "rule_v1":
    from .strategy.rules.rule_v1 import RuleV1Strategy
    rule_strategy = RuleV1Strategy()
elif strategy == "rule_v2":
    from .strategy.rules.rule_v2 import RuleV2Strategy
    rule_strategy = RuleV2Strategy()
else:
    raise typer.Exit(f"未知策略: {strategy}")

# 新
from .strategy import get_strategy
try:
    StrategyCls = get_strategy(strategy)
except ValueError as e:
    raise typer.Exit(str(e))
rule_strategy = StrategyCls()
```

引擎调用 `warmup()` 取预热 bar 数（替代 hardcoded `max(...)`）：

```python
warmup = max(
    engine.bar_strategy.warmup(),
    interval_ms("4h") // interval_ms("5m") * 65,
    ...
)
```

---

## 文件结构

```
src/quant/strategy/
├── __init__.py            # 公开 API + 自动 import rules
├── base.py                # BaseStrategy（新增）
├── registry.py            # register / get_strategy / list_strategies（新增）
├── rules/
│   ├── __init__.py
│   ├── rule_v1.py         # 改为继承 BaseStrategy + @register
│   └── rule_v2.py         # 改为继承 BaseStrategy + @register
└── ml/                    # 未来扩展（暂不实现）
    ├── __init__.py
    └── base_ml.py         # MLBaseStrategy（仅占位）
```

---

## 兼容性

- `RuleV1Strategy` / `RuleV2Strategy` 的对外接口（`name`, `required_intervals`, `on_bar`）完全不变
- 引擎不需要改逻辑，只需要把 `warmup` 从子类读取
- CLI 参数（`--strategy`, `--srsi-oversold` 等）保持不变

---

## 实施步骤

1. 新增 `base.py`、`registry.py`
2. 改造 `rule_v1.py`、`rule_v2.py` 继承 BaseStrategy + @register
3. 改造 `cli.py` 用 `get_strategy` 替换 if/elif
4. 引擎读取 `strategy.warmup()` 替代硬编码（可选）
5. 跑 rule_v1 + rule_v2 回归测试，确认回测结果不变
6. （可选）新建 `ml/base_ml.py` 占位，展示未来扩展路径

---

## 决策点（待确认）

1. **`warmup()` 从策略类读 vs 引擎硬编码**？建议从策略类读，更灵活
2. **`MLBaseStrategy` 是否现在实现**？还是只占位？建议只放占位 + 示例代码，不实际跑 ML
3. **`reset()` 是否每次回测调用**？ML 策略需要重置 buffer，规则策略不需要
