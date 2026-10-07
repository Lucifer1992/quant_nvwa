"""paper 的执行配置与运行参数。

`PaperConfig` 的字段名与默认值必须和 `BacktestConfig` 完全一致（`start`/`end` 除外：
纸面账户没有回测窗口，只有断点）。`backtest/position.py` 的撮合层按字段名取参数，
两份配置只要漂移一个名字，paper 就会静默用错止损倍数 —— 所以这条约束由
`tests/test_paper_runner.py` 的字段对齐用例守住，不靠人肉 review。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

from ..infra.log import get_logger
from ..infra.settings import Settings

_log = get_logger("quant.paper.config")


@dataclass
class PaperConfig:
    """执行层参数：与 `BacktestConfig` 同名同默认值，paper 与回测因此共用同一套撮合口径。"""

    symbol: str
    base_interval: str
    initial_equity: float
    risk_pct: float = 0.01
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    time_stop_bars: int = 96
    allow_short: bool = True
    global_max_leverage: float = 10.0
    taker_fee: float = 0.0005
    slippage_bps: float = 2.0
    include_funding: bool = False
    sl_first_on_conflict: bool = True
    strategy_name: str = "rule_v1"
    pyramid_enabled: bool = True
    entry_frac: float = 0.10
    add_frac: float = 0.20
    add_max_count: int = 2
    add_trigger_r: float = 0.3
    trail_trigger_r: float = 1.0
    trail_atr_mult: float = 1.0
    swing_lookback: int = 12
    tp1_r: float = 1.0
    tp1_close_frac: float = 0.5
    breakeven_after_r: float = 1.0


@dataclass(frozen=True)
class GuardThresholds:
    """`risk.yaml` 里标注 `[paper]` 的闸门阈值。

    这些值**拦截新开仓**（`paper/guards.py`）：日亏/回撤/连亏冷却期间不再试仓，仓位与
    移动止损照常由执行层管理。已平仓的持仓不会被闸门强平 —— 出场口径必须和回测一致，
    否则纸面数字失去参考价值。
    """

    daily_loss_stop_pct: float = 0.03
    max_drawdown_pause_pct: float = 0.15
    max_consecutive_losses: int = 5
    cooldown_hours: float = 24.0
    stale_data_max_gap_min: float = 5.0
    max_new_entries_per_day: int = 4
    max_margin_per_symbol_pct: float = 0.30
    max_notional_usdt: float = 5000.0


@dataclass(frozen=True)
class PaperRuntime:
    """调度与告警参数，与撮合无关。"""

    replay_days: float = 7.0
    poll_sec: float = 20.0
    resync_grace_ms: int = 2000
    resend_cooldown_min: float = 10.0
    guard: GuardThresholds = field(default_factory=GuardThresholds)

    @classmethod
    def from_settings(cls, s: Settings) -> PaperRuntime:
        paper = s.settings.get("paper", {})
        caps = s.risk.get("caps", {})
        guard = s.risk.get("guard", {})
        return cls(
            replay_days=float(paper.get("replay_days", 7.0)),
            poll_sec=float(paper.get("poll_sec", 20.0)),
            resync_grace_ms=int(s.settings.get("network", {}).get("resync_grace_ms", 2000)),
            resend_cooldown_min=float(
                s.notify.get("feishu", {}).get("resend_cooldown_min", 10.0)
            ),
            guard=GuardThresholds(
                daily_loss_stop_pct=float(guard.get("daily_loss_stop_pct", 0.03)),
                max_drawdown_pause_pct=float(guard.get("max_drawdown_pause_pct", 0.15)),
                max_consecutive_losses=int(guard.get("max_consecutive_losses", 5)),
                cooldown_hours=float(guard.get("cooldown_hours", 24.0)),
                stale_data_max_gap_min=float(guard.get("stale_data_max_gap_min", 5.0)),
                max_new_entries_per_day=int(caps.get("max_new_entries_per_day", 4)),
                max_margin_per_symbol_pct=float(caps.get("max_margin_per_symbol_pct", 0.30)),
                max_notional_usdt=float(caps.get("max_notional_usdt", 5000.0)),
            ),
        )

    @property
    def replay_ms(self) -> int:
        return int(self.replay_days * 86_400_000)


def execution_params(
    s: Settings,
    symbol: str,
    base_interval: str,
    strategy_name: str,
    engine_defaults: dict[str, float],
) -> PaperConfig:
    """合成执行层参数，口径与「不带任何开关的 `quant backtest`」完全一致。

    基准取 `PaperConfig` 的字段默认值（= `BacktestConfig` 的字段默认值 = CLI 选项的兜底值），
    再叠加策略自己声明的 `engine_defaults()`：回测 CLI 的判断顺序是
    「用户显式传参 > 策略声明 > 写死的兜底值」，paper 没有人工传参这一环，所以少一层。
    """
    params: dict[str, object] = {
        f.name: f.default for f in fields(PaperConfig) if f.default is not None
    }
    params.update({k: v for k, v in engine_defaults.items() if k in params})
    _warn_yaml_drift(s, strategy_name, params, set(engine_defaults))

    kw = dict(params)
    kw.update({
        "symbol": symbol,
        "base_interval": base_interval,
        "strategy_name": strategy_name,
        "initial_equity": float(s.risk.get("account", {}).get("initial_equity_usdt", 100.0)),
    })
    return PaperConfig(**kw)


def _warn_yaml_drift(
    s: Settings, strategy_name: str, effective: dict[str, object], overridden: set[str]
) -> None:
    """`risk.yaml` 声称的口径与实际生效值不一致时喊出来（被策略覆盖掉的那些不算漂移）。

    回测 CLI 只读 `--sl-atr/--tp-atr/--time-stop-bars` 这些开关，从不读 `costs.*`，
    缺省值就是 `BacktestConfig` 的字段默认值；`risk.yaml` 一旦改成别的数字，两边会静默
    分叉，paper 就不再能和回测对比 —— 而可比性是这套纸面账户唯一的存在意义。
    """
    stops, costs = s.risk.get("stops", {}), s.risk.get("costs", {})
    caps, sizing = s.risk.get("caps", {}), s.risk.get("sizing", {})
    declared = {
        "risk_pct": sizing.get("risk_pct"),
        "sl_atr_mult": stops.get("sl_atr_mult"),
        "tp_atr_mult": stops.get("tp_atr_mult"),
        "time_stop_bars": stops.get("time_stop_bars"),
        "allow_short": stops.get("allow_short"),
        "tp1_r": stops.get("tp1_r"),
        "tp1_close_frac": stops.get("tp1_close_frac"),
        "breakeven_after_r": stops.get("breakeven_after_r"),
        "global_max_leverage": caps.get("global_max_leverage"),
        "taker_fee": costs.get("taker_fee_rate"),
        "slippage_bps": costs.get("slippage_bps"),
        "include_funding": costs.get("include_funding"),
        "sl_first_on_conflict": s.backtest.get("sl_first_on_conflict"),
    }
    drift: list[str] = []
    for key, want in declared.items():
        if want is None or key in overridden:
            continue
        got = effective[key]
        same = bool(got) == bool(want) if isinstance(want, bool) else float(got) == float(want)
        if not same:
            drift.append(f"{key}: 生效={got} vs risk.yaml={want}")
    if drift:
        _log.warning(
            "%s 的执行参数与 risk.yaml 不一致（回测按字段默认值走，两边结果不可比）：%s",
            strategy_name, "；".join(drift),
        )
