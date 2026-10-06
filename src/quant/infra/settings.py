from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


@dataclass(frozen=True)
class Settings:
    """config/ 下各 YAML 的容器。network/proxy 等取自 settings.yaml。"""

    root: Path
    settings: dict[str, Any] = field(default_factory=dict)
    symbols: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    notify: dict[str, Any] = field(default_factory=dict)

    def _get(self, section: str, key: str, default: Any = None) -> Any:
        return self.settings.get(section, {}).get(key, default)

    @property
    def proxy_url(self) -> str | None:
        env = os.getenv("QUANT_PROXY_URL")
        if env is not None:
            return env.strip() or None
        return self._get("network", "proxy_url") or None

    @property
    def rest_base(self) -> str:
        return self._get("network", "rest_base_usdm", "https://fapi.binance.com")

    @property
    def ws_base(self) -> str:
        return self._get("network", "ws_base_usdm", "wss://fstream.binance.com")

    @property
    def timeout_sec(self) -> float:
        return float(self._get("network", "timeout_sec", 20))

    @property
    def weight_budget_per_min(self) -> int:
        return int(self._get("network", "weight_budget_per_min", 1700))

    @property
    def max_retries(self) -> int:
        return int(self._get("network", "max_retries", 5))

    @property
    def log_level(self) -> str:
        return self._get("logging", "level", "INFO")

    @property
    def log_file(self) -> Path:
        return self.root / self._get("logging", "file", "logs/quant.log")

    @property
    def data_dir(self) -> Path:
        return self.root / self._get("paths", "data_dir", "data")

    @property
    def reports_dir(self) -> Path:
        return self.root / self._get("paths", "reports_dir", "reports")

    @property
    def db_path(self) -> Path:
        return self.data_dir / self._get("paths", "db_name", "quant.db")

    @property
    def models_dir(self) -> Path:
        return self.root / self._get("paths", "models_dir", "models")

    @property
    def backtest_default_start(self) -> str:
        return str(self.backtest.get("default_start", "2024-01-01"))

    @property
    def backtest(self) -> dict[str, Any]:
        return self.settings.get("backtest", {})

    def enabled_symbols(self) -> dict[str, Any]:
        return {k: v for k, v in self.symbols.get("symbols", {}).items() if v.get("enabled")}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = project_root()
    load_dotenv(root / ".env", override=False)
    cfg = root / "config"
    return Settings(
        root=root,
        settings=_load_yaml(cfg / "settings.yaml"),
        symbols=_load_yaml(cfg / "symbols.yaml"),
        risk=_load_yaml(cfg / "risk.yaml"),
        notify=_load_yaml(cfg / "notify.yaml"),
    )


def load_strategy_config(name: str) -> dict[str, Any]:
    path = project_root() / "config" / "strategies" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing strategy config: {path}")
    return _load_yaml(path)


def list_strategies() -> list[str]:
    d = project_root() / "config" / "strategies"
    return sorted(p.stem for p in d.glob("*.yaml"))


def validate_config(s: Settings) -> list[str]:
    """跨文件引用完整性校验。返回问题列表；调用方应把它们当错误处理而非警告。"""
    problems: list[str] = []
    risk = s.risk
    caps = risk.get("caps", {})
    stops = risk.get("stops", {})
    sizing = risk.get("sizing", {})
    margin = risk.get("margin")

    if not margin:
        problems.append("risk.yaml 缺少 margin 段（maint_margin_rate/max_leverage_default 是强平与仓位裁剪的必需输入）")
    global_lev = caps.get("global_max_leverage")
    if global_lev is None:
        problems.append("risk.yaml 缺少 caps.global_max_leverage")

    known = set(list_strategies())
    for sym, cfg in s.enabled_symbols().items():
        intervals = list(cfg.get("intervals") or [])
        base = cfg.get("base_interval")
        if not intervals:
            problems.append(f"symbols.{sym}.intervals 为空")
            continue
        if base and base not in intervals:
            problems.append(f"symbols.{sym}.base_interval={base} 不在 intervals={intervals} 中，该周期不会被回填")
        atr_iv = stops.get("atr_interval")
        if atr_iv and atr_iv not in intervals:
            problems.append(
                f"risk.stops.atr_interval={atr_iv} 不在 symbols.{sym}.intervals={intervals} 中 —— "
                f"回填后风控取不到 ATR，止损/仓位会静默按错误值计算"
            )
        lev = cfg.get("leverage_default")
        if lev and global_lev and lev > global_lev:
            problems.append(f"symbols.{sym}.leverage_default={lev} 超过 caps.global_max_leverage={global_lev}")
        if margin and lev and margin.get("max_leverage_default") and lev > margin["max_leverage_default"]:
            problems.append(
                f"symbols.{sym}.leverage_default={lev} 超过 risk.margin.max_leverage_default="
                f"{margin['max_leverage_default']}（该值是 exchangeInfo 不可用时的杠杆上限兜底）"
            )
        for name in cfg.get("strategies") or []:
            if name not in known:
                problems.append(f"symbols.{sym} 引用了不存在的策略 config/strategies/{name}.yaml")
                continue
            scfg = load_strategy_config(name)
            htf_missing = [h for h in (scfg.get("htf") or []) if h not in intervals]
            if htf_missing:
                problems.append(f"strategies/{name}.yaml 的 htf={htf_missing} 未包含在 symbols.{sym}.intervals 中")
            if scfg.get("base_interval") and base and scfg["base_interval"] != base:
                problems.append(f"strategies/{name}.yaml base_interval={scfg['base_interval']} 与 symbols.{sym}={base} 不一致")

    if sizing.get("method") == "fixed_risk" and not sizing.get("risk_pct"):
        problems.append("sizing.method=fixed_risk 需要 sizing.risk_pct")
    tp_mult, sl_mult = stops.get("tp_atr_mult"), stops.get("sl_atr_mult")
    if tp_mult and sl_mult and tp_mult <= sl_mult:
        problems.append(f"stops.tp_atr_mult={tp_mult} <= sl_atr_mult={sl_mult}，盈亏比为负，策略不可能正期望")
    return problems
