"""CLI 选项分发契约：`BaseStrategy.build` 只把"用户显式传了"的选项交给策略。

回归动机（2026-10-06 实测到的污染）：CLI 曾把 rule_v2 调好的 `srsi_oversold=15`
默认值一并传给 rule_v3（它自己的默认是 30），导致 rule_v3 回测笔数从 760 变 668。
策略专属选项默认必须是 None，由策略 dataclass 持有唯一真值默认。
"""
from __future__ import annotations

from pathlib import Path

from quant.strategy.rules.rule_v1 import RuleV1Strategy
from quant.strategy.rules.rule_v2 import RuleV2Strategy
from quant.strategy.rules.rule_v3 import RuleV3Strategy
from quant.strategy.rules.rule_v4 import RuleV4Strategy

# CLI 一次性把所有选项都塞进来，未传的参数值为 None
CLI_OPTS = {
    "symbol": "ETHUSDT",
    "allow_short": True,
    "confirm_bars": None,
    "srsi_oversold": None,
    "srsi_overbought": None,
    "srsi_cross_margin": None,
    "min_signals": None,
    "adx_min": None,
    "pullback_atr": None,
    "tau": None,
    "model_dir": Path("models/lgbm_v1"),
}


def test_none_options_fall_back_to_strategy_defaults():
    assert RuleV3Strategy.build(CLI_OPTS).cfg.srsi_oversold == 30.0
    assert RuleV3Strategy.build(CLI_OPTS).cfg.srsi_overbought == 70.0
    assert RuleV2Strategy.build(CLI_OPTS).cfg.srsi_oversold == 15.0
    assert RuleV4Strategy.build(CLI_OPTS).cfg.adx_min == 32.0


def test_options_do_not_leak_across_strategies():
    """lgbm 专属字段绝不能出现在规则策略的 config 上。"""
    for cls in (RuleV1Strategy, RuleV2Strategy, RuleV3Strategy, RuleV4Strategy):
        cfg = cls.build(CLI_OPTS).cfg
        for foreign in ("model_dir", "tau", "symbol"):
            assert not hasattr(cfg, foreign), f"{cls.name} 被注入了 {foreign}"


def test_explicit_option_is_honoured():
    s = RuleV3Strategy.build({**CLI_OPTS, "min_signals": 3})
    assert s.cfg.min_signals == 3
    s4 = RuleV4Strategy.build({**CLI_OPTS, "adx_min": 40.0})
    assert s4.cfg.adx_min == 40.0


def test_allow_short_false_propagates():
    assert RuleV4Strategy.build({**CLI_OPTS, "allow_short": False}).cfg.allow_short is False


def test_strategy_without_config_cls_ignores_options():
    """无 config_cls 的策略走零参构造，不会因未知选项报错。"""
    from quant.strategy.base import BaseStrategy

    class Plain(BaseStrategy):
        name = "plain_test_strategy"
        config_cls = None

        def _decide(self, *a):  # pragma: no cover
            return None

        def on_bar(self, bar, row, htf, dt):  # pragma: no cover
            return None

    assert isinstance(Plain.build(CLI_OPTS), Plain)
