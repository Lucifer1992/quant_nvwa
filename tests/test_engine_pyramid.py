"""金字塔加仓机制单元测试。

验证目标（撮合与持仓管理已抽到 backtest/position.py，回测与 paper 共用同一份实现）：
  1. open_or_add 试仓逻辑：创建 position、扣 margin、记录 add_levels
  2. open_or_add 加仓逻辑：加权均价、追加 qty、追加 margin
  3. should_add 条件判断：浮盈 R + swing 突破
  4. update_trailing：激活前不动，激活后跟随
  5. settle_price_exits B 模式：用 bar.low/bar.high + 中点时间
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

# 确保 src 在路径中
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.backtest.engine import BacktestConfig, BacktestEngine
from quant.backtest.position import should_add, update_trailing
from quant.core.contracts import Bar, SymbolSpec
from quant.risk.sizing import compute_risk_size
from quant.core.contracts import RiskDecision
from decimal import Decimal


def make_spec() -> SymbolSpec:
    """构造测试用的 ETHUSDT SymbolSpec。"""
    return SymbolSpec(
        symbol="ETHUSDT", status="TRADING",
        tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"), min_notional=5.0,
        maint_margin_rate=0.025, liquidation_fee_rate=0.01,
        max_leverage=20.0, funding_interval_hours=8.0,
        price_precision=2, qty_precision=3,
        fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def make_bar(ts: datetime, o: float, h: float, l: float, c: float) -> Bar:
    """构造 5m 测试 bar。"""
    return Bar(
        symbol="ETHUSDT", interval="5m",
        ts_open=ts, ts_close=ts + timedelta(minutes=5) - timedelta(milliseconds=1),
        open=o, high=h, low=l, close=c, volume=10.0,
        quote_volume=10 * c, taker_buy_volume=5.0,
        trades=100, is_closed=True, src="rest",
    )


@pytest.fixture
def engine():
    """构造一个最小化的 BacktestEngine 用于单元测试。"""
    cfg = BacktestConfig(
        symbol="ETHUSDT", base_interval="5m",
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 12, 31, tzinfo=timezone.utc),
        initial_equity=100.0, risk_pct=0.01, sl_atr_mult=1.5,
        tp_atr_mult=3.0, allow_short=True, global_max_leverage=10.0,
        time_stop_bars=96,
        entry_frac=0.10, add_frac=0.20, add_max_count=2,
        add_trigger_r=0.3, trail_trigger_r=1.0,
    )
    bars = {
        "5m": pd.DataFrame({
            "symbol": ["ETHUSDT"] * 3,
            "interval": ["5m"] * 3,
            "ts_open_ms": [0, 300000, 600000],
            "ts_open": [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i * 5) for i in range(3)],
            "ts_close_ms": [299999, 599999, 899999],
            "open": [100.0, 100.0, 100.0],
            "high": [105.0, 105.0, 105.0],
            "low": [99.0, 99.0, 99.0],
            "close": [104.0, 104.0, 104.0],
            "volume": [10.0] * 3, "quote_volume": [1040.0] * 3,
            "taker_buy_volume": [5.0] * 3, "trades": [100] * 3,
            "is_closed": [1] * 3, "src": ["rest"] * 3,
        })
    }
    return BacktestEngine(cfg, bars, make_spec())


class TestPyramidEntry:
    """测试 open_or_add 试仓逻辑。"""

    def test_first_entry_creates_position(self, engine):
        """第一次入场应该创建 Position、扣 margin、记录 add_levels。"""
        from quant.core.contracts import Action, Mode, Signal
        ts = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        bar = make_bar(ts, 100.0, 105.0, 99.0, 104.0)
        signal = Signal(
            signal_id="t1", ts_signal=ts, symbol="ETHUSDT", timeframe="5m",
            strategy="test", action=Action.OPEN_LONG,
            price_ref=104.0, strength=0.5, reason="test", mode=Mode.BACKTEST,
            meta={"atr": 5.0},
        )
        decision = RiskDecision(
            signal=signal, approved=True, qty=0.05, notional_usdt=500.0,
            leverage=10.0, margin_usdt=50.0, stop_loss=104.0 - 7.5,
            equity_usdt=100.0, risk_usdt=1.0,
        )

        initial_cash = engine.ledger.cash
        engine.ledger.open_or_add(signal, decision, bar)

        assert engine.ledger.position is not None
        assert engine.ledger.position.side == "long"
        assert engine.ledger.position.qty == 0.05
        assert engine.ledger.position.add_count == 0
        assert len(engine.ledger.position.add_levels) == 1
        # cash 减少 margin + fee
        expected_fee = 0.05 * (100 * 1.0002) * 0.0005  # entry_price × taker
        assert abs(engine.ledger.cash - (initial_cash - 50.0 - expected_fee)) < 0.01
        # SL 基于实际入场价调整（不再是 signal.price_ref=104）
        expected_sl = 100 * 1.0002 - 1.5 * 5.0  # entry - 1.5*ATR
        assert abs(engine.ledger.position.stop_loss - expected_sl) < 0.01

    def test_add_increases_qty_and_updates_avg(self, engine):
        """加仓应该累计 qty、加权均价、累计 margin。"""
        from quant.core.contracts import Action, Mode, Signal
        ts1 = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        bar1 = make_bar(ts1, 100.0, 105.0, 99.0, 104.0)
        signal1 = Signal(
            signal_id="t1", ts_signal=ts1, symbol="ETHUSDT", timeframe="5m",
            strategy="test", action=Action.OPEN_LONG,
            price_ref=104.0, strength=0.5, reason="first", mode=Mode.BACKTEST,
            meta={"atr": 5.0},
        )
        decision1 = RiskDecision(
            signal=signal1, approved=True, qty=0.05, notional_usdt=500.0,
            leverage=10.0, margin_usdt=50.0, stop_loss=96.5,
            equity_usdt=100.0, risk_usdt=1.0,
        )
        engine.ledger.open_or_add(signal1, decision1, bar1)

        # 加仓：第二根 bar，价格上涨
        ts2 = ts1 + timedelta(minutes=5)
        bar2 = make_bar(ts2, 110.0, 115.0, 109.0, 113.0)
        signal2 = Signal(
            signal_id="t2", ts_signal=ts2, symbol="ETHUSDT", timeframe="5m",
            strategy="test", action=Action.ADD_TO_LONG,
            price_ref=113.0, strength=0.5, reason="add", mode=Mode.BACKTEST,
            meta={"atr": 5.0},
        )
        decision2 = RiskDecision(
            signal=signal2, approved=True, qty=0.03, notional_usdt=339.0,
            leverage=10.0, margin_usdt=33.9, stop_loss=105.5,
            equity_usdt=100.0, risk_usdt=0.0,
        )
        engine.ledger.open_or_add(signal2, decision2, bar2)

        assert engine.ledger.position.qty == 0.08  # 0.05 + 0.03
        assert engine.ledger.position.add_count == 1
        assert len(engine.ledger.position.add_levels) == 2
        # 加权均价 = (100.02 * 0.05 + 110.02 * 0.03) / 0.08
        # = (5.001 + 3.3006) / 0.08 = 8.3016 / 0.08 = 103.77
        assert abs(engine.ledger.position.avg_entry_price - 103.77) < 0.5


class TestShouldAdd:
    """测试 should_add 加仓条件。"""

    def test_no_position_no_add(self, engine):
        """无持仓时 should_add 应返回 False。"""
        bar = make_bar(datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
                       100, 105, 99, 104)
        # 构造 row with swing_high=102
        row = pd.Series({"close": 104, "swing_high": 102, "swing_low": 99, "atr": 5.0})
        assert should_add(engine.ledger.position, engine.cfg, row, bar) is False

    def test_position_no_float_no_add(self, engine):
        """有持仓但无浮盈时不应加仓。"""
        from quant.core.contracts import Position
        ts = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        bar = make_bar(ts, 100, 105, 99, 104)
        # 构造一个 long 持仓，均价 105（高于当前价）
        engine.ledger.position = Position(
            position_id="t1", symbol="ETHUSDT", side="long",
            qty=0.05, entry_price=105.0, avg_entry_price=105.0,
            entry_ts=ts, margin_usdt=50.0, stop_loss=96.5, take_profits=(),
            entry_atr=5.0,
        )
        row = pd.Series({"close": 104, "swing_high": 110, "swing_low": 99, "atr": 5.0})
        # 当前价 104 < 均价 105 → 浮亏
        assert should_add(engine.ledger.position, engine.cfg, row, bar) is False

    def test_position_with_float_breakout_should_add(self, engine):
        """有持仓 + 浮盈 + 突破 swing high → 应加仓。"""
        from quant.core.contracts import Position
        ts = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        bar = make_bar(ts, 100, 110, 99, 108)  # close=108 > swing_high=106
        engine.ledger.position = Position(
            position_id="t1", symbol="ETHUSDT", side="long",
            qty=0.05, entry_price=100.0, avg_entry_price=100.0,
            entry_ts=ts, margin_usdt=50.0, stop_loss=92.5, take_profits=(),
            entry_atr=5.0,
        )
        # 当前价 108 > 均价 100 → 浮盈 (108-100)/(1.5*5) = 1.07R > 0.3R 阈值
        # 当前 close > swing_high=106 → 突破
        row = pd.Series({"close": 108, "swing_high": 106, "swing_low": 99, "atr": 5.0})
        assert should_add(engine.ledger.position, engine.cfg, row, bar) is True


class TestTrailingStop:
    """测试 update_trailing 移动止损。"""

    def test_no_trail_when_below_trigger(self, engine):
        """浮盈 < trail_trigger_r 时不激活，止损不动。"""
        from quant.core.contracts import Position
        ts = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        bar = make_bar(ts, 100, 101, 99, 100.5)
        engine.ledger.position = Position(
            position_id="t1", symbol="ETHUSDT", side="long",
            qty=0.05, entry_price=100.0, avg_entry_price=100.0,
            entry_ts=ts, margin_usdt=50.0, stop_loss=92.5, take_profits=(),
            entry_atr=5.0, trailing_peak_price=100.5,
        )
        update_trailing(engine.ledger.position, engine.cfg, bar)
        assert engine.ledger.position.trailing_active is False
        assert engine.ledger.position.stop_loss == 92.5  # 不变

    def test_trail_activates_at_threshold(self, engine):
        """浮盈 ≥ trail_trigger_r 时激活，止损开始跟随。"""
        from quant.core.contracts import Position
        ts = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        # 当前价 110，相对均价 100，浮盈 = (110-100)/(1.5*5) = 1.33R ≥ 1.0R
        # bar.high = close = 110（peak = 110）
        bar = make_bar(ts, 100, 110, 99, 110)
        engine.ledger.position = Position(
            position_id="t1", symbol="ETHUSDT", side="long",
            qty=0.05, entry_price=100.0, avg_entry_price=100.0,
            entry_ts=ts, margin_usdt=50.0, stop_loss=92.5, take_profits=(),
            entry_atr=5.0, trailing_peak_price=110.0,
        )
        update_trailing(engine.ledger.position, engine.cfg, bar)
        assert engine.ledger.position.trailing_active is True
        # 新 SL = peak(110) - 1*ATR(5) = 105（向有利方向）
        assert engine.ledger.position.stop_loss == 105.0

    def test_trail_only_moves_in_favorable_direction(self, engine):
        """激活后价格回撤，止损不会向不利方向移动。"""
        from quant.core.contracts import Position
        ts = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        # 当前价 105（回撤），peak 仍是 110（之前的高点）
        bar = make_bar(ts, 100, 105, 99, 105)
        engine.ledger.position = Position(
            position_id="t1", symbol="ETHUSDT", side="long",
            qty=0.05, entry_price=100.0, avg_entry_price=100.0,
            entry_ts=ts, margin_usdt=50.0, stop_loss=105.0, take_profits=(),  # 已激活，SL=105
            entry_atr=5.0, trailing_active=True, trailing_peak_price=110.0,
        )
        update_trailing(engine.ledger.position, engine.cfg, bar)
        # peak 不变（110 > 105），新 SL = 110 - 5 = 105，max(105, 105) = 105（不变）
        assert engine.ledger.position.stop_loss == 105.0


class TestSizing:
    """测试 compute_risk_size 的定仓、保证金与拒绝分支。

    杠杆是策略层的固定政策（设计原则：全程 10×），不是由 notional/equity 反推的
    诊断量，所以 `result.leverage` 恒等于 `global_max_leverage`；真正会变的量是
    notional 与 margin。
    """

    @staticmethod
    def _signal(price_ref: float):
        from quant.core.contracts import Action, Mode, Signal
        return Signal(
            signal_id="t1", ts_signal=datetime.now(tz=timezone.utc),
            symbol="ETHUSDT", timeframe="5m", strategy="t",
            action=Action.OPEN_LONG, price_ref=price_ref,
            strength=0.5, reason="t", mode=Mode.BACKTEST,
        )

    def test_notional_capped_at_max_leverage(self):
        """ATR 极小 → 风险预算推出的名义超过 10× 上限，被 cap 到 equity × leverage。"""
        spec = make_spec()
        signal = self._signal(100.0)
        # ATR=0.05，stop_dist=0.075 → qty=1.0/0.075=13.33 → notional=1333 > 1000
        result = compute_risk_size(
            signal=signal, decision=RiskDecision(signal=signal, approved=False),
            equity=100.0, risk_pct=0.01, atr=0.05,
            sl_atr_mult=1.5, global_max_leverage=10.0,
            symbol_spec=spec,
        )
        assert result.approved is True
        assert result.qty == pytest.approx(10.0)
        assert result.notional_usdt == pytest.approx(1000.0)
        assert result.leverage == 10.0
        # 顶格时 margin 恰好等于全部权益：1000 USDT 名义在 10× 下需要 100 USDT 保证金
        assert result.margin_usdt == pytest.approx(100.0)

    def test_margin_is_notional_over_fixed_leverage_not_equity(self):
        """未触顶时 margin 必须远小于权益。

        回归用例：旧实现把杠杆写成 `notional / equity`，使 margin 恒等于 equity，
        每笔成交都抽干整个账户，ret_pct 退化成对权益的收益率。
        """
        spec = make_spec()
        signal = self._signal(2000.0)
        # ATR=10，stop_dist=15 → qty=1.0/15=0.0666 → step 截断 0.066 → notional=132
        result = compute_risk_size(
            signal=signal, decision=RiskDecision(signal=signal, approved=False),
            equity=100.0, risk_pct=0.01, atr=10.0,
            sl_atr_mult=1.5, global_max_leverage=10.0,
            symbol_spec=spec,
        )
        assert result.approved is True
        assert result.qty == pytest.approx(0.066)
        assert result.notional_usdt == pytest.approx(132.0)
        assert result.margin_usdt == pytest.approx(13.2)
        assert result.margin_usdt < 100.0
        assert result.leverage == 10.0
        # 风险额按定稿后的 qty 回算，而不是照抄 equity × risk_pct
        assert result.risk_usdt == pytest.approx(0.066 * 15.0)
        assert result.est_liq_price < 2000.0

    def test_explicit_qty_bypasses_risk_budget(self):
        """金字塔路径传入 qty 时直接采用，不再用风险预算覆盖。"""
        spec = make_spec()
        signal = self._signal(100.0)
        result = compute_risk_size(
            signal=signal, decision=RiskDecision(signal=signal, approved=False),
            equity=100.0, risk_pct=0.01, atr=10.0,
            sl_atr_mult=1.5, global_max_leverage=10.0,
            symbol_spec=spec, qty=0.5,
        )
        assert result.approved is True
        assert result.qty == pytest.approx(0.5)
        assert result.notional_usdt == pytest.approx(50.0)
        assert result.margin_usdt == pytest.approx(5.0)

    def test_reject_below_min_qty(self):
        spec = make_spec()
        signal = self._signal(100.0)
        # 0.0005 按 step 0.001 向下截断为 0，低于 min_qty
        result = compute_risk_size(
            signal=signal, decision=RiskDecision(signal=signal, approved=False),
            equity=100.0, risk_pct=0.01, atr=10.0,
            sl_atr_mult=1.5, global_max_leverage=10.0,
            symbol_spec=spec, qty=0.0005,
        )
        assert result.approved is False
        assert any("最小下单量" in r for r in result.reject_reasons)

    def test_reject_below_min_notional(self):
        spec = make_spec()
        signal = self._signal(1.0)
        result = compute_risk_size(
            signal=signal, decision=RiskDecision(signal=signal, approved=False),
            equity=100.0, risk_pct=0.01, atr=10.0,
            sl_atr_mult=1.5, global_max_leverage=10.0,
            symbol_spec=spec, qty=0.001,
        )
        assert result.approved is False
        assert any("最小成交额" in r for r in result.reject_reasons)

    def test_reject_without_atr(self):
        """风险预算路径依赖 ATR；ATR 缺失时不能凭空定仓。"""
        spec = make_spec()
        signal = self._signal(100.0)
        result = compute_risk_size(
            signal=signal, decision=RiskDecision(signal=signal, approved=False),
            equity=100.0, risk_pct=0.01, atr=0.0,
            sl_atr_mult=1.5, global_max_leverage=10.0,
            symbol_spec=spec,
        )
        assert result.approved is False
        assert any("ATR" in r for r in result.reject_reasons)

    def test_reject_when_equity_zero(self):
        spec = make_spec()
        signal = self._signal(100.0)
        result = compute_risk_size(
            signal=signal, decision=RiskDecision(signal=signal, approved=False),
            equity=0.0, risk_pct=0.01, atr=10.0,
            sl_atr_mult=1.5, global_max_leverage=10.0,
            symbol_spec=spec,
        )
        assert result.approved is False
        assert any("权益已归零" in r for r in result.reject_reasons)


# 入口
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
