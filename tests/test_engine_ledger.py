"""引擎撮合与账本口径的回归测试。

覆盖 2026-10 整体 review 修掉的执行缺陷：
  1. 同一根 bar 内 TP/SL 双触发时保守取 SL（与 build_labels 一致），可配置反转
  2. 止损按 bar 极值成交、出场时间取 bar 中点
  3. 资金费只累计到 funding_since_entry，平仓时经 net_cash 一次性扣款，不重复计费
  4. margin = notional / 固定杠杆，非金字塔路径不得把整份权益记成保证金
  5. 时间止损/结束平仓也计不利滑点（全 taker 口径）
  6. 账本恒等式 sum(trades.pnl) == Δequity（含资金费与多次反手）
  7. USD 回撤取峰值→谷值，而不是按期初权益折算
  8. 盈亏比在无亏损交易时为 inf，不把平均盈利额当比值
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.backtest.engine import BacktestConfig, BacktestEngine, TradeRecord, _precompute
from quant.core.contracts import Action, Bar, FillReason, Mode, Signal, SymbolSpec

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
IV_MS = 300_000
SLIP = 2.0 / 10000          # cfg.slippage_bps = 2.0


def make_spec() -> SymbolSpec:
    return SymbolSpec(
        symbol="ETHUSDT", status="TRADING",
        tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"), min_notional=5.0,
        maint_margin_rate=0.025, liquidation_fee_rate=0.01,
        max_leverage=20.0, funding_interval_hours=8.0,
        price_precision=2, qty_precision=3,
        fetched_at=BASE,
    )


def _frame(n: int = 40) -> pd.DataFrame:
    """恒定 2000 的 5m 序列：开=收=2000 便于核算成交价，高低价波动让 ATR 有值。"""
    ms0 = int(BASE.timestamp() * 1000)
    return pd.DataFrame({
        "ts_open_ms": [ms0 + i * IV_MS for i in range(n)],
        "ts_open": [BASE + timedelta(minutes=5 * i) for i in range(n)],
        "open": [2000.0] * n,
        "high": [2010.0 + (i % 5) for i in range(n)],
        "low": [1990.0 - (i % 4) for i in range(n)],
        "close": [2000.0] * n,
        "volume": [10.0] * n,
    })


def make_engine(funding: pd.DataFrame | None = None, **cfg_kw) -> BacktestEngine:
    cfg_defaults = dict(
        symbol="ETHUSDT", base_interval="5m", start=BASE,
        end=BASE + timedelta(days=400), initial_equity=100.0, risk_pct=0.01,
        sl_atr_mult=2.0, tp_atr_mult=3.5, time_stop_bars=96,
        global_max_leverage=10.0, slippage_bps=2.0, taker_fee=0.0005,
        entry_frac=0.10, add_frac=0.20, add_max_count=2,
        add_trigger_r=0.3, trail_trigger_r=1.0, trail_atr_mult=1.0,
    )
    cfg_defaults.update(cfg_kw)
    df = _frame()
    eng = BacktestEngine(BacktestConfig(**cfg_defaults), {"5m": df}, make_spec(), funding=funding)
    pre = _precompute(df)
    eng._df5_row = pre.iloc[-1]
    return eng


def bar_at(i: int, o=2000.0, h=2010.0, l=1990.0, c=2000.0) -> Bar:
    t = BASE + timedelta(minutes=5 * i)
    return Bar(
        symbol="ETHUSDT", interval="5m", ts_open=t,
        ts_close=t + timedelta(minutes=5) - timedelta(milliseconds=1),
        open=o, high=h, low=l, close=c, volume=10.0,
        quote_volume=10.0 * o, taker_buy_volume=5.0, is_closed=True,
    )


def sig(action: Action, price_ref=2000.0, **meta) -> Signal:
    return Signal(
        signal_id=f"t_{action.value}_{price_ref}", ts_signal=BASE, symbol="ETHUSDT",
        timeframe="5m", strategy="test", action=action, price_ref=price_ref,
        strength=1.0, reason="test", mode=Mode.BACKTEST, meta=meta,
    )


class TestExitOrdering:
    """同一根 bar 同时触及 SL 与 TP：默认保守取 SL。"""

    def _open_long(self, **kw) -> BacktestEngine:
        eng = make_engine(**kw)
        eng._execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(0), eng._df5_row)
        return eng

    def test_conflict_resolves_as_stop(self):
        eng = self._open_long()
        pos = eng.position
        assert pos.stop_loss < 2000.0 < pos.tp_price
        # 一根 bar 的高低点同时越过 SL 与 TP，无法证明哪一侧先到
        eng._check_exits(bar_at(1, o=2000.0, h=2300.0, l=1800.0, c=2100.0))
        assert len(eng.trades) == 1
        t = eng.trades[0]
        assert t.reason == "sl"
        # B 模式：多头按 bar.low 最坏成交，再扣不利滑点
        assert t.exit_price == pytest.approx(1800.0 * (1 - SLIP), abs=1e-6)

    def test_conflict_can_be_configured_to_tp(self):
        eng = self._open_long(sl_first_on_conflict=False)
        tp = eng.position.tp_price
        eng._check_exits(bar_at(1, o=2000.0, h=2300.0, l=1800.0, c=2100.0))
        assert eng.trades[0].reason == "tp"
        assert eng.trades[0].exit_price == pytest.approx(tp * (1 - SLIP), abs=1e-6)

    def test_stop_exit_time_is_bar_midpoint(self):
        """出场时间不能记成 bar 收盘，否则持仓时长被系统性虚增。"""
        eng = self._open_long()
        b = bar_at(1, h=2300.0, l=1800.0)
        eng._check_exits(b)
        mid = b.ts_open + (b.ts_close - b.ts_open) / 2
        assert eng.trades[0].exit_time == mid


class TestMarginSemantics:
    """margin = notional / 固定杠杆，与权益无关。"""

    def test_non_pyramid_margin_is_not_equity(self):
        """回归用例：旧写法 margin = notional / (notional/equity) ≡ equity，
        每笔试仓都把整个账户记成保证金，ret_pct 退化成对权益的收益率。

        ATR=50、sl_atr_mult=2 → 止损距离 100 → 风险预算 qty = 100×0.01/100 = 0.01，
        名义 20 USDT，10× 下保证金应为 2 USDT，而不是 100。
        """
        eng = make_engine(pyramid_enabled=False)
        eng._execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(0), eng._df5_row)
        pos = eng.position
        assert pos is not None
        assert pos.qty == pytest.approx(0.01)
        # 名义/保证金按信号参考价核算（定仓发生在成交前，只能用 price_ref）；
        # 成交价带 2bps 不利滑点，所以实际占用略高于此值。
        assert pos.margin_usdt == pytest.approx(pos.qty * 2000.0 / 10.0, abs=1e-9)
        assert pos.margin_usdt == pytest.approx(2.0, abs=1e-9)
        assert pos.margin_usdt < 10.0

    def test_pyramid_entry_frac_margin(self):
        eng = make_engine(pyramid_enabled=True)
        eng._execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(0), eng._df5_row)
        pos = eng.position
        # 100 × 0.10 试仓 × 10× 杠杆 / 2000 = 0.05 → 名义 100 → 保证金 10
        assert pos.qty == pytest.approx(0.05)
        assert pos.margin_usdt == pytest.approx(10.0, abs=0.01)

    def test_sizing_uses_equity_known_at_bar_open(self):
        """定仓只能用上一根收盘的净值；本根收盘 MTM 在成交时还没发生。"""
        eng = make_engine(pyramid_enabled=False)
        eng.equity_open = 100.0
        eng.equity = 1000.0          # 假的"本根收盘"净值，若被采用 qty 会放大 10 倍
        eng._execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(0), eng._df5_row)
        assert eng.position.qty == pytest.approx(0.01)


class TestFunding:
    def _funding_df(self, funded_at_ms: int, rate=0.0001, mark=2100.0) -> pd.DataFrame:
        return pd.DataFrame({
            "funded_at_ms": [funded_at_ms],
            "funding_rate": [rate],
            "mark_price": [mark],
        })

    def test_accrues_on_mark_price_and_charges_once(self):
        b2 = bar_at(2)
        eng = make_engine(funding=self._funding_df(int(b2.ts_open.timestamp() * 1000)))
        eng._execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(0), eng._df5_row)
        pos = eng.position
        cash_after_entry = eng.cash

        eng._accrue_funding(int(b2.ts_open.timestamp() * 1000) + IV_MS, 2000.0)
        # 标记价 2100 优先于传入价 2000；多头付正费率
        expected = pos.qty * 2100.0 * 0.0001
        assert eng.funding_since_entry == pytest.approx(expected)
        assert pos.funding_paid == pytest.approx(expected)
        # 现金只在平仓时一次性结算，持仓期内不得重复扣
        assert eng.cash == pytest.approx(cash_after_entry)

        exit_px = 2000.0
        eng._close_trade(bar_at(3), FillReason.SIGNAL, exit_px)
        t = eng.trades[0]
        gross = pos.qty * (exit_px - pos.entry_price)
        fee_exit = pos.qty * exit_px * 0.0005
        assert t.pnl == pytest.approx(gross - t.fee - expected)
        assert eng.cash == pytest.approx(
            cash_after_entry + pos.margin_usdt + gross - fee_exit - expected
        )

    def test_short_receives_positive_rate(self):
        b2 = bar_at(2)
        eng = make_engine(funding=self._funding_df(int(b2.ts_open.timestamp() * 1000)))
        eng._execute_signal(sig(Action.OPEN_SHORT, atr=50.0), bar_at(0), eng._df5_row)
        eng._accrue_funding(int(b2.ts_open.timestamp() * 1000) + IV_MS, 2000.0)
        assert eng.funding_since_entry < 0            # 正费率下空头收钱

    def test_pointer_advances_while_flat(self):
        """空仓期间的结算必须推进指针，否则会被算进之后才开的仓位。"""
        b1 = bar_at(1)
        t1 = int(b1.ts_open.timestamp() * 1000)
        eng = make_engine(funding=self._funding_df(t1))
        eng._accrue_funding(t1 + IV_MS, 2000.0)       # 此时无持仓
        assert eng.funding_since_entry == 0.0
        eng._execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(2), eng._df5_row)
        eng._accrue_funding(t1 + 2 * IV_MS, 2000.0)
        assert eng.funding_since_entry == 0.0         # 旧结算不会被补记

    def test_can_be_disabled(self):
        b2 = bar_at(2)
        eng = make_engine(
            funding=self._funding_df(int(b2.ts_open.timestamp() * 1000)),
            include_funding=False,
        )
        eng._execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(0), eng._df5_row)
        eng._accrue_funding(int(b2.ts_open.timestamp() * 1000) + IV_MS, 2000.0)
        assert eng.funding_since_entry == 0.0


class TestMarketCloseSlippage:
    def test_time_stop_pays_adverse_slip(self):
        """回归用例：时间止损曾经按未滑点的 bar.close 成交，全 taker 口径下少扣一次成本。"""
        n = 3130
        df = _frame(n)
        eng = make_engine(time_stop_bars=1, pyramid_enabled=False)
        eng.bars["5m"] = df
        eng.cfg.start = BASE
        eng.cfg.end = BASE + timedelta(minutes=5 * n)

        calls = {"n": 0}

        def fn(bar, row, htf, dt):
            calls["n"] += 1
            return sig(Action.OPEN_LONG, price_ref=bar.close, atr=50.0) if calls["n"] == 1 else None

        eng.run(fn)
        t = eng.trades[0]
        assert t.reason == "time"
        assert t.exit_price == pytest.approx(2000.0 * (1 - SLIP), abs=1e-6)


class TestLedgerInvariant:
    def test_sum_pnl_equals_equity_change(self):
        """核心恒等式：净值增长必须完全由逐笔盈亏解释（含资金费、手续费、滑点）。"""
        n = 3140
        df = _frame(n)
        ms0 = int(BASE.timestamp() * 1000)
        funding = pd.DataFrame({
            "funded_at_ms": [ms0 + 3125 * IV_MS, ms0 + 3128 * IV_MS],
            "funding_rate": [0.0001, -0.00012],
            "mark_price": [2000.0, 2000.0],
        })
        eng = make_engine(funding=funding)
        eng.bars["5m"] = df
        eng.cfg.start = BASE
        eng.cfg.end = BASE + timedelta(minutes=5 * n)

        plan = {
            1: Action.OPEN_LONG,
            4: Action.CLOSE_LONG,
            6: Action.OPEN_SHORT,
            9: Action.CLOSE_SHORT,
            11: Action.OPEN_LONG,      # 留给结束时强制 FLAT
        }
        calls = {"n": 0}

        def fn(bar, row, htf, dt):
            calls["n"] += 1
            action = plan.get(calls["n"])
            return sig(action, price_ref=bar.close, atr=50.0) if action else None

        result = eng.run(fn)
        assert len(eng.trades) == 3               # 两笔信号平仓 + 收尾强制 FLAT
        assert eng.position is None
        # 收尾结算已回写净值末点，所以末点等于结算后的现金
        assert result.final_equity == pytest.approx(eng.cash, abs=1e-9)
        assert sum(t.pnl for t in eng.trades) == pytest.approx(
            result.final_equity - eng.cfg.initial_equity, abs=1e-6
        )
        # 每笔保证金都必须等于 notional/杠杆，且不超过当时的可用权益
        for t in eng.trades:
            assert t.margin_usdt <= eng.cfg.initial_equity
            assert t.ret_pct == pytest.approx(t.pnl / t.margin_usdt * 100, abs=1e-6)


class TestDrawdownMeasure:
    """USD 回撤按峰值到谷值计，不能折算到期初权益。"""

    def test_peak_to_trough(self):
        eng = make_engine()
        # 2000 根 5m ≈ 7 天：曲线要够长，否则 CAGR 的 1/years 次幂会溢出
        eng.equity_curve = [100.0] * 500 + [200.0] * 500 + [120.0] * 500 + [150.0] * 500
        res = eng._metrics()
        assert res.max_drawdown == pytest.approx(80.0)
        assert res.max_drawdown_pct == pytest.approx(-40.0)


class TestPayoffMeasure:
    """盈亏比是比值：没有亏损交易时给 inf，不能把平均盈利额（USDT）当成比值。"""

    @staticmethod
    def _trade(pnl: float):
        return TradeRecord(
            entry_time=BASE, exit_time=BASE + timedelta(minutes=5), side="long",
            entry_price=2000.0, exit_price=2000.0, qty=0.1, pnl=pnl, fee=0.01, reason="tp",
        )

    def test_all_winners_is_infinite(self):
        eng = make_engine()
        eng.equity_curve = [100.0, 101.0, 102.0]
        eng.trades = [self._trade(1.0), self._trade(1.0)]
        res = eng._metrics()
        assert res.payoff_ratio == float("inf")
        assert res.profit_factor == float("inf")

    def test_ratio_of_avg_pnl(self):
        eng = make_engine()
        eng.equity_curve = [100.0, 101.0, 100.0, 101.0]
        eng.trades = [self._trade(3.0), self._trade(3.0), self._trade(-1.0)]
        res = eng._metrics()
        assert res.payoff_ratio == pytest.approx(3.0)
        assert res.profit_factor == pytest.approx(6.0)

    def test_no_trades_is_zero(self):
        eng = make_engine()
        eng.equity_curve = [100.0, 100.0]
        res = eng._metrics()
        assert res.payoff_ratio == 0.0
