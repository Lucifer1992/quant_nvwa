"""回测引擎执行语义测试：信号在下一根 bar 开盘成交、策略可提供 ATR / 仓位倍数 / 平仓。

覆盖 rule_v4 落地时新增的引擎能力（docs/rule_v4-trend-design.md §5）：
  E1 策略自带 meta["atr"] 不被 5m ATR 覆盖
  E2 CLOSE_* 信号能真正平仓
  E3 meta["size_mult"] 按高周期背景缩小本次仓位
  E6 trail_atr_mult 生效
  反手：持仓中出现对向 OPEN 信号 → 先平后开
  无前视：信号在 bar 收盘产生，成交必须在下一根 bar 开盘
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.backtest.engine import BacktestConfig, BacktestEngine
from quant.backtest.precompute import precompute
from quant.backtest.position import update_trailing
from quant.core.contracts import Action, Bar, Mode, Signal, SymbolSpec

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
IV_MS = 300_000


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


def make_engine(**kw) -> BacktestEngine:
    cfg_kw = dict(
        symbol="ETHUSDT", base_interval="5m", start=BASE,
        end=BASE + timedelta(days=400), initial_equity=100.0, risk_pct=0.01,
        sl_atr_mult=2.0, tp_atr_mult=3.5, time_stop_bars=96,
        global_max_leverage=10.0, slippage_bps=2.0, taker_fee=0.0005,
        entry_frac=0.10, add_frac=0.20, add_max_count=2,
        add_trigger_r=0.3, trail_trigger_r=1.0, trail_atr_mult=1.0,
    )
    cfg_kw.update(kw)
    # 40 根 5m bar：开/收盘恒定便于校验成交价，高低点轻微波动让 ATR 有真实取值
    n = 40
    df = pd.DataFrame({
        "ts_open_ms": [int(BASE.timestamp() * 1000) + i * IV_MS for i in range(n)],
        "ts_open": [BASE + timedelta(minutes=5 * i) for i in range(n)],
        "open": [2000.0] * n,
        "high": [2010.0 + (i % 5) for i in range(n)],
        "low": [1990.0 - (i % 4) for i in range(n)],
        "close": [2000.0] * n, "volume": [10.0] * n,
    })
    engine = BacktestEngine(BacktestConfig(**cfg_kw), {"5m": df}, make_spec())
    pre = precompute(df)
    engine._df5_row = pre.iloc[-1]  # 已出 ATR 值的一行
    engine._df5_nan_row = pre.iloc[0]  # 预热期 ATR 仍为 NaN 的一行
    return engine


def bar_at(i: int, o=2000.0, h=2010.0, l=1990.0, c=2000.0) -> Bar:
    t = BASE + timedelta(minutes=5 * i)
    return Bar(symbol="ETHUSDT", interval="5m", ts_open=t,
               ts_close=t + timedelta(minutes=5) - timedelta(milliseconds=1),
               open=o, high=h, low=l, close=c, volume=10.0,
               quote_volume=10.0 * o, taker_buy_volume=5.0, is_closed=True)


def sig(action: Action, price_ref=2000.0, **meta) -> Signal:
    return Signal(
        signal_id=f"t_{action.value}_{price_ref}", ts_signal=BASE, symbol="ETHUSDT",
        timeframe="5m", strategy="test", action=action, price_ref=price_ref,
        strength=1.0, reason="test", mode=Mode.BACKTEST, meta=meta,
    )


class TestAtrFromStrategy:
    """E1：策略自带 1h ATR 时不得被 5m ATR 覆盖。"""

    def test_meta_atr_survives(self):
        eng = make_engine()
        row = eng._df5_row
        assert row["atr"] > 0 and row["atr"] != 50.0  # 5m ATR 与传入值不同才有意义
        eng.ledger.execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(0), row)
        assert eng.ledger.position is not None
        assert eng.ledger.position.entry_atr == 50.0

    def test_sl_tp_use_strategy_atr(self):
        eng = make_engine()
        eng.ledger.execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(0), eng._df5_row)
        entry = eng.ledger.position.entry_price
        # sl_atr_mult=2.0, tp_atr_mult=3.5，且开仓滑点为不利方向
        assert eng.ledger.position.stop_loss == pytest.approx(entry - 2.0 * 50.0, abs=1e-6)
        assert eng.ledger.position.tp_price == pytest.approx(entry + 3.5 * 50.0, abs=1e-6)

    def test_fallback_to_base_atr(self):
        eng = make_engine()
        row = eng._df5_row
        eng.ledger.execute_signal(sig(Action.OPEN_LONG), bar_at(0), row)
        assert eng.ledger.position.entry_atr == pytest.approx(float(row["atr"]))

    def test_fallback_when_base_atr_still_warming_up(self):
        # 预热期 5m ATR 为 NaN，用 price_ref×0.5% 兜底而不是把 NaN 带进止损
        eng = make_engine()
        eng.ledger.execute_signal(sig(Action.OPEN_LONG), bar_at(0), eng._df5_nan_row)
        assert eng.ledger.position.entry_atr == pytest.approx(2000.0 * 0.005)


class TestSizeMult:
    """E3：per-signal 仓位倍数（4h 软对齐用它减仓，不允许超额）。"""

    def _margin(self, eng, **meta):
        eng.ledger.execute_signal(sig(Action.OPEN_LONG, **meta), bar_at(0), eng._df5_row)
        return eng.ledger.position.margin_usdt

    def test_scales_margin(self):
        assert self._margin(make_engine()) == pytest.approx(10.0, abs=0.01)
        assert self._margin(make_engine(), size_mult=0.7) == pytest.approx(7.0, abs=0.02)

    def test_clamped_to_at_most_one(self):
        assert self._margin(make_engine(), size_mult=5.0) == pytest.approx(10.0, abs=0.01)

    def test_clamped_at_floor(self):
        # 低于 0.2 的倍数被钳到 0.2，避免出现无意义的小额仓位
        assert self._margin(make_engine(), size_mult=0.01) == pytest.approx(2.0, abs=0.01)


class TestCloseSignal:
    """E2：策略主动平仓。"""

    def test_close_long_executes_at_next_bar_with_adverse_slip(self):
        eng = make_engine()
        eng.ledger.execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(0), eng._df5_row)
        assert eng.ledger.position is not None
        eng.ledger.execute_signal(sig(Action.CLOSE_LONG), bar_at(1, o=2050.0), eng._df5_row)
        assert eng.ledger.position is None
        assert len(eng.ledger.trades) == 1
        t = eng.ledger.trades[0]
        assert t.reason == "signal"
        # 多头平仓是卖出：成交价按不利方向打折
        assert t.exit_price == pytest.approx(2050.0 * (1 - 2.0 / 10000), abs=1e-6)

    def test_close_ignored_when_flat(self):
        eng = make_engine()
        eng.ledger.execute_signal(sig(Action.CLOSE_LONG), bar_at(0), eng._df5_row)
        assert eng.ledger.trades == []

    def test_close_ignored_for_opposite_side(self):
        eng = make_engine()
        eng.ledger.execute_signal(sig(Action.OPEN_SHORT, atr=50.0), bar_at(0), eng._df5_row)
        eng.ledger.execute_signal(sig(Action.CLOSE_LONG), bar_at(1), eng._df5_row)
        assert eng.ledger.position is not None and eng.ledger.position.side == "short"


class TestFlip:
    """持仓中出现对向 OPEN：先平后开，同一根 bar 开盘价成交。"""

    def test_opposite_open_replaces_position(self):
        eng = make_engine()
        eng.ledger.execute_signal(sig(Action.OPEN_LONG, atr=50.0), bar_at(0), eng._df5_row)
        first_entry = eng.ledger.position.entry_price
        eng.ledger.execute_signal(sig(Action.OPEN_SHORT, atr=50.0), bar_at(1, o=2000.0), eng._df5_row)
        assert eng.ledger.position is not None and eng.ledger.position.side == "short"
        assert len(eng.ledger.trades) == 1
        assert eng.ledger.trades[0].side == "long"
        assert eng.ledger.trades[0].exit_price == pytest.approx(2000.0 * (1 - 2.0 / 10000), abs=1e-6)
        # 新仓是做空开仓：卖出，成交价打折（与 first_entry 的买入价不同方向）
        assert eng.ledger.position.entry_price == pytest.approx(2000.0 * (1 - 2.0 / 10000), abs=1e-6)
        assert first_entry == pytest.approx(2000.0 * (1 + 2.0 / 10000), abs=1e-6)


class TestTrailingAtrMult:
    """E6：移动止损跟随距离可配。"""

    def test_distance_follows_config(self):
        eng = make_engine(trail_atr_mult=1.5, trail_trigger_r=1.0)
        eng.ledger.execute_signal(sig(Action.OPEN_LONG, atr=10.0), bar_at(0), eng._df5_row)
        pos = eng.ledger.position
        entry = pos.entry_price
        # 浮盈需 ≥ sl_atr_mult×ATR×trail_trigger_r = 20 才激活
        peak_bar = bar_at(1, o=entry, h=entry + 60.0, l=entry, c=entry + 50.0)
        update_trailing(eng.ledger.position, eng.cfg, peak_bar)
        assert pos.trailing_active
        assert pos.stop_loss == pytest.approx(pos.trailing_peak_price - 1.5 * 10.0, abs=1e-6)


class TestNoLookaheadExecution:
    """信号在 bar 收盘产生，必须在下一根 bar 开盘才成交。"""

    def test_fill_uses_next_bar_open(self):
        n = 3125  # 引擎预热 = 4h/5m×65 = 3120 根，之后才开始跑信号
        eng = make_engine()
        ms0 = int(BASE.timestamp() * 1000)
        df = pd.DataFrame({
            "ts_open_ms": [ms0 + i * IV_MS for i in range(n)],
            "ts_open": [BASE + timedelta(minutes=5 * i) for i in range(n)],
            "open": [2000.0] * n, "high": [2010.0] * n,
            "low": [1990.0] * n, "close": [2000.0] * n, "volume": [10.0] * n,
        })
        # 第一根可交易 bar 的 open 故意设为 1900：若按当根成交就会拿到这个便宜价
        first = 3120
        df.loc[first, "open"] = 1900.0
        eng.bars["5m"] = df
        eng.cfg.start = BASE
        eng.cfg.end = BASE + timedelta(minutes=5 * n)

        calls = {"n": 0}

        def fn(bar, row, htf, dt):
            calls["n"] += 1
            if calls["n"] == 1:
                return sig(Action.OPEN_LONG, price_ref=bar.close)
            return None

        result = eng.run(fn)
        assert calls["n"] > 0, "信号函数从未被调用（预热长度或时间窗口不对）"
        entry_fills = [f for f in eng.ledger.fills if f.reason.value == "signal"]
        assert len(entry_fills) == 1
        # 实际成交价 = 下一根 bar 的 open(2000) 加不利滑点，而不是当根的 1900
        assert entry_fills[0].price == pytest.approx(2000.0 * (1 + 2.0 / 10000), abs=1e-6)
        assert result.final_equity > 0
