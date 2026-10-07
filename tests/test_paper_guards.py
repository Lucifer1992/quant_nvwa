"""`[paper]` 事前风控闸门的回归测试。

闸门只拦"新增敞口"，这条边界是 paper 与回测可比的前提，所以每一条都成对验证：
被拦的开仓必须留下 `rejected` 事件且不建仓，已有持仓的出场必须照常执行。

覆盖：
  1. 行情滞后 / 日亏 / 单品种保证金占比 / 名义上限 / 每日试仓次数 —— 持续型否决，不冷却
  2. 回撤与连亏 —— 冷却型闸门，写入 `paper_accounts.pause_until_ms`
  3. 冷却期内不重复评估其余闸门，到期时把净值峰值重新锚到当前净值
  4. 闸门状态跨进程持久化（重启不会忘掉"今天已经亏了 3%"）
  5. 老库缺 peak_equity / pause_until_ms 两列时自动补列
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.backtest.position import PositionLedger, TradeRecord
from quant.core.contracts import Action, Bar, Mode, RiskDecision, Signal, SymbolSpec
from quant.core.timeframe import to_ms
from quant.data.store import Store
from quant.paper.config import GuardThresholds, PaperConfig
from quant.paper.guards import EntryGuards, day_start_ms
from quant.paper.state import KIND_OPEN, PaperState, signal_row, trade_row

BASE = datetime(2026, 6, 10, 3, 0, tzinfo=UTC)  # 北京时间 11:00，同一自然日中段
STEP_MS = 300_000
SYM = "ETHUSDT"
STRAT = "gate_test"


def make_spec() -> SymbolSpec:
    return SymbolSpec(
        symbol=SYM, status="TRADING",
        tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"), min_notional=5.0,
        maint_margin_rate=0.025, liquidation_fee_rate=0.01,
        max_leverage=20.0, funding_interval_hours=8.0,
        price_precision=2, qty_precision=3, fetched_at=BASE,
    )


def make_cfg(**kw) -> PaperConfig:
    defaults = {
        "symbol": SYM, "base_interval": "5m", "initial_equity": 100.0, "strategy_name": STRAT,
        "sl_atr_mult": 2.0, "tp_atr_mult": 3.5, "time_stop_bars": 96,
    }
    defaults.update(kw)
    return PaperConfig(**defaults)


def bar_at(i: int, o=2000.0, h=2010.0, l=1990.0, c=2000.0) -> Bar:
    t = BASE + timedelta(milliseconds=STEP_MS * i)
    return Bar(
        symbol=SYM, interval="5m", ts_open=t,
        ts_close=t + timedelta(milliseconds=STEP_MS - 1),
        open=o, high=h, low=l, close=c, volume=10.0,
        quote_volume=10.0 * o, taker_buy_volume=5.0, is_closed=True,
    )


def sig(action: Action, price_ref=2000.0, ts: datetime | None = None) -> Signal:
    return Signal(
        signal_id=f"t_{action.value}_{price_ref}", ts_signal=ts or BASE, symbol=SYM,
        timeframe="5m", strategy=STRAT, action=action, price_ref=price_ref,
        strength=1.0, reason="test", mode=Mode.TEST, meta={"atr": 20.0},
    )


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "gate.db")
    yield s
    s.close()


def make_guards(store, th: GuardThresholds | None = None, **cfg_kw):
    """建一套（账本 + 闸门 + 状态）：`bind` 已经把否决回调挂进撮合层。"""
    cfg = make_cfg(**cfg_kw)
    state = PaperState(store, SYM, STRAT)
    ledger = PositionLedger(cfg, make_spec())
    guards = EntryGuards(state, th or GuardThresholds()).bind(ledger)
    return guards, ledger, state, cfg


def open_long(ledger, price=2000.0) -> None:
    """建仓时临时摘掉否决回调：闸门只该拦"新增敞口"，测试里要造出已有持仓。"""
    fn = ledger.approve_fn
    ledger.approve_fn = None
    events = ledger.execute_signal(sig(Action.OPEN_LONG, price), bar_at(0), pd.Series({"atr": 20.0}))
    ledger.approve_fn = fn
    assert ledger.position is not None, events


def add_trade(store, *, pnl: float, exit_ms: int, reason: str = "sl") -> None:
    tr = TradeRecord(
        entry_time=BASE, exit_time=datetime.fromtimestamp(exit_ms / 1000, tz=UTC),
        side="long", entry_price=2000.0, exit_price=1900.0, qty=0.1, pnl=pnl, fee=0.1,
        reason=reason, margin_usdt=10.0,
    )
    store.paper_trade(trade_row(tr, f"p{exit_ms}", SYM, STRAT))


class TestPersistentVetoes:
    """持续型否决：只挡新开仓，闸门解除后立刻可以试仓，不写冷却字段。"""

    def test_stale_market_blocks_and_alerts(self, store):
        guards, _led, _, _ = make_guards(store)
        rep = guards.refresh(now_ms=to_ms(BASE), data_gap_ms=6 * 60_000)
        assert rep.blocked and "行情滞后" in rep.blocked[0]
        assert len(rep.alerts) == 1 and rep.alerts[0][0].startswith("stale:")
        assert rep.pause_until_ms == 0, "滞后不该触发冷却"
        assert guards.describe() == "；".join(rep.blocked)

    def test_fresh_market_passes(self, store):
        guards, _led, _, _ = make_guards(store)
        rep = guards.refresh(now_ms=to_ms(BASE), data_gap_ms=4 * 60_000)
        assert rep.blocked == [] and guards.describe() == "放行"

    def test_daily_loss_blocks(self, store):
        # 日亏按"当日期初净值"折算：equity 96 + 当日已实现 -4 → 期初 100，亏 4% > 3%
        guards, led, _, _ = make_guards(store, initial_equity=100.0)
        led.equity = 96.0
        add_trade(store, pnl=-4.0, exit_ms=to_ms(BASE))
        rep = guards.refresh(now_ms=to_ms(BASE), data_gap_ms=0)
        assert any("日亏上限" in r for r in rep.blocked), rep.blocked
        assert rep.pause_until_ms == 0

    def test_yesterday_loss_does_not_block(self, store):
        guards, led, _, _ = make_guards(store)
        led.equity = 96.0
        add_trade(store, pnl=-4.0, exit_ms=day_start_ms(to_ms(BASE)) - 60_000)
        rep = guards.refresh(now_ms=to_ms(BASE), data_gap_ms=0)
        assert rep.blocked == [], "自然日按北京时间切，昨天的亏损不该计入今天"

    def test_daily_entry_cap_counts_open_signals_today(self, store):
        guards, _led, _, _ = make_guards(store, GuardThresholds(max_new_entries_per_day=2))
        for k in range(2):
            row = signal_row(
                signal_id=f"{SYM}:{STRAT}:open{k}", ts_ms=to_ms(BASE), symbol=SYM,
                strategy=STRAT, action=Action.OPEN_LONG.value, price_ref=2000.0,
                reason="r", kind=KIND_OPEN, created_at_ms=to_ms(BASE),
            )
            assert store.insert_paper_signal(row)
        rep = guards.refresh(now_ms=to_ms(BASE), data_gap_ms=0)
        assert any("达到 2 次上限" in r for r in rep.blocked), rep.blocked

    def test_entry_cap_ignores_yesterday(self, store):
        guards, _led, _, _ = make_guards(store, GuardThresholds(max_new_entries_per_day=1))
        row = signal_row(
            signal_id=f"{SYM}:{STRAT}:old", ts_ms=day_start_ms(to_ms(BASE)) - 60_000,
            symbol=SYM, strategy=STRAT, action=Action.OPEN_LONG.value, price_ref=2000.0,
            reason="r", kind=KIND_OPEN, created_at_ms=to_ms(BASE),
        )
        assert store.insert_paper_signal(row)
        assert guards.refresh(now_ms=to_ms(BASE), data_gap_ms=0).blocked == []

    def test_ymy_day_boundary_uses_beijing_time(self):
        # 北京时间 00:30 属于"今天"，UTC 16:30 属于前一天 —— 若按 UTC 切日就会算错闸门
        utc = datetime(2026, 6, 9, 16, 30, tzinfo=UTC)
        assert day_start_ms(to_ms(utc)) == to_ms(datetime(2026, 6, 9, 16, 0, tzinfo=UTC))


class TestCooldownGates:
    """冷却型闸门：回撤与连亏都会写 pause_until_ms。"""

    def test_drawdown_pauses(self, store):
        guards, led, _, _ = make_guards(store)
        led.peak_equity = 100.0
        led.equity = 84.0  # 回撤 16% > 15%
        rep = guards.refresh(now_ms=to_ms(BASE), data_gap_ms=0)
        assert rep.pause_until_ms == to_ms(BASE) + 24 * 3_600_000
        assert any("回撤" in r for r in rep.blocked)
        assert rep.alerts and rep.alerts[0][0].startswith("drawdown:")

    def test_losing_streak_pauses(self, store):
        guards, _led, _, _ = make_guards(store, GuardThresholds(max_consecutive_losses=3))
        for k in range(3):
            add_trade(store, pnl=-1.0, exit_ms=to_ms(BASE) + k * STEP_MS)
        rep = guards.refresh(now_ms=to_ms(BASE), data_gap_ms=0)
        assert rep.pause_until_ms > to_ms(BASE)
        assert any("连续亏损 3 笔" in r for r in rep.blocked)

    def test_win_breaks_the_streak(self, store):
        guards, _led, _, _ = make_guards(store, GuardThresholds(max_consecutive_losses=3))
        add_trade(store, pnl=-1.0, exit_ms=to_ms(BASE))
        add_trade(store, pnl=-1.0, exit_ms=to_ms(BASE) + STEP_MS)
        add_trade(store, pnl=+1.0, exit_ms=to_ms(BASE) + 2 * STEP_MS)
        add_trade(store, pnl=-1.0, exit_ms=to_ms(BASE) + 3 * STEP_MS)
        assert guards.refresh(now_ms=to_ms(BASE), data_gap_ms=0).pause_until_ms == 0

    def test_cooldown_window_does_not_re_evaluate(self, store):
        """冷却期内连亏笔数依然成立，若继续判就会每轮续期，冷却永远走不完。"""
        guards, led, state, _cfg = make_guards(store, GuardThresholds(max_consecutive_losses=1))
        add_trade(store, pnl=-1.0, exit_ms=to_ms(BASE))
        first = guards.refresh(now_ms=to_ms(BASE), data_gap_ms=0)
        assert first.pause_until_ms > 0
        state.commit(led, now_ms=to_ms(BASE), checkpoint_ms=None, dataset_version="v",
                     signals=[], trades=[], pause_until_ms=first.pause_until_ms)
        later = first.pause_until_ms - 60_000
        rep = guards.refresh(now_ms=later, data_gap_ms=0)
        assert rep.pause_until_ms == first.pause_until_ms, "冷却不该被续期"
        assert rep.alerts == [], "冷却中不该重复推卡"
        assert len(rep.blocked) == 1 and "冷却中" in rep.blocked[0], rep.blocked

    def test_cooldown_expiry_rebases_peak(self, store):
        """到期后峰值重新锚到当前净值，否则回撤闸门会在恢复的第一轮再触发一次。"""
        guards, led, state, _ = make_guards(store)
        led.peak_equity = 100.0
        led.equity = 84.0
        first = guards.refresh(now_ms=to_ms(BASE), data_gap_ms=0)
        state.commit(led, now_ms=to_ms(BASE), checkpoint_ms=None, dataset_version="v",
                     signals=[], trades=[], pause_until_ms=first.pause_until_ms)
        assert state.pause_until_ms() == first.pause_until_ms

        led.peak_equity = 100.0  # 模拟重启后从库里恢复到的旧峰值
        # 冷却期内同样的开仓必须先被拦住，才谈得上"到期恢复试仓"
        open_sig = sig(Action.OPEN_LONG)
        assert [e.kind for e in led.execute_signal(
            open_sig, bar_at(1), pd.Series({"atr": 20.0}),
        )] == ["rejected"], "冷却期间不该建仓"
        assert led.position is None

        expiring = guards.refresh(now_ms=first.pause_until_ms + 1, data_gap_ms=0)
        assert expiring.pause_until_ms == 0
        assert expiring.blocked == [], f"峰值应已重新锚定：{expiring.blocked}"
        assert led.peak_equity == led.equity
        led.execute_signal(open_sig, bar_at(2), pd.Series({"atr": 20.0}))
        assert led.position is not None, "冷却到期必须能重新试仓"

    def test_pause_survives_process_restart(self, store):
        """闸门状态存在 SQLite 里：换一个 PaperState（等价于重启进程）依然读得到冷却。"""
        guards, led, state, _ = make_guards(store)
        led.peak_equity = 100.0
        led.equity = 80.0
        rep = guards.refresh(now_ms=to_ms(BASE), data_gap_ms=0)
        state.commit(led, now_ms=to_ms(BASE), checkpoint_ms=None, dataset_version="v",
                     signals=[], trades=[], pause_until_ms=rep.pause_until_ms)

        state2 = PaperState(store, SYM, STRAT)
        assert state2.pause_until_ms() == rep.pause_until_ms
        led2 = state2.open_ledger(make_cfg(), make_spec(), None, resumed_at_ms=None)
        fresh = EntryGuards(state2, GuardThresholds()).bind(led2)
        rep2 = fresh.refresh(now_ms=to_ms(BASE), data_gap_ms=0)
        assert rep2.blocked and "冷却中" in rep2.blocked[0], "重启不该把冷却忘掉"
        assert led2.peak_equity == 100.0, "峰值要跨轮存活，否则重启即清零"


class TestVetoSeam:
    """闸门 → 撮合层的接缝：`approve_fn` 只否决开/加，出场路径必须与回测逐位一致。"""

    def test_blocked_gate_rejects_open_without_position(self, store):
        guards, led, _, _ = make_guards(store)
        guards.blocked = ["测试用否决"]
        events = led.execute_signal(sig(Action.OPEN_LONG), bar_at(0), pd.Series({"atr": 20.0}))
        assert [e.kind for e in events] == ["rejected"]
        assert led.position is None, "被否决的开仓绝不能建仓"
        assert events[0].decision.approved is False
        assert events[0].decision.reject_reasons == ("测试用否决",)

    def test_exit_is_never_vetoed(self, store):
        """已有持仓的平仓信号即使闸门全关也照常执行 —— 否则风控触发时刻的偏差会被当成策略差异。"""
        guards, led, _, _ = make_guards(store)
        open_long(led)
        pos_qty = led.position.qty
        guards.blocked = ["测试用否决"]
        events = led.execute_signal(
            sig(Action.CLOSE_LONG), bar_at(1), pd.Series({"atr": 20.0}),
        )
        assert [e.kind for e in events] == ["close"]
        assert led.position is None and led.trades[-1].qty == pos_qty

    def test_add_is_vetoed(self, store):
        """加仓同样是"新增敞口"：触发条件满足也要被闸门挡住。"""
        guards, led, _, _ = make_guards(store)
        open_long(led)
        guards.blocked = ["测试用否决"]
        events = led.execute_signal(
            sig(Action.ADD_TO_LONG), bar_at(1, c=2100.0),
            pd.Series({"atr": 20.0, "swing_high": 2050.0}),
        )
        assert [e.kind for e in events] == ["rejected"]
        assert led.position.add_count == 0

    def test_backtest_ledger_has_no_veto_hook(self, store):
        """回测侧 approve_fn 恒为 None：同源不等于同参数，纸面才需要事前闸门。"""
        led = PositionLedger(make_cfg(), make_spec())
        assert led.approve_fn is None

    def test_margin_cap_per_symbol(self, store):
        guards, _led, _, _ = make_guards(store, GuardThresholds(max_margin_per_symbol_pct=0.10))
        dec = RiskDecision(
            signal=sig(Action.OPEN_LONG), approved=True, qty=0.1, notional_usdt=200.0,
            leverage=10.0, margin_usdt=20.0, equity_usdt=100.0,
        )
        assert guards.vetoes(dec), "保证金 20 > 净值 100 的 10%"
        assert not guards.vetoes(
            RiskDecision(signal=dec.signal, approved=True, qty=0.01, notional_usdt=100.0,
                         leverage=10.0, margin_usdt=10.0, equity_usdt=100.0),
        )

    def test_existing_margin_counts_toward_the_cap(self, store):
        """加仓时已用保证金要一起算，否则两次各 10% 会合成 20% 敞口。"""
        guards, led, _, _ = make_guards(store, GuardThresholds(max_margin_per_symbol_pct=0.20))
        open_long(led)
        led.approve_fn = guards.vetoes
        used = led.position.margin_usdt
        # 单品种净值 100 → 上限 20；现有仓位已经占掉 used，再加 20 一定超限
        dec = RiskDecision(
            signal=sig(Action.ADD_TO_LONG), approved=True, qty=0.1, notional_usdt=200.0,
            leverage=10.0, margin_usdt=20.0, equity_usdt=100.0,
        )
        reasons = guards.vetoes(dec)
        assert reasons and f"{used + 20:.2f}" in reasons[0]

    def test_notional_cap(self, store):
        guards, _led, _, _ = make_guards(store, GuardThresholds(max_notional_usdt=500.0))
        reasons = guards.vetoes(RiskDecision(
            signal=sig(Action.OPEN_LONG), approved=True, qty=1.0, notional_usdt=2000.0,
            leverage=10.0, margin_usdt=200.0, equity_usdt=10000.0,
        ))
        assert any("名义" in r for r in reasons)

    def test_close_action_skips_cap_checks(self, store):
        """出场决策不带保证金语义，不该被 caps 判据误伤。"""
        guards, _led, _, _ = make_guards(store, GuardThresholds(max_notional_usdt=1.0))
        dec = RiskDecision(
            signal=sig(Action.CLOSE_LONG), approved=True, qty=0.1, notional_usdt=9e9,
            leverage=10.0, margin_usdt=9e9, equity_usdt=100.0,
        )
        assert guards.vetoes(dec) == []

    def test_veto_keeps_original_reasons(self, store):
        """`veto` 是追加而不是覆盖：风控自身的拒单理由不能被闸门抹掉。"""
        from quant.backtest.position import veto
        dec = RiskDecision(signal=sig(Action.OPEN_LONG), approved=False, reject_reasons=("原始拒单",))
        out = veto(dec, ["闸门否决"])
        assert out.reject_reasons == ("原始拒单", "闸门否决")
        assert out.approved is False and dec.reject_reasons == ("原始拒单",)


def legacy_paper_accounts_ddl() -> str:
    """`peak_equity` / `pause_until_ms` 还不存在时期的建表语句，用来模拟老库。"""
    return (
        "CREATE TABLE paper_accounts (symbol TEXT NOT NULL, strategy TEXT NOT NULL,"
        " initial_equity REAL NOT NULL, cash REAL NOT NULL, equity REAL NOT NULL,"
        " equity_open REAL NOT NULL, bars_since_entry INTEGER NOT NULL DEFAULT 0,"
        " funding_since_entry REAL NOT NULL DEFAULT 0, funding_ptr INTEGER NOT NULL DEFAULT 0,"
        " add_count_total INTEGER NOT NULL DEFAULT 0, add_skip_count INTEGER NOT NULL DEFAULT 0,"
        " updated_at_ms INTEGER NOT NULL, PRIMARY KEY (symbol, strategy))"
    )


def make_legacy_db(path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(legacy_paper_accounts_ddl())
    conn.execute(
        "INSERT INTO paper_accounts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (SYM, STRAT, 100.0, 90.0, 90.0, 90.0, 3, 0.0, 0, 1, 0, to_ms(BASE)),
    )
    conn.commit()
    conn.close()


class TestStorePersistence:
    def test_paper_account_columns_migrate(self, tmp_path):
        """老库（没有 peak_equity / pause_until_ms）必须被 `Store.__init__` 补列。"""
        db = tmp_path / "legacy.db"
        make_legacy_db(db)

        with Store(db) as store:
            cols = {r[1] for r in store._conn.execute("PRAGMA table_info(paper_accounts)")}
            assert {"peak_equity", "pause_until_ms"} <= cols
            acct = store.paper_account(SYM, STRAT)
            assert acct["peak_equity"] == 0 and acct["pause_until_ms"] == 0

            # 老库的峰值是 0（DEFAULT），恢复时不能沿用：0 峰值会让回撤闸门永远算不出回撤
            led = PaperState(store, SYM, STRAT).open_ledger(
                make_cfg(), make_spec(), None, resumed_at_ms=None,
            )
            assert led.peak_equity == 90.0
            assert led.cash == 90.0 and led.add_count_total == 1

    def test_second_open_is_no_op(self, tmp_path):
        """重复打开同一个老库不能重复 ALTER：补列是幂等的，否则第二次启动直接崩。"""
        db = tmp_path / "twice.db"
        make_legacy_db(db)

        # 写入走 `commit()` 的生产路径：手工拼行会漏掉 NOT NULL 列，测不出真实行为
        with Store(db) as first:
            state = PaperState(first, SYM, STRAT)
            led = state.open_ledger(make_cfg(), make_spec(), None, resumed_at_ms=None)
            led.peak_equity = 120.0
            state.commit(led, now_ms=to_ms(BASE), checkpoint_ms=None, dataset_version="v",
                         signals=[], trades=[], pause_until_ms=7)

        with Store(db) as second:
            row = second.paper_account(SYM, STRAT)
            assert row["peak_equity"] == 120.0 and row["pause_until_ms"] == 7

    def test_commit_persists_peak_and_pause(self, store):
        """NOT NULL 列必须由 `dump_state()` / `commit()` 无条件 supplying，缺键会整笔事务失败。"""
        _guards, led, state, _ = make_guards(store)
        led.peak_equity = 130.0
        state.commit(led, now_ms=to_ms(BASE), checkpoint_ms=None, dataset_version="v",
                     signals=[], trades=[], pause_until_ms=999)
        row = store.paper_account(SYM, STRAT)
        assert row["peak_equity"] == 130.0 and row["pause_until_ms"] == 999

        fresh = PaperState(store, SYM, STRAT).open_ledger(
            make_cfg(), make_spec(), None, resumed_at_ms=None,
        )
        assert fresh.peak_equity == 130.0, "峰值必须跨轮存活，否则回撤闸门重启即失效"

    def test_migrations_match_the_schema(self):
        """`SCHEMA` 只有 CREATE TABLE IF NOT EXISTS：PAPER_MIGRATIONS 里的列必须也在新建库的
        DDL 里，否则新库有列、老库没列，两边的查询行为会静默分叉。"""
        from quant.data.store import PAPER_MIGRATIONS, PAPER_TABLE_COLS, SCHEMA

        added = [c.split()[0] for c in PAPER_MIGRATIONS["paper_accounts"]]
        for col in added:
            assert col in PAPER_TABLE_COLS["paper_accounts"], f"{col} 没进列清单，写入会被丢弃"
            assert f"{col} " in SCHEMA, f"{col} 只在迁移里声明，新建库不会带上"

