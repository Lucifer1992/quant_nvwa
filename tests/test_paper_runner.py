"""`PaperRunner` 一轮驱动的回归测试：跳过条件、成交落库、重启等价、与回测逐位一致。

纸面账户的全部价值在于"能被当成回测来读"，所以四条主线各自成对验证：
  1. `PaperConfig` 与 `BacktestConfig` 字段/默认值对齐 —— `paper/config.py` 的模块注释
     把这条约束交给了本文件，不靠人肉 review
  2. 数据缺口 / 折模型不可用 / 无本地数据 —— 整轮跳过并推去重告警，绝不带残缺窗口出信号
  3. 事件 → `paper_signals` / `paper_trades` / `paper_accounts` / `paper_checkpoints` 落库，
     确定性主键让重跑不产生第二条流水
  4. 断点续跑（observe 重放）== 一次跑完 == 同参数 `BacktestEngine` 的逐笔流水
"""
from __future__ import annotations

import math
import sys
from dataclasses import MISSING, fields, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.backtest.engine import BacktestConfig, BacktestEngine
from quant.core.contracts import Action, Bar, FundingBar, Mode, Signal, SymbolSpec
from quant.core.timeframe import to_dt, to_ms
from quant.data.store import Store
from quant.infra.settings import Settings
from quant.notify.feishu import FeishuError
from quant.paper.config import PaperConfig, PaperRuntime, execution_params
from quant.paper.runner import PaperRunner
from quant.risk.exchange_meta import spec_to_json
from quant.strategy.base import BaseStrategy

SYM = "ETHUSDT"
STRAT = "runner_test"
BASE = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
STEP = 300_000
DAY_MS = 86_400_000
N_BARS = 700
FIRST_LIVE = 60  # 之前的 bar 只用于指标预热与 observe
REPLAY_BARS = 50  # observe 窗口：必须整段落在已回填的数据里，否则按数据缺口跳过


def open_ms(i: int) -> int:
    return to_ms(BASE) + STEP * i


def close_ms(i: int) -> int:
    return open_ms(i) + STEP - 1


def replay_days(bars: int) -> float:
    return bars * STEP / DAY_MS


def make_spec() -> SymbolSpec:
    return SymbolSpec(
        symbol=SYM, status="TRADING",
        tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"), min_notional=5.0,
        maint_margin_rate=0.025, liquidation_fee_rate=0.01,
        max_leverage=20.0, funding_interval_hours=8.0,
        price_precision=2, qty_precision=3, fetched_at=datetime.now(UTC),
    )


def make_bars(n: int = N_BARS) -> list[Bar]:
    """正弦叠加缓慢上倾：既有 EMA 上穿开仓，也有回落打止损，出场覆盖 SL/TP/移动/时间。"""
    out: list[Bar] = []
    prev = 2000.0
    for i in range(n):
        c = 2000.0 + 80.0 * math.sin(i / 25.0) + 0.05 * i
        o = prev
        out.append(Bar(
            symbol=SYM, interval="5m",
            ts_open=to_dt(open_ms(i)), ts_close=to_dt(close_ms(i)),
            open=round(o, 2), high=round(max(o, c) + 8.0, 2), low=round(min(o, c) - 8.0, 2),
            close=round(c, 2), volume=10.0, quote_volume=10.0 * o, taker_buy_volume=5.0,
            is_closed=True,
        ))
        prev = c
    return out


def make_cfg(**kw) -> PaperConfig:
    defaults: dict[str, Any] = {
        "symbol": SYM, "base_interval": "5m", "initial_equity": 100.0, "strategy_name": STRAT,
        "time_stop_bars": 48,
    }
    defaults.update(kw)
    return PaperConfig(**defaults)


def make_runtime(**kw) -> PaperRuntime:
    # resend_cooldown_min=0：补发队列立刻可重试，否则测试要推进 10 分钟以上的 now_ms
    return replace(
        PaperRuntime(replay_days=replay_days(REPLAY_BARS), resend_cooldown_min=0.0), **kw
    )


class StubStrategy(BaseStrategy):
    """EMA20 上穿开多 + 开仓后冷却 N 根：状态只依赖 bar 序列，重放才可能等价。"""

    name = STRAT
    required_intervals = ("5m",)

    def __init__(self, config: Any = None, *, first_ms: int | None = None, cooldown_bars: int = 30):
        super().__init__(config)
        self.first_ms = open_ms(FIRST_LIVE) if first_ms is None else first_ms
        self.cooldown_ms = cooldown_bars * STEP
        self.last_entry_ms: int | None = None
        self.calls = 0

    def reset(self) -> None:
        self.last_entry_ms = None
        self.calls = 0

    def on_bar(self, bar: Bar, row: pd.Series, htf: dict[str, pd.Series], dt: datetime):
        self.calls += 1
        ts = int(row["ts_open_ms"])
        if ts < self.first_ms:
            return None
        atr, prev_close, prev_ema = row["atr"], row.get("prev_close"), row.get("prev_ema_20")
        if pd.isna(atr) or pd.isna(prev_close) or pd.isna(prev_ema):
            return None
        if self.last_entry_ms is not None and ts - self.last_entry_ms < self.cooldown_ms:
            return None
        if bar.close > row["ema_20"] and float(prev_close) <= float(prev_ema):
            self.last_entry_ms = ts
            return self._signal(bar, float(atr), "ema20 上穿")
        return None

    def _signal(self, bar: Bar, atr: float, reason: str) -> Signal:
        ts = int(bar.ts_open.timestamp() * 1000)
        return Signal(
            signal_id=f"rt_{ts}", ts_signal=bar.ts_close, symbol=SYM, timeframe="5m",
            strategy=STRAT, action=Action.OPEN_LONG, price_ref=round(bar.close, 2),
            strength=1.0, reason=reason, mode=Mode.TEST, meta={"atr": atr},
        )


class EmitOnce(StubStrategy):
    """只在指定那根 bar 出一次信号，用来单独验证 pending 的取舍。"""

    def __init__(self, *, at_bar: int, **kw):
        super().__init__(**kw)
        self.at_ms = open_ms(at_bar)

    def on_bar(self, bar, row, htf, dt):
        if int(row["ts_open_ms"]) != self.at_ms:
            return None
        return self._signal(bar, float(row["atr"]) if not pd.isna(row["atr"]) else 10.0, "test")


def seed_store(tmp_path, name: str = "paper.db") -> Store:
    s = Store(tmp_path / name)
    s.spec_put(SYM, spec_to_json(make_spec()))
    s.upsert_bars(make_bars())
    return s


@pytest.fixture()
def store(tmp_path):
    s = seed_store(tmp_path)
    yield s
    s.close()


def make_runner(store, *, cfg=None, runtime=None, notifier=None, strategy=None) -> PaperRunner:
    return PaperRunner(
        store=store, symbol=SYM, strategy=strategy or StubStrategy(),
        cfg=cfg or make_cfg(), runtime=runtime or make_runtime(), notifier=notifier,
    )


def seed_checkpoint(store, bar_index: int) -> int:
    """模拟"进程已跑到这根 bar 为止"：其后的 bar 才走 live。返回其收盘时刻。"""
    ms = open_ms(bar_index)
    store.save_paper_checkpoint({
        "symbol": SYM, "strategy": STRAT, "last_processed_ts_ms": ms,
        "dataset_version": "seed", "updated_at_ms": ms,
    })
    return ms


def signal_rows(store, kind: str | None = None) -> list[dict]:
    sql = "SELECT * FROM paper_signals WHERE symbol=? AND strategy=?"
    params: list[Any] = [SYM, STRAT]
    if kind:
        sql += " AND kind=?"
        params.append(kind)
    return store._select(sql + " ORDER BY ts_signal_ms", params)


def trade_rows(store) -> list[dict]:
    return store.paper_trades(SYM, STRAT)


def trade_key(r: dict) -> tuple:
    """逐笔流水的可比键：时刻、原因、数量与盈亏都必须与回测逐位一致。"""
    return (
        int(r["entry_time_ms"]), int(r["exit_time_ms"]), str(r["reason"]),
        round(float(r["qty"]), 8), round(float(r["pnl"]), 8),
    )


# ---------------------------------------------------------------- 配置对齐


class TestConfigParity:
    def test_fields_and_defaults_match_backtest_config(self):
        bt = {f.name: f.default for f in fields(BacktestConfig)}
        pp = {f.name: f.default for f in fields(PaperConfig)}
        # 纸面账户没有回测窗口，只有断点：start/end 是唯一允许的差异
        assert set(pp) == set(bt) - {"start", "end"}
        for name, default in pp.items():
            assert bt[name] == default, f"{name} 漂移：paper={default!r} vs backtest={bt[name]!r}"

    def test_required_fields_line_up(self):
        """必填字段两侧都要必填，否则 execution_params 漏传时会静默取兜底值。"""
        bt_required = {f.name for f in fields(BacktestConfig) if f.default is MISSING} - {"start", "end"}
        pp_required = {f.name for f in fields(PaperConfig) if f.default is MISSING}
        assert pp_required == bt_required == {"symbol", "base_interval", "initial_equity"}

    def test_execution_params_equals_bare_backtest(self):
        """不带开关的 `quant backtest` 用的就是字段默认值，paper 必须一个都不差。"""
        s = Settings(root=Path("."), settings={}, symbols={}, risk={}, notify={})
        cfg = execution_params(s, SYM, "5m", STRAT, {})
        assert cfg == make_cfg(initial_equity=100.0, time_stop_bars=96)

    def test_execution_params_applies_strategy_defaults(self):
        """策略声明的 `engine_defaults` 覆盖字段默认值；不认识的字段直接丢弃。"""
        s = Settings(root=Path("."), settings={}, symbols={}, risk={}, notify={})
        cfg = execution_params(s, SYM, "5m", STRAT, {"sl_atr_mult": 2.5, "not_a_field": 1.0})
        assert cfg.sl_atr_mult == 2.5 and cfg.tp_atr_mult == 3.0
        assert not hasattr(cfg, "not_a_field")

    def test_execution_params_reads_initial_equity_from_risk_yaml(self):
        s = Settings(root=Path("."), settings={}, symbols={},
                     risk={"account": {"initial_equity_usdt": 250}}, notify={})
        assert execution_params(s, SYM, "5m", STRAT, {}).initial_equity == 250.0

    def test_execution_params_warns_on_yaml_drift(self, caplog):
        """`risk.yaml` 改了口径而回测 CLI 从不读它，两边结果就不可比 —— 必须喊出来。"""
        s = Settings(root=Path("."), settings={}, symbols={},
                     risk={"stops": {"sl_atr_mult": 4.0}, "costs": {"taker_fee_rate": 0.001}},
                     notify={})
        with caplog.at_level("WARNING", logger="quant.paper.config"):
            execution_params(s, SYM, "5m", STRAT, {})
        text = " ".join(r.message for r in caplog.records)
        assert "sl_atr_mult: 生效=1.5 vs risk.yaml=4.0" in text
        assert "taker_fee" in text

    def test_strategy_override_is_not_drift(self, caplog):
        """策略自己声明的值就是生效值，不该被当成与 yaml 的冲突。"""
        s = Settings(root=Path("."), settings={}, symbols={},
                     risk={"stops": {"sl_atr_mult": 4.0}}, notify={})
        with caplog.at_level("WARNING", logger="quant.paper.config"):
            execution_params(s, SYM, "5m", STRAT, {"sl_atr_mult": 4.0})
        assert "sl_atr_mult" not in " ".join(r.message for r in caplog.records)


# ---------------------------------------------------------------- 整轮跳过


class TestRoundSkipping:
    def test_no_local_data(self, tmp_path):
        empty = Store(tmp_path / "empty.db")
        empty.spec_put(SYM, spec_to_json(make_spec()))
        try:
            rep = make_runner(empty).run_round(now_ms=to_ms(datetime.now(UTC)))
            assert rep.skipped and "quant backfill" in rep.skipped
        finally:
            empty.close()

    def test_gap_skips_and_alerts_once(self, store):
        """残缺窗口里的指标与 bar 间距都不可信，出信号就是拿噪声下注。"""
        seed_checkpoint(store, 20)
        missing = open_ms(30)
        store._conn.execute("DELETE FROM bars WHERE ts_open_ms=?", (missing,))
        store._conn.commit()

        runner = make_runner(store, runtime=make_runtime(replay_days=replay_days(15)))
        rep = runner.run_round(now_ms=close_ms(N_BARS - 1))
        assert rep.skipped and "5m 缺" in rep.skipped
        assert rep.live_bars == 0 and rep.events == []
        alerts = signal_rows(store, "alert")
        assert len(alerts) == 1 and "数据缺口" in alerts[0]["note"]

        # 同一缺口下一轮不重复推卡：告警 id 带观测窗起点，主键去重
        rep2 = runner.run_round(now_ms=close_ms(N_BARS - 1) + STEP)
        assert rep2.skipped
        assert len(signal_rows(store, "alert")) == 1
        assert store.paper_checkpoint_ms(SYM, STRAT) == open_ms(20), "跳过不该推进断点"

    def test_htf_gap_counts_as_gap(self, store):
        """高周期缺柱会让 HTF 游标错位，即使 5m 完整也必须跳过。"""
        seed_checkpoint(store, 20)

        class WithHtf(StubStrategy):
            required_intervals = ("5m", "1h")

        rep = make_runner(store, runtime=make_runtime(replay_days=replay_days(15)),
                          strategy=WithHtf()).run_round(now_ms=close_ms(N_BARS - 1))
        assert rep.skipped and "1h" in rep.skipped

    def test_no_folds_refuses(self, store):
        class ML(StubStrategy):
            def folds(self):
                return []

        rep = make_runner(store, strategy=ML()).run_round(now_ms=close_ms(N_BARS - 1))
        assert rep.skipped and "折模型" in rep.skipped

    def test_fold_load_error_refuses(self, store):
        class ML(StubStrategy):
            def folds(self):
                raise FileNotFoundError("models/runner_test 不存在")

        rep = make_runner(store, strategy=ML()).run_round(now_ms=close_ms(N_BARS - 1))
        assert rep.skipped and "折模型加载失败" in rep.skipped
        assert signal_rows(store, "alert"), "拒绝服务必须推告警卡"
        assert store.paper_checkpoint_ms(SYM, STRAT) is None

    def test_out_of_sample_window_refuses(self, store):
        """折窗口是"模型过期"的判据；拿 dataset_version 当闸门等于永远拒绝服务。"""
        class ML(StubStrategy):
            def folds(self):
                return [SimpleNamespace(test_start_ms=0, test_end_ms=open_ms(10))]

        rep = make_runner(store, strategy=ML()).run_round(now_ms=close_ms(N_BARS - 1))
        assert rep.skipped and "续训" in rep.skipped

    def test_fold_window_covering_now_runs(self, store):
        ref = close_ms(N_BARS - 1)

        class ML(StubStrategy):
            def folds(self):
                return [SimpleNamespace(test_start_ms=ref - 10 * STEP, test_end_ms=ref + STEP)]

        seed_checkpoint(store, FIRST_LIVE - 1)
        rep = make_runner(store, strategy=ML()).run_round(now_ms=ref)
        assert rep.skipped is None and rep.live_bars > 0


# ---------------------------------------------------------------- 一轮的执行


class TestRoundExecution:
    def test_first_round_without_checkpoint_only_observes(self, store):
        """首轮没有断点：整段窗口都是 observe，账本一行不写，否则重启会把历史重放成新成交。"""
        # 窗口取整天（2 天 = 576 根），避开 replay_ms 的浮点边界把首根 bar 排除在外
        rep = make_runner(store, runtime=make_runtime(replay_days=2.0)).run_round(
            now_ms=close_ms(N_BARS - 1)
        )
        assert rep.skipped is None
        assert rep.live_bars == 0 and rep.events == []
        assert rep.observed_bars == 576
        assert trade_rows(store) == [] and signal_rows(store, "open") == []
        assert store.paper_checkpoint_ms(SYM, STRAT) == open_ms(N_BARS - 1)
        assert rep.equity == 100.0
        assert rep.gate == "放行", rep.gate

    def test_first_round_without_checkpoint_with_funding_rows(self, store):
        """首轮没有断点：`seek_funding(None)` 直接喂给 `searchsorted` 会抛 TypeError（实跑踩过）。

        资金费不计入成本，但加载照旧（ML 特征要用），所以这条路径必然被走到。
        """
        store.upsert_funding([
            FundingBar(symbol=SYM, funding_time=to_dt(open_ms(i) + STEP), funding_rate=0.0001,
                       mark_price=2000.0)
            for i in range(0, N_BARS, 96)
        ])
        runner = make_runner(store, runtime=make_runtime(replay_days=2.0))
        rep = runner.run_round(now_ms=close_ms(N_BARS - 1))
        assert rep.skipped is None and rep.live_bars == 0
        assert runner.ledger._funding_ptr == 0, "没有断点就该从第一条结算开始定位"

        # 有断点时指针必须跳到断点之后，否则窗口一滑动就把已计提的结算再记一遍
        seed_checkpoint(store, 300)
        second = make_runner(store)
        second.run_round(now_ms=close_ms(N_BARS - 1))
        times = [int(t) for t in second.ledger.funding["funded_at_ms"]]
        ptr = second.ledger._funding_ptr
        assert ptr > 0 and all(t > open_ms(300) + STEP for t in times[ptr:])
        assert all(t <= open_ms(300) + STEP for t in times[:ptr])

    def test_live_round_fills_and_persists(self, store):
        seed_checkpoint(store, FIRST_LIVE - 1)
        runner = make_runner(store)
        rep = runner.run_round(now_ms=close_ms(N_BARS - 1))

        assert rep.live_bars == N_BARS - FIRST_LIVE
        opens = signal_rows(store, "open")
        assert opens, rep.events
        # 开仓卡 id 由时刻派生：重跑必须撞同一个主键，才能天然去重
        assert {r["signal_id"] for r in opens} == {
            f"{SYM}:{STRAT}:rt_{int(r['ts_signal_ms']) - STEP + 1}:open" for r in opens
        }
        assert all(r["qty"] and r["margin_usdt"] and r["stop_loss"] for r in opens)

        closed = trade_rows(store)
        assert closed
        assert len(closed) == len(runner.ledger.trades)
        acct = store.paper_account(SYM, STRAT)
        assert acct["equity"] == pytest.approx(runner.ledger.equity)
        assert acct["updated_at_ms"] == close_ms(N_BARS - 1)
        assert store.paper_checkpoint_ms(SYM, STRAT) == open_ms(N_BARS - 1)
        # 会计恒等式：净值增长 == 逐笔盈亏之和（期末空仓时）
        if runner.ledger.position is None:
            assert runner.ledger.equity - 100.0 == pytest.approx(
                sum(float(r["pnl"]) for r in closed), abs=1e-6)

    def test_cards_persisted_as_sent_in_dry_run(self, store):
        """`--no-push` 下 notifier=None：落库即视为已处理，不能堆一条永远补不掉的队列。"""
        seed_checkpoint(store, FIRST_LIVE - 1)
        rep = make_runner(store).run_round(now_ms=close_ms(N_BARS - 1))
        rows = signal_rows(store)
        assert rows and rep.sent_ok == len(rows) and rep.sent_failed == 0
        assert all(r["sent_ok"] == 1 for r in rows)
        assert all(r["payload_json"] for r in rows), "卡片快照必须与信号同批落库"
        assert store.paper_signals_unsent(SYM, STRAT) == []

    def test_multiple_exits_of_one_position_stay_distinct(self, store):
        """一笔仓位可以留下 TP1 部分止盈 + 全平多条流水，主键必须带原因与时刻。"""
        seed_checkpoint(store, FIRST_LIVE - 1)
        runner = make_runner(store, cfg=make_cfg(tp1_r=0.3, tp1_close_frac=0.5))
        runner.run_round(now_ms=close_ms(N_BARS - 1))

        rows = trade_rows(store)
        assert rows
        assert len({r["trade_id"] for r in rows}) == len(rows), "trade_id 撞车会覆盖前一条流水"
        grouped: dict[str, list[dict]] = {}
        for r in rows:
            grouped.setdefault(str(r["position_id"]), []).append(r)
        assert any(len(g) > 1 for g in grouped.values()), "TP1 开启后应当出现同仓多条流水"
        for g in grouped.values():
            assert len({(r["reason"], r["exit_time_ms"]) for r in g}) == len(g)

    def test_add_skipped_does_not_push_a_card(self, store):
        """金字塔未达触发条件时每根 bar 都会产生一条，推出去就是把消息流变成日志流。"""
        seed_checkpoint(store, FIRST_LIVE - 1)
        make_runner(store).run_round(now_ms=close_ms(N_BARS - 1))
        assert "add_skipped" not in {r["kind"] for r in signal_rows(store)}

    def test_failed_card_is_retried_in_the_same_round(self, store):
        """补发队列在同一轮末尾就会重试一次：webhook 抖一下不该等到下个节拍。"""
        class OnceBroken:
            def __init__(self):
                self.calls = 0

            def send_payload(self, payload):
                self.calls += 1
                if self.calls == 1:
                    raise FeishuError("429 too many requests")
                return {"code": 0}

        seed_checkpoint(store, FIRST_LIVE - 1)
        rep = make_runner(store, notifier=OnceBroken()).run_round(now_ms=close_ms(N_BARS - 1))
        assert rep.sent_failed == 1 and rep.resent == 1
        assert store.paper_signals_unsent(SYM, STRAT) == []

    def test_persistent_failure_waits_out_the_resend_cooldown(self, store):
        """持续失败不能每轮重推：`resend_cooldown_min` 之内跳过，超了才再试，
        期间卡片快照必须留在库里，否则补发出来的是事后重建的行情。"""
        class AlwaysBroken:
            def __init__(self):
                self.ok = False

            def send_payload(self, payload):
                if not self.ok:
                    raise FeishuError("webhook 关闭")
                return {"code": 0}

        notifier = AlwaysBroken()
        runtime = make_runtime(resend_cooldown_min=10.0)
        seed_checkpoint(store, FIRST_LIVE - 1)
        runner = make_runner(store, notifier=notifier, runtime=runtime)
        end = close_ms(N_BARS - 1)
        rep = runner.run_round(now_ms=end)
        assert rep.sent_failed > 1 and rep.resent == 0

        unsent = store.paper_signals_unsent(SYM, STRAT, 10, attempt_before_ms=end)
        assert unsent and all(r["payload_json"] for r in unsent)

        # 4 分钟后仍未到补发时刻
        assert runner.run_round(now_ms=end + 4 * 60_000).resent == 0
        # 12 分钟后解除限流，补发用的是当初那张卡
        notifier.ok = True
        assert runner.run_round(now_ms=end + 12 * 60_000).resent >= 1

    def test_stale_market_blocks_without_skipping_the_round(self, store):
        """行情滞后只否决新开仓，已有持仓的出场序列照跑 —— 整轮跳过会漏掉止损。"""
        seed_checkpoint(store, N_BARS - 2)
        runner = make_runner(store)
        runner.run_round(now_ms=close_ms(N_BARS - 1))
        opens = len(signal_rows(store, "open"))

        seed_checkpoint(store, FIRST_LIVE - 1)
        lagging = make_runner(store)
        rep = lagging.run_round(now_ms=close_ms(N_BARS - 1) + 10 * 60_000)
        assert rep.skipped is None
        assert "行情滞后" in rep.gate
        assert signal_rows(store, "open") == [] or len(signal_rows(store, "open")) == opens
        assert any("行情滞后" in r["note"] for r in signal_rows(store, "alert"))

    def test_pending_signal_at_breakpoint_is_not_eaten(self, store):
        """断点那根 bar 的信号本该在下一根（第一根 live bar）成交：重启不能吃掉它。"""
        seed_checkpoint(store, FIRST_LIVE)
        rep = make_runner(store, strategy=EmitOnce(at_bar=FIRST_LIVE)).run_round(
            now_ms=close_ms(N_BARS - 1)
        )
        opens = signal_rows(store, "open")
        assert len(opens) == 1, rep.events
        assert opens[0]["ts_signal_ms"] == close_ms(FIRST_LIVE)

    def test_signal_from_an_earlier_observed_bar_is_not_re_executed(self, store):
        """更早的信号上一轮已经成交过，重放时必须被覆盖掉，否则会晚一根 bar 重复开仓。"""
        seed_checkpoint(store, FIRST_LIVE)
        rep = make_runner(store, strategy=EmitOnce(at_bar=FIRST_LIVE - 3)).run_round(
            now_ms=close_ms(N_BARS - 1)
        )
        assert rep.live_bars > 0
        assert signal_rows(store, "open") == []


# ---------------------------------------------------------------- 重启与幂等


class TestRestartIdempotency:
    def test_rerunning_the_same_round_writes_nothing_new(self, store):
        seed_checkpoint(store, FIRST_LIVE - 1)
        runner = make_runner(store)
        runner.run_round(now_ms=close_ms(N_BARS - 1))
        before = (len(signal_rows(store)), len(trade_rows(store)),
                  store.paper_checkpoint_ms(SYM, STRAT))

        rep = runner.run_round(now_ms=close_ms(N_BARS - 1))
        assert rep.live_bars == 0 and rep.events == []
        after = (len(signal_rows(store)), len(trade_rows(store)),
                 store.paper_checkpoint_ms(SYM, STRAT))
        assert after == before

    def test_split_run_equals_continuous_run(self, tmp_path):
        """断点续跑（observe 重放 + 账本恢复）必须与一次跑完逐笔一致。"""
        end = close_ms(N_BARS - 1)
        half = N_BARS // 2
        replay = make_runtime(replay_days=replay_days(REPLAY_BARS))

        base = seed_store(tmp_path, "continuous.db")
        try:
            seed_checkpoint(base, FIRST_LIVE - 1)
            runner = make_runner(base, runtime=replay)
            rep = runner.run_round(now_ms=end)
            assert rep.skipped is None
            expected = [trade_key(r) for r in trade_rows(base)]
            expected_equity = runner.ledger.equity
            expected_opens = len(signal_rows(base, "open"))
        finally:
            base.close()
        assert expected, "样本里必须真的有成交，否则这条用例是空跑"

        split = seed_store(tmp_path, "split.db")
        try:
            seed_checkpoint(split, FIRST_LIVE - 1)
            rep_a = make_runner(split, runtime=replay).run_round(now_ms=close_ms(half))
            assert rep_a.skipped is None and rep_a.live_bars > 0
            assert split.paper_checkpoint_ms(SYM, STRAT) == open_ms(half)

            second = make_runner(split, runtime=replay)  # 换一个实例 = 重启进程
            rep_b = second.run_round(now_ms=end)
            assert rep_b.skipped is None
            assert [trade_key(r) for r in trade_rows(split)] == expected
            assert second.ledger.equity == pytest.approx(expected_equity, abs=1e-6)
            assert len(signal_rows(split, "open")) == expected_opens
            assert rep_b.live_bars == N_BARS - half - 1
        finally:
            split.close()


# ---------------------------------------------------------------- 与回测逐位一致


class TestBacktestParity:
    def test_paper_fills_match_backtest_trade_by_trade(self, store, monkeypatch):
        """同一份数据、同一个策略：paper 的逐笔流水必须与 `BacktestEngine` 完全一致。

        回测用 `warmup_bars()`（3120 根）跳过撮合，paper 的对应物是 checkpoint + observe
        重放。这里把回测的预热长度改成 0、窗口起点设成 paper 的第一根 live bar，两边即从
        同一根 bar 开始撮合。
        """
        from quant.backtest import engine as engine_module

        monkeypatch.setattr(engine_module, "warmup_bars", lambda: 0)
        end = N_BARS - 1
        bt_cfg = BacktestConfig(
            symbol=SYM, base_interval="5m", start=to_dt(open_ms(FIRST_LIVE)),
            end=to_dt(close_ms(end)), initial_equity=100.0, strategy_name=STRAT,
            time_stop_bars=48,
        )
        strategy = StubStrategy()
        eng = BacktestEngine(bt_cfg, {"5m": store.load_bars(SYM, "5m")}, make_spec())
        eng.run(strategy.on_bar, strategy=strategy)
        assert [t for t in eng.ledger.trades if t.reason == "flat"] == [], (
            "期末强平会让回测多出一条 paper 没有的流水，窗口要选在空仓处结束"
        )
        expected = [
            (to_ms(t.entry_time), to_ms(t.exit_time), t.reason, round(t.qty, 8), round(t.pnl, 8))
            for t in eng.ledger.trades
        ]
        assert expected, "回测没有任何成交，这条用例形同空跑"

        seed_checkpoint(store, FIRST_LIVE - 1)
        rep = make_runner(store).run_round(now_ms=close_ms(end))
        assert rep.skipped is None
        assert [trade_key(r) for r in trade_rows(store)] == expected
        assert rep.equity == pytest.approx(eng.ledger.equity, abs=1e-6)


# ---------------------------------------------------------------- 服务调度


def fake_settings(**overrides) -> Settings:
    merged: dict[str, Any] = {
        "settings": {"paper": {"replay_days": replay_days(REPLAY_BARS)}},
        "symbols": {"symbols": {SYM: {
            "enabled": True, "base_interval": "5m", "intervals": ["5m"], "strategies": [STRAT],
        }}},
        "risk": {}, "notify": {},
    }
    merged.update(overrides)
    return Settings(root=Path("."), **merged)


class TestServiceScheduling:
    def service(self, store, **kw) -> Any:
        from quant.paper.service import PaperService

        return PaperService(fake_settings(), push=False, store=store, strategies=[STRAT], **kw)

    def test_account_due_only_on_new_bar_or_staleness(self, store):
        svc = self.service(store)
        svc.runners[(SYM, STRAT)] = make_runner(store)
        runner = svc.runners[(SYM, STRAT)]
        try:
            seed_checkpoint(store, N_BARS - 1)
            latest = store.watermark_ms(SYM, "5m") + STEP
            assert svc._account_due(SYM, runner, latest) is False, (
                "没有新 bar 就不该重算整窗指标"
            )
            assert svc._account_due(SYM, runner, latest + 20 * 60_000) is True, (
                "数据滞后必须进轮，否则 stale 闸门永远得不到评估"
            )
            store._conn.execute("DELETE FROM paper_checkpoints")
            store._conn.commit()
            assert svc._account_due(SYM, runner, latest) is True, "首轮没有断点，必须进轮"
        finally:
            svc.notifier = None

    def test_one_failing_account_does_not_stop_the_others(self, store):
        from quant.paper.service import PaperService

        svc = PaperService(fake_settings(), push=False, store=store, strategies=[STRAT])

        class Boom(PaperRunner):
            def run_round(self, *, now_ms: int):
                raise RuntimeError("撮合层炸了")

        seed_checkpoint(store, FIRST_LIVE - 1)
        svc.runners = {
            (SYM, "boom"): Boom(store=store, symbol=SYM, strategy=StubStrategy(),
                                cfg=make_cfg(strategy_name="boom"), runtime=make_runtime()),
            (SYM, STRAT): make_runner(store),
        }
        reports = svc.run_cycle(now_ms=close_ms(N_BARS - 1))
        assert {r.strategy for r in reports} == {"boom", STRAT}
        failed = [r for r in reports if r.strategy == "boom"]
        assert failed[0].skipped and "执行异常" in failed[0].skipped
        ok = [r for r in reports if r.strategy == STRAT]
        assert ok[0].skipped is None and ok[0].live_bars > 0
        svc.notifier = None

    def test_status_reports_checkpoint_and_push_stats(self, store):
        from quant.paper.service import paper_status

        seed_checkpoint(store, FIRST_LIVE - 1)
        make_runner(store).run_round(now_ms=close_ms(N_BARS - 1))
        rows = paper_status(store)
        assert len(rows) == 1
        row = rows[0]
        assert row["symbol"] == SYM and row["strategy"] == STRAT
        assert row["checkpoint_ms"] == open_ms(N_BARS - 1)
        assert row["behind_bars"] == 0
        assert row["signals"] == len(signal_rows(store))
        assert row["unsent"] == 0
        assert row["closed_trades"] == len(trade_rows(store))
        assert row["realized_pnl"] == pytest.approx(
            sum(float(r["pnl"]) for r in trade_rows(store)), abs=1e-4)
