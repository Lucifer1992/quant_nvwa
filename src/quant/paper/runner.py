"""一个 (symbol, strategy) 纸面账户的"一轮"：加载数据 → 追平策略状态 → 撮合新 bar → 推卡 → 落库。

口径要点（设计文档 §8.3/§8.5/§8.8）：
  * 只处理**已收盘**的 5m bar：交易所未收盘的 K 线随时会改，推出去的信号也会改。
  * checkpoint 之前的一周（`paper.replay_days`）以 `mode="observe"` 重放：只调策略、
    账本一行不写。它的用途是把策略的内部状态（趋势状态机、冷却计数、ML 特征帧）
    推到与断点一致，因此每轮都从 `reset()` 开始，跑第 1 轮和重启第 100 轮等价。
  * checkpoint 之后的 bar 走 `mode="live"`，与回测共用 `BarPipeline`，所以纸面成交
    序列和同参数回测逐位一致。
  * 落库先于推送：崩溃只会留下 `sent_ok=0` 的行由下一轮补发，不会出现"卡片发了、
    账本没记"的幽灵成交。
  * 数据缺口整轮跳过该策略并推告警 —— 残缺窗口里的指标与前一根 bar 的间距都不可信，
    宁可这一轮不出信号。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..backtest.position import BarPipeline, ExecEvent, HtfCursor, PositionLedger, TradeRecord
from ..backtest.precompute import precompute, precompute_htf, row_to_bar, warmup_bars
from ..core.contracts import Action, Signal
from ..core.timeframe import interval_ms, to_dt, to_ms
from ..data.backfill import assert_complete
from ..data.service import MarketDataService
from ..data.store import Store
from ..infra.log import get_logger
from ..notify.cards import build_alert_card, build_card, build_close_card, exit_label
from ..notify.feishu import FeishuBusy, FeishuError
from ..strategy.base import BaseStrategy
from .config import PaperConfig, PaperRuntime
from .guards import EntryGuards
from .state import (
    KIND_ADD,
    KIND_ALERT,
    KIND_CLOSE,
    KIND_OPEN,
    KIND_REJECT,
    PaperState,
    alert_id,
    row_id,
    signal_row,
)

_log = get_logger("quant.paper.runner")

# 事件类型 → paper_signals.kind；add_skipped 不出卡：金字塔加仓没通过触发条件时每根
# bar 都会产生一条，推出去就是把用户的消息流变成日志流
_KIND_BY_EVENT = {
    "open": KIND_OPEN,
    "add": KIND_ADD,
    "close": KIND_CLOSE,
    "partial": KIND_CLOSE,
    "rejected": KIND_REJECT,
}


@dataclass
class RoundReport:
    """一轮的结果，`paper once` 与 status 命令都直接打印它。"""

    symbol: str
    strategy: str
    observed_bars: int = 0
    live_bars: int = 0
    events: list[ExecEvent] = field(default_factory=list)
    inserted: int = 0
    sent_ok: int = 0
    sent_failed: int = 0
    resent: int = 0
    skipped: str | None = None
    gate: str = "放行"
    equity: float = 0.0
    position: str = "空仓"

    def summary(self) -> str:
        if self.skipped:
            return f"{self.symbol} {self.strategy}: 跳过（{self.skipped}）"
        return (
            f"{self.symbol} {self.strategy}: 观察 {self.observed_bars} 根 / "
            f"实撮 {self.live_bars} 根，事件 {len(self.events)} 条，"
            f"推送 {self.sent_ok} 成 / {self.sent_failed} 败 / 补发 {self.resent}，"
            f"净值 {self.equity:,.2f}，{self.position}，闸门 {self.gate}"
        )


class PaperRunner:
    """常驻的一轮驱动。策略实例与账本都活在进程里，跨轮复用。"""

    def __init__(
        self,
        *,
        store: Store,
        symbol: str,
        strategy: BaseStrategy,
        cfg: PaperConfig,
        runtime: PaperRuntime,
        notifier: Any | None = None,
    ):
        self.store = store
        self.svc = MarketDataService(store)
        self.symbol = symbol
        self.strategy = strategy
        self.cfg = cfg
        self.runtime = runtime
        self.notifier = notifier
        self.state = PaperState(store, symbol, cfg.strategy_name)
        self.spec = self.svc.spec(symbol)

        checkpoint = self.state.checkpoint_ms()
        step = interval_ms(cfg.base_interval)
        # 断点那根 bar 的可见时刻是 checkpoint + step（资金费计提按可见时刻定位）
        self.ledger: PositionLedger = self.state.open_ledger(
            cfg, self.spec, None, resumed_at_ms=None if checkpoint is None else checkpoint + step
        )
        self.guards = EntryGuards(self.state, runtime.guard).bind(self.ledger)
        self.pipeline = BarPipeline(cfg, self.ledger, self._signal_fn)

    # ---------- 策略调用 ----------

    def _signal_fn(self, bar, row, htf, dt) -> Signal | None:
        return self.strategy.on_bar(bar, row, htf, dt)

    @property
    def _decision_timeframe(self) -> str:
        return getattr(self.strategy, "decision_interval", self.cfg.base_interval)

    # ---------- 一轮 ----------

    def run_round(self, *, now_ms: int) -> RoundReport:
        cfg = self.cfg
        step = interval_ms(cfg.base_interval)
        report = RoundReport(symbol=self.symbol, strategy=cfg.strategy_name)

        watermark = self.store.watermark_ms(self.symbol, cfg.base_interval)
        if watermark is None:
            report.skipped = f"本地没有 {cfg.base_interval} 数据，先执行 quant backfill"
            return report
        latest_close = watermark + step
        checkpoint = self.state.checkpoint_ms()

        # 观测窗：checkpoint 之前 replay_days 天（首轮没有断点，就以最新收盘 bar 为基准）
        ref_ms = checkpoint if checkpoint is not None else latest_close
        observe_start_ms = ref_ms - self.runtime.replay_ms
        load_start_ms = (
            observe_start_ms
            - warmup_bars() * step
            - int(getattr(self.strategy, "history_lookback", lambda: 0)())
        )

        missing = self._data_gaps(observe_start_ms, latest_close)
        if missing:
            self._push_alert(
                f"gap:{observe_start_ms}", "数据缺口，本轮跳过",
                "；".join(missing), now_ms=now_ms, report=report,
            )
            report.skipped = "；".join(missing)
            return report

        ready = self._model_readiness(latest_close)
        if ready:
            self._push_alert(
                f"model:{latest_close}", "模型不可用，本轮跳过", ready,
                now_ms=now_ms, report=report,
            )
            report.skipped = ready
            return report

        bars, funding = self._load_frames(load_start_ms, now_ms)
        if cfg.base_interval not in bars:
            report.skipped = f"{cfg.base_interval} 在窗口内没有数据"
            return report

        df5 = precompute(
            bars[cfg.base_interval].sort_values("ts_open_ms").reset_index(drop=True),
            swing_window=max(2, int(cfg.swing_lookback)),
        )
        frames = {
            iv: precompute_htf(df.sort_values("ts_open_ms").reset_index(drop=True))
            for iv, df in bars.items() if iv != cfg.base_interval
        }
        cursor = HtfCursor(frames)

        bind = getattr(self.strategy, "bind_history", None)
        if callable(bind):
            bind(bars, funding)
        self.strategy.reset()
        # 每轮窗口起点都会滑动，落库的指针下标没有意义，按断点重定位
        self.ledger.set_funding(funding)
        self.ledger.seek_funding(None if checkpoint is None else checkpoint + step)

        gate = self.guards.refresh(
            now_ms=now_ms, data_gap_ms=max(0, now_ms - latest_close),
        )
        for tag, title, text in gate.alerts:
            self._push_alert(tag, title, text, now_ms=now_ms, report=report)

        last_seen_ms = checkpoint
        for _, row in df5.iterrows():
            ts_open_ms = int(row["ts_open_ms"])
            if ts_open_ms < observe_start_ms:
                continue  # 只用于指标预热
            bar = row_to_bar(row, self.symbol, cfg.base_interval)
            if to_ms(bar.ts_close) > now_ms:
                break  # 交易所还没收盘，不能据此出信号
            htf = cursor.advance(to_ms(bar.ts_close) + 1)
            if checkpoint is None or ts_open_ms <= checkpoint:
                self.pipeline.on_bar(bar, row, htf, mode="observe")
                report.observed_bars += 1
            else:
                report.events += self.pipeline.on_bar(bar, row, htf, mode="live")
                report.live_bars += 1
            last_seen_ms = ts_open_ms

        report.equity = self.ledger.equity
        report.position = "空仓" if self.ledger.position is None else self._describe_position()
        report.gate = self.guards.describe()

        signals, trades = self._derive_records(report.events, now_ms=now_ms)
        inserted = self.state.commit(
            self.ledger,
            now_ms=now_ms,
            checkpoint_ms=last_seen_ms,
            dataset_version=self._dataset_version(),
            signals=signals,
            trades=trades,
            pause_until_ms=gate.pause_until_ms,
        )
        report.inserted = len(inserted)
        self._deliver(inserted, report=report, now_ms=now_ms)
        self._resend_unsent(report=report, now_ms=now_ms)
        return report

    # ---------- 数据 ----------

    def _load_frames(self, start_ms: int, end_ms: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        start, end = to_dt(start_ms), to_dt(end_ms)
        bars: dict[str, pd.DataFrame] = {}
        for iv in self.strategy.required_intervals:
            df = self.svc.bars(self.symbol, iv, start, end)
            if df.empty:
                _log.warning("%s %s 窗口内无数据", self.symbol, iv)
                continue
            bars[iv] = df
        # 资金费不再计入成本，但 ML 特征里要用，加载照旧
        return bars, self.svc.funding(self.symbol, start, end)

    def _data_gaps(self, start_ms: int, end_ms: int) -> list[str]:
        """观测窗 + 实撮区间内任何周期缺柱都算缺口：高周期缺柱会让 HTF 游标错位。"""
        out: list[str] = []
        for iv in self.strategy.required_intervals:
            chk = assert_complete(self.store, self.symbol, iv, start_ms, end_ms)
            if chk["coverage"] < 1.0:
                head = chk["missing_head"][0] if chk["missing_head"] else "?"
                out.append(
                    f"{iv} 缺 {chk['missing']}/{chk['expected']} 根（首个缺失 {head}）"
                )
        return out

    def _dataset_version(self) -> str:
        return self.store.dataset_version(self.symbol, list(self.strategy.required_intervals))

    def _model_readiness(self, ref_ms: int) -> str | None:
        """ML 策略：折模型必须存在且 `ref_ms` 落在某一折的样本外窗口内。

        不用 `dataset_version` 相等做闸门：paper 每轮都在增量拉数据，版本串必然变化，
        拿它当闸门等于永远拒绝服务。特征集漂移由 `MLBaseStrategy._check_features`
        的 `feature_spec_hash` 断言拦住，"模型过期"由折窗口拦住。
        """
        get_folds = getattr(self.strategy, "folds", None)
        if not callable(get_folds):
            return None  # 规则策略无工件依赖
        try:
            folds = get_folds()
        except Exception as e:  # 模型目录缺失/元数据损坏都只是"这个策略不能跑"
            return f"折模型加载失败：{e}"
        if not folds:
            return "没有可用的折模型目录"
        if not any(f.test_start_ms <= ref_ms < f.test_end_ms for f in folds):
            return "当前时刻超出最后一折的样本外窗口，需 quant train 续训"
        return None

    # ---------- 事件 → 落库/卡片 ----------

    def _derive_records(
        self, events: list[ExecEvent], *, now_ms: int
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[tuple[str, TradeRecord]]]:
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        trades: list[tuple[str, TradeRecord]] = []
        cfg = self.cfg
        for ev in events:
            kind = _KIND_BY_EVENT.get(ev.kind)
            if kind is None:
                continue
            row, card = self._event_row(ev, kind=kind, now_ms=now_ms)
            if row is None:
                continue
            pairs.append((row, card))
            if ev.trade is not None:
                pid = ev.position_id or "unknown"
                trades.append((row_id(self.symbol, cfg.strategy_name, pid), ev.trade))
        return pairs, trades

    def _event_row(self, ev: ExecEvent, *, kind: str, now_ms: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        cfg = self.cfg
        if ev.trade is not None:
            tr = ev.trade
            action = Action.CLOSE_LONG if tr.side == "long" else Action.CLOSE_SHORT
            # 一个仓位可以留下 TP1 部分止盈 + 全平多条流水，id 必须带原因与时刻，
            # 否则第二条会把第一条的主键撞掉，卡片也少推一张
            native = f"{ev.position_id}:{tr.reason}:{to_ms(tr.exit_time)}"
            card = build_close_card(
                tr, symbol=self.symbol, strategy=cfg.strategy_name,
                timeframe=self._decision_timeframe, interval=cfg.base_interval,
            )
            row = signal_row(
                signal_id=row_id(self.symbol, cfg.strategy_name, native),
                ts_ms=to_ms(tr.exit_time),
                symbol=self.symbol, strategy=cfg.strategy_name,
                action=action.value, price_ref=tr.exit_price,
                reason=exit_label(tr.reason), kind=kind, created_at_ms=now_ms,
                qty=tr.qty, margin_usdt=tr.margin_usdt,
                stop_loss=None, tp_price=None,
                note=f"pnl {tr.pnl:+.2f} USDT / {tr.ret_pct:+.2f}%",
            )
            return row, card

        sig = ev.signal
        if sig is None:
            return None, {}
        dec = ev.decision
        native = f"{sig.signal_id}:{kind}"
        card = build_card(sig, dec)
        row = signal_row(
            signal_id=row_id(self.symbol, cfg.strategy_name, native),
            ts_ms=to_ms(sig.ts_signal),
            symbol=self.symbol, strategy=cfg.strategy_name,
            action=sig.action.value, price_ref=sig.price_ref,
            reason=sig.reason, kind=kind, created_at_ms=now_ms,
            qty=None if dec is None else dec.qty,
            margin_usdt=None if dec is None else dec.margin_usdt,
            stop_loss=None if dec is None else dec.stop_loss,
            tp_price=None if dec is None or not dec.take_profits else dec.take_profits[0],
            confidence=sig.strength,
            note="" if dec is None or dec.approved else "；".join(dec.reject_reasons),
        )
        return row, card

    def _describe_position(self) -> str:
        pos = self.ledger.position
        if pos is None:
            return "空仓"
        return (
            f"{'多' if pos.side == 'long' else '空'} {pos.qty:g} @ {pos.avg_entry_price:,.2f}"
            f"（SL {pos.stop_loss:,.2f}，加仓 {pos.add_count} 次）"
        )

    # ---------- 推送 ----------

    def _deliver(self, rows: list[dict[str, Any]], *, report: RoundReport, now_ms: int) -> None:
        for row in rows:
            ok = self._send(row["payload_json"])
            if ok:
                report.sent_ok += 1
            else:
                report.sent_failed += 1
            self.store.mark_paper_signal_sent(
                row["signal_id"], ok, now_ms, row["payload_json"]
            )
            if not ok:
                _log.warning("%s %s 卡片投递失败，留待补发", self.symbol, row["signal_id"])

    def _resend_unsent(self, *, report: RoundReport, now_ms: int) -> None:
        before = now_ms - int(self.runtime.resend_cooldown_min * 60_000)
        rows = self.store.paper_signals_unsent(
            self.symbol, self.cfg.strategy_name, 10, attempt_before_ms=before
        )
        for row in rows:
            payload = row.get("payload_json") or ""
            if not payload:
                continue  # 演练模式下没有卡片快照，补发不了也不用补发
            ok = self._send(payload)
            self.store.mark_paper_signal_sent(row["signal_id"], ok, now_ms, payload)
            report.resent += int(ok)

    def _send(self, payload_json: str) -> bool:
        if self.notifier is None:
            return True  # 演练模式（--no-push）：落库即视为已处理，不进补发队列
        try:
            self.notifier.send_payload(json.loads(payload_json))
        except (FeishuBusy, FeishuError) as e:
            _log.warning("飞书投递失败: %s", e)
            return False
        except Exception as e:  # 卡片结构错误、网络栈异常都不能拖垮主循环
            _log.error("飞书投递异常: %s", e)
            return False
        return True

    def _push_alert(
        self, tag: str, title: str, text: str, *, now_ms: int, report: RoundReport
    ) -> None:
        """告警去重靠 `paper_signals` 主键：同一 tag 已存在就不再推第二张。"""
        row = signal_row(
            signal_id=alert_id(self.symbol, self.cfg.strategy_name, tag),
            ts_ms=now_ms, symbol=self.symbol, strategy=self.cfg.strategy_name,
            action="alert", price_ref=0.0, reason=text, kind=KIND_ALERT,
            created_at_ms=now_ms, note=title,
        )
        card = build_alert_card(
            title, text, symbol=self.symbol, strategy=self.cfg.strategy_name,
            ts=to_dt(now_ms),
        )
        inserted = self.state.add_alert(row, card)
        if inserted is None:
            return
        if self._send(inserted["payload_json"]):
            report.sent_ok += 1

    # ---------- status ----------

    def status(self, *, now_ms: int) -> dict[str, Any]:
        checkpoint = self.state.checkpoint_ms()
        step = interval_ms(self.cfg.base_interval)
        watermark = self.store.watermark_ms(self.symbol, self.cfg.base_interval)
        behind = None if checkpoint is None or watermark is None else (watermark - checkpoint) // step
        stats = self.store.paper_signals_stats(self.symbol, self.cfg.strategy_name)
        return {
            "symbol": self.symbol,
            "strategy": self.cfg.strategy_name,
            "checkpoint": None if checkpoint is None else to_dt(checkpoint).isoformat(),
            "behind_bars": behind,
            "equity": round(self.ledger.equity, 4),
            "position": self._describe_position(),
            "gate": self.guards.describe(),
            "pause_until": _iso(self.guards.pause_until_ms),
            "signals": int(stats["n"] or 0),
            "unsent": int((stats["n"] or 0) - (stats["n_sent"] or 0)),
            "last_signal": _iso(int(stats["last_ts"] or 0)) if stats["last_ts"] else None,
        }


def _iso(ms: int) -> str | None:
    return None if not ms else to_dt(ms).isoformat()
