"""`[paper]` 事前风控闸门：只拦"新增敞口"（试仓与加仓），绝不介入已有持仓的出场。

三条口径（设计文档 §8.4）：
  * 回测里没有这些闸门。回测要的是策略裸表现，套上闸门就看不出策略好坏；paper 必须有，
    因为真实资金在连亏之后需要先停一晚。
  * 闸门只在决策进撮合前追加否决理由（`PositionLedger.approve_fn`）。出场路径
    （SL/TP/时间止损/强平）一律由 `backtest/position.py` 决定，与回测逐位一致，
    否则纸面数字失去参考价值。
  * 阈值来自 `risk.yaml` 的 `caps`/`guard`，状态一律从 SQLite 读：进程重启不会把
    "今天已经亏了 3%" 忘掉。冷却到期时刻存在 `paper_accounts.pause_until_ms`。

账户是按 (symbol, strategy) 独立的 100U 纸面账户，所以"单品种保证金占比"这种上限
按本账户净值算 —— 同品种两个策略各花各的保证金，不互相挪用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..backtest.position import PositionLedger
from ..core.contracts import Action, RiskDecision
from ..core.timeframe import BJ_TZ, to_dt, to_ms
from .config import GuardThresholds
from .state import KIND_OPEN, PaperState


@dataclass
class GateReport:
    """一轮闸门评估的结果：新的冷却到期时刻、本轮生效的否决理由、要推的告警。"""

    pause_until_ms: int = 0
    blocked: list[str] = field(default_factory=list)
    alerts: list[tuple[str, str, str]] = field(default_factory=list)  # (tag, title, text)


def day_start_ms(now_ms: int) -> int:
    """北京时间自然日的起点：日亏上限与每日开仓上限都按这个界清零。"""
    dt = to_dt(now_ms).astimezone(BJ_TZ)
    return to_ms(dt.replace(hour=0, minute=0, second=0, microsecond=0))


class EntryGuards:
    """`PositionLedger.approve_fn` 的实现方：`refresh()` 每轮算一次，`vetoes()` 逐笔判。"""

    def __init__(self, state: PaperState, thresholds: GuardThresholds):
        self.state = state
        self.store = state.store
        self.th = thresholds
        self.cooldown_ms = int(thresholds.cooldown_hours * 3_600_000)
        self.ledger: PositionLedger | None = None
        self.blocked: list[str] = []
        self.pause_until_ms = 0

    def bind(self, ledger: PositionLedger) -> EntryGuards:
        """账本建好之后回填引用并把否决回调挂上撮合层。"""
        self.ledger = ledger
        ledger.approve_fn = self.vetoes
        return self

    # ---------- 每轮评估 ----------

    def refresh(self, *, now_ms: int, data_gap_ms: int) -> GateReport:
        """算出本轮的持续型否决理由；冷却型闸门触发时同时记下到期时刻与告警。"""
        led = self.ledger
        th = self.th
        assert led is not None, "EntryGuards.refresh 前必须 bind(ledger)"
        rep = GateReport(pause_until_ms=self.state.pause_until_ms())

        if rep.pause_until_ms > now_ms:
            # 冷却中不再判其余闸门：连亏笔数在下一笔盈利之前会一直成立，
            # 每轮都重触发的话冷却期永远走不完
            rep.blocked.append(f"风控冷却中，{self._hrs(rep.pause_until_ms - now_ms)}后恢复试仓")
            self._publish(rep)
            return rep

        if rep.pause_until_ms:
            # 冷却到期：峰值重新锚到当前净值，否则回撤闸门会在恢复的第一轮再触发一次
            rep.pause_until_ms = 0
            led.peak_equity = led.equity

        if data_gap_ms > th.stale_data_max_gap_min * 60_000:
            # 行情断了就不要开新仓：此时的指标不代表市场。已有持仓的止损照常执行，
            # 拦截只针对新增敞口
            reason = (
                f"行情滞后 {self._mins(data_gap_ms)}，超过 "
                f"{th.stale_data_max_gap_min:g} 分钟阈值"
            )
            rep.blocked.append(reason)
            rep.alerts.append((
                f"stale:{day_start_ms(now_ms)}",
                f"行情滞后，已停止试仓 · {self.state.symbol} {self.state.strategy}",
                reason,
            ))

        pnl, _n = self.store.paper_pnl_since(
            self.state.symbol, self.state.strategy, day_start_ms(now_ms)
        )
        day_open_equity = led.equity - pnl  # 当日期初净值（只按已实现盈亏回推）
        if day_open_equity > 0 and pnl <= -th.daily_loss_stop_pct * day_open_equity:
            rep.blocked.append(
                f"当日已实现 {pnl:+.2f} USDT，超过日亏上限 "
                f"{th.daily_loss_stop_pct * 100:.1f}%（当日 {day_open_equity:.2f} 起算）"
            )

        dd = 0.0 if led.peak_equity <= 0 else (led.peak_equity - led.equity) / led.peak_equity
        if dd >= th.max_drawdown_pause_pct:
            self._pause_for(
                rep, now_ms,
                f"净值 {led.equity:.2f} 相对峰值 {led.peak_equity:.2f} 回撤 "
                f"{dd * 100:.1f}%，超过 {th.max_drawdown_pause_pct * 100:.1f}% 阈值",
                tag="drawdown",
                title="回撤触发冷却",
            )
            self._publish(rep)
            return rep

        losses = self.store.paper_last_consecutive_losses(self.state.symbol, self.state.strategy)
        if losses >= th.max_consecutive_losses:
            self._pause_for(
                rep, now_ms,
                f"连续亏损 {losses} 笔，达到 {th.max_consecutive_losses} 笔阈值",
                tag="losing-streak",
                title="连亏触发冷却",
            )
            self._publish(rep)
            return rep

        opened = self.store.paper_signals_count(
            self.state.symbol, self.state.strategy, (KIND_OPEN,), day_start_ms(now_ms)
        )
        if opened >= th.max_new_entries_per_day:
            rep.blocked.append(
                f"今日已试仓 {opened} 次，达到 {th.max_new_entries_per_day} 次上限"
            )

        self._publish(rep)
        return rep

    def _pause_for(
        self, rep: GateReport, now_ms: int, reason: str, *, tag: str, title: str
    ) -> None:
        rep.pause_until_ms = now_ms + self.cooldown_ms
        rep.blocked.append(f"{reason}，冷却 {self._hrs(self.cooldown_ms)}后恢复试仓")
        # tag 带日期：同一闸门在同一自然日只推一张告警卡（靠 paper_signals 主键去重）
        rep.alerts.append((
            f"{tag}:{day_start_ms(now_ms)}",
            f"{title} · {self.state.symbol} {self.state.strategy}",
            reason,
        ))

    def _publish(self, rep: GateReport) -> None:
        self.blocked = rep.blocked
        self.pause_until_ms = rep.pause_until_ms

    # ---------- 逐笔判 ----------

    def vetoes(self, decision: RiskDecision) -> list[str]:
        """撮合前的否决理由，签名与 `PositionLedger.approve_fn` 一致。"""
        out = list(self.blocked)
        if decision.signal.action not in _ENTRY_ACTIONS:
            return out

        pos = None if self.ledger is None else self.ledger.position
        used = pos.margin_usdt if pos is not None else 0.0
        equity = decision.equity_usdt
        cap = self.th.max_margin_per_symbol_pct * equity
        want = used + decision.margin_usdt
        if want > cap:
            out.append(
                f"保证金将达 {want:.2f} USDT，超过净值 {equity:.2f} 的 "
                f"{self.th.max_margin_per_symbol_pct * 100:.0f}%（上限 {cap:.2f}）"
            )
        if decision.notional_usdt > self.th.max_notional_usdt:
            out.append(
                f"名义 {decision.notional_usdt:.2f} USDT 超过单笔上限 "
                f"{self.th.max_notional_usdt:.2f}"
            )
        return out

    # ---------- status 展示 ----------

    def describe(self) -> str:
        return "；".join(self.blocked) if self.blocked else "放行"

    @staticmethod
    def _hrs(ms: int) -> str:
        hours = max(ms, 0) / 3_600_000
        return f"{hours:.1f} 小时" if hours >= 1 else f"{hours * 60:.0f} 分钟"

    @staticmethod
    def _mins(ms: int) -> str:
        return f"{max(ms, 0) / 60_000:.1f} 分钟"


_ENTRY_ACTIONS = (Action.OPEN_LONG, Action.OPEN_SHORT, Action.ADD_TO_LONG, Action.ADD_TO_SHORT)
