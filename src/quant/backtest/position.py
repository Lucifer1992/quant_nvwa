"""回测与模拟交易（paper）共用的执行层：持仓价位触发 + 成交账本 + 单根 bar 的驱动序列。

为什么单独成模块（设计文档 docs/lgbm-strategy-and-paper-trading-design.md §8.5）：
纸面撮合必须与回测走同一套代码。复制第二份实现迟早漂移，漂移之后 paper 的数字就没有
参考价值了。`BacktestEngine` 与 `src/quant/paper/` 都只通过本模块操作持仓与账本。

口径要点（改动必须同步 tests/test_engine_ledger.py、test_engine_pyramid.py）：
  * 信号在**下一根 bar 开盘**成交，成交价带不利方向滑点；
  * 止损由 bar 极值触发时按 `bar.low`/`bar.high` 成交（最坏情况），出场时间取 **bar 中点**；
  * 同一根 bar 同时触及 SL 与 TP 时，默认保守取 SL（`sl_first_on_conflict`），与
    `features.builder.build_labels` 的标签口径一致；
  * 一根 bar 内的出场顺序：旧止损 → 强平 → TP1 分批止盈（并把止损移到保本）→ 用新止损
    再判一次 → 全平 TP。止损只要在本根 bar 被收紧，就立刻按 bar 极值判触发，与移动止损
    的保守约定一致；
  * 全 taker：开仓腿手续费在成交时从现金扣除并记在 `AddLevel` 上，平仓只结算平仓腿，
    逐笔归属 pnl 再扣往返全部费用；分批止盈按平仓比例分摊开仓腿手续费，
    所以 `sum(trades.pnl) == Δequity` 在部分平仓后依然成立；
  * 定仓用的权益是 `equity_open`（本根 bar 开盘前已知的净值），不是收盘 MTM——后者是
    成交之后才知道的信息；
  * 资金费默认**不计入成本**（`include_funding=False`）：实测 10× 下三天资金费只占保证金
    约 0.24%，而往返 taker 手续费占 1%，留着它只会多一个无关变量。数据仍然照常加载，
    ML 策略的特征里还要用。

cfg 参数一律是"执行参数对象"：`BacktestConfig` 或 `PaperConfig`，两者字段名必须一致
（由 tests/test_paper_service.py 的字段对齐用例守住）。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..core.contracts import (
    Action,
    AddLevel,
    Bar,
    Fill,
    FillReason,
    Position,
    RiskDecision,
    Side,
    Signal,
    SymbolSpec,
)
from ..core.timeframe import interval_ms, to_ms
from ..risk.sizing import decide_position, floor_to

# ---- 纯函数：价位、触发与止损跟随 -------------------------------------------


def fee_qty(qty: float, price: float, rate: float) -> float:
    return round(qty * price * rate, 8)


def bar_mid(bar: Bar) -> datetime:
    return bar.ts_open + (bar.ts_close - bar.ts_open) / 2


def is_long_action(action: Action) -> bool:
    return action in (Action.OPEN_LONG, Action.ADD_TO_LONG)


def entry_price(bar: Bar, action: Action, slippage_bps: float) -> float:
    """在 bar 开盘成交的入场价，滑点方向恒为不利（买入抬高、卖出压低）。"""
    adverse = 1 + slippage_bps / 10000 if is_long_action(action) else 1 - slippage_bps / 10000
    return round(bar.open * adverse, 8)


def exit_price_at_open(bar: Bar, pos: Position, slippage_bps: float) -> float:
    """市价平仓按本根 bar 开盘价成交（信号平仓与反手用）。"""
    adverse = 1 - slippage_bps / 10000 if pos.side == "long" else 1 + slippage_bps / 10000
    return round(bar.open * adverse, 8)


def slipped_close_price(bar: Bar, pos_side: str, slippage_bps: float) -> float:
    """按 bar 收盘价市价平仓的不利滑点成交价（时间止损、期末平仓用）。"""
    adverse = 1 + slippage_bps / 10000 if pos_side == "short" else 1 - slippage_bps / 10000
    return round(float(bar.close) * adverse, 8)


def stop_fill_price(bar: Bar, pos_side: str, slippage_bps: float) -> float:
    """止损被触发时按 bar 极值成交（多头 bar.low / 空头 bar.high，最坏情况）再加不利滑点。"""
    exit_px = float(bar.low if pos_side == "long" else bar.high)
    adverse = 1 - slippage_bps / 10000 if pos_side == "long" else 1 + slippage_bps / 10000
    return round(exit_px * adverse, 8)


def tp_fill_price(pos: Position, slippage_bps: float) -> float:
    adverse = 1 - slippage_bps / 10000 if pos.side == "long" else 1 + slippage_bps / 10000
    return round(float(pos.tp_price) * adverse, 8)


def liq_hit(pos: Position, bar: Bar) -> bool:
    if not pos.liq_price or pos.liq_price <= 0:
        return False
    return bar.low <= pos.liq_price if pos.side == "long" else bar.high >= pos.liq_price


def liq_fill_price(pos: Position, slippage_bps: float) -> float:
    """强平按强平价成交（逐仓被打穿，没有讨价还价的余地），再扣不利滑点。"""
    adverse = 1 - slippage_bps / 10000 if pos.side == "long" else 1 + slippage_bps / 10000
    return round(float(pos.liq_price) * adverse, 8)


def r_price(pos: Position, cfg: Any, r_mult: float) -> float:
    """某个 R 倍数对应的价格（R 以加权均价为原点、`sl_atr_mult × entry_atr` 为单位）。"""
    dist = cfg.sl_atr_mult * (pos.entry_atr or 0.0)
    if dist <= 0:
        return 0.0
    return pos.avg_entry_price + dist * r_mult if pos.side == "long" else pos.avg_entry_price - dist * r_mult


def tp1_hit(pos: Position, bar: Bar, cfg: Any) -> bool:
    if cfg.tp1_r <= 0 or pos.tp1_done or not pos.entry_atr or pos.entry_atr <= 0:
        return False
    px = r_price(pos, cfg, cfg.tp1_r)
    return px > 0 and (bar.high >= px if pos.side == "long" else bar.low <= px)


def apply_breakeven(pos: Position | None, cfg: Any, price: float) -> bool:
    """浮盈达到门槛后把止损移到加权开仓价：这一笔之后最差也是保本出场。"""
    if pos is None or cfg.breakeven_after_r <= 0 or pos.breakeven_done:
        return False
    if not pos.entry_atr or pos.entry_atr <= 0:
        return False
    if unrealized_r(pos, price, cfg.sl_atr_mult) < cfg.breakeven_after_r:
        return False
    pos.breakeven_done = True
    be = pos.avg_entry_price
    if pos.side == "long":
        pos.stop_loss = be if not pos.stop_loss else max(pos.stop_loss, be)
    else:
        pos.stop_loss = be if not pos.stop_loss else min(pos.stop_loss, be)
    return True


def stop_levels(entry: float, action: Action, atr_val: float, cfg: Any) -> tuple[float, float]:
    """试仓时按**实际成交价**重算 SL/TP，避免 open-close gap 一开盘就击穿止损。"""
    if is_long_action(action):
        return entry - cfg.sl_atr_mult * atr_val, entry + cfg.tp_atr_mult * atr_val
    return entry + cfg.sl_atr_mult * atr_val, entry - cfg.tp_atr_mult * atr_val


def stop_hit(pos: Position, bar: Bar) -> bool:
    if not pos.stop_loss:
        return False
    return bar.low <= pos.stop_loss if pos.side == "long" else bar.high >= pos.stop_loss


def tp_hit(pos: Position, bar: Bar) -> bool:
    if pos.tp_price <= 0:
        return False
    return bar.high >= pos.tp_price if pos.side == "long" else bar.low <= pos.tp_price


def exit_reason(pos: Position, bar: Bar, sl_first_on_conflict: bool) -> FillReason | None:
    """本根 bar 是否触发出场；双触发时按保守口径定序。"""
    hit_sl, hit_tp = stop_hit(pos, bar), tp_hit(pos, bar)
    if hit_sl and hit_tp and not sl_first_on_conflict:
        return FillReason.TP
    if hit_sl:
        return FillReason.SL
    if hit_tp:
        return FillReason.TP
    return None


def unrealized_r(pos: Position, price: float, sl_atr_mult: float) -> float:
    """浮盈 R 倍数（基于加权均价与入场时的 ATR 尺度）。"""
    if pos.entry_atr is None or pos.entry_atr <= 0:
        return 0.0
    dist = sl_atr_mult * pos.entry_atr
    move = price - pos.avg_entry_price if pos.side == "long" else pos.avg_entry_price - price
    return move / dist


def should_add(pos: Position | None, cfg: Any, row: pd.Series, bar: Bar) -> bool:
    """加仓条件：浮盈 ≥ `add_trigger_r` 且价格突破 swing 结构位。"""
    if pos is None:
        return False
    if pos.add_count >= cfg.add_max_count:
        return False
    if pos.entry_atr is None or pos.entry_atr <= 0:
        return False
    if unrealized_r(pos, bar.close, cfg.sl_atr_mult) < cfg.add_trigger_r:
        return False
    swing_high = float(row.get("swing_high", 0)) if "swing_high" in row else 0
    swing_low = float(row.get("swing_low", 0)) if "swing_low" in row else 0
    if pos.side == "long" and swing_high > 0:
        return bar.close > swing_high
    if pos.side == "short" and swing_low > 0:
        return bar.close < swing_low
    return False


def update_trailing(pos: Position | None, cfg: Any, bar: Bar) -> None:
    """每根 bar 更新移动止损：先推进极值，浮盈达门槛后按 `trail_atr_mult × entry_atr` 跟随。"""
    if pos is None or pos.entry_atr is None or pos.entry_atr <= 0:
        return
    if pos.side == "long":
        pos.trailing_peak_price = max(pos.trailing_peak_price, bar.high)
    else:
        pos.trailing_peak_price = min(pos.trailing_peak_price, bar.low)

    if not pos.trailing_active and unrealized_r(pos, bar.close, cfg.sl_atr_mult) >= cfg.trail_trigger_r:
        pos.trailing_active = True

    if pos.trailing_active:
        if pos.side == "long":
            trail_sl = pos.trailing_peak_price - cfg.trail_atr_mult * pos.entry_atr
            pos.stop_loss = max(pos.stop_loss, trail_sl)
        else:
            trail_sl = pos.trailing_peak_price + cfg.trail_atr_mult * pos.entry_atr
            pos.stop_loss = min(pos.stop_loss, trail_sl)


def sync_stop_after_add(pos: Position, cfg: Any, entry: float) -> None:
    """加仓后止损跟着新的加权均价上移（只朝有利方向，绝不放松）。"""
    if not pos.entry_atr or pos.entry_atr <= 0:
        return
    if pos.side == "long":
        new_sl = pos.avg_entry_price - cfg.sl_atr_mult * pos.entry_atr
        pos.stop_loss = max(pos.stop_loss, new_sl)
        pos.base_stop_price = pos.stop_loss
        pos.trailing_peak_price = max(pos.trailing_peak_price, entry)
    else:
        new_sl = pos.avg_entry_price + cfg.sl_atr_mult * pos.entry_atr
        pos.stop_loss = min(pos.stop_loss, new_sl)
        pos.base_stop_price = pos.stop_loss
        pos.trailing_peak_price = min(pos.trailing_peak_price, entry)


def mark_to_market(pos: Position, cash: float, close: float, funding_since_entry: float) -> float:
    """持仓期净值：现金 + 浮盈（对加权均价）+ 保证金 − 已计提未结算的资金费负债。"""
    ref = pos.avg_entry_price
    unreal = pos.qty * (close - ref) if pos.side == "long" else pos.qty * (ref - close)
    return cash + unreal + pos.margin_usdt - funding_since_entry


def veto(decision: RiskDecision, reasons: list[str]) -> RiskDecision:
    """给已批准的决策追加否决理由（`RiskDecision` 是 frozen dataclass，只能复制）。"""
    return replace(
        decision,
        approved=False,
        reject_reasons=(*decision.reject_reasons, *reasons),
    )


# ---- 账本 -------------------------------------------------------------------


@dataclass
class TradeRecord:
    """一笔完整开平的逐笔归属结果（报告与 paper 历史共用同一份字段）。"""

    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fee: float
    reason: str
    r_achieved: float = 0.0
    bars_held: int = 0
    add_count: int = 0
    avg_entry_price: float = 0.0
    initial_qty: float = 0.0
    margin_usdt: float = 0.0  # 该笔占用的保证金（含金字塔加仓累计）
    ret_pct: float = 0.0  # 保证金收益率 % = pnl / margin_usdt × 100


@dataclass
class ExecEvent:
    """一次对持仓的改动，paper 侧据此推送卡片与落库；回测侧忽略。

    `position_id` 只有开/加/平才有意义：分批止盈之后一个仓位会留下多条逐笔流水，
    paper 落库要靠它把 `paper_trades.position_id` 对回 `paper_positions`。
    """

    kind: Literal["open", "add", "close", "partial", "rejected", "add_skipped"]
    ts: datetime
    signal: Signal | None = None
    decision: RiskDecision | None = None
    trade: TradeRecord | None = None
    note: str = ""
    position_id: str | None = None


PositionLedgerState = dict[str, Any]


class PositionLedger:
    """现金 / 保证金 / 持仓 / 成交流水 / 逐笔盈亏的唯一记账处。

    回测由 `BarPipeline` 逐根 bar 驱动，paper 在进程重启后从 `paper_*` 表恢复同一份状态
    （见 `src/quant/paper/state.py`）。
    """

    def __init__(self, cfg: Any, spec: SymbolSpec, funding: pd.DataFrame | None = None):
        self.cfg = cfg
        self.spec = spec
        self.funding = funding
        # paper 侧注入的风控闸门：撮合前给决策追加否决理由。回测留 None，
        # 成交序列因此与回测逐位一致（§8.5 的同源要求）
        self.approve_fn: Callable[[RiskDecision], list[str]] | None = None
        self.reset(cfg.initial_equity)

    def reset(self, initial_equity: float) -> None:
        self.cash = initial_equity
        self.equity = initial_equity
        self.equity_open = initial_equity
        # [paper] 回撤闸门的基准：净值高点必须跨轮持久化，否则重启就把回撤清零了
        self.peak_equity = initial_equity
        self.position: Position | None = None
        self.fills: list[Fill] = []
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[float] = []
        self.timestamps: list[datetime] = []
        self.bars_since_entry = 0
        self.funding_since_entry = 0.0
        self.add_count_total = 0
        self.add_skip_count = 0
        self._funding_ptr = 0

    # ---------- 状态快照（paper 持久化用）----------

    def dump_state(self) -> PositionLedgerState:
        return {
            "cash": self.cash,
            "equity": self.equity,
            "equity_open": self.equity_open,
            "peak_equity": self.peak_equity,
            "bars_since_entry": self.bars_since_entry,
            "funding_since_entry": self.funding_since_entry,
            "funding_ptr": self._funding_ptr,
            "add_count_total": self.add_count_total,
            "add_skip_count": self.add_skip_count,
        }

    def restore_state(self, state: PositionLedgerState) -> None:
        self.cash = float(state["cash"])
        self.equity = float(state.get("equity", state["cash"]))
        self.equity_open = float(state.get("equity_open", self.equity))
        self.peak_equity = max(float(state.get("peak_equity") or self.equity), self.equity)
        self.bars_since_entry = int(state.get("bars_since_entry", 0))
        self.funding_since_entry = float(state.get("funding_since_entry", 0.0))
        self._funding_ptr = int(state.get("funding_ptr", 0))
        self.add_count_total = int(state.get("add_count_total", 0))
        self.add_skip_count = int(state.get("add_skip_count", 0))

    def set_funding(self, funding: pd.DataFrame | None) -> None:
        """paper 每轮重新加载资金费帧：指针按 funding_time 定位，不能盲目沿用旧位置。"""
        self.funding = funding
        self._funding_ptr = 0

    def seek_funding(self, t_ms: int | None) -> None:
        """把指针移到第一条 `funded_at_ms > t_ms` 的结算；`t_ms=None` 表示从头开始。

        paper 每轮的加载窗口起点会变，落库存的数组下标没有意义；断点续跑必须按时刻重定位，
        否则窗口一滑动就会把已计提过的结算再记一遍到当前持仓上。
        首轮没有断点，指针就该停在 0 —— 传 None 进来不能喂给 `searchsorted`。
        """
        if self.funding is None or self.funding.empty or t_ms is None:
            self._funding_ptr = 0
            return
        times = self.funding["funded_at_ms"].to_numpy(dtype="int64")
        self._funding_ptr = int(np.searchsorted(times, t_ms, side="right"))

    # ---------- 结算 ----------

    def snapshot(self, bar: Bar) -> float:
        """记录本根 bar 收盘的净值点。"""
        if self.position is not None:
            self.equity = mark_to_market(
                self.position, self.cash, float(bar.close), self.funding_since_entry
            )
        else:
            self.equity = self.cash
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)
        self.timestamps.append(bar.ts_close)
        return self.equity

    def accrue_funding(self, t_vis_ms: int, price: float) -> None:
        """把结算时刻已落在本根 bar 内的资金费计入当前持仓的成本。

        约定：正费率多头付、空头收。费用只累计到 funding_since_entry，现金在平仓时经
        net_cash 一次性扣除，避免持仓期与平仓时重复扣款。指针每根 bar 都要推进，
        否则空仓期间的结算会被算进之后才开的仓位。
        """
        if self.funding is None or self.funding.empty or not self.cfg.include_funding:
            return
        n = len(self.funding)
        while self._funding_ptr < n:
            funded_at = int(self.funding["funded_at_ms"].iloc[self._funding_ptr])
            if funded_at > t_vis_ms:
                break
            rate = float(self.funding["funding_rate"].iloc[self._funding_ptr])
            self._funding_ptr += 1
            pos = self.position
            if pos is None or not np.isfinite(rate) or rate == 0.0:
                continue
            px = price
            if "mark_price" in self.funding.columns:
                mp = float(self.funding["mark_price"].iloc[self._funding_ptr - 1])
                if np.isfinite(mp) and mp > 0:
                    px = mp
            cost = pos.qty * px * rate * (1.0 if pos.side == "long" else -1.0)
            self.funding_since_entry += cost
            pos.funding_paid += cost

    # ---------- 成交 ----------

    def open_or_add(self, signal: Signal, decision: RiskDecision, bar: Bar) -> ExecEvent:
        """试仓或加仓。试仓时 SL/TP 基于实际成交价重算。"""
        cfg = self.cfg
        entry = entry_price(bar, signal.action, cfg.slippage_bps)
        is_add = self.position is not None
        atr_val = float(signal.meta["atr"]) if signal.meta.get("atr") else None

        # 试仓时 SL/TP 基于实际入场价（而非 signal.price_ref=bar.close），避免 open-close
        # gap 一开盘就击穿止损；加仓沿用仓位已有的 tp_price。
        tp_price = None
        if not is_add and atr_val:
            sl, tp_price = stop_levels(entry, signal.action, atr_val, cfg)
            decision = RiskDecision(
                signal=decision.signal, approved=decision.approved,
                qty=decision.qty, notional_usdt=decision.notional_usdt,
                leverage=decision.leverage, margin_usdt=decision.margin_usdt,
                stop_loss=sl, take_profits=decision.take_profits,
                est_liq_price=decision.est_liq_price,
                equity_usdt=decision.equity_usdt, risk_usdt=decision.risk_usdt,
            )

        fee = fee_qty(decision.qty, entry, cfg.taker_fee)
        if not is_add:
            side = "long" if is_long_action(signal.action) else "short"
            pos = Position(
                position_id=signal.signal_id,
                symbol=signal.symbol,
                side=side,
                qty=decision.qty,
                entry_price=entry,
                avg_entry_price=entry,
                entry_ts=bar.ts_close,
                margin_usdt=decision.margin_usdt,
                stop_loss=decision.stop_loss,
                take_profits=[],
                liq_price=decision.est_liq_price,
                opened_by=signal.strategy,
                signal_id=signal.signal_id,
                entry_atr=signal.meta.get("atr"),
                target_qty=decision.qty,
                add_levels=[AddLevel(price=entry, qty=decision.qty, ts=bar.ts_close)],
                base_stop_price=decision.stop_loss or 0.0,
                trailing_peak_price=entry,
                tp_price=tp_price or 0.0,
                entry_fee_outstanding=fee,
            )
            self.position = pos
            self.cash -= decision.margin_usdt
            self.cash -= fee
            pos.add_levels[0].fee_usdt = fee
            self.fills.append(Fill(
                ts=bar.ts_close, symbol=signal.symbol,
                side=Side.BUY if side == "long" else Side.SELL,
                qty=decision.qty, price=entry, fee_usdt=fee,
                reason=FillReason.SIGNAL, position_id=pos.position_id,
            ))
            self.bars_since_entry = 0
            self.funding_since_entry = 0.0
            return ExecEvent("open", bar.ts_close, signal, decision, position_id=pos.position_id)

        pos = self.position
        prev_qty = pos.qty
        new_qty = prev_qty + decision.qty
        pos.avg_entry_price = (pos.avg_entry_price * prev_qty + entry * decision.qty) / new_qty
        pos.qty = new_qty
        pos.add_count += 1
        pos.margin_usdt += decision.margin_usdt
        pos.add_levels.append(AddLevel(price=entry, qty=decision.qty, ts=bar.ts_close))
        self.cash -= decision.margin_usdt
        self.cash -= fee
        pos.add_levels[-1].fee_usdt = fee
        pos.entry_fee_outstanding += fee
        self.fills.append(Fill(
            ts=bar.ts_close, symbol=signal.symbol,
            side=Side.BUY if pos.side == "long" else Side.SELL,
            qty=decision.qty, price=entry, fee_usdt=fee,
            reason=FillReason.SIGNAL, position_id=pos.position_id,
        ))
        sync_stop_after_add(pos, cfg, entry)
        self.add_count_total += 1
        return ExecEvent("add", bar.ts_close, signal, decision, position_id=pos.position_id)

    def close_position(
        self,
        bar: Bar,
        reason: FillReason,
        price: float,
        exit_time: datetime | None = None,
    ) -> ExecEvent | None:
        pos = self.position
        if pos is None:
            return None
        cfg = self.cfg
        ref_price = pos.avg_entry_price if pos.add_count > 0 else pos.entry_price
        if pos.side == "long":
            pnl = pos.qty * (price - ref_price)
            side = Side.SELL
        else:
            pnl = pos.qty * (ref_price - price)
            side = Side.BUY
        # 开仓腿手续费已在成交时从现金扣过，平仓只结算平仓腿，否则同一笔开仓费会被扣两次
        # （现金口径与逐笔 pnl 口径不一致）。分批止盈后剩余仓位只背自己那一份，
        # 所以取 entry_fee_outstanding 而不是 add_levels 的全部。
        fee_exit = fee_qty(pos.qty, price, cfg.taker_fee)
        fee_entry_paid = pos.entry_fee_outstanding
        net_cash = pnl - fee_exit - self.funding_since_entry
        self.cash += pos.margin_usdt + net_cash
        net = pnl - fee_entry_paid - fee_exit - self.funding_since_entry

        r_ach = 0.0
        if pos.entry_atr and pos.entry_atr > 0:
            # 符号跟随盈亏方向：多单涨、空单跌为正 R（不能按 side 直接取反）
            move = price - ref_price if pos.side == "long" else ref_price - price
            r_ach = float(move / (cfg.sl_atr_mult * pos.entry_atr))

        self.fills.append(Fill(
            ts=bar.ts_close, symbol=pos.symbol, side=side,
            qty=pos.qty, price=price, fee_usdt=fee_exit, pnl_usdt=net,
            reason=reason, position_id=pos.position_id,
        ))
        trade = TradeRecord(
            entry_time=pos.entry_ts,
            exit_time=exit_time if exit_time is not None else bar.ts_close,
            side=pos.side, entry_price=pos.entry_price, exit_price=price,
            qty=pos.qty, pnl=net, fee=fee_entry_paid + fee_exit,
            reason=reason.value,
            r_achieved=r_ach, bars_held=self.bars_since_entry,
            add_count=pos.add_count,
            avg_entry_price=pos.avg_entry_price,
            initial_qty=pos.add_levels[0].qty if pos.add_levels else pos.qty,
            margin_usdt=pos.margin_usdt,
            ret_pct=net / pos.margin_usdt * 100 if pos.margin_usdt > 0 else 0.0,
        )
        self.trades.append(trade)
        signal = None
        self.position = None
        self.bars_since_entry = 0
        self.funding_since_entry = 0.0
        return ExecEvent("close", trade.exit_time, signal, None, trade, position_id=pos.position_id)

    def close_partial(
        self,
        bar: Bar,
        qty_part: float,
        price: float,
        reason: FillReason,
        exit_time: datetime | None = None,
    ) -> ExecEvent | None:
        """分批止盈：平掉一部分，剩余仓位继续持有（保证金与费用按平仓比例分摊）。"""
        pos = self.position
        cfg = self.cfg
        if pos is None or qty_part <= 0 or qty_part >= pos.qty:
            return None
        ref_price = pos.avg_entry_price if pos.add_count > 0 else pos.entry_price
        share = qty_part / pos.qty
        gross = qty_part * (price - ref_price) if pos.side == "long" else qty_part * (ref_price - price)
        fee_exit = fee_qty(qty_part, price, cfg.taker_fee)
        fee_entry_part = pos.entry_fee_outstanding * share
        margin_part = pos.margin_usdt * share
        funding_part = self.funding_since_entry * share

        self.cash += margin_part + gross - fee_exit
        net = gross - fee_entry_part - fee_exit - funding_part
        self.funding_since_entry -= funding_part
        pos.funding_paid *= 1.0 - share
        pos.entry_fee_outstanding -= fee_entry_part
        pos.qty -= qty_part
        pos.margin_usdt -= margin_part

        r_ach = 0.0
        if pos.entry_atr and pos.entry_atr > 0:
            move = price - ref_price if pos.side == "long" else ref_price - price
            r_ach = float(move / (cfg.sl_atr_mult * pos.entry_atr))

        self.fills.append(Fill(
            ts=bar.ts_close, symbol=pos.symbol,
            side=Side.SELL if pos.side == "long" else Side.BUY,
            qty=qty_part, price=price, fee_usdt=fee_exit, pnl_usdt=net,
            reason=reason, position_id=pos.position_id,
        ))
        trade = TradeRecord(
            entry_time=pos.entry_ts,
            exit_time=exit_time if exit_time is not None else bar.ts_close,
            side=pos.side, entry_price=pos.entry_price, exit_price=price,
            qty=qty_part, pnl=net, fee=fee_entry_part + fee_exit,
            reason=reason.value,
            r_achieved=r_ach, bars_held=self.bars_since_entry,
            add_count=pos.add_count,
            avg_entry_price=pos.avg_entry_price,
            initial_qty=pos.add_levels[0].qty if pos.add_levels else pos.qty,
            margin_usdt=margin_part,
            ret_pct=net / margin_part * 100 if margin_part > 0 else 0.0,
        )
        self.trades.append(trade)
        return ExecEvent("partial", trade.exit_time, None, None, trade, position_id=pos.position_id)

    def settle_tp1(self, bar: Bar) -> ExecEvent | None:
        """到 `tp1_r` 平掉 `tp1_close_frac`，同时把止损提到保本位。

        拆不出合规数量（余量或平仓量低于 `min_qty`）时只标 `tp1_done`，不重试：
        小本金账户在试仓阶段就会出现这种情况，保本移动仍然生效。
        """
        cfg = self.cfg
        pos = self.position
        if pos is None or not tp1_hit(pos, bar, cfg):
            return None
        pos.tp1_done = True
        px = r_price(pos, cfg, cfg.tp1_r)
        apply_breakeven(pos, cfg, px)
        qty_part = floor_to(pos.qty * cfg.tp1_close_frac, float(self.spec.step_size))
        remain = pos.qty - qty_part
        if qty_part < float(self.spec.min_qty) or remain < float(self.spec.min_qty):
            return None
        adverse = 1 - cfg.slippage_bps / 10000 if pos.side == "long" else 1 + cfg.slippage_bps / 10000
        return self.close_partial(bar, qty_part, round(px * adverse, 8), FillReason.TP1,
                                  exit_time=bar_mid(bar))

    def close_at_open(self, bar: Bar, reason: FillReason = FillReason.SIGNAL) -> ExecEvent | None:
        """按本根 bar 开盘价（不利滑点）平掉当前持仓。信号平仓与反手共用。"""
        pos = self.position
        if pos is None:
            return None
        return self.close_position(bar, reason, exit_price_at_open(bar, pos, self.cfg.slippage_bps))

    def settle_price_exits(self, bar: Bar) -> list[ExecEvent]:
        """价位触发的出场，含双触发的保守定序与 TP1 分批止盈。

        一根 bar 里可以既吃到 1R 的部分止盈、又吃到 3.5R 的全平，所以返回的是列表。
        保本后的新止损在**同一根 bar** 就参与判定（与移动止损收紧后立刻按 bar 极值判
        触发是同一套保守约定）；本根 bar 没能越过旧止损，才轮到强平兜底。
        """
        cfg = self.cfg
        pos = self.position
        if pos is None:
            return []
        events: list[ExecEvent] = []
        reason = exit_reason(pos, bar, cfg.sl_first_on_conflict)
        if reason is FillReason.SL:
            ev = self.close_position(
                bar, FillReason.SL, stop_fill_price(bar, pos.side, cfg.slippage_bps),
                exit_time=bar_mid(bar),
            )
            return [ev] if ev else events
        if reason is None and liq_hit(pos, bar):
            ev = self.close_position(
                bar, FillReason.LIQ, liq_fill_price(pos, cfg.slippage_bps),
                exit_time=bar_mid(bar),
            )
            return [ev] if ev else events

        ev = self.settle_tp1(bar)
        if ev is not None:
            events.append(ev)
        pos = self.position
        if pos is None:
            return events
        if reason is None and stop_hit(pos, bar):
            ev = self.close_position(
                bar, FillReason.SL, stop_fill_price(bar, pos.side, cfg.slippage_bps),
                exit_time=bar_mid(bar),
            )
            if ev:
                events.append(ev)
            return events
        if reason is FillReason.TP or tp_hit(pos, bar):
            ev = self.close_position(
                bar, FillReason.TP, tp_fill_price(pos, cfg.slippage_bps),
                exit_time=bar_mid(bar),
            )
            if ev:
                events.append(ev)
        return events

    def advance_time_stop(self, bar: Bar) -> ExecEvent | None:
        """持仓根数 +1，达到 `time_stop_bars` 则按本根收盘（不利滑点）平仓。"""
        pos = self.position
        if pos is None:
            return None
        self.bars_since_entry += 1
        if self.bars_since_entry >= self.cfg.time_stop_bars:
            return self.close_position(
                bar, FillReason.TIME,
                slipped_close_price(bar, pos.side, self.cfg.slippage_bps),
            )
        return None

    def settle_exits(self, bar: Bar) -> list[ExecEvent]:
        """出场序列：先价位触发（SL/强平/TP1/TP），再时间止损。回测与 paper 同序。"""
        events = self.settle_price_exits(bar)
        ev = self.advance_time_stop(bar)
        if ev is not None:
            events.append(ev)
        return events

    # ---------- 信号 → 成交 ----------

    def execute_signal(self, signal: Signal, bar: Bar, row: pd.Series) -> list[ExecEvent]:
        """执行一根 bar 前收盘产生的信号：平仓 / 试仓 / 加仓。"""
        cfg = self.cfg
        events: list[ExecEvent] = []

        if signal.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
            pos = self.position
            if pos is None or (pos.side == "long") != (signal.action == Action.CLOSE_LONG):
                return events
            ev = self.close_at_open(bar)
            return [ev] if ev else events

        if signal.action not in (
            Action.OPEN_LONG, Action.OPEN_SHORT, Action.ADD_TO_LONG, Action.ADD_TO_SHORT,
        ):
            return events

        if signal.meta.get("atr"):
            atr_val = float(signal.meta["atr"])
        else:
            atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else signal.price_ref * 0.005
            signal.meta["atr"] = atr_val

        eq = self.equity_open
        if self.position is not None and (self.position.side == "long") != is_long_action(signal.action):
            # 反手：先按本根 bar 开盘价平掉对向持仓，再在同一点位开新仓。
            ev = self.close_at_open(bar)
            if ev is not None:
                events.append(ev)
            eq = self.cash

        decision = decide_position(
            signal=signal, equity=eq, atr=atr_val, cfg=cfg, spec=self.spec,
            is_add=self.position is not None,
        )
        if not decision.approved:
            events.append(ExecEvent("rejected", bar.ts_close, signal, decision))
            return events

        # 闸门只拦"新增敞口"（开仓与加仓），绝不介入已有持仓的出场：paper 的平仓路径
        # 必须和回测逐位一致，否则风控触发时刻的偏差会被当成策略差异来解读。
        if self.approve_fn is not None:
            reasons = self.approve_fn(decision)
            if reasons:
                events.append(ExecEvent("rejected", bar.ts_close, signal, veto(decision, reasons)))
                return events

        if self.position is not None:
            if cfg.pyramid_enabled and should_add(self.position, cfg, row, bar):
                events.append(self.open_or_add(signal, decision, bar))
            elif cfg.pyramid_enabled:
                self.add_skip_count += 1
                events.append(ExecEvent("add_skipped", bar.ts_close, signal, decision))
        else:
            events.append(self.open_or_add(signal, decision, bar))
        return events


# ---- 单根 bar 的驱动序列 ----------------------------------------------------


class HtfCursor:
    """高周期游标：按 5m 结算节奏弹出"已收盘"的高周期指标行。

    `t_vis_ms` 是高周期 bar 的可见时刻（ts_open + 周期），只有可见时刻不晚于当前
    基础 bar 收盘的高周期 bar 才能被策略看到，否则就是前视。
    """

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self._frames = {
            iv: df.sort_values("ts_open_ms").reset_index(drop=True)
            for iv, df in frames.items()
            if df is not None and not df.empty
        }
        self._idx = {iv: 0 for iv in self._frames}
        self._row: dict[str, pd.Series] = {}

    def advance(self, t_vis_ms: int) -> dict[str, pd.Series]:
        for iv, df in self._frames.items():
            step = interval_ms(iv)
            n = len(df)
            i = self._idx[iv]
            while i < n:
                row = df.iloc[i]
                if int(row["ts_open_ms"]) + step <= t_vis_ms:
                    self._row[iv] = row
                    i += 1
                else:
                    break
            self._idx[iv] = i
        return dict(self._row)

    def last_visible_ms(self, interval: str) -> int | None:
        row = self._row.get(interval)
        return None if row is None else int(row["ts_open_ms"])


SignalFn = Callable[[Bar, pd.Series, dict[str, pd.Series], datetime], Signal | None]


class BarPipeline:
    """把一根已收盘的基础周期 bar 走完整套执行序列，回测与 paper 共用。

    顺序不能改（每一环都依赖前一环的状态，且都只用本根 bar 收盘前已知的信息）：
      equity_open 定基 → 净值快照 → 资金费计提 → 执行上根 bar  pending 信号
      → 移动止损 → SL/TP/时间止损 → 调策略产生新信号

    mode：
      * `live` —— 完整序列（回测窗口内、paper 新 bar）。
      * `warmup` —— 只做净值快照与资金费计提，撮合与策略都不做（回测窗口最前面那段，
        高周期指标还没脱离 NaN）。
      * `observe` —— 只调策略、丢弃其返回值，账本一行不写。paper 重启后用它把策略的
        内部状态（趋势状态机、冷却计数）推平到断点时刻；这一模式绝不能出现在回测里，
        否则策略的 bar 计数会错位。
    """

    def __init__(
        self,
        cfg: Any,
        ledger: PositionLedger,
        signal_fn: SignalFn | None = None,
    ):
        self.cfg = cfg
        self.ledger = ledger
        self.signal_fn = signal_fn
        self.pending: Signal | None = None

    def on_bar(
        self,
        bar: Bar,
        row: pd.Series,
        htf: dict[str, pd.Series],
        *,
        mode: str = "live",
    ) -> list[ExecEvent]:
        led = self.ledger
        t_vis_ms = to_ms(bar.ts_close) + 1

        if mode == "observe":
            if self.signal_fn is not None:
                # 无条件覆盖（含 None）：只有断点那根 bar 的信号还没被撮合过，留着它才能在
                # 重启后的第一根 live bar 成交。更早的信号已在上一轮 live 里成交，
                # 不覆盖掉就会晚一根 bar 重复开仓。
                self.pending = self.signal_fn(bar, row, htf, bar.ts_close)
            return []

        # 本根 bar 开盘可支配的权益 = 上一根收盘的净值（定仓只用成交前已知的数）
        led.equity_open = led.equity
        led.snapshot(bar)
        led.accrue_funding(t_vis_ms, float(bar.close))

        # 预热期只做净值与资金费的记账，不撮合、不调策略
        if mode != "live":
            return []

        events: list[ExecEvent] = []
        if self.pending is not None:
            events += led.execute_signal(self.pending, bar, row)
            self.pending = None

        update_trailing(led.position, self.cfg, bar)
        events += led.settle_exits(bar)

        if self.signal_fn is not None:
            signal = self.signal_fn(bar, row, htf, bar.ts_close)
            if signal is not None:
                self.pending = signal
        return events

    def finalize(self, bar: Bar | None) -> ExecEvent | None:
        """期末强制平仓（回测收尾用）。"""
        if bar is None or self.ledger.position is None:
            return None
        ev = self.ledger.close_position(
            bar, FillReason.FLAT,
            slipped_close_price(bar, self.ledger.position.side, self.cfg.slippage_bps),
        )
        if self.ledger.equity_curve:
            # 收尾结算必须回写最后一个净值点：循环内快照是持仓 MTM，不含这笔的滑点、
            # 平仓费与资金费，否则 sum(trades.pnl) 与 Δequity 恒等式会在尾部差一个量。
            self.ledger.equity = self.ledger.cash
            self.ledger.equity_curve[-1] = self.ledger.equity
        return ev
