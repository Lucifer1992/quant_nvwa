"""纸面账户状态 ↔ `paper_*` 表的映射。

落库/恢复必须覆盖到"进程随时可以被杀掉"这一程度：`commit()` 把一根 bar 产生的账本、
持仓、成交流水、信号记录与断点放在同一个事务里写，崩溃后重启只会从断点那根 bar 之后
继续，绝不会把同一根 bar 撮两遍。

`paper_accounts.funding_ptr` 只是账本当时的数组下标快照，恢复时**不采用**：每轮的加载
窗口起点会变，下标没有跨轮意义，重定位一律按时刻做（`PositionLedger.seek_funding`）。
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..backtest.position import PositionLedger, TradeRecord
from ..core.contracts import AddLevel, Position, SymbolSpec
from ..core.timeframe import to_dt, to_ms
from ..data.store import Store
from .config import PaperConfig

KIND_OPEN = "open"
KIND_ADD = "add"
KIND_CLOSE = "close"
KIND_REJECT = "reject"
KIND_ALERT = "alert"


def row_id(symbol: str, strategy: str, native_id: str) -> str:
    """策略自己生成的 id 只含策略名与时刻，跨品种会撞车（同分钟两个品种同时出信号，
    后写的会被主键吞掉），落库统一加 (symbol, strategy) 前缀。"""
    return f"{symbol}:{strategy}:{native_id}"


def alert_id(symbol: str, strategy: str, tag: str) -> str:
    """告警也走 `paper_signals` 主键去重：同一原因在同一个断点上只推一次。"""
    return f"{symbol}:{strategy}:alert:{tag}"


def position_row(pos: Position, symbol: str, strategy: str, now_ms: int, status: str = "open") -> dict[str, Any]:
    return {
        "position_id": row_id(symbol, strategy, pos.position_id),
        "symbol": symbol,
        "strategy": strategy,
        "side": pos.side,
        "status": status,
        "qty": pos.qty,
        "entry_price": pos.entry_price,
        "avg_entry_price": pos.avg_entry_price,
        "margin_usdt": pos.margin_usdt,
        "entry_atr": pos.entry_atr,
        "stop_loss": pos.stop_loss,
        "base_stop_price": pos.base_stop_price,
        "tp_price": pos.tp_price,
        "liq_price": pos.liq_price,
        "add_count": pos.add_count,
        "add_levels_json": json.dumps([
            {"price": lv.price, "qty": lv.qty, "ts_ms": to_ms(lv.ts), "fee_usdt": lv.fee_usdt}
            for lv in pos.add_levels
        ]),
        "trailing_active": int(pos.trailing_active),
        "trailing_peak_price": pos.trailing_peak_price,
        "tp1_done": int(pos.tp1_done),
        "breakeven_done": int(pos.breakeven_done),
        "entry_fee_outstanding": pos.entry_fee_outstanding,
        "funding_paid": pos.funding_paid,
        "signal_id": pos.signal_id,
        "opened_at_ms": to_ms(pos.entry_ts),
        "updated_at_ms": now_ms,
    }


def position_from_row(row: dict[str, Any]) -> Position:
    levels = [
        AddLevel(price=float(l["price"]), qty=float(l["qty"]), ts=to_dt(int(l["ts_ms"])),
                 fee_usdt=float(l.get("fee_usdt", 0.0)))
        for l in json.loads(row["add_levels_json"] or "[]")
    ]
    native_id = str(row["signal_id"] or row["position_id"])
    return Position(
        position_id=native_id,
        symbol=str(row["symbol"]),
        side=str(row["side"]),  # type: ignore[arg-type]
        qty=float(row["qty"]),
        entry_price=float(row["entry_price"]),
        entry_ts=to_dt(int(row["opened_at_ms"])),
        margin_usdt=float(row["margin_usdt"]),
        stop_loss=float(row["stop_loss"] or 0.0),
        take_profits=(),
        liq_price=None if row["liq_price"] is None else float(row["liq_price"]),
        opened_by=str(row["strategy"]),
        signal_id=native_id,
        funding_paid=float(row["funding_paid"] or 0.0),
        entry_atr=None if row["entry_atr"] is None else float(row["entry_atr"]),
        status="open",
        target_qty=float(levels[0].qty) if levels else float(row["qty"]),
        avg_entry_price=float(row["avg_entry_price"]),
        add_count=int(row["add_count"] or 0),
        add_levels=levels,
        trailing_active=bool(row["trailing_active"]),
        trailing_peak_price=float(row["trailing_peak_price"] or 0.0),
        base_stop_price=float(row["base_stop_price"] or 0.0),
        tp_price=float(row["tp_price"] or 0.0),
        tp1_done=bool(row["tp1_done"]),
        breakeven_done=bool(row["breakeven_done"]),
        entry_fee_outstanding=float(row["entry_fee_outstanding"] or 0.0),
    )


def trade_row(tr: TradeRecord, position_id: str, symbol: str, strategy: str) -> dict[str, Any]:
    return {
        # 分批止盈会让一笔仓位留下多条成交流水，主键必须带上出场时刻与原因；
        # 只用 position_id 时，TP1 那条会被之后的全平覆盖掉，账目与净值就对不上了。
        "trade_id": f"{position_id}:{tr.reason}:{to_ms(tr.exit_time)}",
        "position_id": position_id,
        "symbol": symbol,
        "strategy": strategy,
        "side": tr.side,
        "entry_time_ms": to_ms(tr.entry_time),
        "exit_time_ms": to_ms(tr.exit_time),
        "entry_price": tr.entry_price,
        "exit_price": tr.exit_price,
        "qty": tr.qty,
        "pnl": tr.pnl,
        "fee": tr.fee,
        "reason": tr.reason,
        "r_achieved": tr.r_achieved,
        "bars_held": tr.bars_held,
        "add_count": tr.add_count,
        "margin_usdt": tr.margin_usdt,
        "ret_pct": tr.ret_pct,
    }


def signal_row(
    *,
    signal_id: str,
    ts_ms: int,
    symbol: str,
    strategy: str,
    action: str,
    price_ref: float,
    reason: str,
    kind: str,
    created_at_ms: int,
    qty: float | None = None,
    margin_usdt: float | None = None,
    stop_loss: float | None = None,
    tp_price: float | None = None,
    confidence: float | None = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "signal_id": signal_id,
        "ts_signal_ms": ts_ms,
        "symbol": symbol,
        "strategy": strategy,
        "action": action,
        "price_ref": price_ref,
        "reason": reason,
        "qty": qty,
        "margin_usdt": margin_usdt,
        "stop_loss": stop_loss,
        "tp_price": tp_price,
        "confidence": confidence,
        "kind": kind,
        "note": note,
        "payload_json": "",
        "sent_ok": 0,
        "sent_at_ms": None,
        "created_at_ms": created_at_ms,
    }


class PaperState:
    """一个 (symbol, strategy) 纸面账户的持久化入口。"""

    def __init__(self, store: Store, symbol: str, strategy: str):
        self.store = store
        self.symbol = symbol
        self.strategy = strategy

    # ---------- 读 ----------

    def account(self) -> dict[str, Any] | None:
        return self.store.paper_account(self.symbol, self.strategy)

    def checkpoint_ms(self) -> int | None:
        return self.store.paper_checkpoint_ms(self.symbol, self.strategy)

    def pause_until_ms(self) -> int:
        acct = self.account()
        return 0 if acct is None else int(acct.get("pause_until_ms") or 0)

    def open_ledger(
        self,
        cfg: PaperConfig,
        spec: SymbolSpec,
        funding: pd.DataFrame | None,
        *,
        resumed_at_ms: int | None,
    ) -> PositionLedger:
        """从库里恢复账本：现金/净值/计数器 + 当前持仓 + 资金费指针。"""
        ledger = PositionLedger(cfg, spec)
        acct = self.account()
        if acct is None:
            ledger.reset(cfg.initial_equity)
        else:
            ledger.reset(float(acct["initial_equity"]))
            ledger.restore_state(acct)
        row = self.store.paper_open_position(self.symbol, self.strategy)
        if row is not None:
            ledger.position = position_from_row(row)
        if funding is not None and not funding.empty:
            ledger.set_funding(funding.sort_values("funded_at_ms").reset_index(drop=True))
        else:
            ledger.set_funding(None)
        # 断点之前（含断点那根 bar 的可见时刻）的结算都已计提过，指针必须跳到其后
        if resumed_at_ms is not None:
            ledger.seek_funding(resumed_at_ms + 1)
        return ledger

    # ---------- 写 ----------

    def record_signal(self, row: dict[str, Any], card: dict[str, Any]) -> dict[str, Any] | None:
        """插入信号行并带上卡片快照；主键已存在（重放/重跑）时返回 None，表示不重发。"""
        payload = json.dumps(card, ensure_ascii=False)
        if not self.store.insert_paper_signal(row):
            return None
        # payload 先落库再推送：投递失败时补发用的就是这张当时的卡片，
        # 事后重建会用到已经变了的行情数据，卡片与成交记录会对不上
        self.store.mark_paper_signal_sent(row["signal_id"], False, 0, payload)
        return {**row, "payload_json": payload}

    def add_alert(self, row: dict[str, Any], card: dict[str, Any]) -> dict[str, Any] | None:
        """告警独立落库：它不改账本，所以不能借 `commit()` 一起写。"""
        with self.store.batch():
            return self.record_signal(row, card)

    def commit(
        self,
        ledger: PositionLedger,
        *,
        now_ms: int,
        checkpoint_ms: int | None,
        dataset_version: str,
        signals: list[tuple[dict[str, Any], dict[str, Any]]],
        trades: list[tuple[str, TradeRecord]],
        pause_until_ms: int = 0,
    ) -> list[dict[str, Any]]:
        """把本轮的账本与派生记录原子落库。

        `signals` 是 (信号行, 卡片) 的配对。返回真正新插入的那些 —— 已存在的行说明
        这根 bar 是重放，不重发。

        `pause_until_ms` 是闸门的冷却到期时刻，和账本写进同一个事务：崩溃只会退回到
        "冷却少算一次"，不会出现账本已回滚、冷却还生效的状态。
        """
        store = self.store
        inserted: list[dict[str, Any]] = []
        with store.batch():
            acct = dict(ledger.dump_state())
            acct.update({
                "symbol": self.symbol,
                "strategy": self.strategy,
                "initial_equity": ledger.cfg.initial_equity,
                "pause_until_ms": pause_until_ms,
                "updated_at_ms": now_ms,
            })
            store.save_paper_account(acct)

            # 先清掉所有 open 行，再按账本的最终状态写回：中途被杀留下的陈旧 open 行
            # 会被归位，本轮开平仓反复也不会留下两条活仓
            store.mark_paper_positions_closed(self.symbol, self.strategy, now_ms)
            if ledger.position is not None:
                store.save_paper_position(
                    position_row(ledger.position, self.symbol, self.strategy, now_ms)
                )

            for position_id, tr in trades:
                store.paper_trade(trade_row(tr, position_id, self.symbol, self.strategy))

            for row, card in signals:
                got = self.record_signal(row, card)
                if got is not None:
                    inserted.append(got)

            if checkpoint_ms is not None:
                store.save_paper_checkpoint({
                    "symbol": self.symbol,
                    "strategy": self.strategy,
                    "last_processed_ts_ms": checkpoint_ms,
                    "dataset_version": dataset_version,
                    "updated_at_ms": now_ms,
                })
        return inserted
