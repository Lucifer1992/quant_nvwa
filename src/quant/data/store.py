from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Self

import pandas as pd

from ..core.contracts import Bar, FundingBar
from ..core.timeframe import source_rank, to_ms

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    time_interval TEXT NOT NULL,
    ts_open_ms INTEGER NOT NULL,
    ts_close_ms INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, quote_volume REAL, taker_buy_volume REAL,
    trades INTEGER,
    is_closed INTEGER NOT NULL,
    src TEXT NOT NULL,
    prio INTEGER NOT NULL,
    PRIMARY KEY (symbol, time_interval, ts_open_ms)
);
CREATE TABLE IF NOT EXISTS funding (
    symbol TEXT NOT NULL,
    funded_at_ms INTEGER NOT NULL,
    funding_rate REAL,
    mark_price REAL,
    rate_type TEXT,
    interval_h REAL,
    src TEXT,
    PRIMARY KEY (symbol, funded_at_ms)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_accounts (
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    initial_equity REAL NOT NULL,
    cash REAL NOT NULL,
    equity REAL NOT NULL,
    equity_open REAL NOT NULL,
    peak_equity REAL NOT NULL DEFAULT 0,
    pause_until_ms INTEGER NOT NULL DEFAULT 0,
    bars_since_entry INTEGER NOT NULL DEFAULT 0,
    funding_since_entry REAL NOT NULL DEFAULT 0,
    funding_ptr INTEGER NOT NULL DEFAULT 0,
    add_count_total INTEGER NOT NULL DEFAULT 0,
    add_skip_count INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (symbol, strategy)
);
CREATE TABLE IF NOT EXISTS paper_positions (
    position_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,
    status TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    margin_usdt REAL NOT NULL,
    entry_atr REAL,
    stop_loss REAL,
    base_stop_price REAL,
    tp_price REAL,
    liq_price REAL,
    add_count INTEGER NOT NULL DEFAULT 0,
    add_levels_json TEXT NOT NULL DEFAULT '[]',
    trailing_active INTEGER NOT NULL DEFAULT 0,
    trailing_peak_price REAL NOT NULL DEFAULT 0,
    tp1_done INTEGER NOT NULL DEFAULT 0,
    breakeven_done INTEGER NOT NULL DEFAULT 0,
    entry_fee_outstanding REAL NOT NULL DEFAULT 0,
    funding_paid REAL NOT NULL DEFAULT 0,
    signal_id TEXT NOT NULL,
    opened_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS paper_positions_open
    ON paper_positions (symbol, strategy, status);
CREATE TABLE IF NOT EXISTS paper_signals (
    signal_id TEXT PRIMARY KEY,
    ts_signal_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    action TEXT NOT NULL,
    price_ref REAL NOT NULL,
    reason TEXT NOT NULL,
    qty REAL,
    margin_usdt REAL,
    stop_loss REAL,
    tp_price REAL,
    confidence REAL,
    kind TEXT NOT NULL DEFAULT 'open',
    note TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '',
    sent_ok INTEGER NOT NULL DEFAULT 0,
    sent_at_ms INTEGER,
    created_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS paper_signals_pending ON paper_signals (sent_ok, created_at_ms);
CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_time_ms INTEGER NOT NULL,
    exit_time_ms INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    qty REAL NOT NULL,
    pnl REAL NOT NULL,
    fee REAL NOT NULL,
    reason TEXT NOT NULL,
    r_achieved REAL NOT NULL DEFAULT 0,
    bars_held INTEGER NOT NULL DEFAULT 0,
    add_count INTEGER NOT NULL DEFAULT 0,
    margin_usdt REAL NOT NULL DEFAULT 0,
    ret_pct REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS paper_trades_by_strategy
    ON paper_trades (symbol, strategy, exit_time_ms);
CREATE TABLE IF NOT EXISTS paper_checkpoints (
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    last_processed_ts_ms INTEGER NOT NULL,
    dataset_version TEXT NOT NULL DEFAULT '',
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (symbol, strategy)
);
"""

BAR_COLS = [
    "symbol", "time_interval", "ts_open_ms", "ts_close_ms", "open", "high", "low", "close",
    "volume", "quote_volume", "taker_buy_volume", "trades", "is_closed", "src", "prio",
]

_UPSERT_BARS = f"""
INSERT INTO bars ({", ".join(BAR_COLS)})
VALUES ({", ".join("?" * len(BAR_COLS))})
ON CONFLICT(symbol, time_interval, ts_open_ms) DO UPDATE SET
    ts_close_ms = excluded.ts_close_ms,
    open = excluded.open, high = excluded.high, low = excluded.low, close = excluded.close,
    volume = excluded.volume, quote_volume = excluded.quote_volume,
    taker_buy_volume = excluded.taker_buy_volume, trades = excluded.trades,
    is_closed = excluded.is_closed, src = excluded.src, prio = excluded.prio
WHERE excluded.prio >= bars.prio
"""

_UPSERT_FUNDING = """
INSERT INTO funding (symbol, funded_at_ms, funding_rate, mark_price, rate_type, interval_h, src)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, funded_at_ms) DO UPDATE SET
    funding_rate = excluded.funding_rate,
    mark_price = COALESCE(excluded.mark_price, funding.mark_price),
    rate_type = excluded.rate_type,
    interval_h = excluded.interval_h,
    src = excluded.src
"""

# 各 paper 表的列清单：_upsert 只接受这里声明过的列名，防止把拼错的键静默写进 SQL
PAPER_TABLE_COLS: dict[str, tuple[str, ...]] = {
    "paper_accounts": (
        "symbol", "strategy", "initial_equity", "cash", "equity", "equity_open",
        "peak_equity", "pause_until_ms",
        "bars_since_entry", "funding_since_entry", "funding_ptr",
        "add_count_total", "add_skip_count", "updated_at_ms",
    ),
    "paper_positions": (
        "position_id", "symbol", "strategy", "side", "status", "qty", "entry_price",
        "avg_entry_price", "margin_usdt", "entry_atr", "stop_loss", "base_stop_price",
        "tp_price", "liq_price", "add_count", "add_levels_json", "trailing_active",
        "trailing_peak_price", "tp1_done", "breakeven_done", "entry_fee_outstanding",
        "funding_paid", "signal_id", "opened_at_ms", "updated_at_ms",
    ),
    "paper_signals": (
        "signal_id", "ts_signal_ms", "symbol", "strategy", "action", "price_ref", "reason",
        "qty", "margin_usdt", "stop_loss", "tp_price", "confidence", "kind", "note",
        "payload_json", "sent_ok", "sent_at_ms", "created_at_ms",
    ),
    "paper_trades": (
        "trade_id", "position_id", "symbol", "strategy", "side", "entry_time_ms",
        "exit_time_ms", "entry_price", "exit_price", "qty", "pnl", "fee", "reason",
        "r_achieved", "bars_held", "add_count", "margin_usdt", "ret_pct",
    ),
    "paper_checkpoints": (
        "symbol", "strategy", "last_processed_ts_ms", "dataset_version", "updated_at_ms",
    ),
}

PAPER_TABLE_PK: dict[str, tuple[str, ...]] = {
    "paper_accounts": ("symbol", "strategy"),
    "paper_positions": ("position_id",),
    "paper_signals": ("signal_id",),
    "paper_trades": ("trade_id",),
    "paper_checkpoints": ("symbol", "strategy"),
}


PAPER_MIGRATIONS: dict[str, tuple[str, ...]] = {
    # 老库里的 paper_accounts 缺这两列：[paper] 回撤闸门要峰值，冷却闸门要暂停到期时刻
    "paper_accounts": (
        "peak_equity REAL NOT NULL DEFAULT 0",
        "pause_until_ms INTEGER NOT NULL DEFAULT 0",
    ),
}


class Store:
    """本地 SQLite 数据仓库。bar 主键保证同 (symbol, interval, ts_open) 唯一，写入天然幂等。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._tx_depth = 0
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._migrate_paper()

    def _migrate_paper(self) -> None:
        """`SCHEMA` 只建缺失表，已存在的表要按列清单补列（新闸门字段走这里）。"""
        for table, additions in PAPER_MIGRATIONS.items():
            have = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            for ddl in additions:
                if ddl.split()[0] not in have:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- write ----------

    def upsert_bars(self, bars: Iterable[Bar]) -> int:
        rows = [
            (
                b.symbol, b.interval, to_ms(b.ts_open), to_ms(b.ts_close),
                b.open, b.high, b.low, b.close, b.volume, b.quote_volume, b.taker_buy_volume,
                int(b.trades), int(bool(b.is_closed)), b.src, source_rank(b.is_closed, b.src),
            )
            for b in bars
        ]
        if not rows:
            return 0
        before = self._conn.total_changes
        with self._conn:
            self._conn.executemany(_UPSERT_BARS, rows)
        # prio 守卫会拒掉低优先级的重复行，len(rows) 是"尝试写入数"而非真实落库数
        return int(self._conn.total_changes - before)

    def upsert_funding(self, rows: Sequence[FundingBar]) -> int:
        data = [
            (
                f.symbol, to_ms(f.funding_time), f.funding_rate, f.mark_price,
                f.rate_type, f.interval_hours, f.src,
            )
            for f in rows
        ]
        if not data:
            return 0
        with self._conn:
            self._conn.executemany(_UPSERT_FUNDING, data)
        return len(data)

    # ---------- read ----------

    def load_bars(
        self, symbol: str, interval: str, start_ms: int | None = None, end_ms: int | None = None,
        closed_only: bool = True,
    ) -> pd.DataFrame:
        sql = f"SELECT {', '.join(BAR_COLS)} FROM bars WHERE symbol = ? AND time_interval = ?"
        params: list[object] = [symbol, interval]
        if start_ms is not None:
            sql += " AND ts_open_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            sql += " AND ts_open_ms < ?"
            params.append(end_ms)
        if closed_only:
            sql += " AND is_closed = 1"
        sql += " ORDER BY ts_open_ms"
        df = pd.read_sql_query(sql, self._conn, params=params)
        if not df.empty:
            # 库里用 time_interval 避开 SQL 保留字风险，取回内存后统一还原成 contracts.Bar 的字段名
            df = df.rename(columns={"time_interval": "interval"})
            df["ts_open"] = pd.to_datetime(df["ts_open_ms"], unit="ms", utc=True)
            df["ts_close"] = pd.to_datetime(df["ts_close_ms"], unit="ms", utc=True)
            df["is_closed"] = df["is_closed"].astype(bool)
        return df

    def load_funding(self, symbol: str, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM funding WHERE symbol = ?"
        params: list[object] = [symbol]
        if start_ms is not None:
            sql += " AND funded_at_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            sql += " AND funded_at_ms <= ?"
            params.append(end_ms)
        sql += " ORDER BY funded_at_ms"
        df = pd.read_sql_query(sql, self._conn, params=params)
        if not df.empty:
            df = df.rename(columns={"interval_h": "interval_hours"})
            df["funding_time"] = pd.to_datetime(df["funded_at_ms"], unit="ms", utc=True)
        return df

    def bar_count(self, symbol: str, interval: str, closed_only: bool = True) -> int:
        sql = "SELECT COUNT(*) FROM bars WHERE symbol=? AND time_interval=?"
        params: list[object] = [symbol, interval]
        if closed_only:
            sql += " AND is_closed = 1"
        return int(self._conn.execute(sql, params).fetchone()[0])

    def watermark_ms(self, symbol: str, interval: str) -> int | None:
        row = self._conn.execute(
            "SELECT MAX(ts_open_ms) FROM bars WHERE symbol=? AND time_interval=? AND is_closed=1", (symbol, interval)
        ).fetchone()
        return None if row[0] is None else int(row[0])

    def ts_range_ms(self, symbol: str, interval: str) -> tuple[int, int] | None:
        row = self._conn.execute(
            "SELECT MIN(ts_open_ms), MAX(ts_open_ms) FROM bars WHERE symbol=? AND time_interval=? AND is_closed=1",
            (symbol, interval),
        ).fetchone()
        if row[0] is None:
            return None
        return int(row[0]), int(row[1])

    def find_gaps(self, symbol: str, interval: str, step_ms: int) -> list[tuple[int, int]]:
        """返回缺失区间的 (起止 ts_open_ms) 列表，按已收盘 bar 网格判断。"""
        have = [
            int(r[0])
            for r in self._conn.execute(
                "SELECT ts_open_ms FROM bars WHERE symbol=? AND time_interval=? AND is_closed=1 ORDER BY ts_open_ms",
                (symbol, interval),
            )
        ]
        if not have:
            return []
        gaps: list[tuple[int, int]] = []
        for prev, cur in pairwise(have):
            if cur - prev > step_ms:
                gaps.append((prev + step_ms, cur))
        return gaps

    # ---------- meta ----------

    def meta_get(self, key: str) -> object | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def meta_set(self, key: str, value: object) -> None:
        with self._conn:
            self._conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))

    def spec_put(self, symbol: str, raw_json: str) -> None:
        """缓存 exchangeInfo 解析出的交易规则，让回测离线也能拿到同一口径。"""
        self.meta_set(f"spec:{symbol}", raw_json)

    def spec_raw(self, symbol: str) -> str | None:
        value = self.meta_get(f"spec:{symbol}")
        return value if isinstance(value, str) else None

    # ---------- paper trading ----------

    @contextmanager
    def batch(self) -> Iterator[None]:
        """把若干次写入合成一个事务提交。

        paper 每根 bar 的账本、持仓、成交流水与断点必须原子落库：分开提交时，进程在
        中间被杀掉会留下"bar 已撮合但断点没推进"的状态，下一轮就会把同一根 bar 撮两遍。
        """
        self._tx_depth += 1
        try:
            yield
        finally:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self._conn.commit()

    def _write(self, sql: str, params: Sequence[object]) -> int:
        cur = self._conn.execute(sql, params)
        # 不在 batch 里就立刻提交；在 batch 里由 batch 统一提交
        if not self._tx_depth:
            self._conn.commit()
        return cur.rowcount

    def _select(self, sql: str, params: Sequence[object] = ()) -> list[dict]:
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def _upsert_paper(self, table: str, row: dict[str, object]) -> None:
        cols = PAPER_TABLE_COLS[table]
        unknown = [k for k in row if k not in cols]
        if unknown:
            raise KeyError(f"{table} 没有这些列：{unknown}")
        names = ", ".join(cols)
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in PAPER_TABLE_PK[table])
        sql = (
            f"INSERT INTO {table} ({names}) VALUES ({', '.join('?' * len(cols))})"
            f" ON CONFLICT({', '.join(PAPER_TABLE_PK[table])}) DO UPDATE SET {updates}"
        )
        self._write(sql, [row.get(c) for c in cols])

    def _paper_get(self, table: str, where: dict[str, object]) -> dict | None:
        cols = PAPER_TABLE_COLS[table]
        keys = list(where)
        sql = f"SELECT {', '.join(cols)} FROM {table} WHERE {' AND '.join(f'{k}=?' for k in keys)}"
        rows = self._select(sql, list(where.values()))
        return rows[0] if rows else None

    def paper_account(self, symbol: str, strategy: str) -> dict | None:
        return self._paper_get("paper_accounts", {"symbol": symbol, "strategy": strategy})

    def save_paper_account(self, row: dict[str, object]) -> None:
        self._upsert_paper("paper_accounts", row)

    def paper_open_position(self, symbol: str, strategy: str) -> dict | None:
        """纸面持仓只有一条：status='open' 里最晚开的那笔。"""
        rows = self.paper_open_positions(symbol, strategy)
        return rows[0] if rows else None

    def paper_open_positions(self, symbol: str, strategy: str) -> list[dict]:
        return self._select(
            "SELECT * FROM paper_positions WHERE symbol=? AND strategy=? AND status='open'"
            " ORDER BY opened_at_ms DESC",
            (symbol, strategy),
        )

    def mark_paper_positions_closed(self, symbol: str, strategy: str, ts_ms: int) -> int:
        """把该策略所有 status='open' 的持仓标记为已平。

        账本已空但库里还留着 open 行只可能是平仓写入被中断，收尾时必须归位，
        否则重启会把一笔已经不存在的仓位恢复进账本。
        """
        return self._write(
            "UPDATE paper_positions SET status='closed', updated_at_ms=?"
            " WHERE symbol=? AND strategy=? AND status='open'",
            (ts_ms, symbol, strategy),
        )

    def save_paper_position(self, row: dict[str, object]) -> None:
        self._upsert_paper("paper_positions", row)

    def paper_trade(self, row: dict[str, object]) -> None:
        self._upsert_paper("paper_trades", row)

    def paper_trades(self, symbol: str, strategy: str, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM paper_trades WHERE symbol=? AND strategy=? ORDER BY exit_time_ms"
        params: list[object] = [symbol, strategy]
        if limit is not None:
            sql += " DESC LIMIT ?"
            params.append(limit)
        return self._select(sql, params)

    def paper_pnl_since(self, symbol: str, strategy: str, since_ms: int) -> tuple[float, int]:
        """`since_ms` 之后已实现盈亏合计与笔数（[paper] 告警闸门的输入）。"""
        row = self._select(
            "SELECT COALESCE(SUM(pnl), 0) AS pnl, COUNT(*) AS n FROM paper_trades"
            " WHERE symbol=? AND strategy=? AND exit_time_ms >= ?",
            (symbol, strategy, since_ms),
        )[0]
        return float(row["pnl"]), int(row["n"])

    def paper_last_consecutive_losses(self, symbol: str, strategy: str) -> int:
        """从最近一笔平仓往前数的连续亏损笔数（最多回看 50 笔）。"""
        losses = 0
        for row in self.paper_trades(symbol, strategy, limit=50):
            if float(row["pnl"]) < 0:
                losses += 1
            else:
                break
        return losses

    def insert_paper_signal(self, row: dict[str, object]) -> bool:
        """signal_id 已存在则不写 —— 重启续跑时同一笔信号不会被推两次。"""
        cols = PAPER_TABLE_COLS["paper_signals"]
        return self._write(
            f"INSERT OR IGNORE INTO paper_signals ({', '.join(cols)})"
            f" VALUES ({', '.join('?' * len(cols))})",
            [row.get(c) for c in cols],
        ) > 0

    def mark_paper_signal_sent(self, signal_id: str, sent_ok: bool, sent_at_ms: int, payload: str) -> None:
        """`sent_at_ms` 是最近一次投递尝试的时刻，成功与失败都写，退避重发按它计时。"""
        self._write(
            "UPDATE paper_signals SET sent_ok=?, sent_at_ms=?, payload_json=? WHERE signal_id=?",
            (int(sent_ok), sent_at_ms, payload, signal_id),
        )

    def paper_signal(self, signal_id: str) -> dict | None:
        return self._paper_get("paper_signals", {"signal_id": signal_id})

    def paper_signals_unsent(
        self,
        symbol: str,
        strategy: str,
        limit: int = 10,
        *,
        attempt_before_ms: int | None = None,
    ) -> list[dict]:
        """待投递信号。给了 `attempt_before_ms` 就只取"从未尝试过或上次尝试够久了"的行，
        否则一条持续失败的消息会每轮都重发。"""
        sql = "SELECT * FROM paper_signals WHERE symbol=? AND strategy=? AND sent_ok=0"
        params: list[object] = [symbol, strategy]
        if attempt_before_ms is not None:
            sql += " AND (sent_at_ms IS NULL OR sent_at_ms <= ?)"
            params.append(attempt_before_ms)
        sql += " ORDER BY created_at_ms LIMIT ?"
        params.append(limit)
        return self._select(sql, params)

    def paper_signals_count(
        self, symbol: str, strategy: str, kinds: tuple[str, ...], since_ms: int
    ) -> int:
        """`since_ms` 以来各 kind 的信号条数（[paper] 每日开仓上限闸门用）。"""
        if not kinds:
            return 0
        marks = ", ".join("?" * len(kinds))
        row = self._select(
            f"SELECT COUNT(*) AS n FROM paper_signals"
            f" WHERE symbol=? AND strategy=? AND ts_signal_ms >= ? AND kind IN ({marks})",
            [symbol, strategy, since_ms, *kinds],
        )[0]
        return int(row["n"])

    def paper_signals_stats(self, symbol: str, strategy: str) -> dict:
        return self._select(
            "SELECT COUNT(*) AS n, SUM(sent_ok) AS n_sent, MAX(ts_signal_ms) AS last_ts"
            " FROM paper_signals WHERE symbol=? AND strategy=?",
            (symbol, strategy),
        )[0]

    def paper_checkpoint(self, symbol: str, strategy: str) -> dict | None:
        return self._paper_get("paper_checkpoints", {"symbol": symbol, "strategy": strategy})

    def paper_checkpoint_ms(self, symbol: str, strategy: str) -> int | None:
        row = self.paper_checkpoint(symbol, strategy)
        return None if row is None else int(row["last_processed_ts_ms"])

    def save_paper_checkpoint(self, row: dict[str, object]) -> None:
        self._upsert_paper("paper_checkpoints", row)

    def paper_accounts(self) -> list[dict]:
        return self._select("SELECT * FROM paper_accounts ORDER BY symbol, strategy")

    # ---------- versioning ----------

    def dataset_version(self, symbol: str, intervals: list[str], include_funding: bool = True) -> str:
        parts: list[str] = []
        for iv in sorted(intervals):
            rng = self.ts_range_ms(symbol, iv)
            n = self.bar_count(symbol, iv)
            first, last = (0, 0) if rng is None else rng
            parts.append(f"{symbol}|{iv}|rows={n}|first={first}|last={last}")
        if include_funding:
            row = self._conn.execute(
                "SELECT COUNT(*), MIN(funded_at_ms), MAX(funded_at_ms) FROM funding WHERE symbol=?", (symbol,)
            ).fetchone()
            parts.append(f"{symbol}|funding|rows={row[0]}|first={row[1] or 0}|last={row[2] or 0}")
        return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]
