from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
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


class Store:
    """本地 SQLite 数据仓库。bar 主键保证同 (symbol, interval, ts_open) 唯一，写入天然幂等。"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)

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
