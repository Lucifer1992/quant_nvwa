from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

# 报表、卡片与 [paper] 闸门的"自然日"都按北京时间切，与用户的盯盘口径一致
BJ_TZ = timezone(timedelta(hours=8))

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def interval_ms(interval: str) -> int:
    try:
        return INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(f"unsupported interval {interval!r}, known: {sorted(INTERVAL_MS)}") from None


def to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def parse_time(s: str) -> datetime:
    """接受 'YYYY-MM-DD' 或 ISO8601，统一返回 UTC。"""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def floor_ms(ms: int, interval: str) -> int:
    step = interval_ms(interval)
    return ms - ms % step


def grid(interval: str, start_ms: int, end_ms: int) -> list[int]:
    """返回 [start_ms, end_ms) 内所有 bar 的 ts_open 毫秒值。"""
    step = interval_ms(interval)
    first = start_ms - start_ms % step
    return list(range(first, end_ms, step))


def source_rank(is_closed: bool, src: str) -> int:
    """upsert 优先级：已收盘覆盖未收盘，REST 覆盖 WS。"""
    return 2 * int(bool(is_closed)) + int(src == "rest")


def utc_now() -> datetime:
    return datetime.now(UTC)
