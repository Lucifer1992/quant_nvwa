from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from quant.core.contracts import Bar
from quant.core.timeframe import interval_ms, to_ms
from quant.data.backfill import Backfiller
from quant.data.store import Store

STEP15 = interval_ms("15m")


def make_bar(ts_open_ms: int, *, closed: bool, src: str = "rest", price: float = 100.0) -> Bar:
    ts_open = datetime.fromtimestamp(ts_open_ms / 1000, tz=UTC)
    ts_close = ts_open + timedelta(milliseconds=STEP15 - 1) if closed else ts_open + timedelta(minutes=5)
    return Bar(
        symbol="ETHUSDT",
        interval="15m",
        ts_open=ts_open,
        ts_close=ts_close,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price + 0.5,
        volume=10.0,
        quote_volume=1000.0,
        taker_buy_volume=4.0,
        trades=7,
        is_closed=closed,
        src=src,  # type: ignore[arg-type]
    )


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


class StubClient:
    """替代 BinanceRestClient：只实现 Backfiller 用到的三个方法，不发网络请求。"""

    def __init__(self, bars: list[Bar], now_ms: int):
        self.bars = bars
        self._now = now_ms
        self.calls: list[tuple[int, int]] = []

    def now_ms(self) -> int:
        return self._now

    def klines(self, symbol: str, interval: str, start_ms: int, end_ms=None, limit: int = 1500) -> list[Bar]:
        self.calls.append((start_ms, limit))
        rows = [b for b in self.bars if to_ms(b.ts_open) >= start_ms]
        return rows[:limit]

    def funding_rates(self, symbol, start_ms, end_ms=None, limit=1000):
        self.calls.append((start_ms, limit))
        return []

    def symbol_info(self, symbol):
        raise AssertionError("测试不应触发 exchangeInfo")


def test_upsert_precedence_closed_rest_wins(store):
    ts = to_ms(datetime(2024, 1, 1, tzinfo=UTC))
    store.upsert_bars([make_bar(ts, closed=False, src="ws", price=1.0)])
    store.upsert_bars([make_bar(ts, closed=True, src="rest", price=2.0)])
    df = store.load_bars("ETHUSDT", "15m")
    assert len(df) == 1 and df.iloc[0]["close"] == 2.5 and df.iloc[0]["src"] == "rest"
    # 未收盘/WS 的低优先级数据不得覆盖已落库的已收盘 REST 数据
    store.upsert_bars([make_bar(ts, closed=False, src="ws", price=9.0)])
    assert store.load_bars("ETHUSDT", "15m").iloc[0]["close"] == 2.5
    assert store.bar_count("ETHUSDT", "15m", closed_only=False) == 1


def test_backfill_paginates_and_drops_unclosed(store):
    t0 = to_ms(datetime(2024, 1, 1, tzinfo=UTC))
    n = 2500
    bars = [make_bar(t0 + i * STEP15, closed=i < n - 1) for i in range(n)]
    client = StubClient(bars, t0 + (n - 1) * STEP15)
    report = Backfiller(client, store, on_progress=lambda _: None).bars("ETHUSDT", "15m", t0, end_ms=t0 + n * STEP15)
    assert report.pages == math.ceil(n / 1500) == 2
    assert report.unclosed_dropped == 1
    assert store.bar_count("ETHUSDT", "15m") == n - 1
    assert report.gaps == []


def test_backfill_resumes_from_watermark_for_recent_history(store):
    t0 = to_ms(datetime(2024, 1, 1, tzinfo=UTC))
    bars = [make_bar(t0 + i * STEP15, closed=True) for i in range(200)]
    bf = Backfiller(StubClient(bars, t0 + 200 * STEP15), store, on_progress=lambda _: None)
    bf.bars("ETHUSDT", "15m", t0, end_ms=t0 + 100 * STEP15)
    client2 = StubClient(bars, t0 + 200 * STEP15)
    report = Backfiller(client2, store, on_progress=lambda _: None).bars("ETHUSDT", "15m", t0, end_ms=t0 + 200 * STEP15)
    # 第二次只需拉尾部，起点应是水位线之后一根，而不是从头重拉
    assert client2.calls and client2.calls[0][0] == t0 + 100 * STEP15
    assert report.rows_written == 100


def test_backfill_still_fetches_earlier_history_than_watermark(store):
    """回归：先回填了近期窗口，再请求更早历史时不能因水位线判据被整体跳过。"""
    t0 = to_ms(datetime(2024, 1, 1, tzinfo=UTC))
    recent = to_ms(datetime(2024, 6, 1, tzinfo=UTC))
    bars = [make_bar(t0 + i * STEP15, closed=True) for i in range(100)]
    bars += [make_bar(recent + i * STEP15, closed=True) for i in range(20)]
    Backfiller(StubClient(bars, recent + 20 * STEP15), store, on_progress=lambda _: None).bars(
        "ETHUSDT", "15m", recent, end_ms=recent + 20 * STEP15
    )
    client = StubClient(bars, recent + 20 * STEP15)
    report = Backfiller(client, store, on_progress=lambda _: None).bars("ETHUSDT", "15m", t0, end_ms=recent)
    assert client.calls[0][0] == t0
    assert report.rows_written == 100


def test_kline_row_close_boundary_is_strict():
    """交易所 closeTime = openTime + 周期 - 1ms，末毫秒仍属成形 bar，不得判为已收盘。"""
    from quant.data.binance_rest import parse_kline_row

    t0 = 1_700_000_100_000
    row = [t0, "1", "2", "0.5", "1.5", "10", t0 + STEP15 - 1, "1000", 7, "4", "400", "0"]
    assert parse_kline_row(row, "ETHUSDT", "15m", t0 + STEP15 - 1).is_closed is False
    forming = parse_kline_row(row, "ETHUSDT", "15m", t0 + STEP15)
    assert forming.is_closed is True
    # 列序按 [openTime,open,high,low,close,volume,closeTime,quoteVolume,trades,takerBuyBase,...]
    assert (forming.open, forming.high, forming.low, forming.close) == (1.0, 2.0, 0.5, 1.5)
    assert (forming.volume, forming.quote_volume, forming.taker_buy_volume, forming.trades) == (10.0, 1000.0, 4.0, 7)
