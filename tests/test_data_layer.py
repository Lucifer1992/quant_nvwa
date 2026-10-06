from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from quant.core.contracts import Bar
from quant.core.timeframe import interval_ms, to_ms
from quant.data.backfill import Backfiller
from quant.data.binance_rest import (
    AccessDenied,
    BinanceRestClient,
    BinanceRestError,
    _kline_weight,
)
from quant.data.quality import htf_consistency
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


def bar_iv(interval: str, ts_open_ms: int, open_: float, close: float) -> Bar:
    """指定周期与开/收盘价的已收盘 bar，用于高周期聚合比对。"""
    t = datetime.fromtimestamp(ts_open_ms / 1000, tz=UTC)
    return Bar(
        symbol="ETHUSDT", interval=interval, ts_open=t,
        ts_close=t + timedelta(milliseconds=interval_ms(interval) - 1),
        open=open_, high=max(open_, close), low=min(open_, close), close=close,
        volume=1.0, quote_volume=close, taker_buy_volume=0.5, trades=1,
        is_closed=True, src="rest",
    )


class TestHtfConsistency:
    def test_edge_bucket_with_incomplete_coverage_is_not_flagged(self, store):
        """被窗口切开的头尾桶没有完整 OHLC，比对必然偏差，不得报成交易所数据错误。"""
        t0 = to_ms(datetime(2024, 1, 1, tzinfo=UTC))
        h1 = interval_ms("1h")
        for i in range(6):  # 15m×6 = 1h30m：第二个 1h 桶只覆盖了一半
            store.upsert_bars([bar_iv("15m", t0 + i * STEP15, 100.0 + i, 100.0 + i)])
        store.upsert_bars([bar_iv("1h", t0, 100.0, 103.0), bar_iv("1h", t0 + h1, 104.0, 999.0)])
        rows = htf_consistency(store, "ETHUSDT", "15m", ["1h"])
        assert len(rows) == 1
        assert rows[0]["compared"] == 1  # 只有覆盖完整的桶参与比对
        assert rows[0]["mismatched"] == 0

    def test_complete_bucket_mismatch_is_flagged(self, store):
        t0 = to_ms(datetime(2024, 1, 1, tzinfo=UTC))
        h1 = interval_ms("1h")
        for i in range(8):  # 补齐到 2h，两个桶都完整
            store.upsert_bars([bar_iv("15m", t0 + i * STEP15, 100.0 + i, 100.0 + i)])
        store.upsert_bars([bar_iv("1h", t0, 100.0, 103.0), bar_iv("1h", t0 + h1, 104.0, 999.0)])
        rows = htf_consistency(store, "ETHUSDT", "15m", ["1h"])
        assert rows[0]["compared"] == 2
        assert rows[0]["mismatched"] == 1
        assert rows[0]["worst_at"] == ["2024-01-01 01:00"]


class TestUpsertWriteCount:
    """upsert_bars 返回真实落库数，不是尝试写入数。"""

    def test_counts_new_updates_and_rejections(self, store):
        t0 = to_ms(datetime(2024, 1, 1, tzinfo=UTC))
        bars = [make_bar(t0 + i * STEP15, closed=True) for i in range(3)]
        assert store.upsert_bars(bars) == 3
        # 守卫是 excluded.prio >= bars.prio，同优先级重写仍然算更新
        assert store.upsert_bars(bars) == 3
        # 低优先级（未收盘 WS）行被守卫拒掉，不能虚报写入
        assert store.upsert_bars([make_bar(t0, closed=False, src="ws")]) == 0
        assert store.upsert_bars([]) == 0
        mixed = [make_bar(t0, closed=False, src="ws"), make_bar(t0 + 3 * STEP15, closed=True)]
        assert store.upsert_bars(mixed) == 1
        assert store.bar_count("ETHUSDT", "15m", closed_only=False) == 4


def test_backfill_reports_gaps_clipped_to_request_window(store):
    """回归：find_gaps 是全库口径，report.gaps 必须裁到本次请求范围内。"""
    t0 = to_ms(datetime(2024, 1, 1, tzinfo=UTC))
    bars = [make_bar(t0 + i * STEP15, closed=True) for i in range(10)]
    bars += [make_bar(t0 + i * STEP15, closed=True) for i in range(20, 30)]
    store.upsert_bars(bars)  # 库里已有 0~9 与 20~29，中间缺 10~19
    client = StubClient(bars, t0 + 30 * STEP15)
    report = Backfiller(client, store, on_progress=lambda _: None).bars(
        "ETHUSDT", "15m", t0 + 15 * STEP15, end_ms=t0 + 25 * STEP15, resume=False
    )
    assert report.rows_written == 5
    assert report.gaps == [(t0 + 15 * STEP15, t0 + 20 * STEP15)]
    assert report.missing_bars == 5
    # 全库口径确实还有范围外的缺口，说明是裁剪生效而非碰巧一致
    assert store.find_gaps("ETHUSDT", "15m", STEP15) == [(t0 + 10 * STEP15, t0 + 20 * STEP15)]


def _client(handler, **kw) -> BinanceRestClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://test.local")
    return BinanceRestClient(client=http, **kw)


class TestRestErrorHandling:
    """HTTP ≥ 400 一律是失败，错误体形状不可信（网关可能返回数组）。"""

    def test_non_dict_error_body_still_raises(self):
        client = _client(lambda request: httpx.Response(400, json=[{"code": -1100}]))
        with pytest.raises(BinanceRestError) as exc:
            client._get("/fapi/v1/klines", {}, weight=1)
        assert exc.value.status == 400
        assert "-1100" in str(exc.value)

    def test_dict_error_body_reports_msg(self):
        client = _client(lambda request: httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."}))
        with pytest.raises(BinanceRestError) as exc:
            client._get("/fapi/v1/klines", {}, weight=1)
        assert exc.value.code == -1121
        assert "Invalid symbol" in str(exc.value)

    def test_403_is_access_denied_without_retry(self):
        calls: list[int] = []

        def handler(request):
            calls.append(1)
            return httpx.Response(403, text="cloudflare block")

        client = _client(handler)
        with pytest.raises(AccessDenied):
            client._get("/fapi/v1/klines", {}, weight=1)
        assert len(calls) == 1

    def test_429_honours_retry_after_then_succeeds(self, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
        responses = [
            httpx.Response(429, headers={"retry-after": "7"}, json={"code": -1015}),
            httpx.Response(200, json=[]),
        ]

        def handler(request):
            return responses.pop(0)

        assert _client(handler)._get("/fapi/v1/klines", {}, weight=1) == []
        assert slept == [7.0]

    def test_5xx_retries_then_raises(self, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

        def handler(request):
            return httpx.Response(503, text="unreadable")

        client = _client(handler, max_retries=1)
        with pytest.raises(BinanceRestError) as exc:
            client._get("/fapi/v1/klines", {}, weight=1)
        assert exc.value.status == 503
        assert len(slept) == 2  # 首次 + 一次重试后各退避一次

    def test_weight_budget_adopts_server_observed_usage(self):
        def handler(request):
            return httpx.Response(200, json=[], headers={"x-mbx-used-weight-1m": "900"})

        client = _client(handler)
        client._get("/fapi/v1/klines", {}, weight=1)
        assert client._budget._used == 900


def test_kline_weight_tiers():
    assert _kline_weight(1) == 1
    assert _kline_weight(99) == 1
    assert _kline_weight(100) == 2  # 边界：100 起跳档
    assert _kline_weight(499) == 2
    assert _kline_weight(500) == 5
    assert _kline_weight(999) == 5
    assert _kline_weight(1000) == 10
    assert _kline_weight(1500) == 10

