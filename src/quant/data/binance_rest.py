from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Self

import httpx

from ..core.contracts import Bar, FundingBar
from ..core.timeframe import interval_ms, to_dt
from ..infra.log import get_logger

_log = get_logger("quant.data.rest")

KLINE_WEIGHT = {1500: 5, 1000: 4, 500: 3, 200: 2, 100: 1}


class BinanceRestError(RuntimeError):
    def __init__(self, msg: str, *, status: int | None = None, code: int | None = None):
        super().__init__(msg)
        self.status = status
        self.code = code


class RateLimited(BinanceRestError):
    def __init__(self, msg: str, *, status: int, retry_after: float | None = None):
        super().__init__(msg, status=status)
        self.retry_after = retry_after


class AccessDenied(BinanceRestError):
    """403/451 通常是代理出口节点被 Cloudflare 拦截或 IP 被封，换节点而非重试。"""


def _kline_weight(limit: int) -> int:
    for threshold, w in sorted(KLINE_WEIGHT.items(), reverse=True):
        if limit >= threshold:
            return w
    return 1


def parse_kline_row(row: list[Any], symbol: str, interval: str, now_ms: int) -> Bar:
    open_ms, close_ms = int(row[0]), int(row[6])
    return Bar(
        symbol=symbol,
        interval=interval,
        ts_open=to_dt(open_ms),
        ts_close=to_dt(close_ms),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        quote_volume=float(row[7]),
        taker_buy_volume=float(row[9]),
        trades=int(row[8]),
        # 交易所的 closeTime = openTime + 周期 - 1ms，故必须严格小于当前时间才算真正收盘，
        # 否则末毫秒仍在成形的 bar 会被当作已收盘写入
        is_closed=close_ms < now_ms,
        src="rest",
    )


def parse_funding_row(row: dict[str, Any], symbol: str) -> FundingBar:
    mark = row.get("markPrice")
    return FundingBar(
        symbol=row.get("symbol", symbol),
        funding_time=to_dt(int(row["fundingTime"])),
        funding_rate=float(row["fundingRate"]),
        mark_price=float(mark) if mark is not None else None,
        rate_type=str(row.get("rateType", "Regular")),
        interval_hours=float(row.get("fundingIntervalHours", 8) or 8),
        src="rest",
    )


class WeightBudget:
    """按分钟滚动窗口控制请求权重，避免触发 429/418。"""

    def __init__(self, budget_per_min: int, sleep: Any = time.sleep):
        self.budget = budget_per_min
        self._sleep = sleep
        self._window_start = time.monotonic()
        self._used = 0

    def observe(self, observed_used: int | None) -> None:
        """用交易所回传的已用权重校正本地计数（实际值可能高于本地估计）。"""
        if observed_used is not None:
            self._used = max(self._used, observed_used)

    def acquire(self, weight: int) -> None:
        now = time.monotonic()
        if now - self._window_start >= 60:
            self._window_start, self._used = now, 0
        if self._used + weight > self.budget:
            wait = 60 - (now - self._window_start) + 0.1
            _log.info("weight budget reached (%s), sleeping %.1fs", self._used, wait)
            self._sleep(max(wait, 0.1))
            self._window_start, self._used = time.monotonic(), 0
        self._used += weight


class BinanceRestClient:
    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        *,
        proxy_url: str | None = None,
        timeout: float = 20.0,
        weight_budget_per_min: int = 1700,
        max_retries: int = 5,
        client: httpx.Client | None = None,
    ):
        kwargs: dict[str, Any] = {
            "base_url": base_url,
            "timeout": timeout,
            "headers": {"User-Agent": "quant-llm/0.1", "Accept": "application/json"},
        }
        if proxy_url:
            kwargs["proxy"] = proxy_url
        self._client = client or httpx.Client(**kwargs)
        self._budget = WeightBudget(weight_budget_per_min)
        self._max_retries = max_retries
        self._time_offset_ms = 0
        self._offset_at = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- transport ----------

    def _get(self, path: str, params: dict[str, Any], *, weight: int = 1) -> Any:
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._budget.acquire(weight)
            try:
                resp = self._client.get(path, params=params)
            except httpx.HTTPError as e:
                last = e
                _log.warning("%s network error (attempt %s): %s", path, attempt + 1, e)
                time.sleep(min(2**attempt, 30))
                continue

            hdr_used = resp.headers.get("x-mbx-used-weight-1m") or resp.headers.get("X-MBX-USED-WEIGHT-1M")
            if hdr_used and hdr_used.isdigit():
                self._budget.observe(int(hdr_used))

            if resp.status_code in (418, 451) or resp.status_code == 403:
                raise AccessDenied(
                    f"{path} http {resp.status_code}: 疑似 IP 被封或代理节点被 Cloudflare 拦截，请更换出口节点（418 封禁可达数小时）",
                    status=resp.status_code,
                )
            if resp.status_code == 429:
                ra = resp.headers.get("retry-after")
                wait = float(ra) if ra and ra.isdigit() else 2 ** (attempt + 1)
                _log.warning("%s rate limited, sleeping %.0fs", path, wait)
                time.sleep(wait)
                last = RateLimited(f"{path} 429", status=429, retry_after=wait)
                continue
            if resp.status_code >= 500:
                last = BinanceRestError(f"{path} http {resp.status_code}: {resp.text[:200]}", status=resp.status_code)
                _log.warning(str(last))
                time.sleep(min(2**attempt, 30))
                continue
            try:
                data = resp.json()
            except ValueError:
                raise BinanceRestError(f"{path} 返回非 JSON（http {resp.status_code}）: {resp.text[:200]}", status=resp.status_code) from None
            if isinstance(data, dict) and data.get("code") and data.get("msg") and resp.status_code >= 400:
                raise BinanceRestError(f"{path} api error {data.get('code')}: {data.get('msg')}", status=resp.status_code, code=data.get("code"))
            return data
        raise last or BinanceRestError(f"{path} failed")

    # ---------- endpoints ----------

    def server_time_ms(self) -> int:
        return int(self._get("/fapi/v1/time", {}, weight=1)["serverTime"])

    def sync_time(self, max_age_sec: float = 1800) -> int:
        """返回本地与交易所的毫秒偏移，过期才重新拉取。"""
        now = time.monotonic()
        if now - self._offset_at > max_age_sec:
            local = time.time() * 1000
            self._time_offset_ms = self.server_time_ms() - int(local)
            self._offset_at = now
            _log.info("server clock offset = %d ms", self._time_offset_ms)
        return self._time_offset_ms

    def now_ms(self) -> int:
        return int(time.time() * 1000) + self.sync_time()

    def exchange_info(self) -> dict[str, Any]:
        """USDT-M 的 exchangeInfo 会忽略 symbol 查询参数并返回全部交易对，只能在本地筛选。"""
        return self._get("/fapi/v1/exchangeInfo", {}, weight=20)

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        for item in self.exchange_info().get("symbols", []):
            if item.get("symbol") == symbol:
                return item
        raise BinanceRestError(f"exchangeInfo 中不存在 {symbol}（可能已下架或拼写错误）")

    def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int | None = None, limit: int = 1500) -> list[Bar]:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms
        rows = self._get("/fapi/v1/klines", params, weight=_kline_weight(limit))
        now = self.now_ms()
        return [parse_kline_row(r, symbol, interval, now) for r in rows]

    def funding_rates(self, symbol: str, start_ms: int, end_ms: int | None = None, limit: int = 1000) -> list[FundingBar]:
        params: dict[str, Any] = {"symbol": symbol, "startTime": start_ms, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms
        rows = self._get("/fapi/v1/fundingRate", params, weight=1)
        return [parse_funding_row(r, symbol) for r in rows]


def day_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def bar_span_ms(interval: str) -> int:
    return interval_ms(interval)
