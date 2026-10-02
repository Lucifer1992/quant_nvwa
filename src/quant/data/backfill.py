from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.contracts import SymbolSpec
from ..core.timeframe import floor_ms, grid, interval_ms, to_dt, to_ms, utc_now
from ..infra.log import get_logger
from ..risk.exchange_meta import parse_symbol_spec, spec_from_json, spec_to_json
from .binance_rest import BinanceRestClient
from .store import Store

_log = get_logger("quant.data.backfill")

KLINE_PAGE = 1500
FUNDING_PAGE = 1000
SPEC_TTL_SEC = 24 * 3600


@dataclass
class BarBackfillReport:
    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    pages: int = 0
    rows_fetched: int = 0
    rows_written: int = 0
    unclosed_dropped: int = 0
    gaps: list[tuple[int, int]] = field(default_factory=list)
    missing_bars: int = 0

    @property
    def expected_bars(self) -> int:
        return len(grid(self.interval, self.start_ms, self.end_ms))

    def summary(self) -> str:
        cov = 1.0 if self.expected_bars == 0 else 1.0 - self.missing_bars / self.expected_bars
        return (
            f"{self.symbol} {self.interval}: pages={self.pages} written={self.rows_written} "
            f"expected={self.expected_bars} coverage={cov:.4%} gaps={len(self.gaps)}"
        )


class Backfiller:
    """REST 分页回填 K 线与资金费率到 SQLite，未收盘 bar 一律不落库。"""

    def __init__(
        self,
        client: BinanceRestClient,
        store: Store,
        *,
        on_progress: Callable[[str], None] | None = None,
    ):
        self.client = client
        self.store = store
        self._on_progress = on_progress or (lambda msg: _log.info(msg))

    # ---------- bars ----------

    def bars(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int | None = None,
        *,
        resume: bool = True,
    ) -> BarBackfillReport:
        step = interval_ms(interval)
        end_ms = floor_ms(end_ms if end_ms is not None else self.client.now_ms(), interval)
        start_ms = floor_ms(start_ms, interval)
        if resume:
            watermark = self.store.watermark_ms(symbol, interval)
            stored_range = self.store.ts_range_ms(symbol, interval)
            # 只有库里的数据已经覆盖到本次请求的起点时才跳过前缀续传；
            # 否则说明请求的是一段更早的历史，必须从头拉，靠 upsert 优先级去重
            if watermark is not None and stored_range is not None and stored_range[0] <= start_ms:
                start_ms = min(watermark + step, end_ms)
        report = BarBackfillReport(symbol=symbol, interval=interval, start_ms=start_ms, end_ms=end_ms)
        if start_ms >= end_ms:
            self._on_progress(f"{symbol} {interval} 已到最新，无需回填")
            return report

        cursor = start_ms
        while cursor < end_ms:
            rows = self.client.klines(symbol, interval, cursor, limit=KLINE_PAGE)
            report.pages += 1
            if not rows:
                break
            # klines 只按 startTime+limit 截断，返回条数可能越过本次请求的右边界；
            # 越界写入会让同一 --end 得到不同数据集，破坏回测可复现性
            rows = [b for b in rows if to_ms(b.ts_open) < end_ms]
            if not rows:
                break
            report.rows_fetched += len(rows)
            closed = [b for b in rows if b.is_closed]
            report.unclosed_dropped += len(rows) - len(closed)
            report.rows_written += self.store.upsert_bars(closed)
            last_open = to_ms(rows[-1].ts_open)
            cursor = last_open + step
            self._on_progress(
                f"{symbol} {interval} 进度 {to_dt(min(cursor, end_ms)):%Y-%m-%d %H:%M} "
                f"(累计写入 {report.rows_written})"
            )
            if len(rows) < KLINE_PAGE:
                break

        report.gaps = self.store.find_gaps(symbol, interval, step)
        report.missing_bars = sum((e - s) // step for s, e in report.gaps)
        return report

    # ---------- funding ----------

    def funding(self, symbol: str, start_ms: int, end_ms: int | None = None) -> int:
        end_ms = end_ms if end_ms is not None else self.client.now_ms()
        cursor, written, pages = start_ms, 0, 0
        while cursor < end_ms:
            rows = self.client.funding_rates(symbol, cursor, limit=FUNDING_PAGE)
            pages += 1
            rows = [f for f in rows if to_ms(f.funding_time) < end_ms]
            if not rows:
                break
            written += self.store.upsert_funding(rows)
            cursor = to_ms(rows[-1].funding_time) + 1
            if len(rows) < FUNDING_PAGE:
                break
        self._on_progress(f"{symbol} funding: pages={pages} written={written}")
        return written

    # ---------- contract spec ----------

    def refresh_spec(self, symbol: str, *, force: bool = False) -> SymbolSpec:
        cached = self.store.spec_raw(symbol)
        if cached and not force:
            spec = spec_from_json(cached)
            if (utc_now() - spec.fetched_at).total_seconds() < SPEC_TTL_SEC:
                return spec
        spec = parse_symbol_spec(self.client.symbol_info(symbol), fetched_at=utc_now())
        self.store.spec_put(symbol, spec_to_json(spec))
        self._on_progress(f"{symbol} 交易规则已刷新：tick={spec.tick_size} step={spec.step_size} "
                          f"maint={spec.maint_margin_rate} max_lev={spec.max_leverage}")
        return spec


def assert_complete(store: Store, symbol: str, interval: str, start_ms: int, end_ms: int) -> dict[str, Any]:
    """数据完整性闸门：回测前必须确认网格无缺口，否则止损触发顺序不可信。"""
    step = interval_ms(interval)
    expected = grid(interval, floor_ms(start_ms, interval), floor_ms(end_ms, interval))
    loaded = store.load_bars(symbol, interval, expected[0] if expected else start_ms,
                             (expected[-1] + step) if expected else end_ms)
    present = set(loaded["ts_open_ms"].tolist()) if not loaded.empty else set()
    missing = [ts for ts in expected if ts not in present]
    return {
        "symbol": symbol,
        "interval": interval,
        "expected": len(expected),
        "present": len(expected) - len(missing),
        "missing": len(missing),
        "coverage": (len(expected) - len(missing)) / len(expected) if expected else 1.0,
        "missing_head": [to_dt(ts).isoformat() for ts in missing[:5]],
    }
