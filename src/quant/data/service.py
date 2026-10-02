from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..core.contracts import Bar, SymbolSpec
from ..core.timeframe import parse_time, to_dt, to_ms
from ..infra.log import get_logger
from ..risk.exchange_meta import spec_from_json
from .store import Store

_log = get_logger("quant.data.service")

SPEC_STALE_WARN_SEC = 24 * 3600


class DataNotReady(RuntimeError):
    """本地库缺少该交易对的数据或交易规则，需要先执行 quant backfill。"""


def _to_ms(value: str | datetime | None) -> int | None:
    if value is None:
        return None
    return to_ms(parse_time(value) if isinstance(value, str) else value)


class MarketDataService:
    """策略与回测唯一的数据入口：只读本地 SQLite，不触网。

    回测要求无网络也能复现同一数据集，因此连交易规则都只读回填时的缓存；
    需要更新时由调用方显式执行 backfill，读路径永远不产生副作用。
    """

    def __init__(self, store: Store):
        self.store = store

    def bars(
        self,
        symbol: str,
        interval: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        *,
        closed_only: bool = True,
    ) -> pd.DataFrame:
        df = self.store.load_bars(symbol, interval, _to_ms(start), _to_ms(end), closed_only=closed_only)
        if df.empty:
            _log.warning("%s %s 本地无已收盘 K 线，请先执行 quant backfill", symbol, interval)
        return df

    def bar_objects(
        self, symbol: str, interval: str, start: str | datetime | None = None, end: str | datetime | None = None
    ) -> list[Bar]:
        fields = set(Bar.__dataclass_fields__)
        return [Bar(**{k: r[k] for k in fields}) for r in self.bars(symbol, interval, start, end).to_dict("records")]

    def last_closed(self, symbol: str, interval: str) -> datetime | None:
        wm = self.store.watermark_ms(symbol, interval)
        return None if wm is None else to_dt(wm)

    def funding(
        self,
        symbol: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        return self.store.load_funding(symbol, _to_ms(start), _to_ms(end))

    def spec(self, symbol: str) -> SymbolSpec:
        raw = self.store.spec_raw(symbol)
        if raw is None:
            raise DataNotReady(f"{symbol} 的交易规则未缓存，请先执行 quant backfill --symbol {symbol}")
        spec = spec_from_json(raw)
        age_sec = (datetime.now(spec.fetched_at.tzinfo) - spec.fetched_at).total_seconds()
        if age_sec > SPEC_STALE_WARN_SEC:
            _log.warning("%s 交易规则已过期 %.1f 天（fetch=%s），回测仍按缓存口径以保持可复现",
                         symbol, age_sec / 86400, spec.fetched_at.date())
        return spec

    def status(self, symbol: str, intervals: list[str]) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for iv in intervals:
            rng = self.store.ts_range_ms(symbol, iv)
            out[iv] = {
                "closed_bars": self.store.bar_count(symbol, iv),
                "first": None if rng is None else to_dt(rng[0]),
                "last": None if rng is None else to_dt(rng[1]),
            }
        out["funding"] = {"rows": len(self.store.load_funding(symbol))}
        return out
