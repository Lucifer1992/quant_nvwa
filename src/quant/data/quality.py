from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.timeframe import interval_ms, to_dt
from .store import Store

CLOSE_TOL = 1e-9


def htf_consistency(
    store: Store,
    symbol: str,
    base_interval: str,
    htf_intervals: list[str],
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[dict[str, Any]]:
    """用基础周期聚合出高周期 OHLC，与交易所直接返回的高周期 bar 比对。

    交易所自身偶有聚合错误（实测 ETHUSDT 2024-10-28 20:00~21:00 两根 1h bar 只覆盖了部分
    15m 数据），而多周期策略直接读交易所的高周期列，因此这类偏差必须显式暴露出来。
    """
    base = store.load_bars(symbol, base_interval, start_ms, end_ms)
    if base.empty:
        return []
    out: list[dict[str, Any]] = []
    for iv in htf_intervals:
        if iv == base_interval:
            continue
        step = interval_ms(iv)
        buckets = (base.ts_open_ms // step) * step
        agg = pd.DataFrame({"close": base.close.groupby(buckets).last(), "open": base.open.groupby(buckets).first()})
        htf = store.load_bars(symbol, iv, start_ms, end_ms).set_index("ts_open_ms")[["open", "close"]]
        joined = agg.join(htf, how="inner", lsuffix="_agg", rsuffix="_ex")
        if joined.empty:
            continue
        close_dev = (joined.close_agg - joined.close_ex).abs()
        open_dev = (joined.open_agg - joined.open_ex).abs()
        bad = joined[(close_dev > CLOSE_TOL) | (open_dev > CLOSE_TOL)]
        out.append({
            "symbol": symbol,
            "base_interval": base_interval,
            "htf_interval": iv,
            "compared": len(joined),
            "mismatched": len(bad),
            "max_close_dev": float(close_dev.max()),
            "max_open_dev": float(open_dev.max()),
            "worst_at": [str(to_dt(int(ts)).strftime("%Y-%m-%d %H:%M")) for ts in bad.index[:5]],
        })
    return out
