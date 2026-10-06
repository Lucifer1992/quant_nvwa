"""报表层：月度分桶与平仓原因标签。

覆盖缺陷：
1. pct_change() 丢首月 → 报表「月度收益明细」永久隐藏首月盈亏；
2. 平仓原因的中文/颜色三处各写一份 → 漏 signal 时表格直接显示英文，图例也只有写死的三项。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quant.backtest.reports import REASON_CN, REASON_COLOR, _monthly_returns
from quant.core.contracts import FillReason

BASE = datetime(2026, 4, 1, tzinfo=UTC)


def _hourly(days: int, start: datetime = BASE):
    return [start + timedelta(hours=i) for i in range(days * 24)]


class TestMonthlyReturns:
    def test_first_month_is_not_dropped(self):
        # 4 月内权益从 100 跌到 90，5 月 90→99
        ts = _hourly(60)
        eq = np.array([100.0 + (90.0 - 100.0) * min(i, 24 * 30) / (24 * 30)
                       + (99.0 - 90.0) * max(0, i - 24 * 30) / (24 * 31) for i in range(len(ts))])
        ret = _monthly_returns(ts, eq, 100.0)
        assert [d.strftime("%Y-%m") for d in ret.index] == ["2026-04", "2026-05"]
        assert ret.iloc[0] == pytest.approx(-0.10, abs=0.02)

    def test_all_months_reconcile_to_total_return(self):
        ts = _hourly(92)
        eq = np.linspace(100.0, 76.25, len(ts))
        ret = _monthly_returns(ts, eq, 100.0)
        compound = float(np.prod(1.0 + ret.values)) - 1.0
        assert compound == pytest.approx(76.25 / 100.0 - 1.0, rel=1e-3)

    def test_buckets_use_beijing_time(self):
        # UTC 04-30 15:00 = 北京 04-30 23:00；UTC 04-30 23:00 = 北京 05-01 07:00。
        # 按 UTC 分桶会把跳变算进 4 月，报表少一整个 5 月。
        ts = [datetime(2026, 4, 30, 15, tzinfo=UTC), datetime(2026, 4, 30, 23, tzinfo=UTC)]
        ret = _monthly_returns(ts, np.array([100.0, 120.0]), 100.0)
        assert [d.strftime("%Y-%m") for d in ret.index] == ["2026-04", "2026-05"]
        assert ret.iloc[0] == pytest.approx(0.0)
        assert ret.iloc[1] == pytest.approx(0.20)

    def test_empty_equity(self):
        assert _monthly_returns([], np.array([]), 100.0).empty


class TestReasonLabels:
    @pytest.mark.parametrize("reason", list(FillReason))
    def test_every_fill_reason_is_localized(self, reason):
        assert reason.value in REASON_CN, f"{reason.value} 缺中文标签，报表会显示英文原值"
        assert reason.value in REASON_COLOR, f"{reason.value} 缺颜色，图例与流水会退回默认灰"
