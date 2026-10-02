from __future__ import annotations

from ..infra.log import get_logger
from ..infra.settings import Settings
from .binance_rest import BinanceRestClient
from .store import Store

_log = get_logger("quant.data.wiring")


def build_rest_client(s: Settings) -> BinanceRestClient:
    """所有币安请求都经 this 代理出口，直连在大陆网络下不可用。"""
    proxy = s.proxy_url
    if not proxy:
        _log.warning("未配置代理（network.proxy_url / QUANT_PROXY_URL），访问 fapi.binance.com 可能失败")
    return BinanceRestClient(
        s.rest_base,
        proxy_url=proxy,
        timeout=s.timeout_sec,
        weight_budget_per_min=s.weight_budget_per_min,
        max_retries=s.max_retries,
    )


def open_store(s: Settings) -> Store:
    return Store(s.db_path)
