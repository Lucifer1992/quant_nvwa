from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Self

import httpx

from ..infra.log import get_logger
from ..infra.settings import Settings

_log = get_logger("quant.notify.feishu")

OK_CODE = 0


class FeishuError(RuntimeError):
    pass


class FeishuBusy(RuntimeError):
    """触发了 webhook 限流，调用方应退避后重试。"""


def make_sign(secret: str, timestamp: int) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), b"", hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


@dataclass(frozen=True)
class FeishuConfig:
    webhook_url: str
    secret: str | None = None
    max_per_min: int = 40
    min_interval_sec: float = 0.25

    @classmethod
    def from_settings(cls, settings: Settings) -> FeishuConfig | None:
        feishu = settings.notify.get("feishu", {})
        if not feishu.get("enabled", False):
            return None
        url_env = feishu.get("webhook_url_env", "FEISHU_WEBHOOK_URL")
        secret_env = feishu.get("secret_env", "FEISHU_SECRET")
        url = (os.getenv(url_env) or feishu.get("webhook_url") or "").strip()
        if not url:
            _log.warning("feishu.enabled=true 但环境变量 %s 未设置", url_env)
            return None
        return cls(
            webhook_url=url,
            secret=(os.getenv(secret_env) or feishu.get("secret") or "").strip() or None,
            max_per_min=int(feishu.get("max_per_min", 40)),
            min_interval_sec=float(feishu.get("min_interval_sec", 0.25)),
        )


class FeishuNotifier:
    def __init__(self, cfg: FeishuConfig, proxy_url: str | None = None, timeout: float = 10.0):
        self.cfg = cfg
        kwargs: dict[str, Any] = {"timeout": timeout, "headers": {"Content-Type": "application/json"}}
        if proxy_url:
            kwargs["proxy"] = proxy_url
        self._client = httpx.Client(**kwargs)
        self._sent_at: deque[float] = deque()
        self._last_send_ts = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        wait = self.cfg.min_interval_sec - (time.monotonic() - self._last_send_ts)
        if wait > 0:
            time.sleep(wait)
        # 睡眠之后必须重取时钟：沿用睡前的 now 会把已经滑出窗口的记录继续计入
        # 一分钟配额，突发几条就会抛出根本不存在的 FeishuBusy。
        now = time.monotonic()
        while self._sent_at and now - self._sent_at[0] > 60.0:
            self._sent_at.popleft()
        if len(self._sent_at) >= self.cfg.max_per_min:
            raise FeishuBusy(f"feishu rate limit {self.cfg.max_per_min}/min reached")

    @staticmethod
    def _parse(resp: httpx.Response) -> dict[str, Any]:
        """飞书在网关错误时会返回 HTML，不能让 resp.json() 抛异常掩盖真实状态码。"""
        try:
            data = resp.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float:
        ra = resp.headers.get("retry-after")
        return float(ra) if ra and ra.replace(".", "", 1).isdigit() else 3.0

    def send_payload(self, payload: dict[str, Any], retries: int = 2) -> dict[str, Any]:
        self._throttle()
        body = dict(payload)
        if self.cfg.secret:
            ts = int(time.time())
            body["timestamp"] = str(ts)
            body["sign"] = make_sign(self.cfg.secret, ts)

        last_err: Exception | None = None
        delay = 0.0
        for attempt in range(retries + 1):
            if attempt:
                time.sleep(delay)
                # 重试同样是对外 POST，必须重新过一遍最小间隔与分钟配额
                self._throttle()
            backoff = 1.5 * 2 ** (attempt - 1)
            try:
                resp = self._client.post(self.cfg.webhook_url, json=body)
            except httpx.HTTPError as e:
                last_err, delay = e, backoff
                _log.warning("飞书请求网络异常 attempt=%s: %s", attempt + 1, e)
                continue
            data = self._parse(resp)
            code = data.get("code", resp.status_code)
            if resp.status_code == 429 or code in (11232, 1123):
                last_err = FeishuBusy(f"feishu rate limited: {data or resp.status_code}")
                delay = self._retry_after(resp)
                _log.warning("%s（%.0fs 后重试）", last_err, delay)
                continue
            if resp.status_code >= 500:
                last_err = FeishuError(f"feishu 5xx: {resp.status_code} {resp.text[:200]}")
                delay = backoff
                _log.warning(str(last_err))
                continue
            if code != OK_CODE:
                # 卡片结构或签名被拒，重试同一 payload 没有意义，直接抛出交调用方处理
                raise FeishuError(f"feishu rejected payload: {data or resp.text[:200]}")
            # 只有真正投递成功才占用分钟配额，失败的重试不应挤占后续通知
            self._sent_at.append(time.monotonic())
            self._last_send_ts = time.monotonic()
            return data
        raise last_err or FeishuError("feishu send failed")
