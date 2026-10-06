"""纯 pandas 技术指标，不内嵌任何业务逻辑，供所有策略复用。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
) -> pd.DataFrame:
    f = ema(close, fast)
    s = ema(close, slow)
    macd_line = f - s
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist}, index=close.index)


def stoch_rsi(
    close: pd.Series, rsi_n: int = 14, stoch_n: int = 14, k: int = 3, d: int = 3,
) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / rsi_n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    min_rsi = rsi.rolling(stoch_n).min()
    max_rsi = rsi.rolling(stoch_n).max()
    srsi_k = 100 * (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    srsi_d = sma(srsi_k, k)
    return pd.DataFrame({"srsi_k": srsi_k, "srsi_d": srsi_d}, index=close.index)


def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0,
) -> pd.DataFrame:
    mid = sma(close, n)
    std = close.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower}, index=close.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def volume_ratio(volume: pd.Series, n: int = 20) -> pd.Series:
    return volume / volume.rolling(n).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr.ewm(alpha=1 / n, adjust=False).mean())
    minus_di = 100 * (minus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr.ewm(alpha=1 / n, adjust=False).mean())
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / n, adjust=False).mean()
