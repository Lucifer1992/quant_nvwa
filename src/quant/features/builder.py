"""LightGBM 策略的特征与标签构建：训练与推理共用同一条代码路径。

设计口径见 docs/lgbm-strategy-and-paper-trading-design.md §3/§4。
所有列只允许使用当前 bar 及更早的已收盘数据（4h 按"整根 4h 已收盘"才可见对齐）。
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from .indicators import adx, atr, bollinger, ema, macd, rsi, stoch_rsi

# 往返成本占名义价格的比例：taker 0.05%/边 + 滑点 2bp/边
ROUND_TRIP_COST_FRAC = 2 * 0.0005 + 2 * 0.0002

_H_MS = 3_600_000
_H4_MS = 4 * _H_MS

# 训练与推理都以这份清单为准（顺序即模型输入顺序），不靠 dtype 猜列
FEATURE_COLUMNS: tuple[str, ...] = (
    "r_1", "r_3", "r_6", "r_12", "r_24",
    "ema20_dist", "ema_span", "ema20_slope",
    "macd_hist_n", "macd_hist_d1",
    "adx_14", "adx_d1", "adx_d3",
    "don_pos", "don_break",
    "rsi_14", "srsi_k", "srsi_d", "srsi_diff",
    "atr_pct", "atr_ratio", "rv_24", "bb_width",
    "vol_ratio", "taker_ratio", "qvol_z",
    "hour_sin", "hour_cos", "dayofweek",
    "h4_ema20_dist", "h4_macd_hist_n", "h4_adx_14", "h4_atr_pct", "h4_basis",
    "funding_rate", "funding_annual", "mark_basis", "hrs_to_funding",
)


def _rolling_pos(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def build_feature_frame(
    bars_1h: pd.DataFrame,
    bars_4h: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """返回 ts_open_ms + 特征列 + 建标签所需的 close/high/low/atr。

    行数与 bars_1h 一致（同一顺序），预热期保留 NaN 不填充。
    """
    df = bars_1h.reset_index(drop=True).copy()
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    a = atr(high, low, close)
    df["atr"] = a
    logc = np.log(close)

    for n in (1, 3, 6, 12, 24):
        df[f"r_{n}"] = logc.diff(n)

    e20, e60 = ema(close, 20), ema(close, 60)
    df["ema20_dist"] = (close - e20) / a
    df["ema_span"] = (e20 - e60) / a
    df["ema20_slope"] = (e20 - e20.shift(6)) / a

    hist = macd(close)["hist"]
    df["macd_hist_n"] = hist / a
    df["macd_hist_d1"] = hist.diff(1) / a

    adx14 = adx(high, low, close)
    df["adx_14"] = adx14
    df["adx_d1"] = adx14.diff(1)
    df["adx_d3"] = adx14 - adx14.shift(3)

    # Donchian：shift(1) 让突破位不含当前 bar，"位置"才表示是否创出区间新高
    don_hi = high.rolling(24).max().shift(1)
    don_lo = low.rolling(24).min().shift(1)
    rng = (don_hi - don_lo).replace(0, np.nan)
    df["don_pos"] = (close - don_lo) / rng
    df["don_break"] = (close - don_hi) / a

    df["rsi_14"] = rsi(close)
    s = stoch_rsi(close)
    df["srsi_k"], df["srsi_d"] = s["srsi_k"], s["srsi_d"]
    df["srsi_diff"] = s["srsi_k"] - s["srsi_d"]

    atr_pct = a / close
    df["atr_pct"] = atr_pct
    df["atr_ratio"] = atr_pct / atr_pct.rolling(100).median()
    df["rv_24"] = logc.diff(1).rolling(24).std()
    bb = bollinger(close)
    df["bb_width"] = (bb["upper"] - bb["lower"]) / a

    df["vol_ratio"] = vol / vol.rolling(20).median().replace(0, np.nan)
    if "taker_buy_volume" in df and "volume" in df:
        df["taker_ratio"] = df["taker_buy_volume"] / vol.replace(0, np.nan)
    if "quote_volume" in df:
        qv = df["quote_volume"]
        df["qvol_z"] = (qv - _rolling_pos(qv, 24)) / qv.rolling(24).std().replace(0, np.nan)

    ts_close_ms = df["ts_open_ms"] + _H_MS - 1
    hours = ((df["ts_open_ms"] // _H_MS) % 24).astype(float)
    df["hour_sin"] = np.sin(2 * math.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * math.pi * hours / 24)
    dow = pd.to_datetime(df["ts_open_ms"], unit="ms", utc=True).dt.dayofweek.astype(float)
    df["dayofweek"] = dow

    if bars_4h is not None and len(bars_4h):
        df = _merge_4h(df, bars_4h)
    if funding is not None and len(funding):
        df = _merge_funding(df, funding, ts_close_ms)

    return df


def _merge_4h(df: pd.DataFrame, bars_4h: pd.DataFrame) -> pd.DataFrame:
    """4h bar 只有在其收盘时刻 ≤ 当前 1h bar 开盘时刻时才可见。"""
    h4 = bars_4h.reset_index(drop=True).copy()
    h4_close = h4["close"]
    h4_a = atr(h4["high"], h4["low"], h4_close)
    src = pd.DataFrame({
        "ts_visible_ms": h4["ts_open_ms"].astype("int64") + _H4_MS,
        "h4_ema20_dist": (h4_close - ema(h4_close, 20)) / h4_a,
        "h4_macd_hist_n": macd(h4_close)["hist"] / h4_a,
        "h4_adx_14": adx(h4["high"], h4["low"], h4_close),
        "h4_atr_pct": h4_a / h4_close,
        "h4_close": h4_close,
    }).sort_values("ts_visible_ms")

    out = pd.merge_asof(
        df.sort_values("ts_open_ms"),
        src,
        left_on="ts_open_ms",
        right_on="ts_visible_ms",
        direction="backward",
        allow_exact_matches=True,
    )
    out["h4_basis"] = out["close"] / out["h4_close"] - 1
    out = out.drop(columns=["h4_close", "ts_visible_ms"])
    return out.sort_index()


def _merge_funding(df: pd.DataFrame, funding: pd.DataFrame, ts_close_ms: pd.Series) -> pd.DataFrame:
    """资金费率/标记价按"已发生的最近一次结算"对齐，并给出距下次结算的小时数。"""
    f = funding.reset_index(drop=True).copy()
    hours_col = "interval_hours" if "interval_hours" in f else "interval_h"
    interval_h = f[hours_col].astype(float)
    src = pd.DataFrame({
        "funded_at_ms": f["funded_at_ms"].astype("int64"),
        "funding_rate": f["funding_rate"].astype(float),
        "funding_annual": f["funding_rate"].astype(float) * (24.0 / interval_h) * 365.0,
        "mark_price": f["mark_price"].astype(float),
        "interval_hours": interval_h.astype(float),
    }).sort_values("funded_at_ms")

    left = pd.DataFrame({"ts_close_ms": ts_close_ms.astype("int64").values, "_i": np.arange(len(df))})
    merged = pd.merge_asof(
        left.sort_values("ts_close_ms"), src,
        left_on="ts_close_ms", right_on="funded_at_ms", direction="backward",
    ).sort_values("_i")

    out = df.copy()
    out["funding_rate"] = merged["funding_rate"].values
    out["funding_annual"] = merged["funding_annual"].values
    out["mark_basis"] = out["close"].values / merged["mark_price"].values - 1.0
    # 下次结算时间按周期外推：数据集末尾（以及 paper 实时场景）最后一次结算总在过去，
    # 直接用 next_funding_ms 会算出负数。
    elapsed_h = (ts_close_ms.values - merged["funded_at_ms"].values) / _H_MS
    interval_vals = merged["interval_hours"].astype(float).values
    out["hrs_to_funding"] = interval_vals - np.mod(elapsed_h, interval_vals)
    return out


_LABEL_COLS = ("ts_open_ms", "close", "high", "low", "atr")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """按 FEATURE_COLUMNS 取列；缺列直接报错，避免训练/推理特征集静默不一致。"""
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"特征帧缺少列：{missing}")
    return list(FEATURE_COLUMNS)


def feature_spec_hash(cols: list[str]) -> str:
    return hashlib.sha1("|".join(cols).encode()).hexdigest()[:12]


def build_labels(
    frame: pd.DataFrame,
    k_up: float = 3.0,
    k_dn: float = 2.0,
    horizon: int = 24,
    cost_frac: float = ROUND_TRIP_COST_FRAC,
    flat_atr_mult: float = 1.0,
) -> pd.Series:
    """Triple-Barrier 三分类：+1 先触上障碍 / -1 先触下障碍 / 0 到期但位移不够。

    同一根 bar 内上下障碍同时命中时保守判 -1（与回测引擎 sl_first_on_conflict 一致）。
    到期未触障碍则要求位移同时超过 `flat_atr_mult*ATR` 和往返成本，否则记 0：
    只按成本判平（0.14% 价格）会让 flat 类几乎为空（实测 2%），模型学不到"不该开仓"。
    末尾 horizon 根无法构造完整标签，返回 NaN。
    """
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    a = frame["atr"].to_numpy(dtype=float)
    n = len(frame)
    y = np.full(n, np.nan)

    for t in range(n - horizon):
        at = a[t]
        entry = close[t]
        if not (np.isfinite(at) and at > 0 and np.isfinite(entry)):
            continue
        up, dn = entry + k_up * at, entry - k_dn * at
        label = 0
        for j in range(t + 1, t + 1 + horizon):
            if low[j] <= dn:
                label = -1
                break
            if high[j] >= up:
                label = 1
                break
        else:
            move = close[t + horizon] - entry
            if abs(move) > max(flat_atr_mult * at, cost_frac * entry):
                label = 1 if move > 0 else -1
        y[t] = label

    return pd.Series(y, index=frame.index, name="label", dtype="float64")
