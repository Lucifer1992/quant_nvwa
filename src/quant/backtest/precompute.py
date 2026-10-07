"""指标预计算与 bar 行转换：回测与 paper 共用的一份。

抽出来只因为一个理由：paper 必须和回测算出**逐位相同**的指标行，否则同一个策略在两边
看到的行情不一样，纸面结果就没有参考价值了。`BacktestEngine` 与 `src/quant/paper/`
都从这里取，不再各自维护第二份。

所有滚动量都只向前看（`.shift(1)` / 只依赖已收盘 bar），无前视；无前视性质由
`tests/test_features_builder.py` 与 `tests/test_engine_ledger.py` 的截断重算用例守住。
"""
from __future__ import annotations

import pandas as pd

from ..core.contracts import Bar
from ..core.timeframe import interval_ms, to_dt
from ..features.indicators import adx, atr, bollinger, ema, macd, rsi, stoch_rsi


def precompute(df: pd.DataFrame, swing_window: int = 12) -> pd.DataFrame:
    """在 DataFrame 上一次性计算所有技术指标（仅向前看，无前瞻）。"""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA 系列
    df["ema_9"] = ema(close, 9)
    df["ema_12"] = ema(close, 12)
    df["ema_20"] = ema(close, 20)
    df["ema_50"] = ema(close, 50)
    df["ema_60"] = ema(close, 60)

    # ATR
    df["atr"] = atr(high, low, close)

    # MACD
    m = macd(close)
    df["macd"] = m["macd"]
    df["macd_signal"] = m["signal"]
    df["macd_hist"] = m["hist"]

    # 前一根 MACD hist（用于检测交叉）
    df["prev_macd_hist"] = df["macd_hist"].shift(1)

    # 前一根 close（用于检测均线下穿）
    df["prev_close"] = df["close"].shift(1)
    df["prev_ema_20"] = df["ema_20"].shift(1)

    # RSI
    df["rsi_14"] = rsi(close)

    # StochRSI（用于 HTF）
    s = stoch_rsi(close)
    df["srsi_k"] = s["srsi_k"]
    df["srsi_d"] = s["srsi_d"]
    df["prev_srsi_k"] = df["srsi_k"].shift(1)

    # Bollinger
    b = bollinger(close)
    df["bb_mid"] = b["mid"]
    df["bb_upper"] = b["upper"]
    df["bb_lower"] = b["lower"]

    # ADX
    df["adx_14"] = adx(high, low, close)

    # ATR 中位数（入场 ATR 过滤用）
    df["atr_median"] = df["atr"].rolling(20).median()

    # 成交量中位数（量能放大信号用）
    df["vol_median"] = df["volume"].rolling(20).median()

    # Swing high/low（最近 N 根 bar 的最高/最低，用于结构突破加仓判定）
    df["swing_high"] = df["high"].rolling(swing_window).max().shift(1)  # shift 避免未来数据
    df["swing_low"] = df["low"].rolling(swing_window).min().shift(1)

    return df


def precompute_htf(df_htf: pd.DataFrame) -> pd.DataFrame:
    """高周期 DataFrame 预计算（MACD / EMA / StochRSI / ATR / ADX / Donchian）。

    所有滚动量都 `.shift(1)`，只用已收盘 bar，无前视。
    """
    close = df_htf["close"]
    high = df_htf["high"]
    low = df_htf["low"]

    df_htf["ema_12"] = ema(close, 12)
    df_htf["ema_20"] = ema(close, 20)
    df_htf["ema_60"] = ema(close, 60)

    m = macd(close)
    df_htf["macd_hist"] = m["hist"]
    df_htf["prev_macd_hist"] = df_htf["macd_hist"].shift(1)

    s = stoch_rsi(close)
    df_htf["srsi_k"] = s["srsi_k"]
    df_htf["srsi_d"] = s["srsi_d"]
    df_htf["prev_srsi_k"] = df_htf["srsi_k"].shift(1)

    df_htf["atr"] = atr(high, low, close)
    df_htf["rsi_14"] = rsi(close)
    df_htf["adx_14"] = adx(high, low, close)
    df_htf["prev_adx_14"] = df_htf["adx_14"].shift(1)

    # 趋势推进检测：N 根高周期 bar 前的 EMA20 / ADX（rule_v4 用它算斜率与强度变化）
    df_htf["ema_20_prev_6"] = df_htf["ema_20"].shift(6)
    df_htf["adx_prev_3"] = df_htf["adx_14"].shift(3)

    # 波动扩张检测
    df_htf["atr_median_100"] = df_htf["atr"].rolling(100).median()

    # 结构突破位（Donchian）
    df_htf["donchian_high_24"] = high.rolling(24).max().shift(1)
    df_htf["donchian_low_24"] = low.rolling(24).min().shift(1)

    return df_htf


def warmup_bars() -> int:
    """窗口最前面不参与撮合的 5m bar 数：等高周期 EMA60 / ADX14 脱离初值偏差。

    paper 用它换算加载窗口的前置长度，保证首个可交易 bar 的指标预热深度不低于回测口径。
    """
    return max(
        interval_ms("4h") // interval_ms("5m") * 65,
        interval_ms("1h") // interval_ms("5m") * 35,
        30,
    )


def row_to_bar(row: pd.Series, symbol: str, interval: str) -> Bar:
    ts_ms = int(row["ts_open_ms"])
    return Bar(
        symbol=symbol,
        interval=interval,
        ts_open=to_dt(ts_ms),
        ts_close=to_dt(ts_ms + interval_ms(interval) - 1),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        quote_volume=float(row.get("quote_volume", 0)),
        taker_buy_volume=float(row.get("taker_buy_volume", 0)),
        trades=int(row.get("trades", 0)),
        is_closed=True,
        src="rest",
    )
