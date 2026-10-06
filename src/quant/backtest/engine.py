"""事件驱动回测引擎：预计算 + 无前视信号 → Fill 账本 → 净值唯一真值。"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..core.contracts import (
    Action,
    AddLevel,
    Bar,
    Fill,
    FillReason,
    Position,
    RiskDecision,
    Side,
    Signal,
    SymbolSpec,
)
from ..core.timeframe import interval_ms, to_dt, to_ms
from ..features.indicators import adx, atr, bollinger, ema, macd, rsi, stoch_rsi
from ..risk.sizing import compute_risk_size


@dataclass
class BacktestConfig:
    symbol: str
    base_interval: str
    start: datetime
    end: datetime
    initial_equity: float
    risk_pct: float = 0.01
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    time_stop_bars: int = 96
    allow_short: bool = True
    global_max_leverage: float = 10.0  # 固定 10x 全程
    taker_fee: float = 0.0005
    slippage_bps: float = 2.0
    include_funding: bool = True
    sl_first_on_conflict: bool = True
    strategy_name: str = "rule_v1"   # 用于报告标题
    # ---- 100U 实盘模拟 + 金字塔加仓参数 ----
    pyramid_enabled: bool = True       # 是否启用金字塔加仓
    entry_frac: float = 0.10           # 试仓比例（满仓的 %）
    add_frac: float = 0.20             # 每次加仓比例下限（满仓的 %）
    add_max_count: int = 2             # 最大加仓次数（不含试仓）
    add_trigger_r: float = 0.3         # 加仓盈利 R 门槛
    trail_trigger_r: float = 1.0       # 移动止损激活 R 门槛
    trail_atr_mult: float = 1.0        # 移动止损跟随距离（× entry_atr）
    swing_lookback: int = 12           # swing high/low 回看 bar 数


@dataclass
class TradeRecord:
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fee: float
    reason: str
    r_achieved: float = 0.0
    bars_held: int = 0
    add_count: int = 0
    avg_entry_price: float = 0.0
    initial_qty: float = 0.0
    margin_usdt: float = 0.0  # 该笔占用的保证金（含金字塔加仓累计）
    ret_pct: float = 0.0  # 保证金收益率 % = pnl / margin_usdt × 100


@dataclass
class BacktestResult:
    equity_curve: list[float]
    timestamps: list[datetime]
    trades: list[TradeRecord]
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    payoff_ratio: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    avg_bars_held: float


# ---- 数据准备 ----------------------------------------------------------------


def _precompute(df: pd.DataFrame, swing_window: int = 12) -> pd.DataFrame:
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


def _precompute_htf(df_htf: pd.DataFrame) -> pd.DataFrame:
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


def _row_to_bar(row: pd.Series, symbol: str, interval: str) -> Bar:
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


def _fee_qty(qty: float, price: float, rate: float) -> float:
    return round(qty * price * rate, 8)


# ---- 引擎 --------------------------------------------------------------------


class BacktestEngine:
    """事件驱动回测，逐根 5m bar 推进，指标全预计算。

    signal_fn 签名:
        (bar: Bar, row: pd.Series, htf: dict[str, pd.Series], dt: datetime) -> Signal | None

    - row: 当前 5m bar 的完整指标行（预计算列全部就绪）
    - htf: 各高周期最近一条已收盘 bar 的指标 Series（key=interval）
    - dt: 信号时间（t_vis = ts_open + 周期 ms）
    """

    def __init__(
        self,
        config: BacktestConfig,
        bars: dict[str, pd.DataFrame],
        spec: SymbolSpec,
        funding: pd.DataFrame | None = None,
    ):
        self.cfg = config
        self.bars = bars
        self.funding = funding
        self.spec = spec
        self.base_ms = interval_ms(config.base_interval)
        self._reset()

    def _reset(self) -> None:
        self.cash = self.cfg.initial_equity
        self.equity = self.cfg.initial_equity
        self.position: Position | None = None
        self.fills: list[Fill] = []
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[float] = []
        self.timestamps: list[datetime] = []
        self.bars_since_entry = 0
        self.funding_since_entry = 0.0
        # 本根 bar 开盘时刻可用来定仓的权益：上一根收盘的 MTM。用本根收盘 MTM 定仓位
        # 等于用成交之后才知道的信息。
        self.equity_open = self.cfg.initial_equity
        self._funding_ptr = 0
        # 金字塔加仓统计
        self._add_count_total = 0
        self._add_skip_count = 0

    def _scale_in(self, signal: Signal, decision: RiskDecision, bar: Bar) -> None:
        """试仓或加仓到已有仓位。

        试仓（self.position is None）：创建新 Position，SL 基于实际成交价（bar.open+滑点）计算
        加仓（self.position is not None）：更新加权均价、追加 qty、追加 margin
        """
        entry_long = signal.action in (Action.OPEN_LONG, Action.ADD_TO_LONG)
        # 滑点方向恒为不利：买入开仓抬高成交价，卖出开仓压低成交价
        adverse = 1 + self.cfg.slippage_bps / 10000 if entry_long else 1 - self.cfg.slippage_bps / 10000
        entry = round(bar.open * adverse, 8)
        is_add = self.position is not None

        # 试仓时，SL 基于实际入场价（而非 signal.price_ref=bar.close）
        # 避免 open-close gap 导致立刻击穿 SL
        # 同时计算 TP（基于实际入场价）
        tp_price = None
        if not is_add and signal.meta.get("atr"):
            atr_val = signal.meta["atr"]
            if signal.action in (Action.OPEN_LONG, Action.ADD_TO_LONG):
                new_sl = entry - self.cfg.sl_atr_mult * atr_val
                tp_price = entry + self.cfg.tp_atr_mult * atr_val
                decision = RiskDecision(
                    signal=decision.signal, approved=decision.approved,
                    qty=decision.qty, notional_usdt=decision.notional_usdt,
                    leverage=decision.leverage, margin_usdt=decision.margin_usdt,
                    stop_loss=new_sl,
                    take_profits=decision.take_profits,
                    est_liq_price=decision.est_liq_price,
                    equity_usdt=decision.equity_usdt, risk_usdt=decision.risk_usdt,
                )
            else:  # OPEN_SHORT / ADD_TO_SHORT
                new_sl = entry + self.cfg.sl_atr_mult * atr_val
                tp_price = entry - self.cfg.tp_atr_mult * atr_val
                decision = RiskDecision(
                    signal=decision.signal, approved=decision.approved,
                    qty=decision.qty, notional_usdt=decision.notional_usdt,
                    leverage=decision.leverage, margin_usdt=decision.margin_usdt,
                    stop_loss=new_sl,
                    take_profits=decision.take_profits,
                    est_liq_price=decision.est_liq_price,
                    equity_usdt=decision.equity_usdt, risk_usdt=decision.risk_usdt,
                )

        if not is_add:
            # ---- 试仓 ----
            side = "long" if signal.action in (Action.OPEN_LONG, Action.ADD_TO_LONG) else "short"
            pos = Position(
                position_id=uuid.uuid4().hex[:8],
                symbol=signal.symbol,
                side=side,
                qty=decision.qty,
                entry_price=entry,
                avg_entry_price=entry,
                entry_ts=bar.ts_close,
                margin_usdt=decision.margin_usdt,
                stop_loss=decision.stop_loss,
                take_profits=[],
                liq_price=decision.est_liq_price,
                opened_by=signal.strategy,
                signal_id=signal.signal_id,
                entry_atr=signal.meta.get("atr"),
                target_qty=decision.qty,
                add_levels=[AddLevel(price=entry, qty=decision.qty, ts=bar.ts_close)],
                base_stop_price=decision.stop_loss or 0.0,
                trailing_peak_price=entry,
                tp_price=tp_price or 0.0,
            )
            self.position = pos
            self.cash -= decision.margin_usdt
            fee = _fee_qty(decision.qty, entry, self.cfg.taker_fee)
            self.cash -= fee
            pos.add_levels[0].fee_usdt = fee
            self.fills.append(Fill(
                ts=bar.ts_close, symbol=signal.symbol,
                side=Side.BUY if side == "long" else Side.SELL,
                qty=decision.qty, price=entry, fee_usdt=fee,
                reason=FillReason.SIGNAL, position_id=pos.position_id,
            ))
            self.bars_since_entry = 0
            self.funding_since_entry = 0.0
        else:
            # ---- 加仓 ----
            pos = self.position
            prev_qty = pos.qty
            new_qty = prev_qty + decision.qty
            pos.avg_entry_price = (
                pos.avg_entry_price * prev_qty + entry * decision.qty
            ) / new_qty
            pos.qty = new_qty
            pos.add_count += 1
            pos.margin_usdt += decision.margin_usdt
            pos.add_levels.append(AddLevel(price=entry, qty=decision.qty, ts=bar.ts_close))

            self.cash -= decision.margin_usdt
            fee = _fee_qty(decision.qty, entry, self.cfg.taker_fee)
            self.cash -= fee
            pos.add_levels[-1].fee_usdt = fee

            self.fills.append(Fill(
                ts=bar.ts_close, symbol=signal.symbol,
                side=Side.BUY if pos.side == "long" else Side.SELL,
                qty=decision.qty, price=entry, fee_usdt=fee,
                reason=FillReason.SIGNAL, position_id=pos.position_id,
            ))

            if pos.entry_atr and pos.entry_atr > 0:
                if pos.side == "long":
                    new_sl = pos.avg_entry_price - self.cfg.sl_atr_mult * pos.entry_atr
                    pos.stop_loss = max(pos.stop_loss, new_sl)
                    pos.base_stop_price = pos.stop_loss
                    pos.trailing_peak_price = max(pos.trailing_peak_price, entry)
                else:
                    new_sl = pos.avg_entry_price + self.cfg.sl_atr_mult * pos.entry_atr
                    pos.stop_loss = min(pos.stop_loss, new_sl)
                    pos.base_stop_price = pos.stop_loss
                    pos.trailing_peak_price = min(pos.trailing_peak_price, entry)

    def _should_add(self, row: pd.Series, bar: Bar) -> bool:
        """判断当前是否满足加仓条件（结构突破 + 浮盈 ≥ R）。"""
        pos = self.position
        if pos is None:
            return False
        if pos.add_count >= self.cfg.add_max_count:
            return False
        if pos.entry_atr is None or pos.entry_atr <= 0:
            return False

        # 计算当前浮盈 R 倍数（基于加权均价）
        if pos.side == "long":
            unreal_r = (bar.close - pos.avg_entry_price) / (self.cfg.sl_atr_mult * pos.entry_atr)
        else:
            unreal_r = (pos.avg_entry_price - bar.close) / (self.cfg.sl_atr_mult * pos.entry_atr)

        if unreal_r < self.cfg.add_trigger_r:
            return False

        # 结构突破（基于 swing high/low）。swing_high/swing_low 由 _precompute 按
        # cfg.swing_lookback 预计算并已 shift(1)，不含当前 bar
        swing_high = float(row.get("swing_high", 0)) if "swing_high" in row else 0
        swing_low = float(row.get("swing_low", 0)) if "swing_low" in row else 0

        if pos.side == "long" and swing_high > 0:
            return bar.close > swing_high
        if pos.side == "short" and swing_low > 0:
            return bar.close < swing_low
        return False

    def _update_trailing_stop(self, bar: Bar) -> None:
        """每根 bar 更新移动止损。"""
        pos = self.position
        if pos is None or pos.entry_atr is None or pos.entry_atr <= 0:
            return

        # 更新极值
        if pos.side == "long":
            pos.trailing_peak_price = max(pos.trailing_peak_price, bar.high)
        else:
            pos.trailing_peak_price = min(pos.trailing_peak_price, bar.low)

        # 当前浮盈 R
        if pos.side == "long":
            unreal_r = (bar.close - pos.avg_entry_price) / (self.cfg.sl_atr_mult * pos.entry_atr)
        else:
            unreal_r = (pos.avg_entry_price - bar.close) / (self.cfg.sl_atr_mult * pos.entry_atr)

        # 激活条件
        if not pos.trailing_active and unreal_r >= self.cfg.trail_trigger_r:
            pos.trailing_active = True

        # 激活后，止损跟随极值（只用 ATR 倍数，作为安全网）
        if pos.trailing_active:
            if pos.side == "long":
                trail_sl = pos.trailing_peak_price - self.cfg.trail_atr_mult * pos.entry_atr
                pos.stop_loss = max(pos.stop_loss, trail_sl)
            else:
                trail_sl = pos.trailing_peak_price + self.cfg.trail_atr_mult * pos.entry_atr
                pos.stop_loss = min(pos.stop_loss, trail_sl)

    def _close_trade(
        self,
        bar: Bar,
        reason: FillReason,
        price: float,
        exit_time: datetime | None = None,
    ) -> None:
        pos = self.position
        if pos is None:
            return
        # 基于加权均价计算 PnL
        ref_price = pos.avg_entry_price if pos.add_count > 0 else pos.entry_price
        if pos.side == "long":
            pnl = pos.qty * (price - ref_price)
            side = Side.SELL
        else:
            pnl = pos.qty * (ref_price - price)
            side = Side.BUY
        # 开仓腿手续费在开仓时已从 cash 扣除并记在 add_levels 上，平仓只结算平仓腿，
        # 否则同一笔开仓费会被扣两次（现金口径与逐笔 pnl 口径不一致）。
        fee_exit = _fee_qty(pos.qty, price, self.cfg.taker_fee)
        fee_entry_paid = sum(lv.fee_usdt for lv in pos.add_levels)
        net_cash = pnl - fee_exit - self.funding_since_entry
        self.cash += pos.margin_usdt + net_cash  # 归还保证金 + 扣费后净盈亏
        net = pnl - fee_entry_paid - fee_exit - self.funding_since_entry  # 往返全成本，用于逐笔归属

        r_ach = 0.0
        if pos.entry_atr and pos.entry_atr > 0:
            # 符号跟随盈亏方向：多单涨、空单跌为正 R（不能按 side 直接取反）
            move = price - ref_price if pos.side == "long" else ref_price - price
            r_ach = float(move / (self.cfg.sl_atr_mult * pos.entry_atr))

        self.fills.append(Fill(
            ts=bar.ts_close, symbol=pos.symbol, side=side,
            qty=pos.qty, price=price, fee_usdt=fee_exit, pnl_usdt=net,
            reason=reason, position_id=pos.position_id,
        ))
        self.trades.append(TradeRecord(
            entry_time=pos.entry_ts,
            exit_time=exit_time if exit_time is not None else bar.ts_close,
            side=pos.side, entry_price=pos.entry_price, exit_price=price,
            qty=pos.qty, pnl=net, fee=fee_entry_paid + fee_exit,
            reason=reason.value,
            r_achieved=r_ach, bars_held=self.bars_since_entry,
            add_count=pos.add_count,
            avg_entry_price=pos.avg_entry_price,
            initial_qty=pos.add_levels[0].qty if pos.add_levels else pos.qty,
            margin_usdt=pos.margin_usdt,
            ret_pct=net / pos.margin_usdt * 100 if pos.margin_usdt > 0 else 0.0,
        ))
        self.position = None
        self.bars_since_entry = 0
        self.funding_since_entry = 0.0

    def _close_position_at_open(self, bar: Bar, reason: FillReason = FillReason.SIGNAL) -> None:
        """按本根 bar 开盘价（不利方向加滑点）平掉当前持仓。信号平仓与反手共用。"""
        pos = self.position
        if pos is None:
            return
        adverse = 1 - self.cfg.slippage_bps / 10000 if pos.side == "long" else 1 + self.cfg.slippage_bps / 10000
        self._close_trade(bar, reason, round(bar.open * adverse, 8))

    def _tp_hit(self, pos: Position, bar: Bar) -> bool:
        if pos.tp_price <= 0:
            return False
        return bar.high >= pos.tp_price if pos.side == "long" else bar.low <= pos.tp_price

    def _stop_hit(self, pos: Position, bar: Bar) -> bool:
        if not pos.stop_loss:
            return False
        return bar.low <= pos.stop_loss if pos.side == "long" else bar.high >= pos.stop_loss

    def _check_exits(self, bar: Bar) -> None:
        """SL / TP 结算。

        同一根 bar 同时触及两个价位时，一根 bar 内无法证明价格先到哪一侧，按 TP 成交
        会把本该止损的交易记成盈利，属于系统性乐观偏差；默认（sl_first_on_conflict=True）
        保守取 SL，与 `features.builder.build_labels` 的标签口径一致。
        """
        pos = self.position
        if pos is None:
            return
        sl_hit = self._stop_hit(pos, bar)
        tp_hit = self._tp_hit(pos, bar)
        if sl_hit and tp_hit and not self.cfg.sl_first_on_conflict:
            tp_hit, sl_hit = True, False
        if sl_hit:
            self._fire_stop(bar, pos)
        elif tp_hit:
            self._fire_tp(bar, pos)

    def _fire_tp(self, bar: Bar, pos: Position) -> None:
        """TP 成交：按止盈价加减不利滑点，出场时间取 bar 中点。"""
        exit_px = float(pos.tp_price)
        if pos.side == "long":
            slipped = exit_px * (1 - self.cfg.slippage_bps / 10000)
        else:
            slipped = exit_px * (1 + self.cfg.slippage_bps / 10000)
        self._close_trade(bar, FillReason.TP, round(slipped, 8), exit_time=self._bar_mid(bar))

    def _fire_stop(self, bar: Bar, pos: Position) -> None:
        """B 模式：止损被 bar 极值触发时，按 bar.low/bar.high 成交（最坏情况），
        出场时间取 bar 中点（避免虚增持仓时长）。taker 单。"""
        exit_px = float(bar.low if pos.side == "long" else bar.high)  # 最坏情况成交
        if pos.side == "long":
            slipped = exit_px * (1 - self.cfg.slippage_bps / 10000)
        else:
            slipped = exit_px * (1 + self.cfg.slippage_bps / 10000)
        self._close_trade(bar, FillReason.SL, round(slipped, 8), exit_time=self._bar_mid(bar))

    @staticmethod
    def _bar_mid(bar: Bar) -> datetime:
        return bar.ts_open + (bar.ts_close - bar.ts_open) / 2

    def _accrue_funding(self, t_vis_ms: int, price: float) -> None:
        """把结算时刻已落在本根 bar 内的资金费计入当前持仓的成本。

        约定：正费率多头付、空头收。费用只累计到 funding_since_entry，现金在平仓时
        经 net_cash 一次性扣除，避免持仓期与平仓时重复扣款。指针每根 bar 都要推进，
        否则空仓期间的结算会被算进之后才开的仓位。
        """
        if self.funding is None or self.funding.empty or not self.cfg.include_funding:
            return
        n = len(self.funding)
        while self._funding_ptr < n:
            funded_at = int(self.funding["funded_at_ms"].iloc[self._funding_ptr])
            if funded_at > t_vis_ms:
                break
            rate = float(self.funding["funding_rate"].iloc[self._funding_ptr])
            self._funding_ptr += 1
            pos = self.position
            if pos is None or not np.isfinite(rate) or rate == 0.0:
                continue
            px = price
            if "mark_price" in self.funding.columns:
                mp = float(self.funding["mark_price"].iloc[self._funding_ptr - 1])
                if np.isfinite(mp) and mp > 0:
                    px = mp
            cost = pos.qty * px * rate * (1.0 if pos.side == "long" else -1.0)
            self.funding_since_entry += cost
            pos.funding_paid += cost

    def _update_mtm(self, bar: Bar) -> None:
        if self.position is None:
            return
        pos = self.position
        mp = bar.close
        # 加仓后成本基准是加权均价，用 entry_price（首次试仓价）会虚增/虚减浮盈
        ref = pos.avg_entry_price
        unreal = pos.qty * (mp - ref) if pos.side == "long" else pos.qty * (ref - mp)
        # funding_since_entry 是已计提未结算的负债（多头为正、空头为负），
        # 不计入就会在结算日之前高估净值。
        self.equity = self.cash + unreal + pos.margin_usdt - self.funding_since_entry

    def run(
        self,
        signal_fn: Callable[[Bar, pd.Series, dict[str, pd.Series], datetime], Signal | None],
        strategy: object | None = None,
    ) -> BacktestResult:
        cfg = self.cfg
        df5 = self.bars.get("5m")
        if df5 is None or df5.empty:
            raise ValueError("5m 数据为空")

        # 资金费按结算时刻定位（iloc 位置索引），必须先规范化索引
        if self.funding is not None and not self.funding.empty:
            self.funding = self.funding.sort_values("funded_at_ms").reset_index(drop=True)

        # E7/E8：策略生命周期钩子。ML 策略需要引擎注入完整历史来预热特征，
        # 规则策略无状态，reset() 只是把内部状态归零。
        if strategy is not None:
            bind = getattr(strategy, "bind_history", None)
            if callable(bind):
                bind(self.bars, self.funding)
            strategy.reset()

        # 预计算 5m 指标（一次性，O(n)）。在整段加载数据上算，指标预热不受回测窗口起点影响
        df5 = df5.sort_values("ts_open_ms").reset_index(drop=True)
        df5 = _precompute(df5, swing_window=max(2, int(cfg.swing_lookback)))

        # 时间窗口过滤（窗口外的 bar 只用于预热指标，不参与撮合）
        start_ms = to_ms(cfg.start)
        end_ms = to_ms(cfg.end)
        df5 = df5[(df5.ts_open_ms >= start_ms) & (df5.ts_open_ms < end_ms)].reset_index(drop=True)

        # 预计算 HTF DataFrame
        df1h = self.bars.get("1h")
        df4h = self.bars.get("4h")
        if df1h is not None and not df1h.empty:
            df1h = df1h.sort_values("ts_open_ms").reset_index(drop=True)
            df1h = _precompute_htf(df1h)
        if df4h is not None and not df4h.empty:
            df4h = df4h.sort_values("ts_open_ms").reset_index(drop=True)
            df4h = _precompute_htf(df4h)

        # 预热期：窗口最前面 3120 根 5m（10.8 天）只让高周期指针与指标就位，不出信号。
        # 规则策略的加载窗口不带前置历史，高周期 EMA60/ADX14 在起点处还是 NaN。
        warmup = max(
            interval_ms("4h") // interval_ms("5m") * 65,
            interval_ms("1h") // interval_ms("5m") * 35,
            30,
        )

        # 高周期弹出指针（高铁 bar 弹出后不再追加）
        idx_1h = 0
        idx_4h = 0
        htf_1h_row: pd.Series | None = None  # 当前可见的最近 1h bar
        htf_4h_row: pd.Series | None = None  # 当前可见的最近 4h bar
        pending_signal: Signal | None = None  # 上一根 bar 收盘产生、待在本根 bar 开盘执行

        for i, row in df5.iterrows():
            ts_ms = int(row["ts_open_ms"])
            t_vis_ms = ts_ms + self.base_ms

            # 弹出已完成收盘的高周期 bar
            if df1h is not None:
                while idx_1h < len(df1h):
                    h_row = df1h.iloc[idx_1h]
                    h_vis = int(h_row["ts_open_ms"]) + interval_ms("1h")
                    if h_vis <= t_vis_ms:
                        htf_1h_row = h_row
                        idx_1h += 1
                    else:
                        break
            if df4h is not None:
                while idx_4h < len(df4h):
                    h_row = df4h.iloc[idx_4h]
                    h_vis = int(h_row["ts_open_ms"]) + interval_ms("4h")
                    if h_vis <= t_vis_ms:
                        htf_4h_row = h_row
                        idx_4h += 1
                    else:
                        break

            # 构建 HTF dict
            htf: dict[str, pd.Series] = {}
            if htf_1h_row is not None:
                htf["1h"] = htf_1h_row
            if htf_4h_row is not None:
                htf["4h"] = htf_4h_row

            # 本根 bar 开盘可支配的权益 = 上一根收盘的净值（定仓只用成交前已知的数）
            self.equity_open = self.equity

            # 净值快照
            if self.position:
                self._update_mtm(_row_to_bar(row, cfg.symbol, cfg.base_interval))
            else:
                self.equity = self.cash
            self.equity_curve.append(self.equity)
            self.timestamps.append(to_dt(ts_ms + self.base_ms - 1))

            # 资金费结算指针每根 bar 都要推进（含预热期），空仓期间的结算不能被跳过
            self._accrue_funding(t_vis_ms, float(row["close"]))

            if i < warmup:
                continue

            bar = _row_to_bar(row, cfg.symbol, cfg.base_interval)
            dt = bar.ts_close

            # ---- 执行上一根 bar 收盘产生的信号：在本根 bar 开盘成交 ----
            # 信号可见时刻是 bar 收盘，若用同一根 bar 的 open 成交即是前视偏差。
            if pending_signal is not None:
                self._execute_signal(pending_signal, bar, row)
                pending_signal = None

            # 每根 bar 更新移动止损
            self._update_trailing_stop(bar)

            # TP / SL 检查（同一根 bar 双触发时按 sl_first_on_conflict 定序）
            self._check_exits(bar)

            if self.position:
                self.bars_since_entry += 1
                if self.bars_since_entry >= cfg.time_stop_bars:
                    self._close_trade(bar, FillReason.TIME, self._slipped_close(bar))

            signal = signal_fn(bar, row, htf, dt)
            if signal is not None:
                pending_signal = signal

        # 平最后持仓
        if self.position and len(df5) > 0:
            last_row = df5.iloc[-1]
            last_bar = _row_to_bar(last_row, cfg.symbol, cfg.base_interval)
            self._close_trade(last_bar, FillReason.FLAT, self._slipped_close(last_bar))
            # 收尾结算必须回写最后一个净值点：循环内的快照是持仓 MTM，不含这笔的
            # 滑点、平仓费与资金费，否则 sum(trades.pnl) 与 Δequity 恒等式会在尾部差一个量。
            self.equity = self.cash
            if self.equity_curve:
                self.equity_curve[-1] = self.equity

        if strategy is not None:
            strategy.finalize()

        return self._metrics()

    def _slipped_close(self, bar: Bar) -> float:
        """按 bar 收盘价市价平仓的不利滑点成交价（全 taker 口径，与其他出场一致）。"""
        short = self.position is not None and self.position.side == "short"
        adverse = 1 + self.cfg.slippage_bps / 10000 if short else 1 - self.cfg.slippage_bps / 10000
        return round(float(bar.close) * adverse, 8)

    def _execute_signal(self, signal: Signal, bar: Bar, row: pd.Series) -> None:
        """执行信号：主动平仓 / 试仓 / 加仓。bar 是成交所在 bar（信号产生后的下一根）。"""
        cfg = self.cfg

        # ---- 策略主动平仓（趋势破位）----
        if signal.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
            pos = self.position
            if pos is None:
                return
            if (pos.side == "long") != (signal.action == Action.CLOSE_LONG):
                return
            self._close_position_at_open(bar)
            return

        if signal.action not in (
            Action.OPEN_LONG, Action.OPEN_SHORT,
            Action.ADD_TO_LONG, Action.ADD_TO_SHORT,
        ):
            return

        # 策略可自带决策周期的 ATR（如 rule_v4 用 1h ATR 设 1h 趋势仓的止损）；
        # 未提供时回落到基础周期 5m ATR。
        if signal.meta.get("atr"):
            atr_val = float(signal.meta["atr"])
        else:
            atr_val = (
                float(row["atr"]) if not pd.isna(row["atr"]) else signal.price_ref * 0.005
            )
            signal.meta["atr"] = atr_val

        # 策略可按高周期背景缩小本次仓位；钳在 [0.2, 1.0]，不允许超额
        size_mult = min(max(float(signal.meta.get("size_mult", 1.0)), 0.2), 1.0)

        # 计算本次 qty
        want_long = signal.action in (Action.OPEN_LONG, Action.ADD_TO_LONG)
        # 定仓用的权益：本根 bar 开盘前已知的净值（收盘 MTM 是成交之后才知道的信息）。
        eq = self.equity_open
        if self.position is not None and (self.position.side == "long") != want_long:
            # 反手：先按本根 bar 开盘价平掉对向持仓，再在同一点位开新仓。
            # 平仓把保证金与净盈亏还回现金，此刻已无持仓，权益就是 cash。
            self._close_position_at_open(bar)
            eq = self.cash
        is_add = self.position is not None
        want_add = is_add and cfg.pyramid_enabled
        frac = cfg.add_frac if want_add else cfg.entry_frac
        target_qty = 0.0
        if cfg.pyramid_enabled:
            target_qty = eq * frac * size_mult * cfg.global_max_leverage / signal.price_ref
        decision = compute_risk_size(
            signal=signal,
            decision=RiskDecision(signal=signal, approved=False),
            equity=eq,
            risk_pct=cfg.risk_pct, atr=atr_val,
            sl_atr_mult=cfg.sl_atr_mult,
            global_max_leverage=cfg.global_max_leverage,
            symbol_spec=self.spec,
            taker_fee_rate=cfg.taker_fee,
            qty=target_qty,
        )
        if not decision.approved:
            return
        # 已持仓：仅在金字塔启用且满足结构突破 + 浮盈时才加仓
        if is_add:
            if cfg.pyramid_enabled and self._should_add(row, bar):
                self._scale_in(signal, decision, bar)
                self._add_count_total += 1
            elif cfg.pyramid_enabled and self.position:
                self._add_skip_count += 1
        else:
            # 无持仓：开新仓
            self._scale_in(signal, decision, bar)

    def _metrics(self) -> BacktestResult:
        eq = pd.Series(self.equity_curve)
        rets = eq.pct_change().dropna()
        log_rets = np.log1p(rets).values

        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        years = len(eq) * interval_ms(self.cfg.base_interval) / (interval_ms("1d") * 365)
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0

        bars_per_year = interval_ms("1d") * 365 / interval_ms(self.cfg.base_interval)
        vol = float(np.std(log_rets) * np.sqrt(bars_per_year))
        sharpe = float(np.mean(log_rets) / np.std(log_rets) * np.sqrt(bars_per_year)) if vol > 0 else 0.0
        downside = log_rets[log_rets < 0]
        sortino = sharpe * np.std(log_rets) / float(np.std(downside)) if len(downside) > 0 and np.std(downside) > 0 else 0.0

        cummax = np.maximum.accumulate(eq.values)
        drawdown = (eq.values - cummax) / cummax
        max_dd = float(np.min(drawdown))
        max_dd_pct = float(max_dd * 100)
        # USD 回撤必须按峰值算（eq - cummax），用 eq[0] 折算会在"先涨后跌"时低估真实亏损额
        max_dd_usdt = float(np.min(eq.values - cummax))

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0
        avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
        avg_loss = abs(float(np.mean([t.pnl for t in losses]))) if losses else 0.0
        # 无亏损交易时给 inf，和 profit_factor 同口径；写死 1.0 兜底会把"平均盈利额（USDT）"
        # 当成比值输出，报表上出现 0.37 这种不可能的盈亏比
        payoff = avg_win / avg_loss if avg_loss > 0 else float("inf") if wins else 0.0
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")

        return BacktestResult(
            equity_curve=list(eq),
            timestamps=self.timestamps,
            trades=self.trades,
            final_equity=float(eq.iloc[-1]),
            total_return=total_return,
            cagr=cagr,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=abs(max_dd_usdt),
            max_drawdown_pct=max_dd_pct,
            win_rate=win_rate,
            payoff_ratio=payoff,
            profit_factor=pf,
            total_trades=len(self.trades),
            winning_trades=len(wins),
            avg_bars_held=float(np.mean([t.bars_held for t in self.trades])) if self.trades else 0.0,
        )
