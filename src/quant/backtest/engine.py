"""事件驱动回测引擎：预计算 + 无前视信号 → 共用执行层 → 净值唯一真值。

撮合与账本不在本文件：仓位管理、触发价、费用与逐笔盈亏全部来自
`backtest/position.py`，paper 服务用的是同一份代码（设计文档 §8.5）。本模块只负责
四件事：指标预计算、高周期可见性游标、预热期、把逐根 bar 交给 `BarPipeline`。

signal_fn 签名:
    (bar: Bar, row: pd.Series, htf: dict[str, pd.Series], dt: datetime) -> Signal | None

- row: 当前基础周期 bar 的完整指标行（预计算列全部就绪）
- htf: 各高周期最近一条**已收盘** bar 的指标 Series（key=interval）
- dt: 信号时间（t_vis = ts_open + 周期 ms）
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..core.contracts import Bar, Signal, SymbolSpec
from ..core.timeframe import interval_ms, to_ms
from .position import BarPipeline, HtfCursor, PositionLedger, TradeRecord
from .precompute import precompute, precompute_htf, row_to_bar, warmup_bars

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "PositionLedger",
    "TradeRecord",
    "warmup_bars",
]


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
    include_funding: bool = False     # 默认不计资金费：10× 下三天只占保证金 0.24%，不如去掉这个无关变量
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
    # ---- 分批止盈与保本 ----
    tp1_r: float = 1.0                 # 到几个 R 平一部分（<=0 关闭）
    tp1_close_frac: float = 0.5        # 分批止盈平掉的比例（占当前持仓）
    breakeven_after_r: float = 1.0     # 浮盈到几个 R 把止损移到开仓价（<=0 关闭）


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


# ---- 引擎 --------------------------------------------------------------------


class BacktestEngine:
    """事件驱动回测，逐根基础周期 bar 推进，指标全预计算。

    持仓与账本状态在 `self.ledger`（`PositionLedger`），逐根 bar 的执行序列在
    `self.pipeline`（`BarPipeline`）；本类的 `run()` 只做数据准备与驱动。
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
        self.ledger = PositionLedger(config, spec, funding)
        self.pipeline: BarPipeline | None = None

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
        funding = None
        if self.funding is not None and not self.funding.empty:
            funding = self.funding.sort_values("funded_at_ms").reset_index(drop=True)

        self.ledger.reset(cfg.initial_equity)
        self.ledger.set_funding(funding)
        self.pipeline = BarPipeline(cfg, self.ledger, signal_fn)

        # E7/E8：策略生命周期钩子。ML 策略需要引擎注入完整历史来预热特征，
        # 规则策略无状态，reset() 只是把内部状态归零。
        if strategy is not None:
            bind = getattr(strategy, "bind_history", None)
            if callable(bind):
                bind(self.bars, funding)
            strategy.reset()

        # 预计算 5m 指标（一次性，O(n)）。在整段加载数据上算，指标预热不受回测窗口起点影响
        df5 = df5.sort_values("ts_open_ms").reset_index(drop=True)
        df5 = precompute(df5, swing_window=max(2, int(cfg.swing_lookback)))

        # 时间窗口过滤（窗口外的 bar 只用于预热指标，不参与撮合）
        start_ms = to_ms(cfg.start)
        end_ms = to_ms(cfg.end)
        df5 = df5[(df5.ts_open_ms >= start_ms) & (df5.ts_open_ms < end_ms)].reset_index(drop=True)

        # 预计算 HTF DataFrame
        frames: dict[str, pd.DataFrame] = {}
        for iv in ("1h", "4h"):
            df = self.bars.get(iv)
            if df is not None and not df.empty:
                frames[iv] = precompute_htf(df.sort_values("ts_open_ms").reset_index(drop=True))
        cursor = HtfCursor(frames)

        # 预热期：窗口最前面 3120 根 5m（10.8 天）只让高周期指针与指标就位，不出信号。
        # 规则策略的加载窗口不带前置历史，高周期 EMA60/ADX14 在起点处还是 NaN。
        warmup = warmup_bars()

        for i, row in df5.iterrows():
            bar = row_to_bar(row, cfg.symbol, cfg.base_interval)
            htf = cursor.advance(to_ms(bar.ts_close) + 1)
            self.pipeline.on_bar(bar, row, htf, mode="live" if i >= warmup else "warmup")

        # 平最后持仓
        if len(df5) > 0:
            self.pipeline.finalize(row_to_bar(df5.iloc[-1], cfg.symbol, cfg.base_interval))

        if strategy is not None:
            strategy.finalize()

        return self._metrics()

    def _metrics(self) -> BacktestResult:
        eq = pd.Series(self.ledger.equity_curve)
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

        trades = self.ledger.trades
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
        avg_loss = abs(float(np.mean([t.pnl for t in losses]))) if losses else 0.0
        # 无亏损交易时给 inf，和 profit_factor 同口径；写死 1.0 兜底会把"平均盈利额（USDT）"
        # 当成比值输出，报表上出现 0.37 这种不可能的盈亏比
        payoff = avg_win / avg_loss if avg_loss > 0 else float("inf") if wins else 0.0
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")

        return BacktestResult(
            equity_curve=list(self.ledger.equity_curve),
            timestamps=self.ledger.timestamps,
            trades=trades,
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
            total_trades=len(trades),
            winning_trades=len(wins),
            avg_bars_held=float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
        )
