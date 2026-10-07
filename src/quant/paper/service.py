"""paper 服务：单进程、多品种 × 多策略的调度循环（设计文档 §8.2/§8.3）。

为什么单进程：数据只拉一次（Binance REST 的分钟级 weight 预算会被多进程重复扣）、
SQLite 单写者更安全、飞书 40/min 的限流只有在单进程里才真正生效。

一轮的节拍：
  1. 用交易所时间（`client.now_ms()`，本地时钟只做日志）判断"最新一根应已收盘的 5m bar"
     是否已经落库，缺了才增量回填；
  2. 只对**有新 bar** 或**数据滞后超阈值**的账户跑 `PaperRunner.run_round`，
     空转的 20 秒不重算 3000 根 bar 的指标；
  3. 任何一个账户抛异常都只影响它自己这一轮，主循环继续 —— 服务停摆比少推一条信号贵得多。
"""
from __future__ import annotations

import signal
import time
from typing import Any, Self

from ..core.timeframe import floor_ms, interval_ms, to_ms, utc_now
from ..data.backfill import Backfiller
from ..data.service import DataNotReady
from ..data.store import Store
from ..infra.log import get_logger
from ..infra.settings import Settings
from ..notify.feishu import FeishuConfig, FeishuNotifier
from ..strategy import get_strategy
from .config import PaperRuntime, execution_params
from .runner import PaperRunner, RoundReport

_log = get_logger("quant.paper.service")


class PaperService:
    """常驻服务主体：持有全部账户的运行态（策略实例、账本、pipeline）。"""

    def __init__(
        self,
        s: Settings,
        *,
        push: bool = True,
        symbols: list[str] | None = None,
        strategies: list[str] | None = None,
        store: Store | None = None,
        client: Any | None = None,
    ):
        self.s = s
        self.runtime = PaperRuntime.from_settings(s)
        self.store = store if store is not None else Store(s.db_path)
        self.client = client
        self.backfiller = Backfiller(client, self.store) if client is not None else None
        self.notifier = self._build_notifier(s) if push else None
        if push and self.notifier is None:
            _log.warning("飞书未启用或缺 FEISHU_WEBHOOK_URL，本轮起只落库不推送")

        targets = s.enabled_symbols()
        if symbols:
            targets = {k: v for k, v in targets.items() if k in symbols}
        if not targets:
            raise ValueError("symbols.yaml 里没有 enabled 品种可用于 paper")

        self.targets: dict[str, dict[str, Any]] = {}
        for name, cfg in targets.items():
            names = [n for n in (cfg.get("strategies") or []) if strategies is None or n in strategies]
            if names:
                self.targets[name] = {**cfg, "strategies": names}
        if not self.targets:
            raise ValueError(
                "没有可跑的 (品种, 策略)：symbols.yaml 的 strategies 为空，"
                "或被 --symbol/--strategy 过滤光了"
            )

        self.runners: dict[tuple[str, str], PaperRunner] = {}
        self._stop = False

    @staticmethod
    def _build_notifier(s: Settings) -> FeishuNotifier | None:
        cfg = FeishuConfig.from_settings(s)
        if cfg is None:
            return None
        # 飞书域名境内直连即可，走 Clash 出口反而可能被路由到境外节点，增加失败率
        return FeishuNotifier(cfg)

    # ---------- 账户装配 ----------

    def _ensure_runners(self) -> None:
        for symbol, cfg in self.targets.items():
            base = cfg.get("base_interval", "5m")
            if base != "5m":
                _log.warning("%s base_interval=%s，已注册策略均按 5m 触发，仍按 5m 跑", symbol, base)
                base = "5m"
            for name in cfg["strategies"]:
                key = (symbol, name)
                if key in self.runners:
                    continue
                try:
                    runner = self._build_runner(symbol, name, base)
                except DataNotReady as e:
                    _log.error("%s 账户无法启动：%s", key, e)
                    continue
                self.runners[key] = runner

    def _build_runner(self, symbol: str, name: str, base: str) -> PaperRunner:
        """按注册表构造策略与执行参数，口径等于"不带任何开关的 quant backtest"。"""
        StrategyCls = get_strategy(name)
        strategy = StrategyCls.build({
            "symbol": symbol,
            "model_dir": self.s.models_dir / name,
        })
        cfg = execution_params(
            self.s, symbol, base, name, strategy.engine_defaults()
        )
        return PaperRunner(
            store=self.store, symbol=symbol, strategy=strategy,
            cfg=cfg, runtime=self.runtime, notifier=self.notifier,
        )

    # ---------- 一轮 ----------

    def now_ms(self) -> int:
        """交易所时间优先：本地时钟漂移会让"这根 bar 收盘了没有"判错。"""
        if self.client is not None:
            return int(self.client.now_ms())
        return to_ms(utc_now())

    def run_cycle(self, *, now_ms: int | None = None) -> list[RoundReport]:
        now_ms = self.now_ms() if now_ms is None else now_ms
        self._ensure_runners()
        self._pull(now_ms)

        reports: list[RoundReport] = []
        for (symbol, name), runner in self.runners.items():
            if not self._account_due(symbol, runner, now_ms):
                continue
            try:
                report = runner.run_round(now_ms=now_ms)
            except Exception as e:  # 单个账户的异常不能带走主循环
                _log.exception("%s %s 本轮失败", symbol, name)
                report = RoundReport(symbol=symbol, strategy=name, skipped=f"执行异常：{e}")
            reports.append(report)
            _log.info(report.summary())
        return reports

    def _account_due(self, symbol: str, runner: PaperRunner, now_ms: int) -> bool:
        """有新 bar 才撮合；数据滞后也必须进轮，否则 stale 闸门永远得不到评估。"""
        base = runner.cfg.base_interval
        step = interval_ms(base)
        watermark = self.store.watermark_ms(symbol, base)
        if watermark is None:
            return False
        checkpoint = runner.state.checkpoint_ms()
        if checkpoint is None or watermark > checkpoint:
            return True
        return now_ms - (watermark + step) > self.runtime.guard.stale_data_max_gap_min * 60_000 + step

    def _pull(self, now_ms: int) -> None:
        if self.backfiller is None:
            return
        for symbol, cfg in self.targets.items():
            base = cfg.get("base_interval", "5m")
            step = interval_ms(base)
            expected_open = floor_ms(now_ms - self.runtime.resync_grace_ms - step, base)
            watermark = self.store.watermark_ms(symbol, base)
            if watermark is not None and watermark >= expected_open:
                continue  # 数据已追到最新收盘 bar，不占 REST 配额
            since = now_ms - self.runtime.replay_ms - 2 * step
            try:
                self.backfiller.refresh_spec(symbol)
                for iv in cfg["intervals"]:
                    self.backfiller.bars(symbol, iv, since, now_ms)
                self.backfiller.funding(symbol, since, now_ms)
            except Exception as e:
                _log.error("%s 增量回填失败（本轮按滞后处理）：%s", symbol, e)

    # ---------- 常驻 ----------

    def run_forever(self, *, once: bool = False) -> list[RoundReport]:
        self._install_signals()
        last: list[RoundReport] = []
        while not self._stop:
            last = self.run_cycle()
            if once:
                break
            self._sleep(self.runtime.poll_sec)
        return last

    def _sleep(self, seconds: float) -> None:
        """分段睡：TERM 之后最多等 1 秒就退出，别让脚本的 30s 优雅期白花。"""
        deadline = time.monotonic() + seconds
        while not self._stop and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))

    def _install_signals(self) -> None:
        def _stop(signum: int, _frame: Any) -> None:
            _log.info("收到信号 %s，当前轮结束后退出", signum)
            self._stop = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _stop)
            except ValueError:
                pass  # 非主线程（测试里从子线程起服务）装不上处理器，不影响功能

    def close(self) -> None:
        if self.notifier is not None:
            self.notifier.close()
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()
        self.store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def paper_status(store: Store) -> list[dict[str, Any]]:
    """每个纸面账户的断点/净值/持仓/推送统计。只读库、不触网，`quant paper status` 用。"""
    step = interval_ms("5m")
    out: list[dict[str, Any]] = []
    for acct in store.paper_accounts():
        symbol, strategy = str(acct["symbol"]), str(acct["strategy"])
        checkpoint = store.paper_checkpoint_ms(symbol, strategy)
        watermark = store.watermark_ms(symbol, "5m")
        pos = store.paper_open_position(symbol, strategy)
        stats = store.paper_signals_stats(symbol, strategy)
        realized, closed_trades = store.paper_pnl_since(symbol, strategy, 0)
        signals = int(stats["n"] or 0)
        sent = int(stats["n_sent"] or 0)
        out.append({
            "symbol": symbol,
            "strategy": strategy,
            "equity": round(float(acct["equity"]), 4),
            "cash": round(float(acct["cash"]), 4),
            "peak_equity": round(float(acct.get("peak_equity") or 0.0), 4),
            "initial_equity": float(acct["initial_equity"]),
            "realized_pnl": round(realized, 4),
            "closed_trades": closed_trades,
            "checkpoint_ms": checkpoint,
            "behind_bars": None if checkpoint is None or watermark is None
            else max(0, (watermark - checkpoint) // step),
            "position": None if pos is None else {
                "side": pos["side"], "qty": float(pos["qty"]),
                "avg_entry_price": float(pos["avg_entry_price"]),
                "stop_loss": pos["stop_loss"], "tp_price": pos["tp_price"],
                "liq_price": pos["liq_price"], "add_count": int(pos["add_count"] or 0),
                "tp1_done": bool(pos["tp1_done"]),
            },
            "pause_until_ms": int(acct.get("pause_until_ms") or 0),
            "signals": signals,
            "unsent": signals - sent,
            "last_signal_ms": None if not stats["last_ts"] else int(stats["last_ts"]),
        })
    return out
