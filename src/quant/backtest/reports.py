"""生成回测报告（CSV + 完整 HTML 报告）。"""
from __future__ import annotations

import csv
import json
import logging
import textwrap
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.timeframe import interval_ms

BJ_TZ = timezone(timedelta(hours=8))  # 北京时间 UTC+8

# 单根 bar 在 tooltip 里最多展开几笔事件，超出折叠成一行提示：
# 浮窗被 confine 在图表容器内，不限条数就会上下都裁切。
MAX_TIP_EVENTS = 10

# K 线聚合源候选周期，从细到粗
_FINER_FIRST = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")

# 平仓原因的标签与颜色：kline tooltip、图例、逐笔流水三处共用一份，
# 各自维护副本会漏项（曾漏 signal，报表上直接显示英文 "signal"）
REASON_CN: dict[str, str] = {
    "signal": "信号平仓",
    "sl": "止损",
    "tp": "止盈",
    "tp1": "止盈 1",
    "tp2": "止盈 2",
    "trail": "移动止损",
    "time": "时间止损",
    "liq": "强平",
    "flip": "反手",
    "flat": "期末平仓",
}
REASON_COLOR: dict[str, str] = {
    "signal": "#3498db",
    "sl": "#e74c3c",
    "tp": "#27ae60",
    "tp1": "#2ecc71",
    "tp2": "#1abc9c",
    "trail": "#9b59b6",
    "time": "#f39c12",
    "liq": "#c0392b",
    "flip": "#e67e22",
    "flat": "#7f8c8d",
}

_LOG = logging.getLogger(__name__)


def to_bj(dt: datetime) -> datetime:
    """tz-aware/naive UTC datetime → 北京时间 datetime（UTC+8，带 tzinfo）。

    naive 视作 UTC。
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(BJ_TZ)


def bj_str(dt: datetime, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """tz-aware/naive UTC datetime → 北京时间格式化字符串。"""
    return to_bj(dt).strftime(fmt)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_equity_csv(path: Path, timestamps: list[datetime], equity: list[float]) -> None:
    write_csv(path, [
        {"timestamp": to_bj(ts).isoformat(), "equity": f"{v:.4f}"}
        for ts, v in zip(timestamps, equity)
    ], ["timestamp", "equity"])


def write_trades_csv(path: Path, trades) -> None:
    write_csv(path, [
        {
            "entry_time": to_bj(t.entry_time).isoformat(),
            "exit_time": to_bj(t.exit_time).isoformat(),
            "side": t.side,
            "entry_price": f"{t.entry_price:.4f}",
            "exit_price": f"{t.exit_price:.4f}",
            "qty": f"{t.qty:.4f}",
            "pnl": f"{t.pnl:.4f}",
            "margin_usdt": f"{t.margin_usdt:.4f}",
            "ret_pct": f"{t.ret_pct:+.4f}",
            "fee": f"{t.fee:.4f}",
            "reason": t.reason,
            "r_achieved": f"{t.r_achieved:.4f}",
            "bars_held": t.bars_held,
            "add_count": t.add_count,
            "avg_entry_price": f"{t.avg_entry_price:.4f}",
            "initial_qty": f"{t.initial_qty:.4f}",
        }
        for t in trades
    ], ["entry_time","exit_time","side","entry_price","exit_price","qty","pnl","margin_usdt","ret_pct","fee","reason","r_achieved","bars_held","add_count","avg_entry_price","initial_qty"])


def write_metrics_json(path: Path, result, config, dataset_version: str) -> None:
    rets = [t.ret_pct for t in result.trades if t.margin_usdt > 0]
    margins = [t.margin_usdt for t in result.trades if t.margin_usdt > 0]
    avg_ret = float(np.mean(rets)) if rets else 0.0
    med_ret = float(np.median(rets)) if rets else 0.0
    data = {
        "run_at": datetime.now(UTC).isoformat(),
        "dataset_version": dataset_version,
        "symbol": config.symbol,
        "interval": config.base_interval,
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "initial_equity": config.initial_equity,
        "final_equity": f"{result.final_equity:.4f}",
        "total_return": f"{result.total_return:.4%}",
        "cagr": f"{result.cagr:.4%}",
        "sharpe": f"{result.sharpe:.4f}",
        "sortino": f"{result.sortino:.4f}",
        "max_drawdown_usdt": f"{result.max_drawdown:.4f}",
        "max_drawdown_pct": f"{result.max_drawdown_pct:.4f}",
        "win_rate": f"{result.win_rate:.4%}",
        "payoff_ratio": f"{result.payoff_ratio:.4f}",
        "profit_factor": f"{result.profit_factor:.4f}",
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "avg_bars_held": f"{result.avg_bars_held:.1f}",
        "avg_margin_usdt": f"{(np.mean(margins) if margins else 0.0):.4f}",
        "avg_trade_ret_on_margin_pct": f"{avg_ret:+.4f}",
        "median_trade_ret_on_margin_pct": f"{med_ret:+.4f}",
        "best_trade_ret_on_margin_pct": f"{(max(rets) if rets else 0.0):+.4f}",
        "worst_trade_ret_on_margin_pct": f"{(min(rets) if rets else 0.0):+.4f}",
        "risk_pct": config.risk_pct,
        "sl_atr_mult": config.sl_atr_mult,
        "tp_atr_mult": config.tp_atr_mult,
        "global_max_leverage": config.global_max_leverage,
        "taker_fee": config.taker_fee,
        "slippage_bps": config.slippage_bps,
        "allow_short": config.allow_short,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 完整 HTML 报告（精致样式 + 指标口径说明）
# ---------------------------------------------------------------------------

_METRICS_GLOSSARY = """
<h2>指标口径说明</h2>
<div class="table-box" style="margin-top:8px;">
<table>
<thead><tr><th>指标</th><th>口径 / 计算公式</th></tr></thead>
<tbody>
<tr><td><strong>总收益率</strong></td><td>(期末净值 / 期初净值 - 1) × 100%</td></tr>
<tr><td><strong>年化收益率 (CAGR)</strong></td><td>(期末净值 / 期初净值)^(365/交易天数) - 1，反应年均复利增速</td></tr>
<tr><td><strong>最大回撤 (Max Drawdown)</strong></td><td>净值从峰值最大跌幅 = max(peak - trough) / peak，USD 表示绝对亏损额</td></tr>
<tr><td><strong>夏普比率 (Sharpe)</strong></td><td>E[日收益] / σ[日收益] × √(年化交易日)，衡量单位风险收益；&gt;1 为优，&lt;0 为差</td></tr>
<tr><td><strong>索提诺比率 (Sortino)</strong></td><td>同 Sharpe，但仅用下行波动率 σ₂（下行偏差），更关注下跌风险</td></tr>
<tr><td><strong>Profit Factor</strong></td><td>总盈利 / 总亏损，绝对值 &gt;1.5 表明策略具有正期望；&lt;1 则整体亏损</td></tr>
<tr><td><strong>盈亏比 (Payoff Ratio)</strong></td><td>平均盈利交易 PnL / 平均亏损交易 |PnL|，配合胜率决定期望 E = WR×Payoff - (1-WR)</td></tr>
<tr><td><strong>胜率 (Win Rate)</strong></td><td>盈利交易次数 / 总交易次数，反映信号转化为盈利的频率</td></tr>
<tr><td><strong>R 倍数</strong></td><td>实际盈亏 / (ATR × SL_ATR倍数)，1R = 一倍止损距离；正 R 盈利，负 R 亏损</td></tr>
<tr><td><strong>期望公式</strong></td><td>E = 胜率 × 盈亏比 - (1 - 胜率)；E &gt; 0 才具有长期正收益</td></tr>
<tr><td><strong>杠杆</strong></td><td>杠杆是固定的策略政策参数，由 BacktestConfig.global_max_leverage 设置；定仓时名义敞口上限 = 权益 × 该杠杆</td></tr>
<tr><td><strong>保证金 (Margin)</strong></td><td>该笔实际占用的本金 = 名义仓位 / 杠杆；金字塔加仓时按每次占用累计，与引擎现金账户口径一致</td></tr>
<tr><td><strong>单笔保证金收益率</strong></td><td>PnL / 保证金 × 100%。分母是保证金而非名义仓位，PnL 已扣除双向 taker 手续费与资金费；{leverage:.0f}× 杠杆下约为价格变动幅度的 {leverage:.0f} 倍</td></tr>
<tr><td><strong>手续费</strong></td><td>开仓 taker fee + 平仓 taker fee，双向计入成本；taker={taker_pct}%</td></tr>
<tr><td><strong>滑点</strong></td><td>入场/出场价格 ± {slippage_bps}bps ({slippage_pct}%)，每笔成交一律取不利方向</td></tr>
<tr><td><strong>止损成交口径</strong></td><td>保守取 bar 极值：多头按 {base_interval} bar 的 low、空头按 high 成交，时间记该 bar 中点。同一根 bar 内止盈止损同时触发时也按止损结算（sl_first_on_conflict），与标签口径一致</td></tr>
<tr><td><strong>资金费 (Funding)</strong></td><td>按已收盘资金费率累计：多头付费、空头收费，cost = qty × 标记价 × 费率，只在平仓时一次性计入 net_cash</td></tr>
</tbody>
</table>
</div>
"""


def _agg_bars(raw: pd.DataFrame, rule: str) -> pd.DataFrame:
    """按 pandas resample 规则把 bar DataFrame 聚合到更粗周期。"""
    r = raw.sort_values("ts_open_ms").reset_index(drop=True).copy()
    r["ts_dt"] = pd.to_datetime(r["ts_open_ms"], unit="ms", utc=True)
    agg = (
        r.set_index("ts_dt")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    agg["ts_open_ms"] = agg["ts_dt"].astype("int64")  # datetime64[ms, UTC] int64 已是毫秒
    return agg


def _render_kline_file(
    kline_df: pd.DataFrame,
    interval_label: str,
    trades,
    timestamps: list[datetime],
    out_path: Path,
) -> None:
    """把单个周期的 K 线（含买卖点标记 + 悬停交易详情）渲染为独立 HTML 文件。"""
    from pyecharts.charts import Kline
    from pyecharts import options as opts
    from pyecharts.commons.utils import JsCode

    kline_df = kline_df.sort_values("ts_open_ms").reset_index(drop=True)
    ts_beijing = pd.to_datetime(
        kline_df["ts_open_ms"], unit="ms", utc=True
    ) + pd.Timedelta(hours=8)
    kline_dates = ts_beijing.dt.strftime("%m-%d %H:%M").tolist()
    kline_full_dates = ts_beijing.dt.strftime("%Y-%m-%d %H:%M").tolist()
    kline_values = kline_df[["open", "close", "low", "high"]].values.tolist()
    ts_sorted = kline_df["ts_open_ms"].tolist()

    def find_kline_idx(ts_ms: int):
        idx = int(np.searchsorted(ts_sorted, ts_ms, side="right")) - 1
        if idx < 0 or idx >= len(ts_sorted):
            return None
        return idx

    entry_marks: list[dict] = []
    exit_marks: list[dict] = []
    trades_by_idx: dict[int, list[str]] = {}
    for i, t in enumerate(trades):
        is_long = t.side == "long"
        # tooltip 可视高度有限：每笔压到 1~2 行，日期用 bar 头给出，这里只留时分。
        entry_html = (
            f"<b style='color:{'#27ae60' if is_long else '#e74c3c'}'>"
            f"{'▲开多' if is_long else '▼开空'} #{i+1}</b>"
            f"<span style='color:#8b92a8'> {bj_str(t.entry_time, '%H:%M')} </span>"
            f"价 <b>{t.entry_price:.4f}</b>"
            f"<span style='color:#8b92a8'> 量 {t.qty:.4f}</span>"
        )
        e_idx = find_kline_idx(int(t.entry_time.timestamp() * 1000))
        if e_idx is not None:
            entry_marks.append({
                "name": "B" if is_long else "S",
                "coord": [e_idx, float(t.entry_price)],
                "value": entry_html,
                "symbol": "triangle",
                "symbolSize": 9,
                "symbolRotate": 0 if is_long else 180,
                "itemStyle": {
                    "color": "#27ae60" if is_long else "#e74c3c",
                    "borderColor": "#ffffff",
                    "borderWidth": 0.5,
                },
            })
            trades_by_idx.setdefault(e_idx, []).append(entry_html)
        exit_color = REASON_COLOR.get(t.reason, "#7f8c8d")
        ret_color = "#27ae60" if t.pnl > 0 else "#e74c3c"
        # 两行为限：保证金/持仓根数在「逐笔交易流水」表里，tooltip 里不重复占高。
        exit_html = (
            f"<b style='color:{exit_color}'>●平仓 #{i+1} "
            f"{REASON_CN.get(t.reason, t.reason.upper())}</b>"
            f"<span style='color:#8b92a8'> {bj_str(t.exit_time, '%H:%M')} </span>"
            f"价 <b>{t.exit_price:.4f}</b><br>"
            f"<span style='color:{ret_color};font-weight:600'>PnL {t.pnl:+.4f}U"
            f" 收益 {t.ret_pct:+.2f}%</span>"
            f"<span style='color:#8b92a8'> (对保证金)</span>"
        )
        x_idx = find_kline_idx(int(t.exit_time.timestamp() * 1000))
        if x_idx is not None:
            exit_marks.append({
                "name": f"出{i+1}",
                "coord": [x_idx, float(t.exit_price)],
                "value": exit_html,
                "symbol": "circle",
                "symbolSize": 5,
                "itemStyle": {
                    "color": exit_color,
                    "borderColor": "#ffffff",
                    "borderWidth": 0.5,
                },
            })
            trades_by_idx.setdefault(x_idx, []).append(exit_html)

    kline_chart = (
        Kline(init_opts=opts.InitOpts(
            width="100%", height="620px",
            bg_color="#1a1d27", theme="dark",
        ))
        .add_xaxis(kline_dates)
        .add_yaxis(
            series_name=f"K线 ({interval_label})",
            y_axis=kline_values,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#27ae60", color0="#e74c3c",
                border_color="#27ae60", border_color0="#e74c3c",
            ),
            markpoint_opts=opts.MarkPointOpts(
                data=entry_marks + exit_marks,
                symbol="triangle",
                symbol_size=9,
                label_opts=opts.LabelOpts(
                    is_show=True,
                    formatter=JsCode(
                        "function(p){var n=p.name;return (n==='B'||n==='S')?n:'';}"
                    ),
                    font_size=7,
                    font_weight="bold",
                    color="#ffffff",
                    position="inside",
                ),
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"K线 · {interval_label} · 买卖点",
                subtitle=f"{bj_str(timestamps[0], '%Y-%m-%d')} ~ {bj_str(timestamps[-1], '%Y-%m-%d')}",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(color="#e8eaf0", font_size=14),
                subtitle_textstyle_opts=opts.TextStyleOpts(color="#8b92a8", font_size=10),
            ),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                axislabel_opts=opts.LabelOpts(color="#8b92a8", font_size=10),
            ),
            yaxis_opts=opts.AxisOpts(
                name="价格 (USDT)",
                name_textstyle_opts=opts.TextStyleOpts(color="#8b92a8"),
                axislabel_opts=opts.LabelOpts(color="#8b92a8"),
                splitline_opts=opts.SplitLineOpts(
                    is_show=True,
                    linestyle_opts=opts.LineStyleOpts(opacity=0.08),
                ),
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(35,39,54,0.96)",
                border_color="#4f8fe8",
                textstyle_opts=opts.TextStyleOpts(color="#e8eaf0"),
                # 单根 bar 上可能叠加多笔开/平仓。三层保证不裁切：
                # ① 每笔压到 1~2 行；② 超过 MAX_TIP_EVENTS 条折叠成提示；
                # ③ position 回调把浮窗整体收进图表可视区（不再贴边被切）。
                is_confine=True,
                is_enterable=True,
                extra_css_text="box-shadow:0 6px 20px rgba(0,0,0,.45);border-radius:8px;",
                position=JsCode(
                    "function(pos,params,dom,rect,size){"
                    "var vw=size.viewSize[0],vh=size.viewSize[1];"
                    "var cw=size.contentSize[0];"
                    "var ch=Math.min(size.contentSize[1],vh-84);"
                    "var x=pos[0]+18;if(x+cw>vw-8){x=pos[0]-cw-18;}if(x<8){x=8;}"
                    "var y=pos[1]-ch/2;if(y<64){y=64;}if(y+ch>vh-36){y=Math.max(64,vh-36-ch);}"
                    "return [x,y];}"
                ),
                formatter=JsCode(
                    "function(params){"
                    "if(!params||!params.length)return '';"
                    "var idx=params[0].dataIndex;"
                    "var d=(window.KL_DATES&&window.KL_DATES[idx])||params[0].axisValueLabel;"
                    "var tr=(window.KL_TRADES&&window.KL_TRADES[idx])||[];"
                    "var H='<div style=\"line-height:1.5\">';"
                    "H+='<div style=\"font-weight:700;color:#c8d4f0\">'+d"
                    "+(tr.length?' <span style=\"color:#8b92a8;font-weight:400\">· '"
                    "+tr.length+' 笔事件</span>':'')+'</div>';"
                    "for(var i=0;i<params.length;i++){"
                    "var v=params[i].value;"
                    "if(v&&v.length>=5){"
                    "H+='<div style=\"color:#8b92a8\">开 '+v[1]+' 收 '+v[2]"
                    "+' 低 '+v[3]+' 高 '+v[4]+'</div>';}}"
                    "if(tr.length){"
                    "H+='<div style=\"border-top:1px solid rgba(79,143,232,.45);margin:6px 0 5px\"></div>';"
                    "var cap=" + str(MAX_TIP_EVENTS) + ";"
                    "for(var k=0;k<Math.min(tr.length,cap);k++){"
                    "H+='<div style=\"margin:0 0 5px;padding-left:6px;"
                    "border-left:2px solid #4f8fe8\">'+tr[k]+'</div>';}"
                    "if(tr.length>cap){H+='<div style=\"color:#f39c12\">…另有 '"
                    "+(tr.length-cap)+' 笔未展开，见下方「逐笔交易流水」表</div>';}}"
                    # ECharts 给 tooltip 根节点写死 white-space:nowrap，长行会被横向裁切；
                    # 这里显式恢复换行，并固定宽度让换行位置稳定。
                    "return '<div style=\"width:330px;max-height:500px;overflow-y:auto;"
                    "white-space:normal;word-break:break-word;padding-right:6px\">'+H+'</div>';"
                    "}"
                ),
            ),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside", xaxis_index=0,
                                   range_start=85, range_end=100),
                opts.DataZoomOpts(type_="slider", xaxis_index=0,
                                   pos_bottom="3%", range_start=85, range_end=100),
            ],
            toolbox_opts=opts.ToolboxOpts(
                pos_right="20px", pos_top="20px",
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(),
                    brush=opts.ToolBoxFeatureBrushOpts(),
                ),
            ),
        )
    )
    kline_chart.add_js_funcs("window.KL_DATES = " + json.dumps(kline_full_dates) + ";")
    kline_chart.add_js_funcs(
        "window.KL_TRADES = "
        + json.dumps({str(k): v for k, v in trades_by_idx.items()}, ensure_ascii=False)
        + ";"
    )
    # 关闭 echarts 渐进渲染：candlestick 在 bar 数超过 progressiveThreshold(默认 10000)
    # 时会分块绘制，15m 等细粒度（近 10 万根）会整层画空。
    for _s in kline_chart.options.get("series", []):
        _s["progressive"] = 0
        _s["progressiveThreshold"] = 10**9
    # category 轴缩放后标签密度超过可用像素会互相压字（如 "04-21" 与 "08:00" 叠成
    # "04-21708:00"）。hideOverlap 让 ECharts 直接丢弃放不下的标签，比旋转更好读。
    for _ax in kline_chart.options.get("xAxis", []):
        _lbl = _ax.get("axisLabel")
        if _lbl is None:
            _ax["axisLabel"] = {"hideOverlap": True}
        elif isinstance(_lbl, dict):
            _lbl["hideOverlap"] = True
        else:  # pyecharts LabelOpts 是只读映射，可写字典在 .opts 里
            _lbl.opts["hideOverlap"] = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kline_chart.render(str(out_path))


def _monthly_returns(
    timestamps: list[datetime],
    equity,
    initial_equity: float,
) -> pd.Series:
    """按北京时间月末分桶的月度收益率；首月用窗口起点权益做基准。

    分桶必须按北京时间切：UTC 月界对应北京每月 1 日 08:00，直接 resample 会把
    每月头 8 小时的盈亏算进上个月。
    pct_change() 会让第一个月变成 NaN 被 drop，等于系统性隐藏首月盈亏
    （实测 lgbm_v1 半年报里 2026-04 的 23 笔 / −8.28 USDT 整月消失）。
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if ts.tz is None:
        ts = ts.tz_localize(UTC)
    ts = ts.tz_convert(BJ_TZ).tz_localize(None)
    monthly = pd.Series(np.asarray(equity, dtype=float), index=ts).resample("ME").last().dropna()
    if monthly.empty:
        return monthly
    base = float(initial_equity) if initial_equity > 0 else float(monthly.iloc[0])
    ret = monthly.pct_change()
    ret.iloc[0] = monthly.iloc[0] / base - 1.0
    return ret.dropna()


def write_html_full_report(
    path: Path,
    result,
    config,
    timestamps: list[datetime],
    equity: list[float],
    trades,
    initial_equity: float,
    bars: dict | None = None,
) -> None:
    """生成完整 HTML 回测报告：精致样式 + 图表 + 逐笔流水 + 指标口径说明。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    eq = np.array(equity)
    eq_norm = eq / eq[0]

    # 净值曲线数据降采样到 ~3000 点（避免 200k+ 点让 HTML 膨胀到几十 MB）
    n = len(eq)
    target = 3000
    if n > target:
        idx = np.linspace(0, n - 1, target, dtype=int)
        ds_timestamps = [timestamps[i] for i in idx]
        ds_eq = eq[idx]
        ds_eq_norm = ds_eq / ds_eq[0]
        ds_cummax = np.maximum.accumulate(ds_eq_norm)
        ds_dd_pct = (ds_eq_norm - ds_cummax) / ds_cummax * 100
    else:
        ds_timestamps = timestamps
        ds_eq = eq
        ds_eq_norm = eq_norm
        ds_cummax = np.maximum.accumulate(ds_eq_norm)
        ds_dd_pct = (ds_eq_norm - ds_cummax) / ds_cummax * 100

    # 核心指标
    total_return = result.total_return * 100
    cagr = result.cagr * 100
    max_dd = result.max_drawdown
    max_dd_pct = result.max_drawdown_pct
    expectancy = result.win_rate * result.payoff_ratio - (1 - result.win_rate)

    # 单笔保证金收益率（分母是该笔占用的保证金，不是名义仓位）
    closed = [t for t in trades if t.margin_usdt > 0]
    rets = [t.ret_pct for t in closed]
    avg_ret_m = float(np.mean(rets)) if rets else 0.0
    med_ret_m = float(np.median(rets)) if rets else 0.0
    avg_margin = float(np.mean([t.margin_usdt for t in closed])) if closed else 0.0

    # 月度收益
    monthly_ret = _monthly_returns(timestamps, eq, initial_equity)

    # 按平仓原因分组
    reason_stats: dict[str, dict] = {}
    for t in trades:
        key = t.reason
        if key not in reason_stats:
            reason_stats[key] = {"count": 0, "pnl_sum": 0.0, "wins": 0}
        reason_stats[key]["count"] += 1
        reason_stats[key]["pnl_sum"] += t.pnl
        if t.pnl > 0:
            reason_stats[key]["wins"] += 1

    # ---- Plotly 图表 ----
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.40, 0.16, 0.22, 0.22],
        subplot_titles=("", "", "", ""),
    )

    # 净值用真实 USDT 数值，纵轴按数据自适应（不再归一化到 1，否则趋势看不出来）
    ds_ts_bj = [to_bj(ts).replace(tzinfo=None) for ts in ds_timestamps]
    fig.add_trace(go.Scatter(
        x=ds_ts_bj, y=list(ds_eq),
        mode="lines", name="净值 (USDT)",
        line=dict(color="#4f8fe8", width=2),
        hovertemplate="时间: %{x}<br>净值: %{y:.2f} USDT<extra></extra>",
    ), row=1, col=1)

    # 回撤单独一行，避免与净值共用坐标系互相压扁
    fig.add_trace(go.Scatter(
        x=ds_ts_bj, y=ds_dd_pct,
        mode="lines", name="回撤 (%)",
        line=dict(color="#e74c3c", width=1.2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.09)",
        hovertemplate="时间: %{x}<br>回撤: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    # 逐笔 PnL
    pnl_colors = ["#27ae60" if t.pnl > 0 else "#e74c3c" for t in trades]
    fig.add_trace(go.Bar(
        x=[to_bj(t.exit_time).replace(tzinfo=None) for t in trades],
        y=[t.pnl for t in trades],
        customdata=[
            [f"{t.margin_usdt:.2f}", f"{t.ret_pct:+.2f}%"] for t in trades
        ],
        marker_color=pnl_colors,
        name="每笔 PnL",
        hovertemplate=(
            "时间: %{x}<br>PnL: %{y:.2f} USDT"
            "<br>保证金: %{customdata[0]} USDT"
            "<br>收益率(对保证金): %{customdata[1]}"
            "<extra></extra>"
        ),
        marker_line_width=0,
    ), row=3, col=1)

    # 月度收益
    if len(monthly_ret) > 0:
        m_colors = ["#27ae60" if r > 0 else "#e74c3c" for r in monthly_ret]
        m_labels = [d.strftime("%Y-%m") for d in monthly_ret.index]
        fig.add_trace(go.Bar(
            x=m_labels, y=[r * 100 for r in monthly_ret],
            marker_color=m_colors,
            name="月度收益 (%)",
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
        ), row=4, col=1)

    fig.update_layout(
        template="plotly_white",
        height=920,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.025, xanchor="right", x=1),
        margin=dict(l=55, r=55, t=20, b=50),
    )
    fig.update_xaxes(title_text="时间", row=4, col=1, showticklabels=True)
    # 净值纵轴用真实 USDT，plotly 按数据自适应，趋势起伏才看得出来
    fig.update_yaxes(
        title_text="净值 (USDT)", row=1, col=1,
        gridcolor="rgba(0,0,0,0.05)", tickformat=".1f",
    )
    fig.update_yaxes(
        title_text="回撤 (%)", row=2, col=1,
        gridcolor="rgba(0,0,0,0.05)", ticksuffix="%",
    )
    fig.update_yaxes(title_text="PnL (USDT)", row=3, col=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title_text="收益 (%)", row=4, col=1, gridcolor="rgba(0,0,0,0.05)")

    chart_html = fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="main-chart",
    )

    # ---- K 线蜡烛图（多周期切换 + 买卖点标注） ----
    candle_html = ""
    if bars:
        # 选最细可用周期作为聚合源：源周期只能向粗降采样，1h 源永远画不出 15m。
        # 判定顺序必须从细到粗，写反了会让 5m 源分支变成死代码。
        finest = None
        finest_label = ""
        for iv in _FINER_FIRST:
            df = bars.get(iv)
            if df is not None and not df.empty:
                finest, finest_label = df, iv
                break

        if finest is not None:
            # 目标周期：(标签, 文件名, resample 规则)，由细到粗
            src_ms = interval_ms(finest_label)
            targets: list[tuple[str, str, str]] = [
                ("15m", "kline_15m.html", "15min"),
                ("1h", "kline_1h.html", "1h"),
                ("4h", "kline_4h.html", "4h"),
                ("1d", "kline_1d.html", "1D"),
            ]
            # 只保留不比源周期更细的目标（等于源周期时直接用源，无需聚合）
            targets = [t for t in targets if interval_ms(t[0]) >= src_ms]
            built: list[tuple[str, str]] = []
            for label, fname, rule in targets:
                try:
                    df_tf = finest if finest_label == label else _agg_bars(finest, rule)
                    if df_tf is None or len(df_tf) < 2:
                        continue
                    _render_kline_file(
                        df_tf, f"{label} (聚合自 {finest_label})",
                        trades, timestamps, path.parent / fname,
                    )
                    built.append((label, fname))
                except Exception as e:
                    # 静默 continue 会让整个 K 线区块凭空消失，报表看起来"成功"但缺图
                    _LOG.warning("K 线 %s 渲染失败，该周期已跳过：%r", label, e)
                    continue
            if built:
                files = [fn for _, fn in built]
                default_file = "kline_4h.html" if "kline_4h.html" in files else files[0]
                _label_cn = {"15m": "15 分钟", "1h": "1 小时", "4h": "4 小时", "1d": "日线"}
                tabs = "".join(
                    f'<button class="tf-tab{" active" if fn == default_file else ""}" '
                    f'data-src="{fn}">{_label_cn.get(lb, lb)}</button>'
                    for lb, fn in built
                )
                # 出场图例按本次真实出现的平仓原因生成：写死三项会漏掉 signal/trail
                exit_legend = "".join(
                    f'<div class="legend-item"><span style="display:inline-block;width:9px;height:9px;'
                    f'background:{REASON_COLOR.get(r, "#7f8c8d")};border-radius:50%"></span> '
                    f'{REASON_CN.get(r, r.upper())}</div>'
                    for r in reason_stats
                )
                candle_html = (
                    '<h2>K 线蜡烛图 · 买卖点标注（pyecharts）</h2>'
                    '<div class="chart-wrap">'
                    '<div class="tf-tabs">' + tabs + '</div>'
                    '<div class="chart-legend">'
                    '<div class="legend-item"><div class="legend-dot" style="background:#27ae60"></div>阳线（收盘≥开盘）</div>'
                    '<div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>阴线（收盘&lt;开盘）</div>'
                    '<div class="legend-item"><span style="color:#27ae60">▲</span> 开多 B</div>'
                    '<div class="legend-item"><span style="color:#e74c3c">▼</span> 开空 S</div>'
                    + exit_legend +
                    f'<div class="legend-item">⇆ <span style="color:#8b92a8">切换周期 / 底部滑块缩放 / 悬停 bar 看交易详情'
                    f'（同根 bar 事件多于 {MAX_TIP_EVENTS} 笔时折叠，完整明细见下方「逐笔交易流水」）</span></div>'
                    '</div>'
                    f'<iframe id="kline-frame" src="{default_file}" style="width:100%;height:680px;'
                    'border:none;border-radius:10px;overflow:hidden"></iframe>'
                    '</div>'
                    '<script>document.querySelectorAll(".tf-tab").forEach(function(b){'
                    'b.addEventListener("click",function(){'
                    'document.querySelectorAll(".tf-tab").forEach(function(x){x.classList.remove("active");});'
                    'b.classList.add("active");'
                    'document.getElementById("kline-frame").setAttribute("src",b.getAttribute("data-src"));'
                    '});});</script>'
                )

    # ---- 逐笔流水 HTML ----
    def pnl_class(v: float) -> str:
        return "win-cell" if v > 0 else "loss-cell"

    def side_html(s: str) -> str:
        color = "#27ae60" if s == "long" else "#e74c3c"
        label = "多" if s == "long" else "空"
        return f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;background:{color};color:white;font-size:0.72rem;font-weight:700;">{label}</span>'

    def reason_html(r: str) -> str:
        color = REASON_COLOR.get(r, "#7f8c8d")
        label = REASON_CN.get(r, r.upper())
        return f'<span style="display:inline-block;padding:2px 7px;border-radius:4px;background:{color};color:white;font-size:0.72rem;font-weight:600;">{label}</span>'

    trade_rows = ""
    for t in trades:
        trade_rows += (
            f'<tr>'
            f'<td style="white-space:nowrap">{bj_str(t.entry_time)}</td>'
            f'<td style="white-space:nowrap">{bj_str(t.exit_time)}</td>'
            f'<td style="text-align:center">{side_html(t.side)}</td>'
            f'<td class="num">{t.entry_price:.4f}</td>'
            f'<td class="num">{t.exit_price:.4f}</td>'
            f'<td class="num">{t.qty:.4f}</td>'
            f'<td class="num {pnl_class(t.pnl)}">{t.pnl:+.4f}</td>'
            f'<td class="num">{t.margin_usdt:.2f}</td>'
            f'<td class="num {pnl_class(t.ret_pct)}">{t.ret_pct:+.2f}%</td>'
            f'<td class="num loss-cell">{t.fee:.4f}</td>'
            f'<td style="text-align:center">{reason_html(t.reason)}</td>'
            f'<td class="num {pnl_class(t.r_achieved)}">{t.r_achieved:+.2f}R</td>'
            f'<td class="num">{t.bars_held}</td>'
            f'</tr>'
        )

    # 月度收益行
    monthly_rows = ""
    for dt, ret in zip(monthly_ret.index, monthly_ret.values):
        color = "#27ae60" if ret > 0 else "#e74c3c"
        sign = "+" if ret > 0 else ""
        monthly_rows += (
            f'<tr>'
            f'<td>{dt.strftime("%Y-%m")}</td>'
            f'<td class="num" style="color:{color};font-weight:600">{sign}{ret*100:.2f}%</td>'
            f'</tr>'
        )

    # 平仓原因行
    reason_rows = ""
    for reason, stats in sorted(reason_stats.items(), key=lambda x: -abs(x[1]["pnl_sum"])):
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
        pnl_color = "#27ae60" if stats["pnl_sum"] > 0 else "#e74c3c"
        reason_rows += (
            f'<tr>'
            f'<td style="text-align:center">{reason_html(reason)}</td>'
            f'<td class="num">{stats["count"]}</td>'
            f'<td class="num" style="color:{pnl_color};font-weight:600">{stats["pnl_sum"]:+.2f}</td>'
            f'<td class="num">{wr:.1f}%</td>'
            f'</tr>'
        )

    # ---- 完整 HTML ----
    # 口径表里的成本参数必须跟着 config 走：写死 0.05%/2bps 会让改了参数的报表说谎
    glossary = textwrap.dedent(_METRICS_GLOSSARY).strip().format(
        taker_pct=f"{config.taker_fee * 100:.4f}".rstrip("0").rstrip("."),
        slippage_bps=config.slippage_bps,
        slippage_pct=f"{config.slippage_bps / 100:.4g}",
        leverage=config.global_max_leverage,
        base_interval=config.base_interval,
    )
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>回测报告 · {config.strategy_name} · {config.symbol} · {bj_str(timestamps[0], "%Y-%m-%d")} ~ {bj_str(timestamps[-1], "%Y-%m-%d")}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #232736;
    --border: #2d3348;
    --accent: #4f8fe8;
    --accent2: #27ae60;
    --danger: #e74c3c;
    --warn: #f39c12;
    --text: #e8eaf0;
    --text2: #8b92a8;
    --text3: #555e75;
    --font: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ height: 100%; }}
  body {{
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
    padding: 0;
  }}

  /* ---- 顶部标题栏 ---- */
  .header {{
    background: linear-gradient(135deg, #1a1d27 0%, #232736 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px;
    position: sticky; top: 0; z-index: 100;
  }}
  .header-inner {{ max-width: 1200px; margin: 0 auto; }}
  .header .tag {{
    display: inline-block;
    background: var(--accent);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
    text-transform: uppercase;
  }}
  .header h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
  }}
  .header .meta {{
    font-size: 0.8rem;
    color: var(--text2);
  }}
  .header .meta span {{ margin-right: 20px; }}

  /* ---- 主内容 ---- */
  .main {{ max-width: 1200px; margin: 0 auto; padding: 28px 40px 60px; }}

  /* ---- 分节标题 ---- */
  h2 {{
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 32px 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}

  /* ---- 指标卡片 ---- */
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 12px;
  }}
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
  }}
  .metric-card:hover {{
    border-color: var(--accent);
    transform: translateY(-1px);
  }}
  .metric-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent);
    border-radius: 12px 12px 0 0;
  }}
  .metric-card.pos::before {{ background: var(--accent2); }}
  .metric-card.neg::before {{ background: var(--danger); }}
  .metric-card.warn::before {{ background: var(--warn); }}
  .metric-card .label {{
    font-size: 0.72rem;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
  }}
  .metric-card .value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
  }}
  .metric-card .sub {{
    font-size: 0.7rem;
    color: var(--text3);
    margin-top: 4px;
  }}
  /* 策略参数行 */
  .params-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 4px;
  }}
  .param-pill {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: var(--text2);
  }}
  .param-pill strong {{ color: var(--text); font-weight: 600; margin-right: 4px; }}

  /* ---- 图表容器 ---- */
  .chart-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 8px;
  }}
  .tf-tabs {{
    display: flex;
    gap: 8px;
    margin-bottom: 14px;
  }}
  .tf-tab {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 18px;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text2);
    cursor: pointer;
    font-family: var(--font);
    transition: all 0.15s;
  }}
  .tf-tab:hover {{ color: var(--text); border-color: var(--accent); }}
  .tf-tab.active {{
    background: var(--accent);
    color: #ffffff;
    border-color: var(--accent);
  }}
  .chart-legend {{
    display: flex;
    gap: 20px;
    margin-bottom: 14px;
    flex-wrap: wrap;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: var(--text2);
  }}
  .legend-dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }}

  /* ---- 表格 ---- */
  .table-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 8px;
  }}
  .table-wrap .table-header {{
    padding: 12px 18px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text2);
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead th {{
    padding: 9px 14px;
    text-align: left;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  tbody tr {{
    border-bottom: 1px solid rgba(45,51,72,0.5);
    transition: background 0.15s;
  }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: rgba(79,143,232,0.04); }}
  td {{ padding: 9px 14px; font-size: 0.82rem; color: var(--text); }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .win-cell {{ color: var(--accent2) !important; font-weight: 600; }}
  .loss-cell {{ color: var(--danger) !important; }}
  .tbl-scroll {{ max-height: 420px; overflow-y: auto; }}
  .tbl-scroll::-webkit-scrollbar {{ width: 6px; }}
  .tbl-scroll::-webkit-scrollbar-track {{ background: var(--surface2); }}
  .tbl-scroll::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

  /* ---- 指标口径表格特殊样式 ---- */
  .glossary-table td:first-child {{ white-space: nowrap; font-weight: 600; color: var(--accent); width: 35%; }}
  .glossary-table td:last-child {{ color: var(--text2); font-size: 0.8rem; }}

  /* ---- 期望计算提示 ---- */
  .expectancy-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 16px;
    font-size: 0.82rem;
    color: var(--text2);
  }}
  .expectancy-box strong {{ color: var(--text); }}
  .expectancy-box .formula {{
    background: var(--surface2);
    border-radius: 6px;
    padding: 8px 14px;
    margin-top: 8px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.85rem;
    color: var(--accent);
    display: inline-block;
  }}
  .expectancy-box .result {{
    margin-left: 12px;
    font-weight: 700;
    color: {{"var(--accent2)" if expectancy > 0 else "var(--danger)"}};
  }}

  /* ---- 响应式 ---- */
  @media (max-width: 768px) {{
    .header {{ padding: 20px; }}
    .main {{ padding: 16px 16px 40px; }}
    .metrics-grid {{ grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }}
    .header h1 {{ font-size: 1.1rem; }}
  }}
</style>
</head>
<body>

<!-- 顶部标题 -->
<div class="header">
  <div class="header-inner">
    <div class="tag">回测报告</div>
    <h1>{config.strategy_name} · {config.symbol} · {bj_str(timestamps[0], "%Y-%m-%d")} ~ {bj_str(timestamps[-1], "%Y-%m-%d")}</h1>
    <div class="meta">
      <span>初始 {initial_equity:,.0f} USDT</span>
      <span>最终 {result.final_equity:,.2f} USDT</span>
      <span>{result.total_trades} 笔交易</span>
      <span>{config.base_interval} 周期</span>
    </div>
  </div>
</div>

<!-- 主内容 -->
<div class="main">

  <!-- 策略参数 -->
  <h2>策略参数</h2>
  <div class="params-row">
    <div class="param-pill"><strong>初始权益</strong>{initial_equity:,.0f} USDT</div>
    <div class="param-pill"><strong>风险比例</strong>{config.risk_pct:.1%}</div>
    <div class="param-pill"><strong>SL ATR</strong>{config.sl_atr_mult}×</div>
    <div class="param-pill"><strong>TP ATR</strong>{config.tp_atr_mult}×</div>
    <div class="param-pill"><strong>最大杠杆</strong>{config.global_max_leverage:.0f}×</div>
    <div class="param-pill"><strong>做空</strong>{"开启" if config.allow_short else "关闭"}</div>
    <div class="param-pill"><strong>手续费</strong>{config.taker_fee*100:.3f}%（双向）</div>
    <div class="param-pill"><strong>滑点</strong>{config.slippage_bps} bps</div>
  </div>

  <!-- 核心指标 -->
  <h2>核心指标</h2>
  <div class="metrics-grid">
    <div class="metric-card {'pos' if total_return >= 0 else 'neg'}">
      <div class="label">总收益率</div>
      <div class="value" style="color:{'#27ae60' if total_return >= 0 else '#e74c3c'}">{total_return:+.2f}%</div>
    </div>
    <div class="metric-card {'pos' if cagr >= 0 else 'neg'}">
      <div class="label">年化收益率 (CAGR)</div>
      <div class="value" style="color:{'#27ae60' if cagr >= 0 else '#e74c3c'}">{cagr:+.2f}%</div>
    </div>
    <div class="metric-card neg">
      <div class="label">最大回撤</div>
      <div class="value" style="color:#e74c3c">{max_dd_pct:.2f}%</div>
      <div class="sub">{max_dd:,.2f} USDT</div>
    </div>
    <div class="metric-card {'pos' if result.sharpe >= 1 else 'warn' if result.sharpe >= 0 else 'neg'}">
      <div class="label">夏普比率 (Sharpe)</div>
      <div class="value">{result.sharpe:.2f}</div>
      <div class="sub">Sortino {result.sortino:.2f}</div>
    </div>
    <div class="metric-card {'pos' if result.profit_factor >= 1.5 else 'warn' if result.profit_factor >= 1 else ''}">
      <div class="label">Profit Factor</div>
      <div class="value" style="color:{'#27ae60' if result.profit_factor >= 1.5 else '#f39c12' if result.profit_factor >= 1 else '#e8eaf0'}">{result.profit_factor:.2f}</div>
      <div class="sub">盈亏比 {result.payoff_ratio:.2f}</div>
    </div>
    <div class="metric-card {'pos' if result.win_rate >= 0.4 else 'warn' if result.win_rate >= 0.3 else ''}">
      <div class="label">胜率 (Win Rate)</div>
      <div class="value">{result.win_rate:.1%}</div>
      <div class="sub">{result.winning_trades}胜 / {result.total_trades - result.winning_trades}负</div>
    </div>
    <div class="metric-card {'pos' if avg_ret_m > 0 else 'neg'}">
      <div class="label">平均单笔保证金收益率</div>
      <div class="value" style="color:{'#27ae60' if avg_ret_m > 0 else '#e74c3c'}">{avg_ret_m:+.2f}%</div>
      <div class="sub">中位 {med_ret_m:+.2f}% · 均保证金 {avg_margin:,.2f}U</div>
    </div>
    <div class="metric-card {'pos' if expectancy > 0 else 'neg'}">
      <div class="label">交易期望 (E)</div>
      <div class="value" style="color:{'#27ae60' if expectancy > 0 else '#e74c3c'}">{expectancy:.3f}</div>
      <div class="sub">E = WR×Payoff - (1-WR)</div>
    </div>
    <div class="metric-card">
      <div class="label">平均持仓时长</div>
      <div class="value">{result.avg_bars_held:.1f}</div>
      <div class="sub">根 ({config.base_interval} bar)</div>
    </div>
  </div>

  <!-- 期望公式说明 -->
  <div class="expectancy-box">
    <strong>期望公式：</strong>E = 胜率 × 盈亏比 - (1 - 胜率)
    <div class="formula">
      E = {result.win_rate:.2f} × {result.payoff_ratio:.2f} - (1 - {result.win_rate:.2f})
      <span class="result">= {expectancy:.4f} {'→ 正期望' if expectancy > 0 else '→ 负期望（长期亏损）'}</span>
    </div>
  </div>

  <!-- 图表 -->
  <h2>净值曲线 &amp; 回撤 &amp; 逐笔 PnL &amp; 月度收益</h2>
  <div class="chart-wrap">
    <div class="chart-legend">
      <div class="legend-item"><div class="legend-dot" style="background:#4f8fe8"></div>净值 (USDT)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>回撤 (%)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#27ae60"></div>盈利交易</div>
      <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>亏损交易</div>
    </div>
    {chart_html}
  </div>

  <!-- K 线蜡烛图 + 买卖点标注 -->
  {candle_html if candle_html else ''}

  <!-- 按平仓原因统计 -->
  <h2>按平仓原因统计</h2>
  <div class="table-wrap">
    <div class="table-header">共 {result.total_trades} 笔，绿色为盈利，红色为亏损</div>
    <table>
      <thead><tr><th>原因</th><th>次数</th><th>累计 PnL (USDT)</th><th>胜率</th></tr></thead>
      <tbody>{reason_rows}</tbody>
    </table>
  </div>

  <!-- 月度收益 -->
  <h2>月度收益明细</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>月份</th><th>收益率</th></tr></thead>
      <tbody>{monthly_rows}</tbody>
    </table>
  </div>

  <!-- 逐笔流水 -->
  <h2>逐笔交易流水</h2>
  <div class="table-wrap">
    <div class="tbl-scroll">
      <table>
        <thead>
          <tr>
            <th>入场时间</th><th>出场时间</th><th>方向</th>
            <th class="num">入场价</th><th class="num">出场价</th><th class="num">数量</th>
            <th class="num">PnL (USDT)</th><th class="num">保证金 (USDT)</th><th class="num">收益率<br><span style="font-weight:400;font-size:0.68rem;opacity:.7">对保证金</span></th><th class="num">手续费</th><th>平仓原因</th>
            <th class="num">R倍数</th><th class="num">持仓根数</th>
          </tr>
        </thead>
        <tbody>{trade_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- 指标口径 -->
  {glossary}

</div>
</body>
</html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def save_report(
    run_dir: Path,
    result,
    config,
    dataset_version: str,
    bars: dict | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_equity_csv(run_dir / "equity.csv", result.timestamps, result.equity_curve)
    write_trades_csv(run_dir / "trades.csv", result.trades)
    write_metrics_json(run_dir / "metrics.json", result, config, dataset_version)
    write_html_full_report(
        run_dir / "report.html",
        result, config,
        result.timestamps, result.equity_curve, result.trades,
        initial_equity=config.initial_equity,
        bars=bars,
    )
