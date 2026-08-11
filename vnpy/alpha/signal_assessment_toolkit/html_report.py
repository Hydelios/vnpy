"""从信号分析和回测产物生成可浏览的 HTML 报告。"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd

from .config import AssessmentConfig


PAGE_STYLE = """
body { margin:0; background:#f4f7fb; color:#172033; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif; }
main { width:min(1500px,calc(100% - 40px)); margin:28px auto 50px; }
header,.panel { background:#fff; border:1px solid #dbe3ee; border-radius:14px; box-shadow:0 5px 18px rgba(15,23,42,.04); }
header { padding:24px 28px; background:linear-gradient(135deg,#fff,#eef5ff); }
h1 { margin:0 0 8px; } h2 { margin:28px 0 12px; } p { color:#64748b; }
.panel { padding:18px; overflow-x:auto; } .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
.card { padding:14px; border:1px solid #dbe3ee; border-radius:10px; background:#f8fafc; }
table { width:100%; border-collapse:collapse; white-space:nowrap; } th,td { padding:10px 12px; border-bottom:1px solid #dbe3ee; text-align:right; }
th { background:#f8fafc; color:#475569; font-size:13px; } th:first-child,td:first-child { text-align:left; }
iframe { width:100%; height:1050px; border:0; border-radius:10px; background:#fff; }
a { color:#2563eb; text-decoration:none; } .back { display:inline-block; margin-bottom:14px; }
img.report { display:block; width:100%; height:auto; border-radius:10px; }
"""


def dataframe_html(path: Path, *, max_rows: int | None = None) -> str:
    """将 CSV 渲染成统一样式的 HTML 表格。"""
    if not path.exists():
        return f"<p>尚未生成：{escape(path.name)}</p>"
    frame = pd.read_csv(path)
    if max_rows is not None:
        frame = frame.head(max_rows)
    numeric = frame.select_dtypes(include=["number"]).columns
    frame[numeric] = frame[numeric].round(4)
    return frame.to_html(index=False, border=0, classes="data")


class HtmlReportBuilder:
    """从磁盘已有产物构建信号页、回测页和汇总首页。"""

    def __init__(self, config: AssessmentConfig) -> None:
        self.config = config

    def build_signal_pages(self) -> dict[str, Path]:
        pages = {}
        for signal_variant in self.config.signal_paths:
            output_dir = self.config.output_root / "signal_analysis" / signal_variant
            summary_path = output_dir / "signal_analysis_summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            simple_dir = output_dir / "simple_backtest_o2o"
            barra_dir = output_dir / "barra_cne5_top300_equal_weight"
            html_text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>{escape(signal_variant)} 信号分析</title><style>{PAGE_STYLE}</style></head>
<body><main><a class='back' href='../../index.html'>← 返回汇总首页</a>
<header><h1>{escape(signal_variant)} · 信号分析</h1><p>区间：{escape(str(summary.get('signal_start')))} 至 {escape(str(summary.get('signal_end')))}</p>
<div class='grid'><div class='card'>信号行数<br><strong>{int(summary.get('signal_rows', 0)):,}</strong></div>
<div class='card'>交易日<br><strong>{int(summary.get('signal_dates', 0)):,}</strong></div>
<div class='card'>股票数<br><strong>{int(summary.get('signal_symbols', 0)):,}</strong></div></div></header>
<h2>AlphaInspect · RankIC</h2><section class='panel'>{dataframe_html(output_dir / 'rank_ic_summary.csv')}
<h3>逐年 RankIC</h3>{dataframe_html(output_dir / 'rank_ic_annual.csv')}</section>
<h2>AlphaInspect · 分层收益、IC 与换手</h2><section class='panel'><img class='report' src='alphainspect_3x2.png' alt='AlphaInspect 3x2'></section>
<h2>SimpleBacktest</h2><section class='panel'><h3>滚动指标</h3>{dataframe_html(simple_dir / 'rolling_metrics.csv')}
<h3>逐年指标</h3>{dataframe_html(simple_dir / 'annual_metrics.csv')}<h3>换手指标</h3>{dataframe_html(simple_dir / 'turnover_metrics.csv')}</section>
<h2>Barra CNE5</h2><section class='panel'><h3>逐年暴露</h3>{dataframe_html(barra_dir / 'barra_cne5_annual.csv')}
<h3>最新暴露</h3>{dataframe_html(barra_dir / 'barra_cne5_latest.csv')}</section>
<p><a href='signal_analysis_summary.json'>信号分析 JSON</a> · <a href='rank_ic_daily.parquet'>RankIC 日频数据</a></p>
</main></body></html>"""
            page_path = output_dir / "index.html"
            page_path.write_text(html_text, encoding="utf-8")
            pages[signal_variant] = page_path
        return pages

    def build_backtest_pages(self) -> dict[tuple[str, str], Path]:
        pages = {}
        labels = {
            "total_return": "总收益",
            "annual_return": "年化收益",
            "max_ddpercent": "最大回撤",
            "sharpe_ratio": "Sharpe",
            "return_drawdown_ratio": "收益回撤比",
            "total_commission": "总手续费",
            "total_turnover": "总成交额",
            "total_trade_count": "总交易数",
        }
        alpha_labels = {
            "alpha_total_return": "Alpha总收益",
            "alpha_annual_return": "Alpha年化",
            "alpha_sharpe": "Alpha Sharpe",
            "alpha_max_drawdown": "Alpha最大回撤",
            "alpha_calmar": "Alpha Calmar",
        }
        for summary_path in sorted(self.config.output_root.glob("*/*/summary.json")):
            output_dir = summary_path.parent
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            signal_variant = summary.get("signal_variant", output_dir.parent.name)
            backtest_mode = summary.get("backtest_mode", output_dir.name)
            statistics = summary.get("statistics", {})
            alpha = summary.get("alpha_statistics", {})
            stat_rows = [
                {"指标": label, "数值": statistics.get(key)}
                for key, label in labels.items()
            ]
            alpha_rows = [
                {"指标": label, "数值": alpha.get(key)}
                for key, label in alpha_labels.items()
            ]
            stat_html = pd.DataFrame(stat_rows).to_html(index=False, border=0)
            alpha_html = pd.DataFrame(alpha_rows).to_html(index=False, border=0)
            yearly_html = dataframe_html(output_dir / "yearly_metrics.csv")
            html_text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>{escape(str(signal_variant))} × {escape(str(backtest_mode))}</title><style>{PAGE_STYLE}</style></head>
<body><main><a class='back' href='../../index.html'>← 返回汇总首页</a>
<header><h1>{escape(str(signal_variant))} × {escape(str(backtest_mode))}</h1>
<p>回测区间：{escape(str(statistics.get('start_date')))} 至 {escape(str(statistics.get('end_date')))}；基准：{escape(self.config.benchmark_symbol)}</p></header>
<h2>回测曲线</h2><section class='panel'><iframe src='performance.html' title='回测曲线'></iframe></section>
<h2>回测统计</h2><div class='grid'><section class='panel'><h3>策略指标</h3>{stat_html}</section>
<section class='panel'><h3>Alpha 指标</h3>{alpha_html}</section></div>
<h2>逐年回测结果</h2><section class='panel'>{yearly_html}</section>
<p><a href='summary.json'>summary.json</a> · <a href='yearly_metrics.csv'>yearly_metrics.csv</a> ·
<a href='performance.parquet'>performance.parquet</a> · <a href='daily_result.parquet'>daily_result.parquet</a></p>
</main></body></html>"""
            page_path = output_dir / "index.html"
            page_path.write_text(html_text, encoding="utf-8")
            pages[(str(signal_variant), str(backtest_mode))] = page_path
        return pages

    @staticmethod
    def _number(value, suffix: str = "") -> str:
        if value is None:
            return "-"
        return f"{float(value):.2f}{suffix}"

    def _load_backtest_records(self) -> list[dict]:
        records = []
        for summary_path in sorted(self.config.output_root.glob("*/*/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            statistics = summary.get("statistics", {})
            alpha = summary.get("alpha_statistics", {})
            records.append(
                {
                    "signal_variant": summary.get(
                        "signal_variant", summary_path.parent.parent.name
                    ),
                    "backtest_mode": summary.get(
                        "backtest_mode", summary_path.parent.name
                    ),
                    "start_date": statistics.get("start_date", ""),
                    "end_date": statistics.get("end_date", ""),
                    "total_return": statistics.get("total_return"),
                    "annual_return": statistics.get("annual_return"),
                    "max_ddpercent": statistics.get("max_ddpercent"),
                    "sharpe_ratio": statistics.get("sharpe_ratio"),
                    "return_drawdown_ratio": statistics.get(
                        "return_drawdown_ratio"
                    ),
                    "alpha_total_return": alpha.get("alpha_total_return"),
                    "alpha_annual_return": alpha.get("alpha_annual_return"),
                    "alpha_sharpe": alpha.get("alpha_sharpe"),
                    "alpha_max_drawdown": alpha.get("alpha_max_drawdown"),
                    "yearly_metrics": summary.get("yearly_metrics", []),
                    "relative_dir": summary_path.parent.relative_to(
                        self.config.output_root
                    ).as_posix(),
                }
            )
        return records

    def build_summary_index(self) -> Path:
        signal_pages = self.build_signal_pages()
        self.build_backtest_pages()
        records = self._load_backtest_records()
        if not records:
            raise RuntimeError(f"没有可汇总的回测结果: {self.config.output_root}")

        rows = []
        for record in records:
            relative_dir = record["relative_dir"]
            rows.append(
                "<tr>"
                f"<td>{escape(str(record['signal_variant']))}</td>"
                f"<td>{escape(str(record['backtest_mode']))}</td>"
                f"<td>{escape(str(record['start_date']))}</td>"
                f"<td>{escape(str(record['end_date']))}</td>"
                f"<td>{self._number(record['total_return'], '%')}</td>"
                f"<td>{self._number(record['annual_return'], '%')}</td>"
                f"<td>{self._number(record['max_ddpercent'], '%')}</td>"
                f"<td>{self._number(record['sharpe_ratio'])}</td>"
                f"<td>{self._number(record['return_drawdown_ratio'])}</td>"
                f"<td>{self._number(record['alpha_total_return'], '%')}</td>"
                f"<td>{self._number(record['alpha_annual_return'], '%')}</td>"
                f"<td>{self._number(record['alpha_sharpe'])}</td>"
                f"<td>{self._number(record['alpha_max_drawdown'], '%')}</td>"
                f"<td class='links'><a href='{relative_dir}/index.html'>综合页</a> "
                f"<a class='secondary' href='{relative_dir}/performance.html'>绩效图</a> "
                f"<a class='secondary' href='{relative_dir}/summary.json'>JSON</a></td>"
                "</tr>"
            )

        signal_cards = []
        for signal_variant in self.config.signal_paths:
            page_path = signal_pages.get(signal_variant)
            if page_path is None:
                signal_cards.append(
                    f"<div class='card'><strong>{escape(signal_variant)}</strong>"
                    "<p>尚未生成信号分析</p></div>"
                )
                continue
            relative_page = page_path.relative_to(self.config.output_root).as_posix()
            signal_cards.append(
                f"<a class='card card-link' href='{relative_page}'>"
                f"<strong>{escape(signal_variant)}</strong>"
                "<span>AlphaInspect · SimpleBacktest · Barra CNE5</span></a>"
            )

        start_dates = sorted({str(record["start_date"]) for record in records})
        end_dates = sorted({str(record["end_date"]) for record in records})
        period_text = (
            f"{start_dates[0]} 至 {end_dates[-1]}"
            if len(start_dates) == 1 and len(end_dates) == 1
            else "各组合区间见表格"
        )
        html_text = self._summary_document(
            period_text=period_text,
            record_count=len(records),
            signal_cards="".join(signal_cards),
            rows="".join(rows),
        )
        index_path = self.config.output_root / "index.html"
        index_path.write_text(html_text, encoding="utf-8")
        return index_path

    def build_all(self) -> Path:
        """构建全部详情页和汇总首页，返回首页路径。"""
        return self.build_summary_index()

    def _summary_document(
        self,
        *,
        period_text: str,
        record_count: int,
        signal_cards: str,
        rows: str,
    ) -> str:
        return f"""<!doctype html>
<html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Calendar / Report / RQTech 信号回测汇总</title>
<style>
body {{ margin:0; background:#f4f7fb; color:#172033; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif; }}
main {{ width:min(1500px,calc(100% - 40px)); margin:32px auto; }}
header,section {{ background:#fff; border:1px solid #dbe3ee; border-radius:14px; box-shadow:0 5px 18px rgba(15,23,42,.04); }}
header {{ padding:26px 30px; margin-bottom:22px; background:linear-gradient(135deg,#fff,#eef5ff); }}
h1 {{ margin:0 0 8px; }} h2 {{ margin:28px 0 12px; }} p {{ color:#64748b; }} section {{ overflow-x:auto; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }}
.card {{ display:flex; flex-direction:column; gap:7px; padding:16px; border:1px solid #dbe3ee; border-radius:12px; background:#fff; }}
.card-link {{ color:#172033; text-decoration:none; }} .card-link:hover {{ border-color:#93c5fd; background:#eff6ff; }} .card span {{ color:#64748b; font-size:13px; }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:12px 13px; border-bottom:1px solid #dbe3ee; text-align:right; }}
th {{ background:#f8fafc; color:#475569; font-size:13px; }}
th:first-child,th:nth-child(2),td:first-child,td:nth-child(2) {{ text-align:left; }}
tbody tr:hover {{ background:#f8fbff; }} .links a {{ display:inline-block; padding:6px 9px; border-radius:7px; background:#eff6ff; color:#2563eb; text-decoration:none; font-weight:650; }}
.links a.secondary {{ background:#fff; color:#475569; border:1px solid #dbe3ee; }}
</style></head><body><main>
<header><h1>Calendar / Report / RQTech 信号回测汇总</h1>
<p>回测区间：{escape(period_text)}；组合数量：{record_count}；Alpha 基准：{escape(self.config.benchmark_symbol)}。</p></header>
<h2>信号分析</h2><div class='cards'>{signal_cards}</div>
<h2>回测分析</h2><section><table><thead><tr>
<th>信号</th><th>回测口径</th><th>开始</th><th>结束</th><th>总收益</th><th>年化</th>
<th>最大回撤</th><th>Sharpe</th><th>收益回撤比</th><th>Alpha总收益</th><th>Alpha年化</th><th>Alpha Sharpe</th><th>Alpha最大回撤</th><th>回看</th>
</tr></thead><tbody>{rows}</tbody></table></section>
<p>原始汇总：<a href='backtest_comparison_all.csv'>backtest_comparison_all.csv</a>；逐年汇总：<a href='yearly_metrics_all.csv'>yearly_metrics_all.csv</a></p>
</main></body></html>"""
