from __future__ import annotations

import base64
import io
import json
import math
import re
import traceback
from datetime import datetime
from dataclasses import asdict, dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from matplotlib import pyplot as plt
from scipy import stats


from .alphainspect import _DATE_, _ASSET_, _QUANTILE_
from .alphainspect.plotting import plot_hist, plot_heatmap_monthly_mean, plot_ts
from .alphainspect.portfolio import calc_cum_return_by_quantile, plot_quantile_portfolio
from .alphainspect.turnover import (
    calc_auto_correlation,
    calc_quantile_turnover,
    plot_factor_auto_correlation,
    plot_turnover_quantile,
)


DEFAULT_LABEL_COLUMNS = ("label_o2o_h1", "label_o2o_h3", "label_o2o_h5")
PREPROCESS_VERSION = "alphainspect_v1"
HTML_TEMPLATE = """\
<html>
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
h1, h2, h3 {{ margin-bottom: 8px; }}
table {{ border-collapse: collapse; margin: 16px 0; width: 100%; }}
th, td {{ border: 1px solid #d9d9d9; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
table.sortable th {{ cursor: pointer; user-select: none; }}
table.sortable th.sort-asc::after {{ content: " ▲"; color: #666; }}
table.sortable th.sort-desc::after {{ content: " ▼"; color: #666; }}
img {{ max-width: 100%; border: 1px solid #d9d9d9; margin: 8px 0 16px; }}
code {{ background: #f5f5f5; padding: 2px 4px; }}
.muted {{ color: #666; }}
.error {{ color: #a40000; white-space: pre-wrap; }}
</style>
<script>
(function () {{
  function parseCell(text) {{
    var raw = (text || "").replace(/\\s+/g, " ").trim();
    if (raw === "" || raw === "NaN" || raw === "inf" || raw === "-inf") {{
      return {{ kind: "empty", value: raw }};
    }}
    var cleaned = raw.replace(/,/g, "");
    var isPercent = cleaned.endsWith("%");
    if (isPercent) {{
      cleaned = cleaned.slice(0, -1);
    }}
    var number = Number(cleaned);
    if (Number.isFinite(number)) {{
      return {{ kind: "number", value: isPercent ? number / 100 : number }};
    }}
    return {{ kind: "text", value: raw.toLowerCase() }};
  }}

  function compareValues(left, right, descending) {{
    if (left.kind === "empty" && right.kind !== "empty") return 1;
    if (right.kind === "empty" && left.kind !== "empty") return -1;
    var result;
    if (left.kind === "number" && right.kind === "number") {{
      result = left.value - right.value;
    }} else {{
      result = String(left.value).localeCompare(String(right.value));
    }}
    return descending ? -result : result;
  }}

  function renumberRows(table) {{
    var headers = Array.from(table.querySelectorAll("thead th"));
    var seqIndex = headers.findIndex(function (th) {{
      return th.textContent.trim() === "序号";
    }});
    if (seqIndex < 0) return;
    Array.from(table.querySelectorAll("tbody tr")).forEach(function (row, idx) {{
      var cell = row.children[seqIndex];
      if (cell) cell.textContent = String(idx + 1);
    }});
  }}

  document.addEventListener("DOMContentLoaded", function () {{
    document.querySelectorAll("table.sortable").forEach(function (table) {{
      var headers = Array.from(table.querySelectorAll("thead th"));
      headers.forEach(function (th, colIndex) {{
        th.addEventListener("click", function () {{
          var descending = !th.classList.contains("sort-desc");
          headers.forEach(function (item) {{
            item.classList.remove("sort-asc", "sort-desc");
          }});
          th.classList.add(descending ? "sort-desc" : "sort-asc");
          var rows = Array.from(table.querySelectorAll("tbody tr")).map(function (row, idx) {{
            return {{ row: row, idx: idx, value: parseCell(row.children[colIndex] ? row.children[colIndex].textContent : "") }};
          }});
          rows.sort(function (a, b) {{
            var result = compareValues(a.value, b.value, descending);
            return result || (a.idx - b.idx);
          }});
          var tbody = table.querySelector("tbody");
          rows.forEach(function (item) {{
            tbody.appendChild(item.row);
          }});
          renumberRows(table);
        }});
      }});
    }});
  }});
}}());
</script>
</head>
<body>
{body}
</body>
</html>
"""


@dataclass
class AssessmentConfig:
    enabled: bool = True
    scope: str = "full"
    label_columns: list[str] = field(default_factory=lambda: list(DEFAULT_LABEL_COLUMNS))
    sort_metric: str = "IC_IR_5D"
    quantiles: int = 10
    html_output: str = "all"
    compute_weighted_rankic: bool = False
    weight_source: str = "turnover"
    weight_transform: str = "sqrt"
    weight_window: int = 5
    weight_lag: int = 1
    turnover_periods: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    min_coverage: float = 0.8
    min_valid_day_ratio: float = 0.8
    sample_filter_column: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "AssessmentConfig":
        data = dict(raw or {})
        label_columns = data.get("label_columns") or DEFAULT_LABEL_COLUMNS
        turnover_periods = data.get("turnover_periods") or [1, 3, 5, 10, 20]
        return cls(
            enabled=bool(data.get("enabled", True)),
            scope=str(data.get("scope", "full")).strip().lower(),
            label_columns=[str(v) for v in label_columns],
            sort_metric=str(data.get("sort_metric", "IC_IR_5D")).strip(),
            quantiles=max(2, int(data.get("quantiles", 10))),
            html_output=str(data.get("html_output", "all")).strip().lower(),
            compute_weighted_rankic=False,
            weight_source=str(data.get("weight_source", "turnover")).strip(),
            weight_transform=str(data.get("weight_transform", "sqrt")).strip().lower(),
            weight_window=max(1, int(data.get("weight_window", 5))),
            weight_lag=max(0, int(data.get("weight_lag", 1))),
            turnover_periods=[max(1, int(v)) for v in turnover_periods],
            min_coverage=min(1.0, max(0.0, float(data.get("min_coverage", 0.8)))),
            min_valid_day_ratio=min(
                1.0,
                max(0.0, float(data.get("min_valid_day_ratio", 0.8))),
            ),
            sample_filter_column=(
                str(data["sample_filter_column"]).strip()
                if data.get("sample_filter_column")
                else None
            ),
        )


@dataclass
class AssessmentArtifacts:
    summary_df: pl.DataFrame
    summary_csv: Path
    summary_parquet: Path
    weighted_summary_csv: Path
    weighted_summary_parquet: Path
    index_html: Path | None
    detail_report_dir: Path | None
    meta_path: Path
    artifacts_meta: dict[str, Any]


def _metric_suffix(label_col: str) -> str:
    match = re.search(r"_h(\d+)$", label_col)
    if match:
        return f"{int(match.group(1))}D"
    trailing = re.search(r"(\d+)$", label_col)
    if trailing:
        return f"{int(trailing.group(1))}D"
    return re.sub(r"[^A-Za-z0-9]+", "_", label_col).strip("_").upper() or "UNKNOWN"


def _sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "factor"


def _fig_to_img(fig: Any, *, format: str = "png") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format=format, bbox_inches="tight")
    return '<img src="data:image/{};base64,{}" />'.format(
        format,
        base64.b64encode(buf.getvalue()).decode(),
    )


def _write_html(path: Path, *, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HTML_TEMPLATE.format(title=escape(title), body=body), encoding="utf-8")


def _with_factor_quantile(
    df: pl.DataFrame,
    factor: str,
    *,
    quantiles: int,
    factor_quantile: str = _QUANTILE_,
) -> pl.DataFrame:
    pdf = df.to_pandas()
    pdf = pdf.sort_values([_DATE_, _ASSET_]).reset_index(drop=True)
    factor_quantiles = pd.Series(np.nan, index=pdf.index, dtype=float)
    for _, group in pdf.groupby(_DATE_, sort=False):
        values = group[factor].replace([np.inf, -np.inf], np.nan)
        mask = values.notna()
        if int(mask.sum()) < 2:
            continue
        try:
            labels = pd.qcut(values[mask], q=quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        factor_quantiles.loc[group.index[mask.to_numpy()]] = labels.astype(float).to_numpy()
    pdf[factor_quantile] = factor_quantiles
    return pl.from_pandas(pdf)


def _create_3x2_sheet(
    df: pl.DataFrame,
    factor: str,
    fwd_ret_1: str,
    *,
    factor_quantile: str = _QUANTILE_,
    periods: Iterable[int] = (1, 3, 5, 10, 20),
    figsize: tuple[int, int] = (14, 15),
) -> tuple[Any, Any, Any, Any, Any, Any]:
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f"{factor} | {fwd_ret_1}", fontsize=14, y=0.985)

    df_ic = _calc_ic_frame(df, factors=[factor], label_cols=[fwd_ret_1])
    col = df_ic.columns[1]
    ic_dict = plot_ts(df_ic, col, ax=axes[0, 0])
    hist_dict = plot_hist(df_ic, col, ax=axes[0, 1])
    plot_heatmap_monthly_mean(df_ic, col, ax=axes[1, 0])

    _, cum, avg, std = calc_cum_return_by_quantile(df, fwd_ret_1, factor_quantile)
    plot_quantile_portfolio(cum, fwd_ret_1, axvlines=(), ax=axes[1, 1])

    df_auto_corr = calc_auto_correlation(df, factor, periods=list(periods))
    df_turnover = calc_quantile_turnover(df, periods=list(periods), factor_quantile=factor_quantile)
    plot_factor_auto_correlation(df_auto_corr, ax=axes[2, 0])

    q_max = int(df_turnover[factor_quantile].max())
    plot_turnover_quantile(df_turnover, quantile=q_max, factor_quantile=factor_quantile, periods=list(periods), ax=axes[2, 1])
    axes[0, 0].set_title(
        "IC Time Series\n"
        f"mean={_format_float(ic_dict.get('ic_mean'))}, "
        f"ic_ir={_format_float(ic_dict.get('ic_ir'))}, "
        f"t={_format_float(ic_dict.get('t_stat'))}, "
        f"p={_format_float(ic_dict.get('p_value'))}",
        fontsize=10,
    )
    axes[0, 1].set_title(
        "IC Distribution\n"
        f"std={_format_float(hist_dict.get('std'))}, "
        f"skew={_format_float(hist_dict.get('skew'))}, "
        f"kurt={_format_float(hist_dict.get('kurt'))}",
        fontsize=10,
    )
    axes[1, 0].set_title("IC Monthly Mean", fontsize=10)
    axes[1, 1].set_title(f"{_metric_suffix(fwd_ret_1)} Quantile Cumulative Return", fontsize=10)
    axes[2, 0].set_title("Factor Auto Correlation", fontsize=10)
    axes[2, 1].set_title(f"Quantile {q_max} Mean Turnover", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.0, w_pad=1.8)
    return fig, ic_dict, hist_dict, cum, avg, std


def _series_to_frame(series: pd.Series, *, value_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, value in series.items():
        factor, label = str(key).rsplit("__", 1)
        rows.append({"feature": factor, "label": label, value_name: value})
    return pd.DataFrame(rows)


def _merge_metric_series(
    summary_df: pd.DataFrame,
    *,
    series: pd.Series,
    label_cols: list[str],
    metric_name: str,
) -> pd.DataFrame:
    metric_frame = _series_to_frame(series, value_name="value")
    for label_col in label_cols:
        suffix = _metric_suffix(label_col)
        subset = metric_frame.loc[metric_frame["label"] == label_col, ["feature", "value"]].rename(
            columns={"value": f"{metric_name}_{suffix}"}
        )
        summary_df = summary_df.merge(subset, on="feature", how="left")
    return summary_df


def _calc_ic_frame(
    df: pl.DataFrame,
    *,
    factors: list[str],
    label_cols: list[str],
) -> pl.DataFrame:
    return (
        df.group_by(_DATE_)
        .agg(
            [
                pl.corr(factor, label_col, method="spearman", ddof=0, propagate_nans=False).alias(f"{factor}__{label_col}")
                for factor in factors
                for label_col in label_cols
            ]
        )
        .sort(_DATE_)
        .fill_nan(None)
    )


def _calc_ic_stats(df_ic: pl.DataFrame) -> dict[str, pd.Series]:
    pdf = df_ic.to_pandas().set_index(_DATE_)
    ic_mean = pdf.mean(axis=0, skipna=True)
    ic_std = pdf.std(axis=0, ddof=0, skipna=True)
    ic_ir = ic_mean / ic_std.replace(0, np.nan)
    ic_valid_days = pdf.notna().sum(axis=0)
    win_rate = (pdf > 0).mean(axis=0, skipna=True)
    t_stat, p_value = stats.ttest_1samp(pdf.reset_index(drop=True), 0, nan_policy="omit")
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ic_valid_days": ic_valid_days,
        "win_rate": win_rate,
        "t_stat": pd.Series(t_stat, index=pdf.columns),
        "p_value": pd.Series(p_value, index=pdf.columns),
    }


def _vectorized_rank_corr(
    x_rank: np.ndarray,
    y_rank: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-12
    valid = np.isfinite(x_rank) & np.isfinite(y_rank[:, None])
    cnt = valid.sum(axis=0)
    x_safe = np.where(valid, x_rank, 0.0)
    y_safe = np.where(valid, y_rank[:, None], 0.0)
    x_mean = np.divide(x_safe.sum(axis=0), cnt, out=np.full(x_rank.shape[1], np.nan), where=cnt > 1)
    y_mean = np.divide(y_safe.sum(axis=0), cnt, out=np.full(x_rank.shape[1], np.nan), where=cnt > 1)
    x_center = np.where(valid, x_rank - x_mean, 0.0)
    y_center = np.where(valid, y_rank[:, None] - y_mean, 0.0)
    cov = (x_center * y_center).sum(axis=0)
    std_x = np.sqrt((x_center * x_center).sum(axis=0))
    std_y = np.sqrt((y_center * y_center).sum(axis=0))
    corr = np.divide(
        cov,
        std_x * std_y,
        out=np.full(x_rank.shape[1], np.nan),
        where=(std_x > eps) & (std_y > eps) & (cnt > 2),
    )

    valid_w = valid & np.isfinite(weights)[:, None] & (weights[:, None] > 0)
    w = np.where(valid_w, weights[:, None], 0.0)
    w_sum = w.sum(axis=0)
    x_safe_w = np.where(valid_w, x_rank, 0.0)
    y_safe_w = np.where(valid_w, y_rank[:, None], 0.0)
    x_mean_w = np.divide((w * x_safe_w).sum(axis=0), w_sum, out=np.full(x_rank.shape[1], np.nan), where=w_sum > 0)
    y_mean_w = np.divide((w * y_safe_w).sum(axis=0), w_sum, out=np.full(x_rank.shape[1], np.nan), where=w_sum > 0)
    x_center_w = np.where(valid_w, x_rank - x_mean_w, 0.0)
    y_center_w = np.where(valid_w, y_rank[:, None] - y_mean_w, 0.0)
    cov_w = np.divide(
        (w * x_center_w * y_center_w).sum(axis=0),
        w_sum,
        out=np.full(x_rank.shape[1], np.nan),
        where=w_sum > 0,
    )
    var_x_w = np.divide(
        (w * x_center_w * x_center_w).sum(axis=0),
        w_sum,
        out=np.full(x_rank.shape[1], np.nan),
        where=w_sum > 0,
    )
    var_y_w = np.divide(
        (w * y_center_w * y_center_w).sum(axis=0),
        w_sum,
        out=np.full(x_rank.shape[1], np.nan),
        where=w_sum > 0,
    )
    corr_w = np.divide(
        cov_w,
        np.sqrt(var_x_w * var_y_w),
        out=np.full(x_rank.shape[1], np.nan),
        where=(var_x_w > eps) & (var_y_w > eps) & (w_sum > 0),
    )
    return corr, corr_w


def _factor_stats(frame: pl.DataFrame, factor: str) -> dict[str, Any]:
    valid_expr = pl.col(factor).is_not_null() & pl.col(factor).is_finite()
    stats = frame.select(
        pl.len().alias("total_rows"),
        valid_expr.sum().alias("valid_rows"),
        pl.col(factor).filter(valid_expr).mean().alias("mean"),
        pl.col(factor).filter(valid_expr).std(ddof=0).alias("std"),
        pl.col(factor).filter(valid_expr).min().alias("min"),
        pl.col(factor).filter(valid_expr).max().alias("max"),
        pl.col(factor).filter(valid_expr).n_unique().alias("unique_values"),
    ).to_dicts()[0]
    total_rows = int(stats["total_rows"])
    valid_rows = int(stats["valid_rows"])
    coverage = float(valid_rows / total_rows) if total_rows else float("nan")
    return {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "coverage": coverage,
        "missing_ratio": float(1.0 - coverage) if np.isfinite(coverage) else float("nan"),
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "unique_values": int(stats["unique_values"]),
    }


def _select_label_columns(available: Iterable[str], requested: Iterable[str]) -> list[str]:
    available_list = [str(v) for v in available]
    available_set = set(available_list)
    requested_list = [str(v) for v in requested]
    missing = [v for v in requested_list if v not in available_set]
    if missing:
        raise ValueError(f"factor assessment 缺少 label 列: {missing}")
    return requested_list


def prepare_assessment_frame(
    *,
    master_df: pl.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    valid_period: tuple[str, str] | None,
    config: AssessmentConfig,
) -> pl.DataFrame:
    if master_df.is_empty():
        raise ValueError("master_df 为空，无法执行因子评价")

    required_base = {"datetime", "vt_symbol", "close", "turnover", config.weight_source}
    missing_base = sorted(required_base - set(master_df.columns))
    if missing_base:
        raise ValueError(f"factor assessment 缺少基础列: {missing_base}")

    missing_features = [col for col in feature_cols if col not in master_df.columns]
    if missing_features:
        raise ValueError(f"factor assessment 缺少因子列: {missing_features[:20]}")

    selected_labels = _select_label_columns(master_df.columns, label_cols)

    source = master_df
    if config.sample_filter_column:
        if config.sample_filter_column not in source.columns:
            raise ValueError(
                f"factor assessment 缺少公共样本过滤列: {config.sample_filter_column}"
            )
        source = source.filter(
            pl.col(config.sample_filter_column)
            .cast(pl.Boolean, strict=False)
            .fill_null(False)
        )

    keep_cols = ["datetime", "vt_symbol", "close", config.weight_source, *feature_cols, *selected_labels]
    keep_cols = list(dict.fromkeys(keep_cols))
    frame = source.select(keep_cols).sort(["datetime", "vt_symbol"])

    if config.scope == "valid":
        if valid_period is None:
            raise ValueError("scope=valid 时必须提供 valid_period")
        start, end = valid_period
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        frame = frame.filter(
            (pl.col("datetime") >= pl.lit(start_dt))
            & (pl.col("datetime") <= pl.lit(end_dt))
        )
    elif config.scope == "full":
        pass
    else:
        raise ValueError(f"暂不支持的 factor assessment scope: {config.scope}")

    if frame.is_empty():
        raise ValueError("factor assessment 在当前区间内没有可用样本")

    numeric_cols = [col for col in frame.columns if col not in {"datetime", "vt_symbol"}]
    finite_exprs = []
    for col in numeric_cols:
        casted = pl.col(col).cast(pl.Float64, strict=False)
        finite_exprs.append(
            pl.when(casted.is_finite()).then(casted).otherwise(None).alias(col)
        )
    return (
        frame.rename({"datetime": _DATE_, "vt_symbol": _ASSET_})
        .with_columns(finite_exprs)
        .sort([_DATE_, _ASSET_])
    )


def compute_coverage_summary(frame: pl.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    total_rows = frame.height
    exprs = [
        ((pl.col(col).is_not_null() & pl.col(col).is_finite()).mean()).alias(col)
        for col in feature_cols
    ]
    coverage_row = frame.select(exprs).to_dicts()[0]
    rows = []
    for col in feature_cols:
        coverage = float(coverage_row.get(col, np.nan))
        rows.append(
            {
                "feature": col,
                "coverage": coverage,
                "missing_ratio": float(1.0 - coverage) if np.isfinite(coverage) else np.nan,
                "sample_count": int(total_rows),
            }
        )
    return pd.DataFrame(rows)


def compute_xsec_diagnostics(frame: pl.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # XSEC diagnostics are intentionally disabled; keep a feature-only frame for old callers.
    return pd.DataFrame({"feature": feature_cols})


def compute_rankic_summary(
    *,
    frame: pl.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
) -> pd.DataFrame:
    df_ic = _calc_ic_frame(frame.select([_DATE_, *feature_cols, *label_cols]), factors=feature_cols, label_cols=label_cols)
    stats = _calc_ic_stats(df_ic)
    summary_df = pd.DataFrame({"feature": feature_cols})
    metric_map = {
        "ic_mean": "IC",
        "ic_std": "IC_STD",
        "ic_ir": "IC_IR",
        "ic_valid_days": "IC_VALID_DAYS",
        "win_rate": "WIN_RATE",
        "t_stat": "T_STAT",
        "p_value": "P_VALUE",
    }
    for series_name, metric_name in metric_map.items():
        summary_df = _merge_metric_series(summary_df, series=stats[series_name], label_cols=label_cols, metric_name=metric_name)
    return summary_df.sort_values("feature").reset_index(drop=True)


def apply_quality_gate(
    *,
    summary_df: pd.DataFrame,
    frame: pl.DataFrame,
    label_cols: list[str],
    config: AssessmentConfig,
) -> tuple[pd.DataFrame, int]:
    """标记适合进入正式排名的常规因子，同时保留稀疏因子的诊断报告。"""
    total_days = int(frame.select(pl.col(_DATE_).n_unique()).item())
    min_valid_days = int(math.ceil(total_days * config.min_valid_day_ratio))
    valid_day_columns = [
        f"IC_VALID_DAYS_{_metric_suffix(label_col)}" for label_col in label_cols
    ]

    result = summary_df.copy()
    result["min_ic_valid_days"] = result[valid_day_columns].min(axis=1)
    result["required_ic_valid_days"] = min_valid_days
    result["eligible"] = (
        (result["coverage"] >= config.min_coverage)
        & (result["min_ic_valid_days"] >= min_valid_days)
    )

    def exclusion_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if float(row["coverage"]) < config.min_coverage:
            reasons.append(f"coverage<{config.min_coverage:.0%}")
        if int(row["min_ic_valid_days"]) < min_valid_days:
            reasons.append(f"valid_days<{min_valid_days}")
        return ";".join(reasons)

    result["assessment_group"] = np.where(result["eligible"], "regular", "sparse")
    result["exclusion_reason"] = result.apply(exclusion_reason, axis=1)
    return result, total_days


def _build_weighted_frame(
    *,
    frame: pl.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    config: AssessmentConfig,
) -> pl.DataFrame:
    # Deprecated: WRankIC is disabled in run_factor_assessment and this helper is kept only for old ad-hoc callers.
    if config.weight_source not in frame.columns:
        raise ValueError(f"weighted rankic 需要列: {config.weight_source}")

    weighted_frame = frame.sort([_ASSET_, _DATE_]).with_columns(
        pl.col(config.weight_source)
        .cast(pl.Float64, strict=False)
        .shift(config.weight_lag)
        .rolling_mean(window_size=config.weight_window, min_samples=1)
        .over(_ASSET_)
        .alias("_weight_base")
    )
    if config.weight_transform == "sqrt":
        weight_expr = pl.when(pl.col("_weight_base") > 0).then(pl.col("_weight_base").sqrt()).otherwise(None)
    elif config.weight_transform == "log1p":
        weight_expr = pl.when(pl.col("_weight_base") > 0).then(pl.col("_weight_base").log1p()).otherwise(None)
    elif config.weight_transform in {"none", "identity"}:
        weight_expr = pl.when(pl.col("_weight_base") > 0).then(pl.col("_weight_base")).otherwise(None)
    else:
        raise ValueError(f"不支持的 weight_transform: {config.weight_transform}")

    cast_exprs = [pl.col(col).cast(pl.Float32, strict=False) for col in feature_cols if col in weighted_frame.columns]
    cast_exprs += [pl.col(col).cast(pl.Float32, strict=False) for col in label_cols if col in weighted_frame.columns]
    return weighted_frame.with_columns([*cast_exprs, weight_expr.alias("_weight")]).sort([_DATE_, _ASSET_])


def compute_weighted_rankic_summary(
    *,
    frame: pl.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    config: AssessmentConfig,
) -> pd.DataFrame:
    # WRank series metrics are intentionally disabled; keep a feature-only frame for old callers.
    return pd.DataFrame({"feature": feature_cols})


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{float(value):.6f}"
    return str(value)


def _resolve_sort_metric(summary_pdf: pd.DataFrame, requested_metric: str) -> str:
    if requested_metric in summary_pdf.columns:
        return requested_metric
    if not requested_metric.startswith("WRankIC"):
        raise ValueError(f"factor assessment sort_metric 不存在: {requested_metric}")

    fallback_candidates = ["IC_IR_5D"]
    fallback_candidates.extend(sorted(col for col in summary_pdf.columns if col.startswith("IC_IR_")))
    fallback_candidates.append("IC_5D")
    fallback_candidates.extend(
        sorted(
            col
            for col in summary_pdf.columns
            if col.startswith("IC_")
            and not col.startswith("IC_IR_")
            and not col.startswith("IC_STD_")
            and not col.startswith("IC_VALID_DAYS_")
        )
    )
    for candidate in dict.fromkeys(fallback_candidates):
        if candidate in summary_pdf.columns:
            logger.warning(
                "[factor_assessment] WRankIC 已禁用，sort_metric={} fallback 到 {}",
                requested_metric,
                candidate,
            )
            return candidate
    raise ValueError(f"factor assessment sort_metric 不存在且无可用 IC fallback: {requested_metric}")


def _build_factor_metric_table(
    *,
    summary_row: dict[str, Any],
    label_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for label_col in label_cols:
        suffix = _metric_suffix(label_col)
        rows.append(
            {
                "label": label_col,
                "IC": summary_row.get(f"IC_{suffix}"),
                "IC_STD": summary_row.get(f"IC_STD_{suffix}"),
                "IC_IR": summary_row.get(f"IC_IR_{suffix}"),
                "IC_VALID_DAYS": summary_row.get(f"IC_VALID_DAYS_{suffix}"),
                "WIN_RATE": summary_row.get(f"WIN_RATE_{suffix}"),
                "T_STAT": summary_row.get(f"T_STAT_{suffix}"),
                "P_VALUE": summary_row.get(f"P_VALUE_{suffix}"),
            }
        )
    return pd.DataFrame(rows)


def generate_factor_detail_report(
    *,
    frame: pl.DataFrame,
    factor: str,
    summary_row: dict[str, Any],
    label_cols: list[str],
    config: AssessmentConfig,
    report_path: Path,
) -> list[str]:
    errors: list[str] = []
    factor_stats = _factor_stats(frame, factor)
    metric_table = _build_factor_metric_table(summary_row=summary_row, label_cols=label_cols)

    stats_html = pd.DataFrame([factor_stats]).to_html(index=False, float_format=_format_float)
    metrics_html = metric_table.to_html(index=False, float_format=_format_float)

    if factor_stats["unique_values"] < 2 or factor_stats["valid_rows"] < config.quantiles:
        body = "\n".join(
            [
                f"<h1>{escape(factor)}</h1>",
                "<p class='muted'>因子离散度不足，跳过分层图表，仅保留统计信息。</p>",
                "<h2>Basic Stats</h2>",
                stats_html,
                "<h2>IC Summary</h2>",
                metrics_html,
            ]
        )
        _write_html(report_path, title=factor, body=body)
        return errors

    base_cols = [_DATE_, _ASSET_, "close", config.weight_source, factor, *label_cols]
    factor_frame = _with_factor_quantile(
        frame.select(base_cols),
        factor,
        quantiles=config.quantiles,
    )

    chart_blocks: list[str] = []
    for label_col in label_cols:
        suffix = _metric_suffix(label_col)
        try:
            fig, _, _, _, _, _ = _create_3x2_sheet(
                factor_frame,
                factor,
                label_col,
                periods=config.turnover_periods,
            )
            chart_blocks.append(f"<h2>{escape(label_col)} ({suffix})</h2>")
            chart_blocks.append(_fig_to_img(fig))
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - defensive reporting
            errors.append(f"{factor}:{label_col}:{exc}")
            chart_blocks.append(
                f"<h2>{escape(label_col)} ({suffix})</h2><p class='error'>{escape(traceback.format_exc())}</p>"
            )
            plt.close("all")

    body = "\n".join(
        [
            f"<h1>{escape(factor)}</h1>",
            "<h2>Basic Stats</h2>",
            stats_html,
            "<h2>IC Summary</h2>",
            metrics_html,
            *chart_blocks,
        ]
    )
    _write_html(report_path, title=factor, body=body)
    return errors


def generate_index_page(
    *,
    output_dir: Path,
    run_name: str,
    summary_pdf: pd.DataFrame,
    config: AssessmentConfig,
    label_cols: list[str],
    report_failures: list[str],
    effective_sort_metric: str,
) -> Path:
    render_df = summary_pdf.copy()
    render_df.insert(0, "序号", np.arange(1, len(render_df) + 1))
    if "report_file" in render_df.columns:
        render_df["feature"] = render_df.apply(
            lambda row: f'<a href="{escape(str(row["report_file"]))}">{escape(str(row["feature"]))}</a>',
            axis=1,
        )
        render_df = render_df.drop(columns=["report_file"])

    summary_html = render_df.to_html(
        index=False,
        escape=False,
        float_format=_format_float,
        table_id="summary-table",
        classes="sortable",
    )
    failures_html = ""
    if report_failures:
        failures_html = "<h2>Report Failures</h2><ul>{}</ul>".format(
            "".join(f"<li>{escape(item)}</li>" for item in report_failures)
        )

    body = "\n".join(
        [
            f"<h1>{escape(run_name)} Factor Assessment</h1>",
            "<p class='muted'>默认区间: <code>{}</code> | 默认排序: <code>{}</code> | 标签: <code>{}</code></p>".format(
                escape(config.scope),
                escape(effective_sort_metric),
                escape(", ".join(label_cols)),
            ),
            "<p class='muted'>预处理: <code>{}</code> | 正式排名门槛: coverage ≥ {:.0%}, IC有效交易日 ≥ {:.0%}</p>".format(
                PREPROCESS_VERSION,
                config.min_coverage,
                config.min_valid_day_ratio,
            ),
            "<h2>Summary</h2>",
            summary_html,
            failures_html,
        ]
    )
    index_path = output_dir / "index.html"
    _write_html(index_path, title=f"{run_name} factor assessment", body=body)
    return index_path


def run_factor_assessment(
    *,
    master_df: pl.DataFrame,
    feature_cols: list[str],
    valid_period: tuple[str, str] | None,
    output_dir: str | Path,
    config: dict[str, Any] | AssessmentConfig | None = None,
    run_name: str = "factor_assessment",
    label_cols: list[str] | None = None,
) -> AssessmentArtifacts:
    assessment_config = config if isinstance(config, AssessmentConfig) else AssessmentConfig.from_mapping(config)
    if not assessment_config.enabled:
        raise ValueError("factor assessment 已禁用，不应调用 run_factor_assessment")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "assessment_summary.csv"
    summary_parquet = output_dir / "assessment_summary.parquet"
    weighted_summary_csv = output_dir / "assessment_weighted_summary.csv"
    weighted_summary_parquet = output_dir / "assessment_weighted_summary.parquet"
    meta_path = output_dir / "assessment_meta.json"
    report_dir = output_dir / "reports"

    requested_labels = label_cols or assessment_config.label_columns
    frame = prepare_assessment_frame(
        master_df=master_df,
        feature_cols=feature_cols,
        label_cols=requested_labels,
        valid_period=valid_period,
        config=assessment_config,
    )
    selected_label_cols = _select_label_columns(frame.columns, requested_labels)

    coverage_df = compute_coverage_summary(frame, feature_cols)
    rankic_df = compute_rankic_summary(frame=frame, feature_cols=feature_cols, label_cols=selected_label_cols)
    if assessment_config.compute_weighted_rankic:
        logger.warning(
            "[factor_assessment] compute_weighted_rankic 已废弃，本次忽略 WRankIC 计算请求"
        )

    summary_pdf = (
        coverage_df
        .merge(rankic_df, on="feature", how="left")
    )
    summary_pdf, total_days = apply_quality_gate(
        summary_df=summary_pdf,
        frame=frame,
        label_cols=selected_label_cols,
        config=assessment_config,
    )
    effective_sort_metric = _resolve_sort_metric(summary_pdf, assessment_config.sort_metric)

    report_failures: list[str] = []
    report_files: dict[str, str] = {}
    if assessment_config.html_output == "all":
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = {row["feature"]: row for row in summary_pdf.to_dict(orient="records")}
        for factor in feature_cols:
            safe_name = _sanitize_filename(factor)
            report_path = report_dir / f"{safe_name}.html"
            try:
                errors = generate_factor_detail_report(
                    frame=frame,
                    factor=factor,
                    summary_row=summary_rows[factor],
                    label_cols=selected_label_cols,
                    config=assessment_config,
                    report_path=report_path,
                )
                report_failures.extend(errors)
                report_files[factor] = str(report_path.relative_to(output_dir))
            except Exception as exc:  # pragma: no cover - defensive reporting
                report_failures.append(f"{factor}:{exc}")
                error_html = f"<h1>{escape(factor)}</h1><p class='error'>{escape(traceback.format_exc())}</p>"
                _write_html(report_path, title=factor, body=error_html)
                report_files[factor] = str(report_path.relative_to(output_dir))

    summary_pdf["report_file"] = summary_pdf["feature"].map(report_files.get)
    summary_pdf = summary_pdf.sort_values(
        ["eligible", effective_sort_metric],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    weighted_only_pdf = summary_pdf[["feature"]].copy()

    summary_pl = pl.from_pandas(summary_pdf)
    weighted_pl = pl.from_pandas(weighted_only_pdf)
    summary_pl.write_csv(summary_csv)
    summary_pl.write_parquet(summary_parquet, compression="zstd")
    weighted_pl.write_csv(weighted_summary_csv)
    weighted_pl.write_parquet(weighted_summary_parquet, compression="zstd")

    index_html = None
    if assessment_config.html_output == "all":
        index_html = generate_index_page(
            output_dir=output_dir,
            run_name=run_name,
            summary_pdf=summary_pdf,
            config=assessment_config,
            label_cols=selected_label_cols,
            report_failures=report_failures,
            effective_sort_metric=effective_sort_metric,
        )

    artifacts_meta = {
        "run_name": run_name,
        "preprocess_version": PREPROCESS_VERSION,
        "scope": assessment_config.scope,
        "sort_metric": effective_sort_metric,
        "requested_sort_metric": assessment_config.sort_metric,
        "weighted_rankic_enabled": False,
        "feature_count": len(feature_cols),
        "label_columns": selected_label_cols,
        "sample_count": int(frame.height),
        "trading_day_count": total_days,
        "eligible_feature_count": int(summary_pdf["eligible"].sum()),
        "sparse_feature_count": int((~summary_pdf["eligible"]).sum()),
        "summary_csv": str(summary_csv),
        "summary_parquet": str(summary_parquet),
        "weighted_summary_csv": str(weighted_summary_csv),
        "weighted_summary_parquet": str(weighted_summary_parquet),
        "index_html": str(index_html) if index_html else None,
        "detail_report_dir": str(report_dir) if assessment_config.html_output == "all" else None,
        "report_failure_count": len(report_failures),
        "report_failures": report_failures,
        "config": asdict(assessment_config),
    }
    meta_path.write_text(json.dumps(artifacts_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return AssessmentArtifacts(
        summary_df=summary_pl,
        summary_csv=summary_csv,
        summary_parquet=summary_parquet,
        weighted_summary_csv=weighted_summary_csv,
        weighted_summary_parquet=weighted_summary_parquet,
        index_html=index_html,
        detail_report_dir=report_dir if assessment_config.html_output == "all" else None,
        meta_path=meta_path,
        artifacts_meta=artifacts_meta,
    )
