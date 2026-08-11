from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

REPO = Path("/home/hyd/research/vnpy_hub")
sys.path.insert(0, str(REPO / "vnpy"))

from vnpy.alpha import AlphaLab
from vnpy.alpha.analysis.alphainspect.ic import calc_ic
from vnpy.alpha.analysis.alphainspect_backend import prepare_alphainspect_input
from vnpy.trader.constant import Interval


SIGNAL_PATH = REPO / (
    "playground/alpha_research/trade_real/reports/"
    "old167_label_indcap_ols_equal_cv5_train_filters_20260713/"
    "notebook_formal_backtest_tail_filter/signals/"
    "old167_indcap_ols_equal_cv5_train_cap30/tail_only.parquet"
)
LAB_PATH = REPO / "playground/alpha_research/trade_real/lab/all_stocks"
COMPONENT_DIR = LAB_PATH / "component"
OUTPUT_DIR = SIGNAL_PATH.parents[2] / "index_universe_ic_cap30_tail_only"
HORIZONS = (1, 3, 5, 10, 20)
RETURNS = tuple(f"FWD_RET_OO_{h}" for h in HORIZONS)
INDEX_FILES = {
    "300": "indexweight_000300.SSE.parquet",
    "500": "indexweight_000905.SSE.parquet",
    "1000": "indexweight_000852.SSE.parquet",
    "2000": "indexweight_932000.SSE.parquet",
}
UNIVERSE_INDICES = {
    "800": ("300", "500"),
    "1800": ("300", "500", "1000"),
    "3800": ("300", "500", "1000", "2000"),
}


def load_component(name: str, start: pd.Timestamp, end: pd.Timestamp) -> pl.DataFrame:
    return (
        pl.scan_parquet(COMPONENT_DIR / INDEX_FILES[name])
        .select(
            pl.col("date")
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            .cast(pl.Datetime)
            .alias("date"),
            pl.col("vt_symbol").alias("asset"),
        )
        .filter(
            (pl.col("date") >= start.to_pydatetime())
            & (pl.col("date") <= end.to_pydatetime())
        )
        .unique(["date", "asset"])
        .collect()
    )


def finite_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def summarize(
    label: str,
    universe: str,
    daily_ic: pl.DataFrame,
    daily_counts: pl.DataFrame,
) -> list[dict[str, object]]:
    ic_pd = daily_ic.to_pandas().set_index("date")
    counts_pd = daily_counts.to_pandas().set_index("date")
    rows: list[dict[str, object]] = []
    for forward_return in RETURNS:
        column = f"signal__{forward_return}"
        values = ic_pd[column].dropna().astype(float)
        sample_counts = counts_pd.loc[values.index, f"{forward_return}_n"].astype(float)
        t_stat, p_value = stats.ttest_1samp(values, 0, nan_policy="omit")
        mean = finite_or_nan(values.mean())
        std = finite_or_nan(values.std(ddof=1))
        rows.append(
            {
                "period": label,
                "universe": universe,
                "forward_return": forward_return,
                "start": values.index.min(),
                "end": values.index.max(),
                "ic_days": int(values.count()),
                "rank_ic_mean": mean,
                "rank_ic_std": std,
                "rank_ic_ir": mean / std if std > 0 else float("nan"),
                "win_rate": finite_or_nan((values > 0).mean()),
                "t_stat": finite_or_nan(t_stat),
                "p_value": finite_or_nan(p_value),
                "avg_valid_n": finite_or_nan(sample_counts.mean()),
                "median_valid_n": finite_or_nan(sample_counts.median()),
                "min_valid_n": finite_or_nan(sample_counts.min()),
                "max_valid_n": finite_or_nan(sample_counts.max()),
            }
        )
    return rows


def summarize_annual(
    universe: str,
    daily_ic: pl.DataFrame,
    daily_counts: pl.DataFrame,
) -> list[dict[str, object]]:
    column = "signal__FWD_RET_OO_1"
    frame = daily_ic.select("date", column).join(
        daily_counts.select("date", "FWD_RET_OO_1_n"),
        on="date",
        how="left",
    ).to_pandas()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    rows: list[dict[str, object]] = []
    for year, group in frame.groupby("year", sort=True):
        values = group[column].dropna().astype(float)
        if values.empty:
            continue
        t_stat, p_value = stats.ttest_1samp(values, 0, nan_policy="omit")
        mean = finite_or_nan(values.mean())
        std = finite_or_nan(values.std(ddof=1))
        rows.append(
            {
                "universe": universe,
                "year": int(year),
                "start": group["date"].min(),
                "end": group["date"].max(),
                "ic_days": int(values.count()),
                "rank_ic_mean": mean,
                "rank_ic_std": std,
                "rank_ic_ir": mean / std if std > 0 else float("nan"),
                "win_rate": finite_or_nan((values > 0).mean()),
                "t_stat": finite_or_nan(t_stat),
                "p_value": finite_or_nan(p_value),
                "avg_valid_n": finite_or_nan(group["FWD_RET_OO_1_n"].mean()),
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lab = AlphaLab(str(LAB_PATH))
    prepared = prepare_alphainspect_input(
        lab,
        SIGNAL_PATH,
        factor="signal",
        horizons=HORIZONS,
        interval=Interval.DAILY,
        extended_days=400,
    )
    analysis = prepared.analysis_frame
    start = pd.Timestamp(prepared.start)
    end = pd.Timestamp(prepared.end)
    components = {
        name: load_component(name, start, end)
        for name in INDEX_FILES
    }
    component_starts = {
        name: frame["date"].min()
        for name, frame in components.items()
    }
    common_start = max(component_starts.values())

    native_summary_rows: list[dict[str, object]] = []
    common_summary_rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    daily_ic_frames: list[pl.DataFrame] = []
    daily_count_frames: list[pl.DataFrame] = []

    for universe, index_names in UNIVERSE_INDICES.items():
        universe_start = max(component_starts[name] for name in index_names)
        membership = (
            pl.concat([components[name] for name in index_names])
            .unique(["date", "asset"])
            .filter(pl.col("date") >= universe_start)
            .sort(["date", "asset"])
        )
        joined = analysis.join(membership, on=["date", "asset"], how="inner")
        daily_counts = (
            joined.group_by("date")
            .agg(
                pl.len().alias("signal_n"),
                *[
                    pl.col(column).is_not_null().sum().alias(f"{column}_n")
                    for column in RETURNS
                ],
            )
            .sort("date")
        )
        daily_ic = calc_ic(
            joined,
            factors=["signal"],
            forward_returns=RETURNS,
            method="rank_ic",
        )
        daily_ic_frames.append(daily_ic.with_columns(pl.lit(universe).alias("universe")))
        daily_count_frames.append(
            daily_counts.with_columns(pl.lit(universe).alias("universe"))
        )
        native_summary_rows.extend(
            summarize("native_available", universe, daily_ic, daily_counts)
        )
        annual_rows.extend(summarize_annual(universe, daily_ic, daily_counts))

        common_daily_ic = daily_ic.filter(pl.col("date") >= common_start)
        common_daily_counts = daily_counts.filter(pl.col("date") >= common_start)
        common_summary_rows.extend(
            summarize("common_since_2000_start", universe, common_daily_ic, common_daily_counts)
        )
        membership_counts = membership.group_by("date").len().rename({"len": "members"})
        joined_counts = daily_counts.select("date", "signal_n", "FWD_RET_OO_1_n")
        cov = membership_counts.join(joined_counts, on="date", how="left")
        coverage_rows.append(
            {
                "universe": universe,
                "membership_start": membership["date"].min(),
                "membership_end": membership["date"].max(),
                "dates": membership["date"].n_unique(),
                "avg_members": cov["members"].mean(),
                "avg_signal_n": cov["signal_n"].mean(),
                "avg_fwd_oo1_n": cov["FWD_RET_OO_1_n"].mean(),
            }
        )

    native_summary = pl.from_dicts(native_summary_rows)
    common_summary = pl.from_dicts(common_summary_rows)
    annual = pl.from_dicts(annual_rows)
    coverage = pl.from_dicts(coverage_rows)
    daily_ic_all = pl.concat(daily_ic_frames, how="diagonal_relaxed")
    daily_counts_all = pl.concat(daily_count_frames, how="diagonal_relaxed")

    native_summary.write_csv(OUTPUT_DIR / "rank_ic_summary_native_period.csv")
    common_summary.write_csv(OUTPUT_DIR / "rank_ic_summary_common_period.csv")
    annual.write_csv(OUTPUT_DIR / "rank_ic_annual_fwd_oo_1.csv")
    coverage.write_csv(OUTPUT_DIR / "universe_coverage.csv")
    daily_ic_all.write_parquet(OUTPUT_DIR / "rank_ic_daily.parquet")
    daily_counts_all.write_parquet(OUTPUT_DIR / "daily_valid_counts.parquet")

    print("SIGNAL", prepared.signal_summary)
    print("COVERAGE", prepared.coverage)
    print("COMPONENT_STARTS", component_starts)
    print("COMMON_START", common_start)
    print("UNIVERSE_COVERAGE")
    print(coverage)
    print("NATIVE_OO1")
    print(native_summary.filter(pl.col("forward_return") == "FWD_RET_OO_1"))
    print("COMMON_OO1")
    print(common_summary.filter(pl.col("forward_return") == "FWD_RET_OO_1"))
    print("ANNUAL_OO1")
    print(annual)
    print("ALL_HORIZONS_NATIVE")
    print(native_summary)
    print("ALL_HORIZONS_COMMON")
    print(common_summary)
    print("OUTPUT_DIR", OUTPUT_DIR)


if __name__ == "__main__":
    main()
