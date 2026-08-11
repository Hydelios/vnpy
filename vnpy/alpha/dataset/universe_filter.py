"""Daily cross-sectional universe filters shared by training and signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


KEY_COLUMNS = ["datetime", "vt_symbol"]


@dataclass(frozen=True)
class UniverseFilterProfile:
    """Tail-filter configuration."""

    name: str
    cap_ratio: float
    volume_ratio: float | None = None


@dataclass
class UniverseFilterResult:
    """Filtered rows and reconciliation artifacts."""

    filtered: pl.DataFrame
    daily_audit: pl.DataFrame
    summary: dict[str, Any]


FILTER_PROFILES: dict[str, UniverseFilterProfile] = {
    "cap10_volume10": UniverseFilterProfile(
        name="cap10_volume10",
        cap_ratio=0.10,
        volume_ratio=0.10,
    ),
    "cap30": UniverseFilterProfile(
        name="cap30",
        cap_ratio=0.30,
        volume_ratio=None,
    ),
}


def resolve_filter_profile(
    profile: str | UniverseFilterProfile,
) -> UniverseFilterProfile:
    """Resolve and validate a named or explicit filter profile."""
    resolved = FILTER_PROFILES.get(profile) if isinstance(profile, str) else profile
    if resolved is None:
        choices = ", ".join(sorted(FILTER_PROFILES))
        raise ValueError(f"unknown universe filter profile {profile!r}; choices: {choices}")
    if not 0 <= resolved.cap_ratio < 1:
        raise ValueError("cap_ratio must be in [0, 1)")
    if resolved.volume_ratio is not None and not 0 <= resolved.volume_ratio < 1:
        raise ValueError("volume_ratio must be in [0, 1)")
    return resolved


def _require_columns(frame: pl.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _normalize_keys(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    _require_columns(frame, set(KEY_COLUMNS), name)
    normalized = frame.with_columns(
        pl.col("datetime").cast(pl.Datetime),
        pl.col("vt_symbol").cast(pl.Utf8),
    ).drop_nulls(KEY_COLUMNS)
    if normalized.select(pl.struct(KEY_COLUMNS).is_duplicated().any()).item():
        raise ValueError(f"{name} contains duplicate datetime/vt_symbol keys")
    return normalized


def _ordinal_rank_panel(
    frame: pl.DataFrame,
    *,
    metric: str,
    ratio: float,
    prefix: str,
) -> pl.DataFrame:
    """Rank ascending by metric, breaking ties deterministically by symbol."""
    return (
        frame.filter(pl.col(metric).is_finite() & (pl.col(metric) > 0))
        .select([*KEY_COLUMNS, metric])
        .sort(["datetime", metric, "vt_symbol"])
        .with_columns(
            (pl.col("vt_symbol").cum_count().over("datetime") - 1).alias(f"_{prefix}_rank"),
            pl.len().over("datetime").alias(f"_{prefix}_valid_count"),
        )
        .with_columns(
            (pl.col(f"_{prefix}_valid_count") * ratio)
            .floor()
            .cast(pl.UInt32)
            .alias(f"_{prefix}_cutoff")
        )
        .select(
            [
                *KEY_COLUMNS,
                f"_{prefix}_rank",
                f"_{prefix}_valid_count",
                f"_{prefix}_cutoff",
            ]
        )
    )


def filter_universe(
    frame: pl.DataFrame,
    *,
    cap_panel: pl.DataFrame,
    volume_panel: pl.DataFrame | None = None,
    status_panel: pl.DataFrame | None = None,
    profile: str | UniverseFilterProfile,
    filter_status: bool,
    stage: str,
) -> UniverseFilterResult:
    """Physically delete invalid rows and daily cap/volume tails.

    Status deletion is applied before tail ranks. Cap and volume ranks are
    calculated independently, and their tails are deleted as a union.
    """
    spec = resolve_filter_profile(profile)
    source = _normalize_keys(frame, "frame")
    if source.is_empty():
        raise ValueError("frame is empty")

    cap = _normalize_keys(cap_panel, "cap_panel")
    _require_columns(cap, {"cap"}, "cap_panel")
    cap = cap.select(*KEY_COLUMNS, pl.col("cap").cast(pl.Float64))

    if spec.volume_ratio is not None:
        if volume_panel is None:
            raise ValueError(f"profile {spec.name!r} requires volume_panel")
        volume = _normalize_keys(volume_panel, "volume_panel")
        _require_columns(volume, {"volume"}, "volume_panel")
        volume = volume.select(*KEY_COLUMNS, pl.col("volume").cast(pl.Float64))
    else:
        volume = None

    attached_names = {
        "cap",
        "volume",
        "is_st",
        "is_suspended",
        "has_bar",
        "is_one_word_limit_up",
        "is_one_word_limit_down",
    }
    output_columns = [column for column in source.columns if column not in attached_names]
    joined = source.drop(attached_names, strict=False).join(
        cap,
        on=KEY_COLUMNS,
        how="left",
        validate="1:1",
    )
    if volume is not None:
        joined = joined.join(volume, on=KEY_COLUMNS, how="left", validate="1:1")
    else:
        joined = joined.with_columns(pl.lit(None, dtype=pl.Float64).alias("volume"))

    if filter_status:
        if status_panel is None:
            raise ValueError("filter_status=True requires status_panel")
        status = _normalize_keys(status_panel, "status_panel")
        status_columns = {
            "is_st",
            "is_suspended",
            "has_bar",
            "is_one_word_limit_up",
            "is_one_word_limit_down",
        }
        _require_columns(status, status_columns, "status_panel")
        joined = joined.join(
            status.select(*KEY_COLUMNS, *sorted(status_columns)),
            on=KEY_COLUMNS,
            how="left",
            validate="1:1",
        ).with_columns(
            (
                pl.col("is_st").is_null()
                | pl.col("is_suspended").is_null()
                | pl.col("has_bar").is_null()
                | ~pl.col("has_bar").fill_null(False)
            ).alias("_missing_or_no_bar"),
            pl.col("is_st").fill_null(False).alias("_is_st"),
            pl.col("is_suspended").fill_null(False).alias("_is_suspended"),
            pl.col("is_one_word_limit_up").fill_null(False).alias("_one_up"),
            pl.col("is_one_word_limit_down").fill_null(False).alias("_one_down"),
        )
    else:
        joined = joined.with_columns(
            pl.lit(False).alias("_missing_or_no_bar"),
            pl.lit(False).alias("_is_st"),
            pl.lit(False).alias("_is_suspended"),
            pl.lit(False).alias("_one_up"),
            pl.lit(False).alias("_one_down"),
        )

    joined = joined.with_columns(
        (
            pl.col("_missing_or_no_bar")
            | pl.col("_is_st")
            | pl.col("_is_suspended")
            | pl.col("_one_up")
            | pl.col("_one_down")
        ).alias("_drop_status")
    )
    status_eligible = joined.filter(~pl.col("_drop_status"))

    cap_rank = _ordinal_rank_panel(
        status_eligible,
        metric="cap",
        ratio=spec.cap_ratio,
        prefix="cap",
    )
    ranked = joined.join(cap_rank, on=KEY_COLUMNS, how="left", validate="1:1").with_columns(
        (~pl.col("_drop_status") & pl.col("_cap_rank").is_null()).alias("_missing_cap"),
        (
            ~pl.col("_drop_status")
            & pl.col("_cap_rank").is_not_null()
            & (pl.col("_cap_rank") < pl.col("_cap_cutoff"))
        ).alias("_drop_cap_tail"),
    )

    if spec.volume_ratio is not None:
        volume_rank = _ordinal_rank_panel(
            status_eligible,
            metric="volume",
            ratio=spec.volume_ratio,
            prefix="volume",
        )
        ranked = ranked.join(
            volume_rank,
            on=KEY_COLUMNS,
            how="left",
            validate="1:1",
        ).with_columns(
            (~pl.col("_drop_status") & pl.col("_volume_rank").is_null()).alias("_missing_volume"),
            (
                ~pl.col("_drop_status")
                & pl.col("_volume_rank").is_not_null()
                & (pl.col("_volume_rank") < pl.col("_volume_cutoff"))
            ).alias("_drop_volume_tail"),
        )
    else:
        ranked = ranked.with_columns(
            pl.lit(False).alias("_missing_volume"),
            pl.lit(False).alias("_drop_volume_tail"),
        )

    ranked = ranked.with_columns(
        (
            pl.col("_drop_status")
            | pl.col("_missing_cap")
            | pl.col("_missing_volume")
            | pl.col("_drop_cap_tail")
            | pl.col("_drop_volume_tail")
        ).alias("_drop_any")
    )
    daily = (
        ranked.group_by("datetime")
        .agg(
            pl.len().alias("candidate_count"),
            pl.col("_missing_or_no_bar").sum().alias("missing_or_no_bar_count"),
            pl.col("_is_st").sum().alias("st_count"),
            pl.col("_is_suspended").sum().alias("suspended_count"),
            pl.col("_one_up").sum().alias("one_word_limit_up_count"),
            pl.col("_one_down").sum().alias("one_word_limit_down_count"),
            pl.col("_drop_status").sum().alias("status_union_count"),
            pl.col("_missing_cap").sum().alias("missing_cap_count"),
            pl.col("_missing_volume").sum().alias("missing_volume_count"),
            pl.col("_drop_cap_tail").sum().alias("cap_tail_count"),
            pl.col("_drop_volume_tail").sum().alias("volume_tail_count"),
            (pl.col("_drop_cap_tail") & pl.col("_drop_volume_tail"))
            .sum()
            .alias("tail_overlap_count"),
            pl.col("_drop_any").sum().alias("removed_total_count"),
            (~pl.col("_drop_any")).sum().alias("kept_count"),
        )
        .sort("datetime")
    )

    filtered = (
        ranked.filter(~pl.col("_drop_any"))
        .select(output_columns)
        .sort(KEY_COLUMNS)
    )
    summary = {
        "profile": spec.name,
        "filter_stage": stage,
        "status_filter_enabled": filter_status,
        "status_filter": (
            "T-day ST, suspension, missing/no bar, one-word limit up/down physically deleted"
            if filter_status
            else "disabled"
        ),
        "one_word_definition": "round(high,2)==round(low,2)==round(limit_up/down,2)",
        "cap_ratio": spec.cap_ratio,
        "volume_ratio": spec.volume_ratio,
        "tail_combination": "union" if spec.volume_ratio is not None else "cap only",
        "tail_ranking_population": (
            "after T-day status/one-word deletion" if filter_status else "all input signal candidates"
        ),
        "rows_before": source.height,
        "rows_after": filtered.height,
        "missing_or_no_bar_rows": int(ranked["_missing_or_no_bar"].sum()),
        "st_rows": int(ranked["_is_st"].sum()),
        "suspended_rows": int(ranked["_is_suspended"].sum()),
        "one_word_limit_up_rows": int(ranked["_one_up"].sum()),
        "one_word_limit_down_rows": int(ranked["_one_down"].sum()),
        "status_union_rows": int(ranked["_drop_status"].sum()),
        "missing_cap_rows": int(ranked["_missing_cap"].sum()),
        "missing_volume_rows": int(ranked["_missing_volume"].sum()),
        "cap_tail_rows": int(ranked["_drop_cap_tail"].sum()),
        "volume_tail_rows": int(ranked["_drop_volume_tail"].sum()),
        "tail_overlap_rows": int((ranked["_drop_cap_tail"] & ranked["_drop_volume_tail"]).sum()),
        "removed_total_rows": int(ranked["_drop_any"].sum()),
        "dates": daily.height,
        "avg_candidate_count": float(daily["candidate_count"].mean()),
        "avg_kept_count": float(daily["kept_count"].mean()),
        "kept_ratio": filtered.height / source.height,
    }
    if summary["removed_total_rows"] != source.height - filtered.height:
        raise RuntimeError("universe-filter reconciliation failed")
    if spec.volume_ratio is None and (
        summary["missing_volume_rows"] or summary["volume_tail_rows"]
    ):
        raise RuntimeError("cap-only profile unexpectedly applied a volume filter")
    return UniverseFilterResult(filtered=filtered, daily_audit=daily, summary=summary)
