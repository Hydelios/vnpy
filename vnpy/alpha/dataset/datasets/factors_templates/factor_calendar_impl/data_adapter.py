from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from typing import Iterable

import polars as pl

from .engine import CalendarDataBundle


BAR_FIELDS = {"open", "high", "low", "close", "volume", "turnover", "open_interest", "vwap"}
COMMON_FILES = (
    "factors_valuation.csv",
    "factors_operation.csv",
    "factors_financial.csv",
    "factors_growth.csv",
    "factors_cashflow.csv",
    "industry.csv",
)


class CalendarLabDataAdapter:
    """Read existing AlphaLab files without changing AlphaLab or creating folders."""

    def __init__(self, lab_path: str | Path, *, industry_source: str = "sws") -> None:
        self.lab_path = Path(lab_path)
        if not self.lab_path.exists():
            raise FileNotFoundError(self.lab_path)
        self._common_columns: dict[str, Path] | None = None
        self.industry_source = industry_source

    def load_bars(
        self,
        symbols: Iterable[str],
        *,
        frequency: str,
        start: str | datetime,
        end: str | datetime,
    ) -> pl.DataFrame:
        folder_map = {
            "1d": self.lab_path / "daily",
            "1m": self.lab_path / "minute" / "1m",
            "10m": self.lab_path / "minute" / "10m",
            "60m": self.lab_path / "minute" / "60m",
        }
        if frequency not in folder_map:
            raise ValueError(f"unsupported bar frequency: {frequency}")
        start_value = _as_datetime(start, end_of_day=False)
        end_value = _as_datetime(end, end_of_day=True)
        frames = []
        for symbol in symbols:
            path = folder_map[frequency] / f"{symbol}.parquet"
            if not path.exists():
                continue
            frame = (
                pl.scan_parquet(path)
                .filter(pl.col("datetime").is_between(start_value, end_value, closed="both"))
                .with_columns(pl.lit(symbol).alias("vt_symbol"))
                .collect()
            )
            if not frame.is_empty():
                frames.append(frame)
        if not frames:
            return pl.DataFrame()
        out = pl.concat(frames, how="diagonal_relaxed").sort(["vt_symbol", "datetime"])
        if {"turnover", "volume"}.issubset(out.columns):
            out = out.with_columns(
                pl.when(pl.col("volume") > 0)
                .then(pl.col("turnover") / pl.col("volume"))
                .otherwise(None)
                .alias("vwap")
            )
        return out

    def common_column_map(self) -> dict[str, Path]:
        if self._common_columns is not None:
            return dict(self._common_columns)
        mapping: dict[str, Path] = {}
        common = self.lab_path / "common"
        for filename in COMMON_FILES:
            path = common / filename
            if not path.exists():
                continue
            header = pl.read_csv(path, n_rows=0)
            columns = list(header.columns)
            if columns and columns[0].startswith("\ufeff"):
                columns[0] = columns[0].lstrip("\ufeff")
            for column in columns:
                if column not in {"trade_date", "vt_symbol"}:
                    mapping.setdefault(column, path)
        self._common_columns = mapping
        return dict(mapping)

    def load_common_fields(
        self,
        symbols: Iterable[str],
        fields: Iterable[str],
        *,
        start: str | datetime,
        end: str | datetime,
    ) -> pl.DataFrame | None:
        symbols = list(symbols)
        mapping = self.common_column_map()
        by_file: dict[Path, list[str]] = {}
        for field in fields:
            path = mapping.get(field)
            if path is not None:
                by_file.setdefault(path, []).append(field)
        merged: pl.DataFrame | None = None
        for path, columns in by_file.items():
            header = pl.read_csv(path, n_rows=0)
            first = header.columns[0]
            rename = {first: first.lstrip("\ufeff")} if first.startswith("\ufeff") else {}
            scan = pl.scan_csv(path, infer_schema_length=1000)
            if rename:
                scan = scan.rename(rename)
            selected = ["trade_date", "vt_symbol", *columns]
            is_industry = path.name == "industry.csv" and "source" in header.columns
            if is_industry and "source" not in selected:
                selected.append("source")
            frame_scan = scan.select(*selected)
            if is_industry:
                frame_scan = frame_scan.filter(pl.col("source") == self.industry_source)
            frame = (
                frame_scan
                .filter(pl.col("vt_symbol").is_in(symbols))
                .with_columns(
                    pl.col("trade_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).cast(pl.Datetime).alias("datetime")
                )
                .drop("trade_date")
                .filter(
                    pl.col("datetime").is_between(
                        _as_datetime(start, end_of_day=False),
                        _as_datetime(end, end_of_day=True),
                        closed="both",
                    )
                )
                .collect()
            )
            if is_industry and "source" not in columns:
                frame = frame.drop("source")
            merged = frame if merged is None else merged.join(frame, on=["datetime", "vt_symbol"], how="outer_coalesce")
        return merged

    def load_market_return(
        self,
        *,
        start: str | datetime,
        end: str | datetime,
        benchmark: str = "000300.SSE",
    ) -> pl.DataFrame:
        bars = self.load_bars([benchmark], frequency="1d", start=start, end=end)
        if bars.is_empty():
            return pl.DataFrame(schema={"datetime": pl.Datetime, "market_ret": pl.Float64})
        return bars.sort("datetime").select(
            "datetime",
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("market_ret"),
        )

    def load_bundle(
        self,
        symbols: Iterable[str],
        *,
        start: str | datetime,
        end: str | datetime,
        required_fields: Iterable[str] = (),
        with_minute: bool = True,
        benchmark: str = "000300.SSE",
    ) -> CalendarDataBundle:
        symbols = list(symbols)
        daily = self.load_bars(symbols, frequency="1d", start=start, end=end)
        minute = self.load_bars(symbols, frequency="1m", start=start, end=end) if with_minute else None
        if minute is not None and not minute.is_empty():
            benchmark_minute = self.load_bars([benchmark], frequency="1m", start=start, end=end)
            if not benchmark_minute.is_empty():
                benchmark_minute = benchmark_minute.sort("datetime").select(
                    "datetime",
                    (pl.col("close") / pl.col("close").shift(1) - 1).alias("market_minute_ret"),
                )
                minute = minute.join(benchmark_minute, on="datetime", how="left")
        fields = set(required_fields).difference(BAR_FIELDS).difference({"market_ret"})
        common = self.load_common_fields(symbols, fields, start=start, end=end) if fields else None
        if common is not None:
            daily = daily.join(common, on=["datetime", "vt_symbol"], how="left")
        market = self.load_market_return(start=start, end=end, benchmark=benchmark)
        if not market.is_empty():
            daily = daily.join(market, on="datetime", how="left")
        return CalendarDataBundle(daily=daily, minute=minute)


def _as_datetime(value: str | datetime, *, end_of_day: bool) -> datetime:
    if isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(value)
    if end_of_day and "T" not in value and " " not in value:
        return datetime.combine(parsed.date(), time.max)
    return parsed
