from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import polars as pl

from .catalog import CalendarFactorCatalog, YEARS


def audit_lab(lab_path: str | Path, *, sample_symbol: str = "000001.SZSE") -> dict:
    root = Path(lab_path)
    frequencies = {
        "1d": root / "daily",
        "1m": root / "minute" / "1m",
        "10m": root / "minute" / "10m",
        "30m": root / "minute" / "30m",
        "60m": root / "minute" / "60m",
    }
    bars = {}
    for frequency, folder in frequencies.items():
        files = list(folder.glob("*.parquet")) if folder.exists() else []
        sample = folder / f"{sample_symbol}.parquet"
        detail = {"file_count": len(files), "sample_symbol": sample_symbol}
        if sample.exists():
            stats = pl.scan_parquet(sample).select(
                pl.len().alias("rows"),
                pl.col("datetime").min().alias("start"),
                pl.col("datetime").max().alias("end"),
            ).collect().row(0, named=True)
            detail.update({key: str(value) for key, value in stats.items()})
            detail["columns"] = list(pl.scan_parquet(sample).collect_schema().names())
        bars[frequency] = detail

    common = {}
    common_dir = root / "common"
    for path in sorted(common_dir.glob("*.csv")):
        header = pl.read_csv(path, n_rows=0)
        common[path.name] = {
            "columns": len(header.columns),
            "size_bytes": path.stat().st_size,
            "field_names": [name.lstrip("\ufeff") for name in header.columns],
        }
    return {"lab_path": str(root.resolve()), "bars": bars, "common": common}


def audit_catalog(catalog: CalendarFactorCatalog | None = None) -> dict:
    catalog = catalog or CalendarFactorCatalog()
    years = {}
    for year in YEARS:
        definitions = catalog.build_year(year)
        years[str(year)] = {
            "definitions": len(definitions),
            "status": dict(Counter(item.implementation_status.value for item in definitions)),
            "unrecovered_titles": sum(item.title.startswith("未恢复标题_") for item in definitions),
            "categories": dict(Counter(item.normalized_category for item in definitions)),
        }
    return {"years": years, "total_definitions": sum(item["definitions"] for item in years.values())}


def write_audit_report(output: str | Path, payload: dict) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

