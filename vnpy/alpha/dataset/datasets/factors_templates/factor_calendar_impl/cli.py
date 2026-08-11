from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .audit import audit_catalog, audit_lab, write_audit_report
from .catalog import CalendarFactorCatalog, YEARS
from .data_adapter import CalendarLabDataAdapter
from .engine import CalendarFactorEngine
from .metadata import ImplementationStatus


DEFAULT_LAB = Path("playground/alpha_research/lab/all_stocks")
DEFAULT_MANIFESTS = Path(__file__).resolve().parent / "manifests"
SAMPLE_SYMBOLS = [
    "000001.SZSE", "000002.SZSE", "000063.SZSE", "000333.SZSE", "000651.SZSE",
    "000725.SZSE", "000858.SZSE", "000895.SZSE", "000977.SZSE", "001979.SZSE",
    "002027.SZSE", "002230.SZSE", "002415.SZSE", "002594.SZSE", "002714.SZSE",
    "300014.SZSE", "300059.SZSE", "300122.SZSE", "300274.SZSE", "300750.SZSE",
    "600000.SSE", "600036.SSE", "600276.SSE", "600519.SSE", "600887.SSE",
    "601012.SSE", "601318.SSE", "601398.SSE", "601857.SSE", "603259.SSE",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent factor-calendar tooling")
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifests")
    manifest.add_argument("--output-dir", type=Path, default=DEFAULT_MANIFESTS)
    audit = sub.add_parser("audit")
    audit.add_argument("--lab-path", type=Path, default=DEFAULT_LAB)
    audit.add_argument("--output", type=Path, default=DEFAULT_MANIFESTS / "audit.json")
    sample = sub.add_parser("sample")
    sample.add_argument("--lab-path", type=Path, default=DEFAULT_LAB)
    sample.add_argument("--year", type=int, choices=YEARS, default=2026)
    sample.add_argument("--start", default="2024-01-02")
    sample.add_argument("--end", default="2024-06-28")
    sample.add_argument("--max-factors", type=int, default=30)
    sample.add_argument("--output", type=Path, default=Path("/tmp/factor_calendar_sample.parquet"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    catalog = CalendarFactorCatalog()
    if args.command == "manifests":
        paths = catalog.export_manifests(args.output_dir)
        print(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2))
        return
    if args.command == "audit":
        payload = {"catalog": audit_catalog(catalog), "data": audit_lab(args.lab_path)}
        print(write_audit_report(args.output, payload))
        return
    definitions = [
        item for item in catalog.build_year(args.year)
        if item.implementation_status in {ImplementationStatus.EXACT, ImplementationStatus.PROXY}
    ][: args.max_factors]
    fields = {field for item in definitions for field in item.required_fields}
    adapter = CalendarLabDataAdapter(args.lab_path)
    bundle = adapter.load_bundle(
        SAMPLE_SYMBOLS, start=args.start, end=args.end, required_fields=fields,
        with_minute=any(item.input_frequency == "1m" for item in definitions),
    )
    result = CalendarFactorEngine().compute(definitions, bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.output)
    print(json.dumps({"output": str(args.output), "shape": result.shape, "factors": len(definitions)}, indent=2))


if __name__ == "__main__":
    main()
