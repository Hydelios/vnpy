from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..catalog import CalendarFactorCatalog
from .backend import CALENDAR_2025_SPECIAL_IDS, CalendarCppBackend, SPECIAL_KEY_TITLES
from .library import classify_cpp_coverage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Titan Calendar compatibility CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    coverage = sub.add_parser("coverage")
    coverage.add_argument("--output", type=Path)
    library = sub.add_parser("library")
    library.add_argument("--output", type=Path, required=True)
    compute = sub.add_parser("compute")
    compute.add_argument("--lab-path", type=Path, required=True)
    compute.add_argument("--start", required=True)
    compute.add_argument("--end", required=True)
    compute.add_argument("--symbols", required=True, help="逗号分隔 vt_symbol")
    compute.add_argument("--mode", choices=("daily", "special", "all"), default="all")
    compute.add_argument("--industry-source", default="sws")
    compute.add_argument("--num-threads", type=int, default=0)
    compute.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    backend = CalendarCppBackend(num_threads=getattr(args, "num_threads", 0))
    if args.command == "coverage":
        coverage = classify_cpp_coverage()
        catalog = CalendarFactorCatalog()
        special_2026_names = {
            backend.titan_name(item.name) for item in catalog.build_year(2026)
            if item.title in set(SPECIAL_KEY_TITLES.values())
        }
        special_2025_names = {
            backend.titan_name(item.name) for item in catalog.build_year(2025)
            if item.calendar_id in CALENDAR_2025_SPECIAL_IDS
        }
        special_names = special_2025_names | special_2026_names
        unsupported = {
            backend.titan_name(name): reason for name, reason in coverage.unsupported.items()
            if backend.titan_name(name) not in special_names
        }
        expressions = backend.available_daily_expressions()
        payload = {
            "titan_expression_rows": len(expressions),
            "titan_custom_rows": len(special_names),
            "total_titan_rows": len(expressions) + len(special_names),
            "unsupported_rows": len(unsupported),
            "expressions": expressions,
            "unsupported": unsupported,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text)
        return
    if args.command == "library":
        print(backend.export_library(args.output))
        return

    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    catalog = CalendarFactorCatalog()
    expression_names = set(backend.available_internal_daily_expressions())
    daily_definitions = [item for item in catalog.build_all() if item.name in expression_names]
    special_2025_definitions = [
        item for item in catalog.build_year(2025)
        if item.calendar_id in CALENDAR_2025_SPECIAL_IDS
    ]
    special_2026_definitions = [
        item for item in catalog.build_year(2026) if item.title in set(SPECIAL_KEY_TITLES.values())
    ]
    special_definitions = [*special_2025_definitions, *special_2026_definitions]
    definitions = daily_definitions if args.mode == "daily" else special_definitions
    if args.mode == "all":
        definitions = [*daily_definitions, *special_definitions]
    bundle = backend.prepare_pit_bundle(
        args.lab_path,
        symbols,
        start=args.start,
        end=args.end,
        definitions=definitions,
        with_minute=args.mode in {"special", "all"},
        industry_source=args.industry_source,
    )
    result = bundle.daily.select("datetime", "vt_symbol")
    if args.mode in {"daily", "all"}:
        result = result.join(
            backend.compute_daily(bundle.daily, [item.name for item in daily_definitions]),
            on=["datetime", "vt_symbol"], how="left",
        )
    if args.mode in {"special", "all"}:
        result = result.join(
            backend.compute_2025_special(bundle), on=["datetime", "vt_symbol"], how="left",
        ).join(
            backend.compute_special(bundle), on=["datetime", "vt_symbol"], how="left",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.output)
    print(json.dumps({"output": str(args.output), "shape": result.shape}, ensure_ascii=False))


if __name__ == "__main__":
    main()
