"""Compatibility import for Titan's Calendar factor registry."""

from __future__ import annotations

import sys
from pathlib import Path


def _add_titan_python_path() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "playground" / "alpha_research" / "titan_jinja" / "python"
        if candidate.is_dir():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
            return
    raise RuntimeError("无法定位 titan_jinja/python")


_add_titan_python_path()

from factors.calendar_family import (  # noqa: E402
    CALENDAR_FAMILY,
    CppCoverage,
    build_calendar_daily_library,
    calendar_name_maps,
    classify_cpp_coverage,
    to_titan_calendar_name,
)

__all__ = [
    "CALENDAR_FAMILY",
    "CppCoverage",
    "build_calendar_daily_library",
    "calendar_name_maps",
    "classify_cpp_coverage",
    "to_titan_calendar_name",
]
