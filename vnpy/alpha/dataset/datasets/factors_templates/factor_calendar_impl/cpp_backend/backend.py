"""Compatibility import for the Titan-owned Calendar execution service.

Calendar C++ definitions, compilation and caching live exclusively in
``playground/alpha_research/titan_jinja``.  This module intentionally contains
no factor implementation.
"""

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

from factors.calendar_backend import (  # noqa: E402
    CALENDAR_2025_SPECIAL_IDS,
    SPECIAL_KEY_TITLES,
    CalendarCppBackend,
    TitanCalendarService,
)

__all__ = [
    "CALENDAR_2025_SPECIAL_IDS",
    "SPECIAL_KEY_TITLES",
    "CalendarCppBackend",
    "TitanCalendarService",
]
