from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ImplementationStatus(StrEnum):
    EXACT = "exact"
    PROXY = "proxy"
    PENDING_IMPLEMENTATION = "pending_implementation"
    BLOCKED_DATA = "blocked_data"
    BLOCKED_FORMULA = "blocked_formula"
    BLOCKED_SOURCE = "blocked_source"


@dataclass(frozen=True)
class CalendarFactorDef:
    """Traceable definition and implementation metadata for one calendar item."""

    name: str
    calendar_year: int
    calendar_id: str
    title: str
    category: str
    sub_category: str = ""
    raw_category: str = ""
    normalized_category: str = ""
    source_file: str = ""
    source_start_line: int = 0
    source_end_line: int = 0
    input_frequency: str = "unknown"
    output_frequency: str = "1d"
    available_at: str = "T_close"
    required_fields: tuple[str, ...] = ()
    original_formula: str = ""
    implemented_formula: str = ""
    implementation_status: ImplementationStatus = ImplementationStatus.BLOCKED_FORMULA
    implementation_notes: str = ""
    implementation_key: str = ""
    proxy_of: str = ""
    duplicate_refs: tuple[str, ...] = ()
    duplicate_type: str = ""
    default_params: dict[str, Any] = field(default_factory=dict)
    warmup_requirement: int = 0

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["implementation_status"] = self.implementation_status.value
        for key in ("required_fields", "duplicate_refs"):
            row[key] = "|".join(row[key])
        return row
