from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


HEADING_RE = re.compile(
    r"^(#{1,3})\s+([0-9]+(?:\.[0-9]+){0,2})\s+(.+?)\s*$",
    re.MULTILINE,
)
SEPARATOR_RE = re.compile(r"^---\s*$", re.MULTILINE)
GENERIC_TITLES = {"其中", "说明", "参考文献", "计算步骤", "02"}
MANUAL_RECOVERY = {
    (2022, 236): "毛利率环比增长率",
    (2024, 1): "资本利得突出量",
    (2024, 361): "持仓机构个数",
}
DATE_LINE_RE = re.compile(
    r"^(?:Jan\.|Feb\.|Mar\.|Apr\.|May\.|Jun\.|Jul\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.|"
    r"\d{1,2}月|\d{1,2}|星期[一二三四五六日天]|正月|冬月|腊月|闰?\w+月)"
)


@dataclass(frozen=True)
class ParsedFactor:
    calendar_id: str
    title: str
    category: str
    sub_category: str
    raw_category: str
    body: str
    start_line: int
    end_line: int


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _clean_title(value: str) -> str:
    value = re.sub(r"\s*🏷️.*$", "", value).strip()
    value = re.sub(r"\s*<span.*$", "", value).strip()
    value = re.sub(r"<[^>]+>", "", value).strip()
    value = re.sub(r"\s*[\[（(](?:高频|技术|流动性|规模|估值|动量|波动率)[^\]）)]*因子[^\]）)]*[\]）)]?\s*$", "", value).strip()
    return value.strip("# []　")


def _category_from_body(body: str) -> str:
    patterns = (
        r"🏷️\s*([^\n|>]+)",
        r"^\s*[（(\[]([^\n\]）)]*因子[^\n\]）)]*)[）)\]]\s*$",
        r"<[^>]*>([^<>]*因子[^<>]*)</",
    )
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
    return ""


def _body_formula(body: str) -> str:
    equations = re.findall(r"\$\$(.*?)\$\$", body, flags=re.DOTALL)
    if equations:
        return "\n\n".join(eq.strip() for eq in equations if eq.strip())
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith(("#", "-")):
            lines.append(stripped)
    return "\n".join(lines[:12])


def formula_from_body(body: str) -> str:
    return _body_formula(body)


def parse_hierarchical(path: Path, year: int) -> list[ParsedFactor]:
    """Parse 2023/2026, whose factor identifiers are hierarchical headings."""
    text = path.read_text(encoding="utf-8")
    matches = list(HEADING_RE.finditer(text))
    category = ""
    sub_category = ""
    factors: list[ParsedFactor] = []
    for index, match in enumerate(matches):
        identifier, title = match.group(2), _clean_title(match.group(3))
        depth = identifier.count(".")
        if depth == 0:
            category = title
            sub_category = ""
            continue
        if depth == 1 and year == 2026:
            sub_category = title
            continue
        if depth == 1 and year == 2023 and identifier not in {"17.1", "17.2"}:
            sub_category = title
            continue
        if depth < 2 and not (year == 2023 and identifier in {"17.1", "17.2"}):
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        factors.append(
            ParsedFactor(
                calendar_id=identifier,
                title=title,
                category=category,
                sub_category=sub_category or category,
                raw_category=sub_category or category,
                body=body,
                start_line=_line_number(text, match.start()),
                end_line=_line_number(text, end),
            )
        )
    return factors


def _candidate_title(block: str, fallback: str) -> str:
    headings = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            value = re.sub(r"^#{1,4}\s*", "", stripped)
            value = re.sub(r"^\d+(?:\.\d+)*\s+", "", value)
            value = _clean_title(value)
            if value and value not in GENERIC_TITLES:
                headings.append(value)
    if headings:
        return headings[0]
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    for line in lines[:15]:
        clean = _clean_title(line)
        if (
            clean
            and not DATE_LINE_RE.match(clean)
            and clean not in GENERIC_TITLES
            and not clean.startswith(("---", ">", "$$", "$", "<", "由因子"))
            and "因子日历" not in clean
            and len(clean) <= 60
        ):
            return clean
    return fallback


def _looks_like_factor_block(block: str) -> bool:
    if re.search(r"^#{1,3}\s+\d+\s+", block, re.MULTILINE):
        return True
    markers = sum(token in block for token in ("$$", "说明", "参考文献", "因子"))
    return markers >= 2 and len(block.strip()) > 120


def parse_daily_calendar(path: Path, year: int, expected: int) -> list[ParsedFactor]:
    """Parse date-oriented 2022/2024 calendars and repair missing heading IDs.

    Explicit numeric headings are used as anchors.  Factor-like unnumbered blocks
    between two anchors receive the missing sequential identifiers.  Any still
    missing identifiers are retained as blocked placeholders so completeness is
    never confused with formula availability.
    """
    text = path.read_text(encoding="utf-8")
    raw_blocks = []
    cursor = 0
    for match in SEPARATOR_RE.finditer(text):
        raw_blocks.append((cursor, match.start(), text[cursor:match.start()]))
        cursor = match.end()
    raw_blocks.append((cursor, len(text), text[cursor:]))

    blocks: list[dict] = []
    for start, end, body in raw_blocks:
        if not _looks_like_factor_block(body):
            continue
        ids = []
        for heading in HEADING_RE.finditer(body):
            if "." not in heading.group(2):
                title = _clean_title(heading.group(3))
                if title not in GENERIC_TITLES:
                    ids.append((int(heading.group(2)), title))
        explicit_id = ids[0][0] if ids else None
        explicit_title = ids[0][1] if ids else ""
        blocks.append(
            {
                "start": start,
                "end": end,
                "body": body.strip(),
                "id": explicit_id,
                "title": explicit_title,
            }
        )

    # Assign unnumbered factor blocks from their ordered positions between anchors.
    anchor_positions = [(i, b["id"]) for i, b in enumerate(blocks) if b["id"]]
    for (left_pos, left_id), (right_pos, right_id) in zip(anchor_positions, anchor_positions[1:]):
        if right_id <= left_id + 1:
            continue
        candidates = [b for b in blocks[left_pos + 1:right_pos] if b["id"] is None]
        missing_ids = list(range(left_id + 1, right_id))
        for identifier, block in zip(missing_ids, candidates):
            block["id"] = identifier

    explicit: dict[int, dict] = {}
    for block in blocks:
        identifier = block["id"]
        if isinstance(identifier, int) and 1 <= identifier <= expected and identifier not in explicit:
            explicit[identifier] = block

    # A few source blocks have no numeric heading at all (or contain a month-day
    # heading that masks the real ID).  Recover them by their unique source title.
    for identifier in range(1, expected + 1):
        if identifier in explicit:
            continue
        recovery_title = MANUAL_RECOVERY.get((year, identifier))
        if not recovery_title:
            continue
        for block in blocks:
            if recovery_title in block["body"] and block not in explicit.values():
                recovered = dict(block)
                recovered["id"] = identifier
                recovered["title"] = recovery_title
                explicit[identifier] = recovered
                break
        if identifier not in explicit:
            offset = text.find(recovery_title)
            if offset >= 0:
                start = text.rfind("\n---", 0, offset)
                end = text.find("\n---", offset)
                start = 0 if start < 0 else start + 4
                end = len(text) if end < 0 else end
                explicit[identifier] = {
                    "start": start,
                    "end": end,
                    "body": text[start:end].strip(),
                    "id": identifier,
                    "title": recovery_title,
                }

    factors: list[ParsedFactor] = []
    for identifier in range(1, expected + 1):
        block = explicit.get(identifier)
        if block is None:
            factors.append(
                ParsedFactor(
                    calendar_id=str(identifier),
                    title=f"未恢复标题_{identifier:03d}",
                    category="未分类",
                    sub_category="",
                    raw_category="",
                    body="",
                    start_line=0,
                    end_line=0,
                )
            )
            continue
        body = block["body"]
        title = block["title"] or _candidate_title(body, f"未恢复标题_{identifier:03d}")
        raw_category = _category_from_body(body)
        factors.append(
            ParsedFactor(
                calendar_id=str(identifier),
                title=title,
                category=raw_category or "未分类",
                sub_category="",
                raw_category=raw_category,
                body=body,
                start_line=_line_number(text, block["start"]),
                end_line=_line_number(text, block["end"]),
            )
        )
    return factors


def parse_calendar(path: Path, year: int) -> list[ParsedFactor]:
    if year in {2023, 2026}:
        return parse_hierarchical(path, year)
    expected = 366 if year == 2024 else 365
    return parse_daily_calendar(path, year, expected)
