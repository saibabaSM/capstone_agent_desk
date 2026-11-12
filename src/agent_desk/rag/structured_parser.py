"""Split a 10-K HTML file into Item-labelled blocks, keeping tables as tables.

The starter parser (filing_parser.py) flattens every table row into a line of text, which loses
which number belongs to which column/year. This parser emits two kinds of block, in document order:

    Block(kind="text",  item="7", text="Operating income increased ...")
    Block(kind="table", item="7", text="| | 2025 | 2024 |\n|---|---|---|\n| Revenue | $ 281,724 | $ 245,122 |")

SEC tables put "$", ")" and "%" in their own cells and pad with empty spacer cells; those are
merged so a cell reads "$ 281,724" or "(1,234)". `table_aware=False` reproduces the plain-text
behaviour (E4 ablation): table rows become text lines inside the surrounding prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag

ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[\.\:\-–—]?\s*(.*)$", re.IGNORECASE)
MIN_TEXT_CHARS = 40
MAX_TABLE_ROWS = 60


@dataclass
class Block:
    kind: str  # text | table
    item: str  # "1A", "7", "front"
    item_title: str
    text: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _row_cells(row: Tag) -> list[str]:
    raw = [_clean(cell.get_text(" ")) for cell in row.find_all(["td", "th"])]
    cells: list[str] = []
    for cell in raw:
        if not cell:
            continue
        if cells and cell in (")", "%", ")%", "%)"):
            cells[-1] += cell
        elif cells and cells[-1] in ("$", "(", "$(", "($"):
            cells[-1] = f"{cells[-1]} {cell}".replace("( ", "(")
        else:
            cells.append(cell)
    return cells


def table_to_markdown(table: Tag) -> str:
    rows = [cells for cells in (_row_cells(r) for r in table.find_all("tr")) if cells]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows[:MAX_TABLE_ROWS]]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def table_to_text(table: Tag) -> str:
    return "\n".join(" ".join(cells) for cells in (_row_cells(r) for r in table.find_all("tr")) if cells)


def _is_numeric_table(md: str) -> bool:
    return len(re.findall(r"\d", md)) >= 6


# Some filers (Intel) do not use "Item 1A." headings at all and give a cross-reference index instead.
# For those, standard section titles are mapped to Items; other known titles close the section.
TITLE_ITEMS = [
    (re.compile(r"^risk factors(?: and other key information)?$", re.I), "1A", "Risk Factors"),
    (re.compile(r"^management.s discussion and analysis", re.I), "7", "Management's Discussion and Analysis"),
    # Intel opens MD&A with these sections instead of an MD&A heading
    (re.compile(r"^(?:a year in review|significant events and trends impacting results)$", re.I), "7", "Management's Discussion and Analysis"),
    (re.compile(r"^quantitative and qualitative disclosures? about market risk", re.I), "7A", "Market Risk"),
    (re.compile(r"^(?:consolidated )?financial statements(?: and supplementary data)?$", re.I), "8", "Financial Statements"),
    (re.compile(r"^cybersecurity$", re.I), "1C", "Cybersecurity"),
    (re.compile(r"^legal proceedings$", re.I), "3", "Legal Proceedings"),
    (re.compile(r"^(?:properties|unresolved staff comments|controls and procedures|exhibits.*|other key information|"
                r"market for (?:the )?registrant.*|executive officers.*|form 10-k cross-reference index|"
                r"supplemental details|financial and operational highlights|our (?:business|strategy|products).*|"
                r"fiscal year in review|auditor.s report.*|report of independent registered public accounting firm)$", re.I),
     "other", "Other"),
]


def _title_item(text: str) -> tuple[str, str] | None:
    if len(text) > 90:
        return None
    for rx, item, title in TITLE_ITEMS:
        if rx.match(text.strip().rstrip(".:")):
            return item, title
    return None


def parse_filing(path: Path, table_aware: bool = True) -> list[Block]:
    blocks = _parse(path, table_aware, by_title=False)
    if blocks and all(b.item == "front" for b in blocks):
        blocks = _parse(path, table_aware, by_title=True)
    return blocks


def _parse(path: Path, table_aware: bool, by_title: bool) -> list[Block]:
    soup = BeautifulSoup(Path(path).read_bytes(), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:header$")):
        tag.decompose()

    blocks: list[Block] = []
    item, title = "front", "Front matter"
    pending_plain: list[str] = []

    def flush_plain() -> None:
        if pending_plain:
            blocks.append(Block("text", item, title, "\n".join(pending_plain)))
            pending_plain.clear()

    for node in soup.find_all(["table", "div", "p"]):
        if node.name != "table" and node.find_parent("table") is not None:
            continue
        if node.name == "table":
            if node.find_parent("table") is not None:
                continue
            if node.find("a", href=True) and not _is_numeric_table(node.get_text()):
                continue  # table of contents
            cells_text = _clean(node.get_text(" "))
            if by_title and (titled := _title_item(cells_text)):  # running page header naming the section
                if titled[0] != item:
                    flush_plain()
                    item, title = titled
                continue
            heading = ITEM_RE.match(cells_text)
            if heading and len(cells_text) < 200:  # some filers put Item headings in a 1-row table
                flush_plain()
                item, title = heading.group(1).upper(), _clean(heading.group(2)).rstrip(".")
                continue
            if table_aware:
                md = table_to_markdown(node)
                if md and _is_numeric_table(md):
                    blocks.append(Block("table", item, title, md))
                elif md:
                    pending_plain.append(_clean(node.get_text(" ")))
            else:
                flat = table_to_text(node)
                if flat:
                    pending_plain.append(flat)
            continue
        if node.find(["div", "p", "table"]):
            continue  # only leaf blocks, so nested divs are not read twice
        text = _clean(node.get_text(" "))
        if not text:
            continue
        if by_title and (titled := _title_item(text)) and not node.find("a", href=True):
            flush_plain()
            item, title = titled
            continue
        heading = ITEM_RE.match(text)
        if heading and len(text) < 200 and not node.find("a", href=True):
            flush_plain()
            item, title = heading.group(1).upper(), _clean(heading.group(2)).rstrip(".")
            continue
        if len(text) >= MIN_TEXT_CHARS:
            if table_aware:
                blocks.append(Block("text", item, title, text))
            else:
                pending_plain.append(text)
    flush_plain()
    return blocks
