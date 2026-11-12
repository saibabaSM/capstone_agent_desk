"""Extract Item-labelled text passages from a saved SEC 10-K HTML file."""

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

DEFAULT_FILING = Path("data/filings/apple/2025/aapl-20251227.html")

ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\.?\s*(.*)$", re.IGNORECASE)
BLOCK_TAGS = ["div", "p", "tr"]
MIN_PASSAGE_CHARS = 40


@dataclass
class Passage:
    item: str
    text: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def load_passages(path: Path) -> list[Passage]:
    """Split the filing into text passages labelled with their Item.

    The table of contents lists every Item as a link; those rows are skipped
    so only the real section headings change the current Item.
    """
    # Bytes let BeautifulSoup honor the charset declared in the filing's meta tag.
    soup = BeautifulSoup(Path(path).read_bytes(), "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(["ix:header"]):
        tag.decompose()

    passages: list[Passage] = []
    current_item = "Front matter"
    for block in soup.find_all(BLOCK_TAGS):
        # Only leaf-level blocks, so nested divs are not counted twice.
        if block.find(BLOCK_TAGS):
            continue
        if block.find_parent("tr") is not None and block.name != "tr":
            continue
        text = _clean(block.get_text(" "))
        if not text:
            continue
        heading = ITEM_RE.match(text)
        if heading and not block.find("a", href=True) and len(text) < 200:
            current_item = f"Item {heading.group(1).upper()}"
            if heading.group(2):
                current_item += f" - {heading.group(2).rstrip('.')}"
            continue
        if len(text) >= MIN_PASSAGE_CHARS:
            passages.append(Passage(current_item, text))
    return passages
