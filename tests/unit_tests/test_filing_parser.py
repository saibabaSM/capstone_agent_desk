from agent_desk.rag.filing_parser import load_passages

SAMPLE = """
<html><body>
<table><tr><td><a href="#a">Item 1.</a></td><td><a href="#a">Business</a></td></tr></table>
<div>PART I</div>
<div>Item 1.&nbsp;&nbsp;Business</div>
<div>The Company designs smartphones, personal computers and tablets worldwide.</div>
<div>Item 1A.&nbsp;&nbsp;Risk Factors</div>
<div>Supply chain disruptions could materially affect the Company's operations.</div>
</body></html>
"""


def _passages(tmp_path):
    path = tmp_path / "filing.html"
    path.write_text(SAMPLE, encoding="utf-8")
    return load_passages(path)


def test_passages_are_labelled_with_item(tmp_path):
    items = {p.item for p in _passages(tmp_path)}
    assert "Item 1 - Business" in items
    assert "Item 1A - Risk Factors" in items


def test_table_of_contents_does_not_change_item(tmp_path):
    first = _passages(tmp_path)[0]
    assert first.item != "Item 1 - Business" or "designs" in first.text
