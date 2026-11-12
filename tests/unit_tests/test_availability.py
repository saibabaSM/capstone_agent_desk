from agent_desk.agents import research_lead_agent as lead
from agent_desk.rag import risk_section


def make_filing(root, company, year, name="filing.htm"):
    folder = root / company / str(year)
    folder.mkdir(parents=True)
    (folder / name).write_text("<html></html>", encoding="utf-8")


def test_available_filings_lists_years_with_a_filing(monkeypatch, tmp_path):
    make_filing(tmp_path, "microsoft", 2024)
    make_filing(tmp_path, "microsoft", 2023)
    make_filing(tmp_path, "apple", 2025, "aapl.html")
    (tmp_path / "apple" / "2024").mkdir()  # an empty year folder does not count
    (tmp_path / "tesla").mkdir()  # a company with no filings at all is left out
    make_filing(tmp_path, "alphabet", 2025, "notes.txt")  # not an HTML filing, so it does not count
    monkeypatch.setattr(risk_section, "FILINGS_DIR", tmp_path)

    assert risk_section.available_filings() == {"apple": [2025], "microsoft": [2023, 2024]}


def test_available_filings_without_a_data_folder_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(risk_section, "FILINGS_DIR", tmp_path / "missing")

    assert risk_section.available_filings() == {}


def test_availability_table_names_the_latest_year_per_company(monkeypatch):
    monkeypatch.setattr(
        lead, "available_filings", lambda: {"apple": [2025], "microsoft": [2023, 2024, 2025]}
    )

    assert lead._availability_table() == (
        "  - apple: latest fiscal year 2025; filings for 2025\n"
        "  - microsoft: latest fiscal year 2025; filings for 2023, 2024, 2025"
    )


def test_availability_table_when_nothing_is_stored(monkeypatch):
    monkeypatch.setattr(lead, "available_filings", lambda: {})

    assert lead._availability_table() == "  (no filings are stored)"


def test_prompt_uses_the_table_and_never_the_current_date(monkeypatch):
    monkeypatch.setattr(lead, "available_filings", lambda: {"alphabet": [2023, 2024, 2025]})

    prompt = lead._system_prompt()

    assert "  - alphabet: latest fiscal year 2025; filings for 2023, 2024, 2025" in prompt
    assert "Today's date" not in prompt
    assert "__AVAILABLE__" not in prompt and "__TODAY__" not in prompt
    assert not hasattr(lead, "date")  # the date import is gone
