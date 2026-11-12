import pytest

from agent_desk.rag import risk_compare as rc
from agent_desk.rag.risk_extract import Risk
from agent_desk.rag.risk_section import RiskPassage


def passage(year, n, text):
    return RiskPassage(f"co-{year}-1A-{n}", text)


def risk(year, title, numbers):
    return Risk(
        title=title,
        summary=f"{title} summary",
        passage_ids=[f"co-{year}-1A-{n}" for n in numbers],
        quote="",
    )


OLD = [
    passage(2024, 1, "Cyber old wording about attacks."),
    passage(2024, 2, "Tax law may raise our tax rate."),
    passage(2024, 3, "We depend on key talent."),
    passage(2024, 4, "Sustainability targets may not be met."),
]
NEW = [
    passage(2025, 1, "Cyber new wording about attacks and AI-driven threats."),
    passage(2025, 2, "Tax law may raise our tax rate."),
    passage(2025, 3, "New AI regulation may restrict our products."),
    passage(2025, 4, "We depend on key talent."),
]
OLD_RISKS = [risk(2024, "Cyber", [1]), risk(2024, "Tax", [2]), risk(2024, "Talent", [3]), risk(2024, "Sustainability", [4])]
NEW_RISKS = [risk(2025, "Cyber", [1]), risk(2025, "Tax", [2]), risk(2025, "AI regulation", [3]), risk(2025, "People risk", [4])]

# Cyber old/new have cosine 0.6; tax and talent are identical across years; the rest are unrelated.
VECTORS = {
    OLD[0].text: [1, 0, 0, 0, 0, 0],
    NEW[0].text: [0.6, 0.8, 0, 0, 0, 0],
    OLD[1].text: [0, 0, 1, 0, 0, 0],
    NEW[1].text: [0, 0, 1, 0, 0, 0],
    OLD[2].text: [0, 0, 0, 1, 0, 0],
    NEW[3].text: [0, 0, 0, 1, 0, 0],
    OLD[3].text: [0, 0, 0, 0, 1, 0],
    NEW[2].text: [0, 0, 0, 0, 0, 1],
}


def fake_embed(texts):
    return [VECTORS[t] for t in texts]


class FakeLLM:
    """Answers the proposal call with `proposal` and the verification call with `verdicts`."""

    def __init__(self, proposal, verdicts=None, parsed_none=False):
        self.proposal = proposal
        self.verdicts = verdicts if verdicts is not None else []
        self.parsed_none = parsed_none
        self.schemas = []
        self.user_messages = []

    def with_structured_output(self, schema, **kwargs):
        llm = self

        class Structured:
            def invoke(self, messages):
                llm.schemas.append(schema.__name__)
                llm.user_messages.append(messages[1][1])
                if llm.parsed_none:
                    return {"raw": None, "parsed": None, "parsing_error": "bad json"}
                parsed = llm.proposal if schema is rc.RawComparison else {"verdicts": llm.verdicts}
                return {"raw": None, "parsed": parsed, "parsing_error": None}

        return Structured()


def compare(llm):
    return rc.compare_risks("co", 2024, 2025, OLD, NEW, OLD_RISKS, NEW_RISKS, llm=llm, embed=fake_embed)


def test_genuine_added_removed_and_changed_claims_are_kept():
    proposal = {
        "added": [{"description": "New AI regulation risk", "risks": ["N3"]}],
        "removed": [{"description": "Sustainability risk removed", "risks": ["O4"]}],
        "changed": [{"description": "Cyber changed", "risks": ["O1", "N1"]}],
    }
    llm = FakeLLM(proposal, [{"claim": 1, "material": True, "change": "Now also names AI-driven threats."}])

    diff = compare(llm)

    assert [f["passage_ids"] for f in diff["added"]] == [["co-2025-1A-3"]]
    assert diff["added"][0]["quote"].startswith("New AI regulation")
    assert [f["passage_ids"] for f in diff["removed"]] == [["co-2024-1A-4"]]
    assert diff["changed"][0]["description"] == "Now also names AI-driven threats."
    assert diff["changed"][0]["passage_ids"] == ["co-2024-1A-1", "co-2025-1A-1"]
    assert diff["changed"][0]["evidence"] == [
        {"old_id": "co-2024-1A-1", "new_id": "co-2025-1A-1", "similarity": 0.6}
    ]
    assert diff["rejected"] == []
    assert (diff["company"], diff["year_from"], diff["year_to"]) == ("co", 2024, 2025)
    assert llm.schemas == ["RawComparison", "VerdictList"]


def test_regrouped_risks_are_rejected_as_not_added_or_removed():
    proposal = {
        "added": [{"description": "People risk is new", "risks": ["N4"]}],
        "removed": [{"description": "Talent risk removed", "risks": ["O3"]}],
        "changed": [],
    }

    diff = compare(FakeLLM(proposal))

    assert diff["added"] == [] and diff["removed"] == []
    reasons = {r["description"]: r["reason"] for r in diff["rejected"]}
    assert "near-identical passage in FY2024" in reasons["People risk is new"]
    assert "near-identical passage in FY2025" in reasons["Talent risk removed"]


def test_claim_with_exactly_half_matching_passages_is_rejected():
    old = [passage(2024, 1, "Shared sentence."), passage(2024, 2, "Other old text.")]
    new = [passage(2025, 1, "Shared sentence."), passage(2025, 2, "Brand new text.")]
    vectors = {
        "Shared sentence.": [1, 0, 0],
        "Other old text.": [0, 1, 0],
        "Brand new text.": [0, 0, 1],
    }
    new_risks = [risk(2025, "Mixed", [1, 2])]
    proposal = {"added": [{"description": "Mixed is new", "risks": ["N1"]}], "removed": [], "changed": []}

    diff = rc.compare_risks(
        "co", 2024, 2025, old, new, [risk(2024, "Old", [1, 2])], new_risks,
        llm=FakeLLM(proposal), embed=lambda texts: [vectors[t] for t in texts],
    )

    assert diff["added"] == []
    assert "50% of the cited passages" in diff["rejected"][0]["reason"]


def test_unchanged_text_and_bad_citations_are_rejected():
    proposal = {
        "added": [{"description": "bogus", "risks": ["N99"]}, {"description": "wrong side", "risks": ["O1"]}],
        "removed": [],
        "changed": [
            {"description": "Tax changed", "risks": ["O2", "N2"]},
            {"description": "only old cited", "risks": ["O1"]},
        ],
    }

    llm = FakeLLM(proposal)
    diff = compare(llm)

    reasons = {r["description"]: r["reason"] for r in diff["rejected"]}
    assert reasons["bogus"] == "no valid risk labels cited"
    assert reasons["wrong side"] == "no valid risk labels cited"  # an added claim may cite only N labels
    assert "every cited passage is near-identical" in reasons["Tax changed"]
    assert "at least one older and one newer" in reasons["only old cited"]
    assert diff["added"] == diff["removed"] == diff["changed"] == []
    assert llm.schemas == ["RawComparison"]  # nothing reached the verification call


def test_changed_claim_the_verifier_calls_immaterial_is_rejected():
    proposal = {"added": [], "removed": [], "changed": [{"description": "Cyber changed", "risks": ["O1", "N1"]}]}
    llm = FakeLLM(proposal, [{"claim": 1, "material": False, "change": ""}])

    diff = compare(llm)

    assert diff["changed"] == []
    assert diff["rejected"][0]["reason"] == "the cited passages are not materially different"
    assert "OLD [co-2024-1A-1]: Cyber old wording" in llm.user_messages[1]
    assert "NEW [co-2025-1A-1]: Cyber new wording" in llm.user_messages[1]


def test_unusable_model_reply_raises():
    with pytest.raises(ValueError, match="no usable output"):
        compare(FakeLLM({}, parsed_none=True))


def test_proposal_prompt_labels_both_years_risks():
    llm = FakeLLM({"added": [], "removed": [], "changed": []})
    compare(llm)
    prompt = llm.user_messages[0]
    assert "Older year (FY2024) risks:" in prompt and "Newer year (FY2025) risks:" in prompt
    assert "[O1] Cyber: Cyber summary" in prompt and "[N3] AI regulation: AI regulation summary" in prompt


def test_compare_company_years_orders_years_and_loads_both(monkeypatch):
    loaded, compared = [], {}

    def fake_load(company, year):
        loaded.append((company, year))
        return OLD if year == 2024 else NEW

    monkeypatch.setattr(rc, "load_risk_section", fake_load)
    monkeypatch.setattr(rc, "extract_risks", lambda passages, llm: OLD_RISKS if passages is OLD else NEW_RISKS)
    monkeypatch.setattr(rc, "compare_risks", lambda *args, **kwargs: compared.update(args=args) or "diff")

    assert rc.compare_company_years(" Microsoft ", "2025", 2024) == "diff"

    assert loaded == [(" Microsoft ", 2024), (" Microsoft ", 2025)]
    assert compared["args"][:3] == ("microsoft", 2024, 2025)


def test_compare_company_years_rejects_same_year_and_passes_missing_filing_error(monkeypatch):
    with pytest.raises(ValueError, match="two different fiscal years"):
        rc.compare_company_years("microsoft", 2025, 2025)

    def missing(company, year):
        raise FileNotFoundError("No alphabet 10-K for fiscal year 2026. Available years: 2023, 2024, 2025.")

    monkeypatch.setattr(rc, "load_risk_section", missing)
    with pytest.raises(FileNotFoundError, match="Available years"):
        rc.compare_company_years("alphabet", 2025, 2026)
