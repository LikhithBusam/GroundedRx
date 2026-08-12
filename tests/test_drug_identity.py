"""Regression tests for the Drug Identity Gate's matching logic, ported from
GroundedRx_Colab.ipynb's "Safety Improvement: Drug Identity Gate" self-check.

Pure functions, plain strings and dicts -- no GPU, no model, no network.
"""

from groundedrx.drug_identity import (
    DRUG_REGISTRY,
    chunk_matches_drug,
    extract_drug_identity,
    filter_chunks_by_drug,
)


def test_extracts_generic_name_english():
    assert extract_drug_identity("What is the lisinopril 10 mg dose?", "en")["drug"] == "lisinopril"
    assert extract_drug_identity("enalapril side effects", "en")["drug"] == "enalapril"
    assert extract_drug_identity("ramipril dosage for adults", "en")["drug"] == "ramipril"


def test_extracts_drug_name_arabic():
    assert extract_drug_identity("ما هي جرعة ليزينوبريل؟", "ar")["drug"] == "lisinopril"
    result = extract_drug_identity("ما هي الآثار الجانبية للوجينون؟", "ar")
    assert result["drug"] == "levonorgestrel_ethinylestradiol"


def test_spelling_variant_typo_is_matched():
    assert extract_drug_identity("lisinoprol dose", "en")["drug"] == "lisinopril"


def test_brand_name_resolves_to_generic():
    assert extract_drug_identity("Linopril side effects", "en")["drug"] == "lisinopril"


def test_no_explicit_medication_returns_none_not_an_error():
    result = extract_drug_identity("What are the side effects of this medication?", "en")
    assert result["drug"] is None
    assert result["all_drugs"] == []


def test_multiple_medications_in_one_query():
    result = extract_drug_identity("Compare lisinopril and enalapril side effects", "en")
    assert "lisinopril" in result["all_drugs"]
    assert "enalapril" in result["all_drugs"]


def test_chunk_matching_lisinopril_vs_enalapril():
    lisinopril_chunk = {"text": "Linopril (lisinopril) is used to treat high blood pressure."}
    enalapril_chunk = {"text": "Enalapril is an ACE inhibitor used for hypertension."}
    assert chunk_matches_drug(lisinopril_chunk, "lisinopril") is True
    assert chunk_matches_drug(lisinopril_chunk, "enalapril") is False
    assert chunk_matches_drug(enalapril_chunk, "enalapril") is True
    assert chunk_matches_drug(enalapril_chunk, "lisinopril") is False


def test_chunk_matching_lisinopril_vs_ramipril():
    ramipril_chunk = {"text": "Ramipril tablets are used to lower blood pressure."}
    assert chunk_matches_drug(ramipril_chunk, "ramipril") is True
    assert chunk_matches_drug(ramipril_chunk, "lisinopril") is False


def test_filter_chunks_by_drug_keeps_only_matching_chunks():
    chunks = [
        {"text": "Lisinopril is used to treat high blood pressure."},
        {"text": "Enalapril is an ACE inhibitor."},
        {"text": "Take lisinopril once daily."},
    ]
    kept = filter_chunks_by_drug(chunks, "lisinopril")
    assert len(kept) == 2
    assert all("lisinopril" in c["text"].lower() for c in kept)


def test_registry_covers_expected_drugs():
    # Documents the current seed-list scope -- not exhaustive over the full
    # 464-document corpus (see drug_identity.py docstring).
    assert set(DRUG_REGISTRY.keys()) == {
        "lisinopril",
        "enalapril",
        "ramipril",
        "levonorgestrel_ethinylestradiol",
        "desogestrel_ethinylestradiol",
        "batlor",
    }
