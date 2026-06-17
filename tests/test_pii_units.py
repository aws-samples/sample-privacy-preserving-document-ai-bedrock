"""
Unit tests for the pure pipeline logic — no AWS credentials or network required.

Covers: pseudonymization + round-trip reassembly, national-ID partial masking,
text chunking, TSV/JSON response parsing, and the KO/EN regex supplements.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "pii-service"))
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import pseudonymizer  # noqa: E402
import pipeline  # noqa: E402
import prompts  # noqa: E402


# ── Pseudonymization ──────────────────────────────────────────────────────
def test_pseudonymize_masks_and_is_consistent():
    text = "Name: John Doe. Contact John Doe at john@test.com."
    entities = [
        {"entity_group": "p_nm", "word": "John Doe"},
        {"entity_group": "p_em", "word": "john@test.com"},
    ]
    masked, mapping = pseudonymizer.pseudonymize_text(text, entities)
    assert "John Doe" not in masked
    assert "john@test.com" not in masked
    # Same value -> same token everywhere.
    assert masked.count("[PERSON_1]") == 2
    assert "[EMAIL_1]" in masked


def test_pseudonymize_no_entities_returns_original():
    masked, mapping = pseudonymizer.pseudonymize_text("nothing here", [])
    assert masked == "nothing here"
    assert mapping == {}


# ── Reassembly + national-ID partial masking ──────────────────────────────
def test_reassemble_round_trip_keeps_national_id_masked():
    mappings = [
        {"token": "PERSON_1", "original": "John Doe"},
        {"token": "SSN_1", "original": "521-84-6390"},
        {"token": "RRN_1", "original": "850315-1234567"},
    ]
    report = "Applicant [PERSON_1] (SSN [SSN_1], RRN [RRN_1]) is approved."
    final, replaced, unreplaced = pseudonymizer.reassemble(report, mappings)

    assert "John Doe" in final          # name fully restored
    assert "521-**-****" in final       # SSN: only the leading group kept, rest masked
    assert "850315-*******" in final    # RRN: birth-date prefix kept, rest masked
    assert "6390" not in final          # raw SSN tail never reappears
    assert "1234567" not in final       # raw RRN tail never reappears
    assert replaced == 3
    assert unreplaced == []


def test_reassemble_reports_unreplaced_tokens():
    final, replaced, unreplaced = pseudonymizer.reassemble("[PERSON_9] missing", [])
    assert replaced == 0
    assert unreplaced == ["[PERSON_9]"]


# ── Chunking ───────────────────────────────────────────────────────────────
def test_chunk_text_respects_max_chars():
    text = "\n".join(f"line {i} " + "x" * 100 for i in range(50))
    chunks = pipeline._chunk_text(text, max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 120 for c in chunks)  # +overlap headroom


def test_chunk_text_force_splits_long_line():
    chunks = pipeline._chunk_text("y" * 5000, max_chars=1000, overlap=100)
    assert len(chunks) > 1


# ── Response parsing ─────────────────────────────────────────────────────--
def test_parse_tsv_response():
    content = "PERSON\tJohn Doe\nSSN\t521-84-6390\nignored line without tab\n"
    items = pipeline._parse_pii_list_from_qwen_response(content)
    assert {"type": "PERSON", "original": "John Doe"} in items
    assert {"type": "SSN", "original": "521-84-6390"} in items
    assert len(items) == 2  # the non-TSV line is dropped


def test_parse_json_fallback():
    content = 'prose... {"pii_list": [{"type": "PERSON", "original": "Jane"}]} trailing'
    items = pipeline._parse_pii_list_from_qwen_response(content)
    assert items == [{"type": "PERSON", "original": "Jane"}]


def test_parse_strips_think_block():
    content = "<think>reasoning here</think>\nPERSON\tJohn Doe"
    items = pipeline._parse_pii_list_from_qwen_response(content)
    assert items == [{"type": "PERSON", "original": "John Doe"}]


# ── Regex supplements (both languages) ─────────────────────────────────────
def test_regex_en_finds_us_pii():
    text = ("SSN: 521-84-6390  Phone: (415) 555-2389  Email: a@b.com  "
            "Card: 4532-7165-9082-3471")
    found = pipeline._supplement_with_regex([], text, "en")
    types = {item["type"] for item in found}
    assert {"SSN", "PHONE", "EMAIL", "CARD"} <= types


def test_regex_ko_finds_korean_pii():
    text = "주민등록번호: 850315-1234567 연락처: 010-2345-6789 이메일: a@b.com"
    found = pipeline._supplement_with_regex([], text, "ko")
    types = {item["type"] for item in found}
    assert {"RRN", "PHONE", "EMAIL"} <= types


def test_regex_skips_masked_national_id():
    found = pipeline._supplement_with_regex([], "SSN: 521-**-****", "en")
    assert not any(item["type"] == "SSN" for item in found)


# ── Prompt config integrity ────────────────────────────────────────────────
@pytest.mark.parametrize("lang", prompts.SUPPORTED_LANGS)
def test_prompts_and_regex_exist_for_each_lang(lang):
    assert "$TEXT$" in prompts.PII_PROMPTS[lang]
    assert prompts.REGEX_SETS[lang]
    assert set(prompts.ANALYSIS_PROMPTS[lang]) == set(pipeline.VALID_SCENARIOS)
    assert "{anonymized_text}" in prompts.ANALYSIS_WRAPPER[lang]
