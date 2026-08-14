"""
The sample's actual value proposition, pinned as a test rather than left as a
README claim: no raw PII value reaches the Bedrock `invoke_model` request body.

This runs the real `run_pipeline()` orchestration logic end to end, but fakes
every network boundary (the sLLM call, the pii-service HTTP calls, and
`bedrock_runtime.invoke_model` itself) — no AWS credentials, DynamoDB, vLLM
endpoint, or network access required. What's under test is the wiring: does
`_bedrock_analysis()` actually only ever receive the masked/anonymized text,
never the raw values Stage 3 detected.
"""

import asyncio
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestrator"))

import pipeline  # noqa: E402


class _FakeResponse:
    def __init__(self, json_data):
        self.status_code = 200
        self._json = json_data
        self.text = json.dumps(json_data)

    def json(self):
        return self._json


def test_no_raw_pii_reaches_bedrock_invoke_model(monkeypatch):
    raw_person = "John Doe"
    raw_ssn = "521-84-6390"
    raw_text = f"Applicant {raw_person}, SSN {raw_ssn}, approved for coverage."
    masked_text = "Applicant [PERSON_1], SSN [SSN_1], approved for coverage."
    pii_data = {"PERSON_1": raw_person, "SSN_1": raw_ssn}

    # 1) Skip the real sLLM HTTP call — feed a canned detection result, as if
    #    the model had already extracted these two items.
    async def fake_detect_sllm(text, lang):
        return {
            "pii_list": [
                {"type": "PERSON", "original": raw_person},
                {"type": "SSN", "original": raw_ssn},
            ],
            "chunk_total": 1, "chunk_failed": 0, "chunk_parse_failed": 0,
        }
    monkeypatch.setattr(pipeline, "_detect_sllm", fake_detect_sllm)

    # 2) Fake the pii-service's four endpoints. This is the contract
    #    src/pii-service/main.py actually implements, reproduced here rather
    #    than run live, so a real /pseudonymize call is what turns raw values
    #    into pii_data — same as production, just not over the network.
    async def fake_post(url, json=None, **kwargs):  # noqa: A002 - matches httpx's kwarg name
        if url.endswith("/pseudonymize"):
            return _FakeResponse({"final_text": masked_text, "pii_data": pii_data, "pii_count": 2})
        if url.endswith("/guardrails"):
            return _FakeResponse({"extra_pii": [], "verified_text": masked_text, "skipped": True})
        if url.endswith("/store"):
            return _FakeResponse({"stored": len(pii_data), "session_id": "test-session"})
        if url.endswith("/reassemble"):
            return _FakeResponse({
                "final_text": f"Report: {raw_person} approved. Ref: {raw_ssn}.",
                "replaced_count": 2, "unreplaced_tokens": [],
            })
        raise AssertionError(f"unexpected POST {url}")
    monkeypatch.setattr(pipeline.http_client, "post", fake_post)

    # 3) Capture exactly what's sent to Bedrock, and hand back a canned
    #    response that still carries the PII tokens (as a real analysis would).
    captured = {}

    def fake_invoke_model(**kwargs):
        captured.update(kwargs)
        body = {
            "content": [{"text": "Report: [PERSON_1] is approved. SSN on file: [SSN_1]."}],
            "usage": {"input_tokens": 42, "output_tokens": 17},
            "stop_reason": "end_turn",
        }
        return {"body": io.BytesIO(json.dumps(body).encode("utf-8"))}
    monkeypatch.setattr(pipeline.bedrock_runtime, "invoke_model", fake_invoke_model)

    result = asyncio.run(pipeline.run_pipeline("insurance", raw_text, lang="en"))

    # --- The actual assertion: what got sent to Bedrock ---
    sent_body = captured["body"]
    if isinstance(sent_body, bytes):
        sent_body = sent_body.decode("utf-8")
    assert raw_person not in sent_body, "raw PERSON value leaked into the Bedrock request body"
    assert raw_ssn not in sent_body, "raw SSN value leaked into the Bedrock request body"
    assert "[PERSON_1]" in sent_body
    assert "[SSN_1]" in sent_body

    # Sanity: the pipeline still completed and restored PII into the final report
    # (via the faked /reassemble above) — this isn't a no-op that trivially passes.
    assert result["piiData"] == pii_data
    assert raw_person in result["finalText"]
    assert result["summary"]["piiDetected"] == 2
