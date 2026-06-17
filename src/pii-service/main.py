"""
PII Service — pseudonymization, Guardrails verification, storage, and reassembly.

This service holds the PII-handling primitives that the orchestrator calls over HTTP.
It is CPU-only and has no ML model dependency.

Endpoints:
  POST /pseudonymize  convert detected PII (type/original pairs) into [PERSON_1]-style tokens
  POST /guardrails    double-check masked text with Amazon Bedrock Guardrails
  POST /store         persist token -> original mappings in DynamoDB (with TTL)
  POST /reassemble    restore tokens in analysis output (national IDs stay partially masked)
  GET  /health
"""

import asyncio
import logging
import os
import time

import boto3
from boto3.dynamodb.conditions import Key
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pseudonymizer import LABEL_DESCRIPTIONS, pseudonymize_text, reassemble

# Detection type label (from the sLLM) -> internal detection label used by the
# pseudonymizer. Keep in sync with prompts.VALID_PII_TYPES (orchestrator).
TYPE_TO_LABEL = {
    "PERSON": "p_nm",
    "RRN": "p_rrn",
    "SSN": "p_ssn",
    "PHONE": "p_ph",
    "EMAIL": "p_em",
    "ADDRESS": "p_add",
    "ACCOUNT": "p_acn",
    "IP": "p_ip",
    "PASSPORT": "p_pp",
    "DATE": "p_dt",
    "ORG": "p_org",
    "LOCATION": "p_loc",
    "URL": "p_url",
    "CARD": "p_card",
    "DOB": "p_dob",
    "REL": "p_rel",
}

# Token prefixes that are national IDs — Stage 7 keeps these partially masked even
# after reassembly so the final document never exposes the full identifier.
NATIONAL_ID_PREFIXES = ("RRN", "SSN")

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Privacy-Preserving Document AI — PII Service",
    version="1.0.0",
    description="Pseudonymization, Bedrock Guardrails verification, DynamoDB storage, and PII reassembly.",
)

# ── Configuration ──
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "DRAFT")
PII_TABLE = os.environ.get("PII_TABLE", "pii-mappings")
PII_TTL_SECONDS = int(os.environ.get("PII_TTL_SECONDS", "3600"))

bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
pii_table = dynamodb.Table(PII_TABLE)


# ── Request / Response models ──
class QwenPiiItem(BaseModel):
    type: str = Field(..., description="Detection type (PERSON/RRN/SSN/PHONE/EMAIL/...)")
    original: str = Field(..., description="Original PII text")


class PseudonymizeRequest(BaseModel):
    text: str = Field(..., description="Original text")
    pii_list: list[QwenPiiItem] = Field(..., description="Detected PII candidates")


class PseudonymizeResponse(BaseModel):
    final_text: str = Field(..., description="Text with PII replaced by tokens")
    pii_data: dict[str, str] = Field({}, description="token -> original mapping")
    pii_count: int = Field(0)


class GuardrailsRequest(BaseModel):
    masked_text: str


class StoreRequest(BaseModel):
    session_id: str
    scenario_id: str
    pii_data: dict[str, str]


class ReassembleRequest(BaseModel):
    session_id: str
    analysis_result: str


# ── Endpoints ──
@app.get("/health")
async def health():
    return {"status": "ok", "service": "pii-service", "entity_types": list(LABEL_DESCRIPTIONS.keys())}


@app.post("/pseudonymize", response_model=PseudonymizeResponse)
async def pseudonymize(req: PseudonymizeRequest):
    """Convert detected (type, original) pairs into [PERSON_1]-style tokens.

    The detector reports type + value (no positions), so the pseudonymizer applies
    global string substitution over the document.
    """
    entities = [
        {"entity_group": TYPE_TO_LABEL.get(item.type, "p_unknown"), "word": item.original}
        for item in req.pii_list
        if item.original.strip()
    ]
    masked_text, mapping = await asyncio.to_thread(pseudonymize_text, req.text, entities)
    pii_data = {info["tag"].strip("[]"): original for original, info in mapping.items()}
    logger.info("pseudonymize complete: %d tokens", len(pii_data))
    return JSONResponse({"final_text": masked_text, "pii_data": pii_data, "pii_count": len(pii_data)})


@app.post("/guardrails")
async def verify_guardrails(req: GuardrailsRequest):
    """Double-check the masked text with Bedrock Guardrails (skipped if unconfigured)."""
    if not GUARDRAIL_ID:
        return JSONResponse({"extra_pii": [], "verified_text": req.masked_text, "skipped": True})
    try:
        response = await asyncio.to_thread(
            bedrock_runtime.apply_guardrail,
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION,
            source="INPUT",
            content=[{"text": {"text": req.masked_text}}],
        )
        extra_pii = []
        for assessment in response.get("assessments", []):
            policy = assessment.get("sensitiveInformationPolicy", {})
            for item in policy.get("piiEntities", []):
                extra_pii.append({
                    "type": item.get("type", "UNKNOWN"),
                    "match": item.get("match", ""),
                    "action": item.get("action", "ANONYMIZED"),
                    "source": "builtin",
                })
            for item in policy.get("regexes", []):
                extra_pii.append({
                    "type": item.get("name", "regex"),
                    "match": item.get("match", ""),
                    "action": item.get("action", "ANONYMIZED"),
                    "source": "regex",
                })
        outputs = response.get("outputs", [])
        verified_text = outputs[0].get("text", req.masked_text) if outputs else req.masked_text
        return JSONResponse({"extra_pii": extra_pii, "verified_text": verified_text})
    except Exception as e:
        # Guardrails failure is non-fatal — pass the masked text through.
        return JSONResponse({"extra_pii": [], "verified_text": req.masked_text, "error": str(e)})


@app.post("/store")
async def store_pii(req: StoreRequest):
    """Persist token -> original mappings in DynamoDB with a TTL."""
    ttl = int(time.time()) + PII_TTL_SECONDS

    def _write():
        with pii_table.batch_writer() as batch:
            for token, original in req.pii_data.items():
                batch.put_item(Item={
                    "session_id": req.session_id,
                    "token": token,
                    "original_value": original,
                    "scenario_id": req.scenario_id,
                    "ttl": ttl,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

    await asyncio.to_thread(_write)
    return JSONResponse({"stored": len(req.pii_data), "session_id": req.session_id})


@app.post("/reassemble")
async def reassemble_pii(req: ReassembleRequest):
    """Restore PII tokens in the analysis output via deterministic string replacement.

    National IDs (RRN/SSN) are restored only partially — the trailing digits stay
    masked so the final document never re-exposes the full identifier.
    """
    ddb_response = await asyncio.to_thread(
        pii_table.query, KeyConditionExpression=Key("session_id").eq(req.session_id)
    )
    pii_mappings = [
        {"token": item["token"], "original": item["original_value"]}
        for item in ddb_response["Items"]
    ]

    result_text, replaced_count, unreplaced = reassemble(
        req.analysis_result, pii_mappings, NATIONAL_ID_PREFIXES
    )
    logger.info("reassembly complete: %d restored, %d unreplaced", replaced_count, len(unreplaced))
    return JSONResponse({
        "final_text": result_text,
        "replaced_count": replaced_count,
        "unreplaced_tokens": unreplaced,
        "mappings": [{"token": m["token"], "type": m["token"].split("_")[0]} for m in pii_mappings],
    })


if __name__ == "__main__":
    import uvicorn

    # Binds all interfaces by design — the service runs inside a container. nosec B104
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8082")))  # nosec B104
