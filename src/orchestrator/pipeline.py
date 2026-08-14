"""
Privacy-preserving document-AI pipeline (post-OCR).

Given OCR'd text, this orchestrator runs five stages so that a large model only
ever sees anonymized text:

  Stage 3  PII detection    — an sLLM (Qwen via vLLM) + regex extract PII; the
                              pii-service pseudonymizes it into [PERSON_1]-style tokens.
  Stage 4  Guardrails       — Amazon Bedrock Guardrails double-checks the masked text.
  Stage 5  PII store        — token -> original mappings are stored in DynamoDB (TTL).
  Stage 6  Bedrock analysis — Amazon Bedrock (Claude) analyzes the anonymized text.
  Stage 7  Reassembly       — tokens in the analysis output are restored by
                              deterministic string replacement (national IDs stay masked).

(Stages 1-2, document load and OCR, are out of scope for this sample — the input is
already OCR'd text. See ``sample-documents/``.)

Stage numbering is preserved from the reference architecture for readability.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import Awaitable, Callable

import boto3
import httpx
import prompts
from botocore.config import Config as _BotoConfig

logger = logging.getLogger("orchestrator.pipeline")

VALID_SCENARIOS = ("insurance", "mortgage", "creditcard", "stock")

# An emit callback reports per-stage progress. It receives (stage_name, data).
EmitFn = Callable[[str, dict], Awaitable[None]]


# ── Configuration (environment, with neutral defaults) ──────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("MODEL_ID", "global.anthropic.claude-sonnet-5")

# Optional — attaches the same Bedrock Guardrail to the direct invoke_model call
# in Stage 6, in addition to the out-of-band ApplyGuardrail check the pii-service
# already runs in Stage 4. This is defense-in-depth, not a substitute: Stage 4
# checks the text *before* it is sent; this checks the model invocation itself,
# which also covers PII the model might echo back into its own output. Leave
# unset to skip (the pipeline still runs Stage 4 either way).
GUARDRAIL_ID_FOR_INVOKE = os.environ.get("GUARDRAIL_ID", "")
GUARDRAIL_VERSION_FOR_INVOKE = os.environ.get("GUARDRAIL_VERSION", "")

PII_SERVICE_URL = os.environ.get("PII_SERVICE_URL", "http://localhost:8082")
SLLM_ENDPOINT = os.environ.get("SLLM_ENDPOINT", "http://localhost:8000")
SLLM_MODEL_NAME = os.environ.get("SLLM_MODEL_NAME", "Qwen/Qwen3-8B")

# Optional L2 result cache (DynamoDB). Unset table name => caching disabled.
QWEN_CACHE_TABLE = os.environ.get("QWEN_CACHE_TABLE", "")
QWEN_CACHE_TTL_DAYS = int(os.environ.get("QWEN_CACHE_TTL_DAYS", "1"))

# AWS clients. invoke_model can take a couple of minutes for long reports, so the
# botocore read timeout is raised well above the default 60s.
_bedrock_config = _BotoConfig(read_timeout=600, connect_timeout=10,
                              retries={"max_attempts": 2, "mode": "standard"})
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=_bedrock_config)

_dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION) if QWEN_CACHE_TABLE else None
qwen_cache_table = _dynamodb.Table(QWEN_CACHE_TABLE) if _dynamodb else None

# Shared async HTTP client for pii-service and vLLM calls.
http_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0))


# Map a full Bedrock model id to a short label for display, e.g.
# ``global.anthropic.claude-sonnet-5`` -> ``sonnet-5``, and
# ``global.anthropic.claude-sonnet-4-6`` -> ``sonnet-4.6`` (older generations
# carry a minor version segment; newer ones don't).
_MODEL_ALIAS_PATTERN = re.compile(
    r"(?:global|us|apac|eu)\.anthropic\.claude-(sonnet|opus|haiku)-(\d)(?:-(\d))?(?:-.*)?$"
)


def _short_model_alias(model_id: str) -> str:
    if not model_id:
        return model_id
    m = _MODEL_ALIAS_PATTERN.match(model_id)
    if m:
        family, major, minor = m.group(1), m.group(2), m.group(3)
        return f"{family}-{major}.{minor}" if minor else f"{family}-{major}"
    return model_id


# ── Stage 3 helpers: chunking, regex supplement, response parsing ───────────

def _chunk_text(raw_text: str, max_chars: int = 8000, overlap: int = 400) -> list[str]:
    """Split text into chunks at line boundaries, with overlap to avoid cutting PII.

    A single line longer than ``max_chars`` is force-split to prevent the model
    server from hanging or running out of memory.
    """
    if overlap >= max_chars:
        overlap = max(0, max_chars // 8)
    raw_lines = raw_text.split("\n")
    lines: list[str] = []
    for rl in raw_lines:
        rl = rl.strip()
        if not rl:
            continue
        if len(rl) > max_chars:
            for i in range(0, len(rl), max_chars - overlap):
                lines.append(rl[i:i + max_chars])
        else:
            lines.append(rl)

    if not lines:
        return [raw_text[:max_chars]] if raw_text.strip() else []

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))
            overlap_lines: list[str] = []
            overlap_len = 0
            for prev_line in reversed(current_lines):
                if overlap_len + len(prev_line) + 1 > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_len += len(prev_line) + 1
            current_lines = overlap_lines
            current_len = overlap_len
        current_lines.append(line)
        current_len += line_len
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks if chunks else [raw_text[:max_chars]]


def _supplement_with_regex(pii_list: list, text: str, lang: str) -> list:
    """Add fixed-format PII (national ID, phone, email, account, ...) the model missed.

    Uses the language-specific pattern set from ``prompts.REGEX_SETS``. Items are
    deduplicated against the existing list by whitespace-insensitive comparison.
    """
    existing = {re.sub(r"\s+", "", item.get("original", "")) for item in pii_list}
    supplemented = list(pii_list)

    for pii_type, pattern in prompts.REGEX_SETS.get(lang, []):
        for m in pattern.finditer(text):
            matched = m.group(1) if m.groups() else m.group(0)

            if pii_type in ("RRN", "SSN"):
                # Skip already-masked identifiers.
                if re.search(r"[*○O]", matched):
                    continue
            if pii_type == "ACCOUNT":
                digits_only = re.sub(r"\D", "", matched)
                if len(digits_only) < 10:
                    continue
                # Exclude business-registration / tax-id and national-id shapes.
                if re.fullmatch(r"\d{3}\s*-\s*\d{2}\s*-\s*\d{5}", matched.strip()):
                    continue
                if re.fullmatch(r"\d{6}\s*-\s*\d{7}", matched.strip()):
                    continue
            if pii_type == "ADDRESS":
                matched = matched.strip().rstrip(",. ")

            normalized = re.sub(r"\s+", "", matched)
            if normalized and normalized not in existing:
                existing.add(normalized)
                supplemented.append({"type": pii_type, "original": matched})

    return supplemented


def _parse_pii_list_from_qwen_response(content: str) -> list | None:
    """Parse the model's response into ``[{"type", "original"}, ...]`` or None.

    Primary format: TSV (``TYPE<TAB>ORIGINAL`` per line). Falls back to a legacy
    JSON ``{"pii_list": [...]}`` shape. Any ``<think>`` reasoning block is stripped.
    """
    clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Primary: TSV. Non-TSV lines (headers/footers) are ignored automatically.
    tsv_items: list = []
    for raw_line in clean.splitlines():
        line = raw_line.strip()
        if not line or "\t" not in line:
            continue
        t, _, v = line.partition("\t")
        t, v = t.strip(), v.strip()
        if t in prompts.VALID_PII_TYPES and v:
            tsv_items.append({"type": t, "original": v})
    if tsv_items:
        return tsv_items

    # Fallback: JSON (absorbs cached/legacy responses).
    json_match = re.search(r'\{[^{}]*"pii_list"\s*:\s*\[.*?\]\s*\}', clean, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group()).get("pii_list", [])
        except json.JSONDecodeError:
            pass
    for m in re.finditer(r"\{", clean):
        try:
            candidate = json.loads(clean[m.start():])
            if "pii_list" in candidate:
                return candidate["pii_list"]
        except (json.JSONDecodeError, ValueError):
            continue
    return None


async def _call_sllm_for_chunk(chunk: str, lang: str) -> dict:
    """Call the Qwen vLLM endpoint for a single chunk, with one retry.

    Sampling params follow vLLM guidance for structured extraction:
      - ``temperature=0`` (then 0.2 on retry) for near-deterministic output
      - ``repetition_penalty=1.1`` / ``frequency_penalty=0.2`` to break the
        degenerate "repeat the same token to max_tokens" loop that greedy decoding
        can fall into.
    On repeated failure returns ``{"pii_list": [], "parse_error": True}`` so the
    caller can apply its partial-failure threshold.
    """
    prompt = prompts.PII_PROMPTS[lang].replace("$TEXT$", chunk)
    base_request_body = {
        "model": SLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "frequency_penalty": 0.2,
        "repetition_penalty": 1.1,
    }

    MAX_ATTEMPTS = 2
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request_body = dict(base_request_body)
        if attempt > 1:
            request_body["temperature"] = 0.2
            request_body["seed"] = 7919
        try:
            resp = await http_client.post(
                f"{SLLM_ENDPOINT}/v1/chat/completions",
                json=request_body,
                timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
            )
        except Exception as e:  # transient network error
            logger.warning("Qwen HTTP error (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, type(e).__name__)
            if attempt < MAX_ATTEMPTS:
                continue
            return {"pii_list": [], "parse_error": True}

        if resp.status_code != 200:
            if attempt < MAX_ATTEMPTS:
                continue
            return {"pii_list": [], "parse_error": True}

        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        pii_list = _parse_pii_list_from_qwen_response(content)
        if pii_list is not None:
            return {"pii_list": pii_list, "parse_error": False}

        logger.warning("Qwen parse failure (attempt %d/%d)", attempt, MAX_ATTEMPTS)
        if attempt < MAX_ATTEMPTS:
            continue
        return {"pii_list": [], "parse_error": True}

    return {"pii_list": [], "parse_error": True}


async def _detect_sllm(raw_text: str, lang: str) -> dict:
    """Detect PII with a regex-first, sLLM-supplement strategy.

    1) regex extracts fixed-format PII first
    2) the sLLM detects the rest, chunk by chunk (concurrency-limited)
    3) sLLM results are filtered for false positives and verified against the
       source text (a substring check) to drop hallucinations
    4) regex + verified-sLLM results are combined

    Returns ``{"pii_list", "chunk_total", "chunk_failed", "chunk_parse_failed"}`` or
    ``{"error", "pii_list": []}`` on a total failure (non-fatal to caller).
    """
    try:
        regex_results = _supplement_with_regex([], raw_text, lang)
        chunks = _chunk_text(raw_text, max_chars=8000, overlap=400)

        sem = asyncio.Semaphore(4)

        async def _limited_call(chunk):
            async with sem:
                return await _call_sllm_for_chunk(chunk, lang)

        chunk_results = await asyncio.gather(
            *[_limited_call(c) for c in chunks], return_exceptions=True
        )

        seen = {(item["type"], re.sub(r"\s+", "", item["original"])) for item in regex_results}
        sllm_new = []
        fail_count = 0
        parse_fail_count = 0
        for result in chunk_results:
            if isinstance(result, Exception):
                fail_count += 1
                continue
            if result.get("parse_error"):
                parse_fail_count += 1
            for item in result.get("pii_list", []):
                key = (item.get("type", ""), re.sub(r"\s+", "", item.get("original", "")))
                if key not in seen:
                    seen.add(key)
                    sllm_new.append(item)

        def _is_false_positive(item) -> bool:
            orig = item.get("original", "")
            item_type = item.get("type", "")
            if item_type in prompts.FIXED_FORMAT_TYPES:
                if re.search(r"\*{2,}|[O○0]{2}", orig):
                    return True
            if item_type == "ACCOUNT" and re.fullmatch(r"\d{3}-\d{2}-\d{5}", orig.strip()):
                return True
            return False

        sllm_filtered = [p for p in sllm_new if not _is_false_positive(p)]

        normalized_text = re.sub(r"\s+", "", raw_text)
        verified = []
        for p in sllm_filtered:
            norm_orig = re.sub(r"\s+", "", p.get("original", ""))
            if norm_orig in normalized_text:
                verified.append(p)
            elif p.get("type") == "ADDRESS" and len(norm_orig) > 10:
                # Addresses are often re-wrapped by OCR; accept on a prefix match.
                if norm_orig[:len(norm_orig) // 2] in normalized_text:
                    verified.append(p)

        final = regex_results + verified
        return {
            "pii_list": final,
            "chunk_total": len(chunks),
            "chunk_failed": fail_count,
            "chunk_parse_failed": parse_fail_count,
        }
    except Exception as e:
        logger.warning("sLLM PII detection failed (non-fatal): %s", e)
        return {"error": str(e), "pii_list": []}


# ── Masking-residue gate ─────────────────────────────────────────────────────
def _detect_masking_residue(text: str, pii_data: dict) -> set:
    """Check whether any known original PII value survived masking.

    This runs against ``pii_data`` — the same token -> original mapping Stage 5
    stores — not against a PII-shape heuristic, so it cannot false-positive on
    text that merely *looks* like PII. Two checks:

      (a) the full original value (2+ chars) appears verbatim in ``text``.
      (b) a 5+ digit run *derived from* an original value (e.g. the tail of a
          national ID or account number) appears as a substring in ``text`` —
          catches partial tokenization where only part of a fixed-format ID
          got masked.

    Returns the set of token names with suspected residue (never the original
    values themselves — callers may log this set). An empty result does not
    guarantee zero residue (this is a best-effort net, not a formal proof), but
    any hit is a real signal that Stage 3 masking was incomplete.
    """
    residue: set = set()
    for token, original in pii_data.items():
        original_s = str(original)
        if len(original_s) >= 2 and original_s in text:
            residue.add(token)
            continue
        for run in re.findall(r"\d{5,}", original_s):
            if run in text:
                residue.add(token)
                break
    return residue


def _assert_no_masking_residue(stage_label: str, text: str, pii_data: dict) -> None:
    """Abort the pipeline if any known PII value survived masking.

    Called before every point where text leaves this trust boundary (Stage 4's
    call to the managed Guardrails API, and Stage 6's call to Bedrock). Amazon
    Bedrock Guardrails is a second line of defense, not a substitute for
    complete Stage 3 masking — this gate is what actually enforces "the large
    model never sees PII" when Stage 3 missed something.
    """
    residue = _detect_masking_residue(text, pii_data)
    if residue:
        raise RuntimeError(
            f"masking residue detected before {stage_label}: {len(residue)} "
            f"token(s) still have their original value present in the text "
            f"(tokens: {sorted(residue)[:5]}) — aborting rather than risk "
            f"forwarding unmasked PII"
        )


# ── Stage 3: optional L2 result cache (DynamoDB) ────────────────────────────
_PROMPT_FINGERPRINT = {
    lang: hashlib.sha256(p.encode()).hexdigest()[:8] for lang, p in prompts.PII_PROMPTS.items()
}


def _qwen_cache_key(scenario_id: str, raw_text: str, lang: str) -> str:
    """Stable cache key for (model, prompt, lang, scenario, normalized text)."""
    normalized = re.sub(r"\s+", " ", raw_text).strip()
    payload = f"{SLLM_MODEL_NAME}|{_PROMPT_FINGERPRINT.get(lang, '')}|{lang}|{scenario_id}:{normalized}"
    return hashlib.sha256(payload.encode()).hexdigest()


async def _qwen_cache_get(cache_key: str) -> dict | None:
    if qwen_cache_table is None or QWEN_CACHE_TTL_DAYS <= 0:
        return None
    try:
        item = (await asyncio.to_thread(qwen_cache_table.get_item, Key={"cache_key": cache_key})).get("Item")
        if not item:
            return None
        expires_at = int(item.get("expires_at", 0))
        if expires_at and int(time.time()) > expires_at:
            return None
        return {
            "final_text": item["final_text"],
            "pii_data": {k: str(v) for k, v in item.get("pii_data", {}).items()},
            "pii_count": int(item.get("pii_count", 0)),
        }
    except Exception as e:
        logger.warning("Qwen cache get failed (treat as miss): %s", e)
        return None


async def _qwen_cache_put(cache_key: str, scenario_id: str, result: dict) -> None:
    if qwen_cache_table is None or QWEN_CACHE_TTL_DAYS <= 0:
        return
    try:
        await asyncio.to_thread(
            qwen_cache_table.put_item,
            Item={
                "cache_key": cache_key,
                "scenario_id": scenario_id,
                "final_text": result["final_text"],
                "pii_data": result["pii_data"],
                "pii_count": result["pii_count"],
                "created_at": int(time.time()),
                "expires_at": int(time.time()) + QWEN_CACHE_TTL_DAYS * 86400,
            },
        )
    except Exception as e:
        logger.warning("Qwen cache put failed (non-fatal): %s", e)


async def _detect_pii(scenario_id: str, raw_text: str, lang: str) -> dict:
    """Stage 3 core: detect PII and pseudonymize into tokens.

    Flow: cache lookup -> sLLM+regex detection -> pii-service /pseudonymize -> cache store.
    Returns ``{"final_text", "pii_data", "pii_count", "cache_hit"}``.

    Raises RuntimeError if detection fails entirely or if too many chunks failed —
    proceeding would risk leaking unmasked PII to the large model.
    """
    cache_key = _qwen_cache_key(scenario_id, raw_text, lang)
    cached = await _qwen_cache_get(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    sllm_result = await _detect_sllm(raw_text, lang)
    if "error" in sllm_result and not sllm_result.get("pii_list"):
        raise RuntimeError(f"PII detection failed: {sllm_result.get('error')}")

    total_chunks = sllm_result.get("chunk_total", 0)
    failed = sllm_result.get("chunk_failed", 0) + sllm_result.get("chunk_parse_failed", 0)
    # If half or more of the chunks failed, refuse to proceed: PII coverage cannot
    # be guaranteed and unmasked text might reach the large model.
    if total_chunks > 0 and failed / total_chunks >= 0.5:
        raise RuntimeError(
            f"PII detection partial failure exceeded threshold: {failed}/{total_chunks} chunks failed"
        )

    pii_list_for_pseudo = [
        {"type": item.get("type", ""), "original": item.get("original", "")}
        for item in sllm_result.get("pii_list", [])
        if item.get("original", "").strip()
    ]
    resp = await http_client.post(
        f"{PII_SERVICE_URL}/pseudonymize",
        json={"text": raw_text, "pii_list": pii_list_for_pseudo},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"/pseudonymize returned {resp.status_code}: {resp.text[:300]}")
    pseudo = resp.json()
    pseudo["cache_hit"] = False
    await _qwen_cache_put(cache_key, scenario_id, pseudo)
    return pseudo


# ── Stage 6: Bedrock analysis (direct invoke_model) ─────────────────────────
async def _bedrock_analysis(scenario_id: str, anonymized_text: str, lang: str, emit: EmitFn) -> dict:
    """Analyze the anonymized text with Amazon Bedrock (Claude)."""
    scenario_prompt = prompts.ANALYSIS_PROMPTS[lang].get(
        scenario_id, prompts.ANALYSIS_PROMPTS[lang]["insurance"]
    )
    user_prompt = prompts.ANALYSIS_WRAPPER[lang].format(
        scenario_prompt=scenario_prompt, anonymized_text=anonymized_text
    )

    request_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": user_prompt}],
    })

    await emit("BEDROCK_INPUT", {"anonymizedText": user_prompt})

    try:
        invoke_kwargs = dict(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=request_body.encode("utf-8"),
        )
        if GUARDRAIL_ID_FOR_INVOKE and GUARDRAIL_VERSION_FOR_INVOKE:
            invoke_kwargs["guardrailIdentifier"] = GUARDRAIL_ID_FOR_INVOKE
            invoke_kwargs["guardrailVersion"] = GUARDRAIL_VERSION_FOR_INVOKE
        response = await asyncio.to_thread(bedrock_runtime.invoke_model, **invoke_kwargs)
        response_body = json.loads(response["body"].read().decode("utf-8"))
        content_blocks = response_body.get("content") or []
        if not content_blocks:
            raise RuntimeError(f"Bedrock returned empty content: {response_body}")
        full_report = content_blocks[0].get("text", "")
        usage = response_body.get("usage", {})
        stop_reason = response_body.get("stop_reason")
        if stop_reason == "refusal":
            raise RuntimeError(f"Bedrock refused analysis: {full_report[:200]}")
        if stop_reason == "max_tokens":
            logger.warning("Bedrock invoke_model hit max_tokens cap (output truncated)")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Bedrock invoke_model failed: {e}")

    model_label = _short_model_alias(BEDROCK_MODEL_ID)
    await emit("BEDROCK_COMPLETE", {
        "model": model_label,
        "inputTokens": usage.get("input_tokens", 0),
        "outputTokens": usage.get("output_tokens", 0),
        "outputText": full_report,
    })
    return {
        "fullReport": full_report,
        "bedrockStats": {
            "model": model_label,
            "totalInputTokens": usage.get("input_tokens", 0),
            "totalOutputTokens": usage.get("output_tokens", 0),
        },
    }


# ── Stages 4, 5, 7: thin pii-service calls ──────────────────────────────────
async def _stage_guardrails(masked_text: str, emit: EmitFn) -> dict:
    """Stage 4: double-check the masked text with Bedrock Guardrails (non-fatal).

    Three distinct outcomes are reported, and treated as three distinct things —
    a "successful" pipeline run can still be running degraded or unconfigured:
      - ``skipped``  — GUARDRAIL_ID is not set; the pii-service never called AWS.
      - ``degraded`` — the pii-service called Guardrails and it failed (bad ARN,
        missing permissions, throttling, network error); the masked text passes
        through unverified.
      - neither      — Guardrails ran and verified the text.
    """
    await emit("GUARDRAIL_START", {})
    try:
        resp = await http_client.post(f"{PII_SERVICE_URL}/guardrails", json={"masked_text": masked_text})
        result = resp.json()
        await emit("GUARDRAIL_COMPLETE", {
            "extraPiiFound": len(result.get("extra_pii", [])),
            "skipped": result.get("skipped", False),
            "degraded": result.get("degraded", False),
        })
        return result
    except Exception as e:
        # A transport-level failure (the pii-service itself unreachable) is the
        # same "degraded" outcome as a Guardrails-API failure the pii-service
        # already caught — either way the text passes through unverified.
        logger.warning("Guardrails call failed, proceeding (degraded): %s", e)
        await emit("GUARDRAIL_COMPLETE", {"degraded": True, "error": str(e)})
        return {"extra_pii": [], "verified_text": masked_text, "degraded": True}


async def _stage_store(session_id: str, scenario_id: str, pii_data: dict, emit: EmitFn) -> int:
    """Stage 5: store token -> original mappings in DynamoDB (with TTL)."""
    await emit("PII_STORE_START", {})
    resp = await http_client.post(f"{PII_SERVICE_URL}/store", json={
        "session_id": session_id, "scenario_id": scenario_id, "pii_data": pii_data,
    })
    stored = resp.json().get("stored", 0)
    await emit("PII_STORE_COMPLETE", {"stored": stored})
    return stored


async def _stage_reassemble(session_id: str, analysis_result: str, emit: EmitFn) -> dict:
    """Stage 7: restore PII tokens in the analysis output (national IDs stay masked)."""
    await emit("REASSEMBLY_START", {})
    resp = await http_client.post(f"{PII_SERVICE_URL}/reassemble", json={
        "session_id": session_id, "analysis_result": analysis_result,
    })
    result = resp.json()
    await emit("REASSEMBLY_COMPLETE", {
        "replacedCount": result.get("replaced_count", 0),
        "unreplacedTokens": result.get("unreplaced_tokens", []),
    })
    return result


# ── Orchestrator ────────────────────────────────────────────────────────────
async def run_pipeline(
    scenario_id: str,
    raw_text: str,
    lang: str = prompts.DEFAULT_LANG,
    emit: EmitFn | None = None,
) -> dict:
    """Run Stages 3-7 on OCR'd text and return the result.

    Args:
        scenario_id: one of ``VALID_SCENARIOS``.
        raw_text:    the OCR'd document text.
        lang:        ``"ko"`` or ``"en"`` — selects detection/analysis prompts and regex.
        emit:        optional async callback ``emit(stage, data)`` for progress; every
                     event is also collected into the returned ``events`` list.

    Returns ``{sessionId, scenario, lang, finalText, piiData, summary, events}``.
    """
    if scenario_id not in VALID_SCENARIOS:
        raise ValueError(f"invalid scenario_id={scenario_id!r}; expected one of {VALID_SCENARIOS}")
    lang = prompts.normalize_lang(lang)

    events: list[dict] = []

    async def _emit(stage: str, data: dict) -> None:
        event = {"stage": stage, "timestamp": int(time.time() * 1000), "data": data}
        events.append(event)
        logger.info("stage=%s %s", stage, {k: v for k, v in data.items() if k != "anonymizedText"})
        if emit is not None:
            await emit(stage, data)

    session_id = str(uuid.uuid4())
    pipeline_start = time.time()
    await _emit("PIPELINE_START", {"sessionId": session_id, "scenarioId": scenario_id, "lang": lang})

    # Stage 3: PII detection + pseudonymization
    await _emit("PII_DETECT_START", {})
    detect_result = await _detect_pii(scenario_id, raw_text, lang)
    pii_data = detect_result.get("pii_data", {})
    masked_text = detect_result["final_text"]
    await _emit("PII_DETECT_COMPLETE", {
        "piiCount": detect_result.get("pii_count", len(pii_data)),
        "cacheHit": detect_result.get("cache_hit", False),
    })

    # Gate: refuse to send anything past this point to an external API (the
    # Guardrails managed endpoint, then Bedrock) if masking is incomplete.
    _assert_no_masking_residue("Stage 4 (Guardrails)", masked_text, pii_data)

    # Stage 4: Guardrails verification
    guard_result = await _stage_guardrails(masked_text, _emit)
    anonymized_text = guard_result.get("verified_text", masked_text)

    # Same gate again on Guardrails' own output — it may rewrite the text, and
    # a rewrite is exactly the kind of change this check exists to catch.
    _assert_no_masking_residue("Stage 6 (Bedrock analysis)", anonymized_text, pii_data)

    # Stage 5: store PII mappings
    stored_count = await _stage_store(session_id, scenario_id, pii_data, _emit)

    # Stage 6: Bedrock analysis on anonymized text
    await _emit("AGENT_START", {})
    agent_result = await _bedrock_analysis(scenario_id, anonymized_text, lang, _emit)

    # Stage 7: reassemble PII into the report
    reassembly_result = await _stage_reassemble(session_id, agent_result.get("fullReport", ""), _emit)

    elapsed_ms = int((time.time() - pipeline_start) * 1000)
    final_text = reassembly_result.get("final_text", agent_result.get("fullReport", ""))
    summary = {
        "piiDetected": detect_result.get("pii_count", len(pii_data)),
        "guardrailsExtra": len(guard_result.get("extra_pii", [])),
        "guardrailsSkipped": guard_result.get("skipped", False),
        "guardrailsDegraded": guard_result.get("degraded", False),
        "piiStored": stored_count,
        "reassembledTokens": reassembly_result.get("replaced_count", 0),
        "totalTimeMs": elapsed_ms,
    }
    await _emit("PIPELINE_COMPLETE", {"summary": summary})

    return {
        "sessionId": session_id,
        "scenario": scenario_id,
        "lang": lang,
        "finalText": final_text,
        "piiData": pii_data,
        "summary": summary,
        "events": events,
    }
