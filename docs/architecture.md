# Architecture

## Goal

Run AI analysis on financial documents **without ever exposing personal information
to the large language model**. PII is detected and masked first; the model sees only
anonymized text; then PII is restored deterministically in the final output.

## Pipeline

```mermaid
flowchart LR
    IN[PDF] --> S2

    subgraph Orchestrator
      S2[Stage 2<br/>OCR]
      S3[Stage 3<br/>PII detection]
      S4[Stage 4<br/>Guardrails verify]
      S5[Stage 5<br/>PII store]
      S6[Stage 6<br/>Bedrock analysis]
      S7[Stage 7<br/>PII reassembly]
    end

    S2 -->|text| S3
    IN2["OCR'd text<br/>(skip Stage 2)"] --> S3
    S3 -->|masked text| S4 --> S5 --> S6 -->|report w/ tokens| S7 --> OUT[Final report<br/>PII restored, IDs masked]

    S2 -.parse.-> OCR[OCR Service]
    S3 <-->|detect| SLLM[(vLLM sLLM<br/>Qwen3-8B)]
    S3 -.pseudonymize.-> PII[PII Service]
    S4 -.apply_guardrail.-> GR[Bedrock Guardrails]
    S5 -.put.-> DDB[(DynamoDB<br/>token to original)]
    S6 -.invoke_model.-> BR[Bedrock Claude]
    S7 -.query + restore.-> DDB
```

There are two entry points into the orchestrator, not one: `POST /pipeline/run`
takes already-OCR'd text directly into Stage 3 (`IN2` above); `POST
/pipeline/run-file` takes a PDF and runs Stage 2 first (`IN` above). Both paths
converge at Stage 3 onward — everything below that point is identical either way.

## Stages

| Stage | Where | What happens |
|-------|-------|--------------|
| 2 — OCR | ocr-service | Extracts text (and per-page layout elements) from a PDF. No PII handling here — this is pure text extraction, and its output is exactly what Stage 3 would otherwise receive directly. Skippable: `POST /pipeline/run` bypasses it entirely. |
| 3 — PII detection | orchestrator + pii-service | A regex pass plus an sLLM (Qwen via vLLM) extract PII end-to-end. The pii-service `/pseudonymize` turns each value into a stable token (`[PERSON_1]`, `[SSN_1]`, …). If too many detection chunks fail, the pipeline aborts rather than risk leaking unmasked text. |
| 4 — Guardrails | pii-service | Amazon Bedrock Guardrails re-scans the masked text as a second line of defense. Non-fatal: passes through (marked `skipped` or `degraded`) if unconfigured or on error — see [Security notes](../README.md#security-notes) for why that distinction matters. |
| 5 — PII store | pii-service | The token→original mapping is written to DynamoDB with a short TTL. |
| 6 — Bedrock analysis | orchestrator | Amazon Bedrock (Claude) analyzes the **anonymized** text and writes a report that keeps the PII tokens verbatim. |
| 7 — Reassembly | pii-service | Tokens in the report are replaced by their originals via deterministic string replacement. National IDs (RRN/SSN) are only **partially** restored — trailing digits stay masked. |

## Why "the large model never sees PII"

- Detection + masking (Stages 3–4) happen **before** any call to the large model.
- A masking-residue check re-verifies this immediately before each external call
  (to Guardrails, then to Bedrock) by re-scanning the text against the known
  token → original mapping — so this isn't just "masking runs first," it's
  actively checked at the boundary, and the pipeline aborts rather than send
  through a hit.
- The large model (Stage 6) is invoked only on the masked/verified text.
- Restoration (Stage 7) is a pure string operation on the model's output, using the
  mapping stored in Stage 5 — the model is never involved in un-masking.
- OCR (Stage 2) sits entirely outside this boundary — it never receives or
  produces anonymized data, it just extracts the raw text that Stage 3 then masks.

## Components

- **ocr-service** (`src/ocr-service`) — FastAPI service exposing `POST /parse`. Two engines behind `OCR_ENGINE`: `paddleocr` (default, CPU-only classic PaddleOCR) or `paddleocr-vl` (opt-in, GPU, layout-aware — see [`deploy/ocr-gpu/`](../deploy/ocr-gpu/)).
- **orchestrator** (`src/orchestrator`) — FastAPI service exposing `POST /pipeline/run` (text) and `POST /pipeline/run-file` (PDF); runs Stages 2–7, calling the ocr-service, the sLLM, and Bedrock directly, and the pii-service over HTTP.
- **pii-service** (`src/pii-service`) — FastAPI service with `/pseudonymize`, `/guardrails`, `/store`, `/reassemble`. CPU-only, no ML model dependency.
- **prompts** (`src/orchestrator/prompts.py`) — all language-specific assets (detection prompt, regex set, analysis prompt), keyed by `lang` (`ko` | `en`).

## Language support

The same Qwen sLLM serves both Korean and English (it is multilingual). The `lang`
parameter selects the detection prompt, regex supplement, and analysis prompt. The
Korean set detects RRN / Korean addresses; the English set detects SSN / US addresses.
