# Privacy-Preserving Document AI with Amazon Bedrock

Run AI analysis on financial documents **without ever exposing personal information
(PII) to the large language model.** PII is detected and masked first, the model
analyzes only anonymized text, and the original values are restored deterministically
in the final output — so the large model never sees who the document is about.

This sample implements the **post-OCR** stages of such a pipeline: given the text of
a document (OCR is assumed already done), it detects and masks PII, verifies the
masking with Amazon Bedrock Guardrails, runs analysis on Amazon Bedrock (Claude),
and reassembles the PII into the report.

> Sample input documents are provided in both **Korean** and **English**. The PII
> detection prompt, regex set, and analysis prompt are selected by a `lang` parameter.

## How it works

```
OCR'd text ─▶ [3] PII detection ─▶ [4] Guardrails ─▶ [5] PII store ─▶ [6] Bedrock analysis ─▶ [7] Reassembly ─▶ report
                 detect + mask        verify masking     DynamoDB         anonymized text only      restore tokens
```

| Stage | Description |
|-------|-------------|
| 3. PII detection | A regex pass + an sLLM (Qwen via vLLM) extract PII; values become `[PERSON_1]`, `[SSN_1]`, … tokens. |
| 4. Guardrails | Amazon Bedrock Guardrails re-scans the masked text (second line of defense). |
| 5. PII store | The token→original map is stored in DynamoDB with a short TTL. |
| 6. Bedrock analysis | Amazon Bedrock (Claude) analyzes the **anonymized** text. |
| 7. Reassembly | Tokens in the report are restored; national IDs (RRN/SSN) stay partially masked. |

See [`docs/architecture.md`](docs/architecture.md) for a diagram and details.

## Repository layout

```
src/orchestrator/   FastAPI service — runs Stages 3–7 (POST /pipeline/run)
src/pii-service/    FastAPI service — pseudonymize / guardrails / store / reassemble
sample-documents/   pre-OCR'd sample inputs (ko/ and en/)
scripts/            run_pipeline.py CLI
deploy/             docker-compose + minimal CDK (DynamoDB + Guardrail + IAM)
tests/              pure-function unit tests (no AWS/network needed)
```

## Prerequisites

1. **AWS account** with Amazon Bedrock model access enabled for a Claude model in
   your region (e.g. `global.anthropic.claude-sonnet-4-6`), and credentials available
   to the services (env vars, `~/.aws`, or an IAM role).
2. **A vLLM endpoint** serving the PII-detection sLLM. This sample uses Qwen3-8B:
   ```bash
   # on a GPU host
   pip install vllm
   vllm serve Qwen/Qwen3-8B --port 8000
   ```
   Set `SLLM_ENDPOINT` to its URL (default `http://localhost:8000`).
3. **A DynamoDB table** for PII mappings (and optionally one for the sLLM cache).
   Create them with the CDK stack below, or via the CLI:
   ```bash
   aws dynamodb create-table --table-name pii-mappings \
     --attribute-definitions AttributeName=session_id,AttributeType=S AttributeName=token,AttributeType=S \
     --key-schema AttributeName=session_id,KeyType=HASH AttributeName=token,KeyType=RANGE \
     --billing-mode PAY_PER_REQUEST
   aws dynamodb update-time-to-live --table-name pii-mappings \
     --time-to-live-specification "Enabled=true,AttributeName=ttl"
   ```
4. *(Optional)* **A Bedrock Guardrail** for Stage 4. If `GUARDRAIL_ID` is unset, Stage 4
   passes through (the pipeline still runs). If you do set it, also set
   `GUARDRAIL_VERSION` — the pii-service refuses to start with one set and not
   the other.

## Provision infrastructure (optional, CDK)

```bash
cd deploy/cdk
npm install
npx cdk deploy        # creates pii-mappings, qwen-cache, a Guardrail, and an IAM policy
```
Outputs include the table names and the Guardrail ID to plug into the env below.

## Run locally

```bash
# 1. Make sure your vLLM endpoint and AWS credentials are available.
export AWS_REGION=us-east-1
export GUARDRAIL_ID=<your-guardrail-id>      # optional
export GUARDRAIL_VERSION=<version>           # required if GUARDRAIL_ID is set
export SLLM_ENDPOINT=http://host.docker.internal:8000

# 2. Start both services.
docker compose -f deploy/docker-compose.yml up --build

# 3. Run a sample document through the pipeline.
python scripts/run_pipeline.py --scenario insurance --lang ko
python scripts/run_pipeline.py --scenario mortgage  --lang en
```

`run_pipeline.py` prints the per-stage events, a summary, and the final report with
PII restored (national IDs partially masked).

### Run the services without Docker

```bash
pip install -r src/pii-service/requirements.txt
( cd src/pii-service && uvicorn main:app --port 8082 ) &

pip install -r src/orchestrator/requirements.txt
( cd src/orchestrator && PII_SERVICE_URL=http://localhost:8082 uvicorn main:app --port 8080 ) &
```

## API

`POST /pipeline/run`

```json
{ "scenario": "insurance", "text": "<OCR'd document text>", "lang": "en" }
```

Response (abridged):

```json
{
  "sessionId": "…",
  "finalText": "…report with PII restored…",
  "piiData": { "PERSON_1": "Michael Carter", "SSN_1": "521-84-6390" },
  "summary": { "piiDetected": 9, "reassembledTokens": 9, "totalTimeMs": 4210 },
  "events": [ { "stage": "PII_DETECT_COMPLETE", "data": { … } }, … ]
}
```

## Configuration

| Variable | Service | Default | Purpose |
|----------|---------|---------|---------|
| `AWS_REGION` | both | `us-east-1` | AWS region |
| `MODEL_ID` | orchestrator | `global.anthropic.claude-sonnet-5` | Bedrock model for analysis |
| `SLLM_ENDPOINT` | orchestrator | `http://localhost:8000` | vLLM OpenAI-compatible endpoint |
| `SLLM_MODEL_NAME` | orchestrator | `Qwen/Qwen3-8B` | sLLM model name |
| `PII_SERVICE_URL` | orchestrator | `http://localhost:8082` | pii-service URL |
| `QWEN_CACHE_TABLE` | orchestrator | *(empty = disabled)* | optional DynamoDB result cache |
| `GUARDRAIL_ID` | both | *(empty = skip)* | Bedrock Guardrail id — pii-service uses it for Stage 4 `ApplyGuardrail`; the orchestrator, if also set, attaches the same guardrail to the Stage 6 `invoke_model` call as defense-in-depth |
| `GUARDRAIL_VERSION` | both | *(empty)* | Guardrail version to verify against. Required if `GUARDRAIL_ID` is set — the pii-service refuses to start on a half-configured pair rather than silently defaulting to `"DRAFT"` |
| `PII_TABLE` | pii-service | `pii-mappings` | DynamoDB table for mappings |
| `PII_TTL_SECONDS` | pii-service | `3600` | TTL for stored mappings |

## Tests

```bash
pip install pytest
pytest -q          # pure-function tests; no AWS or network required
```

## Adding another language

1. Add `sample-documents/<lang>/<scenario>.txt`.
2. In `src/orchestrator/prompts.py`, add a `<lang>` entry to `PII_PROMPTS`,
   `REGEX_SETS`, `ANALYSIS_PROMPTS`, and `ANALYSIS_WRAPPER`.
3. Pass `lang=<lang>` to `/pipeline/run`.

## Extending this sample

The focus of this sample is the **privacy-preserving handling of PII** (detection,
masking, storage, and reassembly) — Stages 3–5 and 7. Stage 6 (analysis) is
intentionally a **single Amazon Bedrock `invoke_model` call** on the anonymized text,
which is all that is needed to demonstrate that the large model never sees PII.

If your use case needs richer analysis, Stage 6 is the natural extension point. You
can replace the single call with, for example:

- a multi-step chain of Bedrock calls (e.g. analyst → assessor → reporter), or
- a multi-agent runtime such as Amazon Bedrock AgentCore.

As long as the analysis only ever receives the anonymized text and preserves the PII
tokens, the privacy guarantee is unchanged. The swap is isolated to
`_bedrock_analysis` in `src/orchestrator/pipeline.py`.

## Cleanup

```bash
docker compose -f deploy/docker-compose.yml down
cd deploy/cdk && npx cdk destroy
```

## Security notes

- The pipeline aborts Stage 3 if too many detection chunks fail, rather than risk
  forwarding unmasked text to the large model.
- Before both Stage 4 (Guardrails) and Stage 6 (Bedrock), a masking-residue check
  re-scans the text against the known token → original mapping and aborts the
  pipeline if any original value is still present — Guardrails is a second line
  of defense, not a substitute for Stage 3 catching everything itself.
- Stage 4's outcome is reported as one of three distinct states: verified,
  `skipped` (Guardrail not configured — never called AWS), or `degraded`
  (Guardrail was configured but the call failed) — the latter two both mean the
  text reached Bedrock *unverified* by Guardrails, which is worth knowing even
  though the pipeline still completes either way.
- Stored PII mappings carry a short TTL; restoration is a pure string operation that
  never re-invokes the model.
- The minimal IAM policy in the CDK stack uses `bedrock:InvokeModel` on `*` for
  brevity — **narrow it to your model / inference-profile ARN in production.**
- All sample names, IDs, accounts, and figures are fictitious.

## License

This sample is licensed under the MIT-0 License. See [LICENSE](LICENSE).
