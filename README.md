# Privacy-Preserving Document AI with Amazon Bedrock

Run AI analysis on financial documents **without ever exposing personal information
(PII) to the large language model.** PII is detected and masked first, the model
analyzes only anonymized text, and the original values are restored deterministically
in the final output — so the large model never sees who the document is about.

This sample implements a full document-to-report pipeline: given a PDF (or
already-OCR'd text — OCR is optional, not assumed), it extracts text, detects and
masks PII, verifies the masking with Amazon Bedrock Guardrails, runs analysis on
Amazon Bedrock (Claude), and reassembles the PII into the report.

> Sample input documents are provided in both **Korean** and **English**, as both
> `.txt` (already-OCR'd) and `.pdf` (run through Stage 2 yourself). The PII
> detection prompt, regex set, and analysis prompt are selected by a `lang` parameter.

## How it works

```
PDF ─▶ [2] OCR ─┐
                ├─▶ [3] PII detection ─▶ [4] Guardrails ─▶ [5] PII store ─▶ [6] Bedrock analysis ─▶ [7] Reassembly ─▶ report
OCR'd text ─────┘      detect + mask        verify masking     DynamoDB         anonymized text only      restore tokens
```

| Stage | Description |
|-------|-------------|
| 2. OCR *(optional)* | Extracts text from a PDF. Skip this and call `/pipeline/run` directly if you already have OCR'd text. |
| 3. PII detection | A regex pass + an sLLM (Qwen via vLLM) extract PII; values become `[PERSON_1]`, `[SSN_1]`, … tokens. |
| 4. Guardrails | Amazon Bedrock Guardrails re-scans the masked text (second line of defense). |
| 5. PII store | The token→original map is stored in DynamoDB with a short TTL. |
| 6. Bedrock analysis | Amazon Bedrock (Claude) analyzes the **anonymized** text. |
| 7. Reassembly | Tokens in the report are restored; national IDs (RRN/SSN) stay partially masked. |

See [`docs/architecture.md`](docs/architecture.md) for a diagram and details.

## Repository layout

```
src/ocr-service/    FastAPI service — Stage 2 (POST /parse); CPU by default, GPU optional
src/orchestrator/   FastAPI service — runs Stages 2–7 (POST /pipeline/run, /pipeline/run-file)
src/pii-service/    FastAPI service — pseudonymize / guardrails / store / reassemble
sample-documents/   sample inputs: pre-OCR'd .txt and matching .pdf, ko/ and en/
scripts/            run_pipeline.py CLI, generate_sample_pdfs.mjs (regenerates the PDFs)
deploy/             docker-compose (local) + CDK (ECS Fargate) + k8s/ (plain manifests) + ocr-gpu/ (opt-in GPU OCR image)
tests/              pure-function unit tests (no AWS/network needed)
```

## Prerequisites

1. **AWS account** with Amazon Bedrock model access enabled for a Claude model in
   your region (e.g. `global.anthropic.claude-sonnet-5`), and credentials available
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

**Nothing extra for Stage 2 (OCR) by default** — `docker compose up` alone gets you
a working ocr-service on CPU (classic PaddleOCR). A GPU OCR engine
(`paddleocr-vl`, layout-aware) is available but opt-in — see
["GPU OCR (optional)"](#gpu-ocr-optional) below; it costs real money per hour
it runs, so it's not part of the default path.

## Provision infrastructure (optional, CDK)

```bash
cd deploy/cdk
npm install
npx cdk deploy        # creates pii-mappings, qwen-cache, a Guardrail (+ a published version), and an IAM policy
```
Outputs include the table names and the Guardrail ID to plug into the env below.

`cdk deploy` (no flags) also stands up the three services on ECS Fargate behind a
public ALB — see ["Deploy to AWS"](#deploy-to-aws) below for what that includes and
what it costs.

## Run locally

```bash
# 1. Make sure your vLLM endpoint and AWS credentials are available.
export AWS_REGION=us-east-1
export GUARDRAIL_ID=<your-guardrail-id>      # optional
export GUARDRAIL_VERSION=<version>           # required if GUARDRAIL_ID is set
export SLLM_ENDPOINT=http://host.docker.internal:8000

# 2. Start all three services (ocr-service, pii-service, orchestrator).
docker compose -f deploy/docker-compose.yml up --build

# 3. Run a sample document through the pipeline — either already-OCR'd text...
python scripts/run_pipeline.py --scenario insurance --lang ko
python scripts/run_pipeline.py --scenario mortgage  --lang en

# ...or a PDF, which also exercises Stage 2:
python scripts/run_pipeline.py --scenario insurance --lang ko --pdf sample-documents/ko/insurance.pdf
```

`run_pipeline.py` prints the per-stage events, a summary, and the final report with
PII restored (national IDs partially masked).

### Run the services without Docker

```bash
pip install -r src/ocr-service/requirements.txt   # also needs poppler-utils (pdf2image) — see src/ocr-service/Dockerfile
( cd src/ocr-service && uvicorn main:app --port 8083 ) &

pip install -r src/pii-service/requirements.txt
( cd src/pii-service && uvicorn main:app --port 8082 ) &

pip install -r src/orchestrator/requirements.txt
( cd src/orchestrator && OCR_SERVICE_URL=http://localhost:8083 PII_SERVICE_URL=http://localhost:8082 uvicorn main:app --port 8080 ) &
```

## API

`POST /pipeline/run` — Stages 3–7, given text you already OCR'd yourself:

```json
{ "scenario": "insurance", "text": "<OCR'd document text>", "lang": "en" }
```

`POST /pipeline/run-file` — Stage 2 (OCR) then Stages 3–7, given a PDF
(`multipart/form-data`: `scenario`, `lang`, `file`).

Both return the same shape (abridged):

```json
{
  "sessionId": "…",
  "finalText": "…report with PII restored…",
  "piiData": { "PERSON_1": "Michael Carter", "SSN_1": "521-84-6390" },
  "summary": { "piiDetected": 9, "reassembledTokens": 9, "totalTimeMs": 4210 },
  "events": [ { "stage": "PII_DETECT_COMPLETE", "data": { … } }, … ]
}
```

`ocr-service` itself exposes `POST /parse` (`multipart/form-data`: `file`),
returning `{engine, text, pages: [{width, height, elements: [{category, text, bbox}]}]}`
— that's what the orchestrator calls internally for Stage 2, but you can call
it directly if you only want OCR.

## Configuration

| Variable | Service | Default | Purpose |
|----------|---------|---------|---------|
| `AWS_REGION` | orchestrator, pii-service | `us-east-1` | AWS region |
| `MODEL_ID` | orchestrator | `global.anthropic.claude-sonnet-5` | Bedrock model for analysis |
| `OCR_SERVICE_URL` | orchestrator | `http://localhost:8083` | ocr-service URL, used by `/pipeline/run-file` |
| `SLLM_ENDPOINT` | orchestrator | `http://localhost:8000` | vLLM OpenAI-compatible endpoint |
| `SLLM_MODEL_NAME` | orchestrator | `Qwen/Qwen3-8B` | sLLM model name |
| `PII_SERVICE_URL` | orchestrator | `http://localhost:8082` | pii-service URL |
| `QWEN_CACHE_TABLE` | orchestrator | *(empty = disabled)* | optional DynamoDB result cache |
| `GUARDRAIL_ID` | orchestrator, pii-service | *(empty = skip)* | Bedrock Guardrail id — pii-service uses it for Stage 4 `ApplyGuardrail`; the orchestrator, if also set, attaches the same guardrail to the Stage 6 `invoke_model` call as defense-in-depth |
| `GUARDRAIL_VERSION` | orchestrator, pii-service | *(empty)* | Guardrail version to verify against. Required if `GUARDRAIL_ID` is set — the pii-service refuses to start on a half-configured pair rather than silently defaulting to `"DRAFT"` |
| `PII_TABLE` | pii-service | `pii-mappings` | DynamoDB table for mappings |
| `PII_TTL_SECONDS` | pii-service | `3600` | TTL for stored mappings |
| `OCR_ENGINE` | ocr-service | `paddleocr` | `paddleocr` (default, CPU) or `paddleocr-vl` (opt-in, GPU) |
| `OCR_LANG` | ocr-service | `korean` | Language model for the `paddleocr` engine (see [PaddleOCR's supported languages](https://github.com/PaddlePaddle/PaddleOCR)) |
| `OCR_DPI` | ocr-service | `200` | PDF→image render resolution for the `paddleocr` engine |
| `PADDLE_VL_ENDPOINT` | ocr-service | `http://localhost:8080` | vLLM PaddleOCR-VL endpoint, used only when `OCR_ENGINE=paddleocr-vl` |

## Tests

```bash
pip install pytest
pytest -q          # pure-function tests; no AWS or network required
```

## Deploy to AWS

Two options — pick one, they're independent of each other and both build the
same three container images from source.

### ECS Fargate (CDK)

```bash
cd deploy/cdk
npm install
npx cdk deploy
```

This is the same `cdk deploy` from ["Provision infrastructure"](#provision-infrastructure-optional-cdk)
above — it also stands up:
- A VPC (public subnets only, no NAT Gateway — see the cost note below) and an ECS cluster.
- The three services on Fargate, wired together over ECS Service Connect.
  ocr-service and pii-service are reachable only from inside the cluster.
- A public Application Load Balancer in front of the orchestrator only —
  the `OrchestratorUrl` stack output.
- The Bedrock Guardrail's numbered version and IDs are auto-wired into
  pii-service's and the orchestrator's environment — no manual "plug the
  Guardrail ID in" step, unlike the docker-compose path.

**What it costs**: no NAT Gateway, so idle cost is close to just the Fargate
tasks (3 small tasks, ~0.25 vCPU/0.5GB each) and the ALB — order of $30–50/month
if left running continuously. `cdk destroy` when you're done (see
["Cleanup"](#cleanup)).

**Not created**: the vLLM sLLM endpoint (bring your own GPU host — this is a
separate model from OCR, see [Prerequisites](#prerequisites)).

### Kubernetes (plain manifests)

```bash
# Build and push the three images yourself first (e.g. to ECR), then:
cd deploy/k8s
kustomize edit set image IMAGE_PLACEHOLDER/ocr-service=<your-repo>/ocr-service:<tag>
kustomize edit set image IMAGE_PLACEHOLDER/pii-service=<your-repo>/pii-service:<tag>
kustomize edit set image IMAGE_PLACEHOLDER/orchestrator=<your-repo>/orchestrator:<tag>
kubectl apply -k .
```

Plain `Deployment`/`Service` manifests — no Helm chart, no operator, no GitOps
controller, no cluster-specific extras (works on EKS or any other cluster; the
`ServiceAccount`'s IRSA annotation is EKS-specific and safe to remove/replace
on another cluster's credential model). See [`deploy/k8s/kustomization.yaml`](deploy/k8s/kustomization.yaml)
for the full instructions, including the optional GPU OCR deployment.

### GPU OCR (optional)

`paddleocr-vl` (layout-aware OCR — tables, figures, reading order) needs a
GPU and is off by default in both paths above:

- **ECS**: `cdk deploy -c enableGpuOcr=true` adds a `g4dn.xlarge`-backed ECS
  EC2 service running [`deploy/ocr-gpu/`](deploy/ocr-gpu/). This bills by the
  hour the instance is running, not by request — it's a real, continuous
  charge the moment it's on, unlike the Fargate services above.
- **Kubernetes**: apply [`deploy/k8s/ocr-service-gpu.yaml`](deploy/k8s/ocr-service-gpu.yaml)
  separately, on a cluster with a GPU node pool and the NVIDIA device plugin.

Either way, point `ocr-service` at it afterward: `OCR_ENGINE=paddleocr-vl` and
`PADDLE_VL_ENDPOINT=http://paddleocr-vl:<port>`.

## Adding another language

1. Add `sample-documents/<lang>/<scenario>.txt` (and, if you want to exercise
   Stage 2 OCR too, a matching `.pdf` — see
   ["Regenerating the sample PDFs"](#regenerating-the-sample-pdfs)).
2. In `src/orchestrator/prompts.py`, add a `<lang>` entry to `PII_PROMPTS`,
   `REGEX_SETS`, `ANALYSIS_PROMPTS`, and `ANALYSIS_WRAPPER`.
3. Pass `lang=<lang>` to `/pipeline/run` or `/pipeline/run-file`.

## Regenerating the sample PDFs

The `.txt` files under `sample-documents/` are the source of truth; the
matching `.pdf` files are generated from them:

```bash
npx playwright install chromium
node scripts/generate_sample_pdfs.mjs
```

Rendering Korean text needs a CJK-capable font available to the browser doing
the rendering (e.g. `fonts-noto-cjk` on Debian/Ubuntu) — without one, Hangul
renders as blank boxes. Most contributors won't need to run this: only if
you're adding or editing a sample document.

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

# If you ran `cdk deploy` (ECS Fargate path):
cd deploy/cdk && npx cdk destroy

# If you ran `kubectl apply -k` (Kubernetes path):
kubectl delete -k deploy/k8s/
kubectl delete -f deploy/k8s/ocr-service-gpu.yaml   # if you applied it
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
- OCR (Stage 2) does no PII handling and sits entirely outside the "large model
  never sees PII" boundary — it just extracts raw text, the same raw text you'd
  otherwise hand to `/pipeline/run` yourself. Don't expose `ocr-service`
  publicly on its own; the ECS/Kubernetes deploy options above don't.
- All sample names, IDs, accounts, and figures are fictitious.

## License

This sample is licensed under the MIT-0 License. See [LICENSE](LICENSE).
Third-party model licenses (all Apache-2.0) are listed in [THIRD-PARTY.md](THIRD-PARTY.md).
