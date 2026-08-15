"""
Orchestrator service — FastAPI entry point.

Exposes the privacy-preserving pipeline as a simple REST API:

  POST /pipeline/run       run Stages 3-7 on OCR'd text and return the result
  POST /pipeline/run-file  run Stage 2 (OCR, via the ocr-service) then Stages 3-7 on a PDF
  GET  /health             readiness/liveness probe
"""

import logging
import os

import prompts
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pipeline import VALID_SCENARIOS, run_pipeline, run_pipeline_from_pdf
from pydantic import BaseModel, Field

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI(
    title="Privacy-Preserving Document AI — Orchestrator",
    version="1.0.0",
    description="Detect and mask PII before analysis, run Bedrock on anonymized text, then reassemble PII.",
)

# CORS — set ALLOWED_ORIGINS (comma-separated) for browser clients; defaults to localhost.
_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRequest(BaseModel):
    scenario: str = Field(..., description=f"One of {VALID_SCENARIOS}")
    text: str = Field(..., description="OCR'd document text")
    lang: str = Field(prompts.DEFAULT_LANG, description='Language: "ko" or "en"')


@app.get("/health")
async def health():
    return {"status": "ok", "service": "orchestrator", "scenarios": list(VALID_SCENARIOS)}


@app.post("/pipeline/run")
async def pipeline_run(req: PipelineRequest):
    if req.scenario not in VALID_SCENARIOS:
        return JSONResponse(
            {"error": f"Invalid scenario: {req.scenario}", "valid": list(VALID_SCENARIOS)},
            status_code=400,
        )
    try:
        result = await run_pipeline(req.scenario, req.text, lang=req.lang)
        return JSONResponse(result)
    except Exception as e:
        logging.exception("pipeline failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/pipeline/run-file")
async def pipeline_run_file(
    scenario: str = Form(...),
    lang: str = Form(prompts.DEFAULT_LANG),
    file: UploadFile = File(...),
):
    if scenario not in VALID_SCENARIOS:
        return JSONResponse(
            {"error": f"Invalid scenario: {scenario}", "valid": list(VALID_SCENARIOS)},
            status_code=400,
        )
    try:
        pdf_bytes = await file.read()
        result = await run_pipeline_from_pdf(scenario, pdf_bytes, lang=lang)
        return JSONResponse(result)
    except Exception as e:
        logging.exception("pipeline failed")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    # Binds all interfaces by design — the service runs inside a container. nosec B104
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8080")))  # nosec B104
