"""
OCR Service — Stage 2 of the privacy-preserving pipeline.

POST /parse: receives a PDF, returns extracted text plus per-page layout
elements (text + normalized bbox). This is purely text extraction — no PII
handling happens here. The orchestrator hands the returned ``text`` to Stage 3
(PII detection) unchanged.

Engine selection (OCR_ENGINE):
  paddleocr     (default) — classic PaddleOCR, CPU-only. No GPU needed, so
                 `docker compose up` reproduces the full pipeline end to end.
  paddleocr-vl  (opt-in, GPU) — layout-aware OCR via a self-hosted vLLM
                 PaddleOCR-VL endpoint. See deploy/ocr-gpu/.
Both engines are Apache-2.0.
"""

import asyncio
import logging
import os

import engine_paddleocr
import engine_paddleocr_vl
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

OCR_ENGINE = os.environ.get("OCR_ENGINE", "paddleocr").strip().lower()
if OCR_ENGINE not in ("paddleocr", "paddleocr-vl"):
    raise ValueError(f"OCR_ENGINE must be 'paddleocr' or 'paddleocr-vl', got {OCR_ENGINE!r}")

app = FastAPI(
    title="Privacy-Preserving Document AI — OCR Service",
    version="1.0.0",
    description="PDF -> text + layout elements. No PII handling; that is Stage 3, in the pii-service.",
)


class OcrElement(BaseModel):
    category: str = Field(..., description='e.g. "paragraph", "table", "figure"')
    text: str
    bbox: list[float] = Field(..., description="[x0, y0, x1, y1], normalized 0..1")


class OcrPage(BaseModel):
    width: float
    height: float
    elements: list[OcrElement]


class OcrResponse(BaseModel):
    engine: str
    text: str = Field(..., description="Full document text, in reading order")
    pages: list[OcrPage]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ocr-service", "engine": OCR_ENGINE}


@app.post("/parse", response_model=OcrResponse)
async def parse(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    if OCR_ENGINE == "paddleocr-vl":
        result = await engine_paddleocr_vl.run(pdf_bytes)
    else:
        # CPU-bound (image decode + model inference) — offload so it doesn't
        # block the event loop for other requests.
        result = await asyncio.to_thread(engine_paddleocr.run, pdf_bytes)
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    # Binds all interfaces by design — the service runs inside a container. nosec B104
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8083")))  # nosec B104
