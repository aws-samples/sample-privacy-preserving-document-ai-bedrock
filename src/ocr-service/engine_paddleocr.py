"""CPU OCR engine — classic PaddleOCR (PP-OCRv5 detection + recognition).

This is the sample's default engine: no GPU required, so ``docker compose up``
alone reproduces the full pipeline end to end. It returns line-level text and
boxes (no layout classification — every element is category "paragraph"),
which is enough to produce the OCR'd text the rest of the pipeline consumes.

For higher-fidelity layout-aware OCR (tables, figures, reading order), see
``engine_paddleocr_vl.py`` (GPU, opt-in).
"""

from __future__ import annotations

import functools
import os

import numpy as np
from pdf2image import convert_from_bytes

OCR_LANG = os.environ.get("OCR_LANG", "korean")
OCR_DPI = int(os.environ.get("OCR_DPI", "200"))


@functools.lru_cache(maxsize=1)
def _get_ocr():
    # Imported lazily and cached: constructing PaddleOCR loads its models from
    # disk/cache, which is slow enough that doing it once per process (not per
    # request) matters, and importing it eagerly would slow down every import
    # of this module (including in engine_paddleocr_vl-only deployments).
    from paddleocr import PaddleOCR

    return PaddleOCR(use_textline_orientation=True, lang=OCR_LANG)


def run(pdf_bytes: bytes) -> dict:
    """Extract text from a PDF. Synchronous/CPU-bound — call via asyncio.to_thread."""
    ocr = _get_ocr()
    images = convert_from_bytes(pdf_bytes, dpi=OCR_DPI)

    pages: list[dict] = []
    text_parts: list[str] = []
    for image in images:
        width, height = image.size
        results = ocr.predict(np.array(image))

        elements: list[dict] = []
        for res in results:
            texts = res.get("rec_texts", []) or []
            boxes = res.get("rec_polys", []) or res.get("dt_polys", []) or []
            for text, box in zip(texts, boxes):
                # box is a numpy array of [x, y] corner points — cast to plain
                # floats, since numpy scalars aren't JSON-serializable as-is.
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                bbox = [min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height]
                elements.append({"category": "paragraph", "text": str(text), "bbox": bbox})
                text_parts.append(str(text))

        pages.append({"width": float(width), "height": float(height), "elements": elements})

    return {"engine": "paddleocr", "text": "\n".join(text_parts), "pages": pages}
