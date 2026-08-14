"""GPU OCR engine — PaddleOCR-VL via a self-hosted vLLM ``/layout-parsing`` endpoint.

Opt-in: set ``OCR_ENGINE=paddleocr-vl`` and point ``PADDLE_VL_ENDPOINT`` at your
GPU host. See ``deploy/ocr-gpu/`` for a Dockerfile that pre-bakes the model
into a vLLM-serving image (PaddleOCR-VL is served, not called as a hosted API
— there is no managed endpoint for it).

This gives layout-aware OCR (tables, figures, reading order) instead of
``engine_paddleocr.py``'s plain line-by-line text. Both are Apache-2.0.

The ``result.layoutParsingResults[]`` response shape follows PaddleOCR-VL's
published API docs; if your served version differs, adjust the field lookups
below (they degrade gracefully to empty text/elements rather than raising, so
a shape mismatch shows up as empty OCR output rather than a crash).
"""

from __future__ import annotations

import base64
import os

import httpx

PADDLE_VL_ENDPOINT = os.environ.get("PADDLE_VL_ENDPOINT", "http://localhost:8080")

# PaddleOCR-VL's layout label -> a small set of display categories. Unmapped
# labels pass through as-is.
_CATEGORY_MAP = {
    "text": "paragraph",
    "paragraph_title": "paragraph",
    "abstract": "paragraph",
    "reference": "paragraph",
    "doc_title": "paragraph",
    "table": "table",
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "header": "header",
    "footer": "footer",
    "formula": "equation",
    "figure_title": "caption",
    "table_title": "caption",
    "figure_caption": "caption",
    "table_caption": "caption",
}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _bbox_to_norm(bbox: list, page_width: float, page_height: float) -> list[float]:
    """PaddleX ``block_bbox`` ([x1, y1, x2, y2] in pixels) -> normalized 0..1."""
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    w = page_width or 1.0
    h = page_height or 1.0
    return [_clamp(x1 / w), _clamp(y1 / h), _clamp(x2 / w), _clamp(y2 / h)]


async def run(pdf_bytes: bytes) -> dict:
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)) as client:
        resp = await client.post(
            f"{PADDLE_VL_ENDPOINT}/layout-parsing",
            json={"file": pdf_b64, "fileType": 0, "visualize": False},
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})

    pages: list[dict] = []
    text_parts: list[str] = []
    for page in result.get("layoutParsingResults", []):
        md_text = page.get("markdown", {}).get("text", "").strip()
        if md_text:
            text_parts.append(md_text)

        pruned = page.get("prunedResult", {})
        page_w = pruned.get("input_img_w") or 1.0
        page_h = pruned.get("input_img_h") or 1.0

        elements: list[dict] = []
        for block in pruned.get("parsing_res_list", []):
            # Isolated per block: one malformed block (missing bbox, etc.)
            # drops that block, not the whole page/document.
            try:
                bbox = block.get("block_bbox")
                content = (block.get("block_content") or "").strip()
                if not bbox or len(bbox) < 4 or not content:
                    continue
                label = block.get("block_label", "")
                elements.append({
                    "category": _CATEGORY_MAP.get(label, label),
                    "text": content,
                    "bbox": _bbox_to_norm(bbox, page_w, page_h),
                })
            except Exception:
                continue

        pages.append({"width": float(page_w), "height": float(page_h), "elements": elements})

    return {"engine": "paddleocr-vl", "text": "\n\n".join(text_parts), "pages": pages}
