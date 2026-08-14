"""
Unit tests for the ocr-service's pure logic — no PDF rendering, model, or
network required. (engine_paddleocr.py needs paddleocr/pdf2image installed
and is exercised via `docker compose up` + scripts/run_pipeline.py --pdf
instead — see the README.)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "ocr-service"))

import engine_paddleocr_vl  # noqa: E402


def test_bbox_to_norm_scales_and_clamps():
    # 100x200 pixel box inside a 200x400 page -> exactly the [0.5, 0.5] quadrant.
    bbox = engine_paddleocr_vl._bbox_to_norm([0, 0, 100, 200], page_width=200, page_height=400)
    assert bbox == [0.0, 0.0, 0.5, 0.5]


def test_bbox_to_norm_clamps_out_of_range():
    # A box extending past the page (bad upstream data) still lands in 0..1.
    bbox = engine_paddleocr_vl._bbox_to_norm([-10, -10, 300, 300], page_width=200, page_height=200)
    assert bbox == [0.0, 0.0, 1.0, 1.0]


def test_category_map_known_and_unknown_labels():
    assert engine_paddleocr_vl._CATEGORY_MAP["table"] == "table"
    assert engine_paddleocr_vl._CATEGORY_MAP["doc_title"] == "paragraph"
    assert "made_up_label" not in engine_paddleocr_vl._CATEGORY_MAP  # falls through as-is in run()
