#!/usr/bin/env python3
"""
CLI to run a sample document through the pipeline.

By default, reads ``sample-documents/<lang>/<scenario>.txt`` and POSTs it to the
orchestrator's ``/pipeline/run`` endpoint (Stages 3-7, already-OCR'd text). With
``--pdf``, instead POSTs a PDF to ``/pipeline/run-file`` (Stage 2 OCR + Stages 3-7).
Either way, prints the per-stage events and the final report.

Examples:
    python scripts/run_pipeline.py --scenario insurance --lang ko
    python scripts/run_pipeline.py --scenario mortgage --lang en
    python scripts/run_pipeline.py -s stock -l en --pdf sample-documents/en/stock.pdf
    ORCHESTRATOR_URL=http://localhost:8080 python scripts/run_pipeline.py -s stock -l en
"""

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

SCENARIOS = ("insurance", "mortgage", "creditcard", "stock")
LANGS = ("ko", "en")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_multipart(fields: dict, file_field: str, file_path: Path) -> tuple[bytes, str]:
    """Minimal multipart/form-data encoder (stdlib-only, no extra dependency)."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_path.name}"\r\nContent-Type: {content_type}\r\n\r\n'.encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a sample document through the pipeline.")
    parser.add_argument("-s", "--scenario", choices=SCENARIOS, default="insurance")
    parser.add_argument("-l", "--lang", choices=LANGS, default="ko")
    parser.add_argument("--url", default=os.environ.get("ORCHESTRATOR_URL", "http://localhost:8080"))
    parser.add_argument("--file", help="Override: path to a text file to send instead of the sample.")
    parser.add_argument("--pdf", help="Send a PDF through /pipeline/run-file instead (runs OCR too).")
    args = parser.parse_args()

    # Only allow http(s) — guard against file:// or custom schemes via --url.
    if urlparse(args.url).scheme not in ("http", "https"):
        print(f"error: --url must use http or https: {args.url}", file=sys.stderr)
        return 1

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
            return 1
        body, content_type = _build_multipart(
            {"scenario": args.scenario, "lang": args.lang}, "file", pdf_path
        )
        endpoint = f"{args.url}/pipeline/run-file"
        request = urllib.request.Request(endpoint, data=body, headers={"Content-Type": content_type})
    else:
        doc_path = Path(args.file) if args.file else REPO_ROOT / "sample-documents" / args.lang / f"{args.scenario}.txt"
        if not doc_path.exists():
            print(f"error: document not found: {doc_path}", file=sys.stderr)
            return 1
        text = doc_path.read_text(encoding="utf-8")
        payload = json.dumps({"scenario": args.scenario, "text": text, "lang": args.lang}).encode("utf-8")
        endpoint = f"{args.url}/pipeline/run"
        request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})

    print(f"→ POST {endpoint}  (scenario={args.scenario}, lang={args.lang})\n")
    try:
        with urllib.request.urlopen(request, timeout=900) as resp:  # nosec B310 - scheme validated above
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"error: {e.code} {e.reason}\n{e.read().decode('utf-8', 'replace')}", file=sys.stderr)
        return 1

    if "error" in result:
        print(f"pipeline error: {result['error']}", file=sys.stderr)
        return 1

    print("Stages:")
    for ev in result.get("events", []):
        print(f"  • {ev['stage']}")

    print("\nSummary:")
    for k, v in result.get("summary", {}).items():
        print(f"  {k}: {v}")

    print("\n=== Final report (PII reassembled) ===\n")
    print(result.get("finalText", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
