#!/usr/bin/env python3
"""
CLI to run a sample document through the pipeline.

Reads ``sample-documents/<lang>/<scenario>.txt`` and POSTs it to the orchestrator's
``/pipeline/run`` endpoint, then prints the per-stage events and the final report.

Examples:
    python scripts/run_pipeline.py --scenario insurance --lang ko
    python scripts/run_pipeline.py --scenario mortgage --lang en
    ORCHESTRATOR_URL=http://localhost:8080 python scripts/run_pipeline.py -s stock -l en
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

SCENARIOS = ("insurance", "mortgage", "creditcard", "stock")
LANGS = ("ko", "en")
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a sample document through the pipeline.")
    parser.add_argument("-s", "--scenario", choices=SCENARIOS, default="insurance")
    parser.add_argument("-l", "--lang", choices=LANGS, default="ko")
    parser.add_argument("--url", default=os.environ.get("ORCHESTRATOR_URL", "http://localhost:8080"))
    parser.add_argument("--file", help="Override: path to a text file to send instead of the sample.")
    args = parser.parse_args()

    # Only allow http(s) — guard against file:// or custom schemes via --url.
    if urlparse(args.url).scheme not in ("http", "https"):
        print(f"error: --url must use http or https: {args.url}", file=sys.stderr)
        return 1

    doc_path = Path(args.file) if args.file else REPO_ROOT / "sample-documents" / args.lang / f"{args.scenario}.txt"
    if not doc_path.exists():
        print(f"error: document not found: {doc_path}", file=sys.stderr)
        return 1
    text = doc_path.read_text(encoding="utf-8")

    payload = json.dumps({"scenario": args.scenario, "text": text, "lang": args.lang}).encode("utf-8")
    request = urllib.request.Request(
        f"{args.url}/pipeline/run", data=payload, headers={"Content-Type": "application/json"}
    )
    print(f"→ POST {args.url}/pipeline/run  (scenario={args.scenario}, lang={args.lang})\n")
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
