#!/usr/bin/env bash
# Fails if any tracked file contains a value that must never appear in this
# public sample: real AWS account IDs, internal domains/usernames, a
# customer name, the excluded OCR vendor, or live resource IDs.
#
# This script does not distinguish "old" vs "new" — any match anywhere in a
# tracked file is a failure. It intentionally checks tracked content only
# (`git grep`), not history — this repo's history starts clean.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Extended-regex patterns. Keep each one specific enough to avoid false
# positives (e.g. "sg-" alone would match too much prose, so it is scoped to
# the AWS security-group ID shape).
PATTERNS=(
  '180294183052'
  '836533914648'
  '013503698282'
  'atomai\.click'
  'Atom-oh'
  '[Ii]sengard'
  '[Kk]akao[Bb]ank'
  '카카오'
  '[Uu]pstage'
  'solar\.box'
  'sg-[0-9a-f]{8,17}'
  'vpc-[0-9a-f]{8,17}'
  'subnet-[0-9a-f]{8,17}'
  'arn:aws:elasticloadbalancing:[a-z0-9-]+:[0-9]{12}'
  'arn:aws:bedrock-agentcore:[a-z0-9-]+:[0-9]{12}'
  'f6b6907a-5747-4039-967a-a8c7c73116a7'
  'd5f951df-418a-4124-8d06-e55a2079bd26'
)

fail=0
for p in "${PATTERNS[@]}"; do
  if matches=$(git grep -nIE "$p" -- . ':!scripts/check-sanitized.sh' 2>/dev/null); then
    echo "FORBIDDEN pattern matched: $p"
    echo "$matches" | sed 's/^/  /'
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo
  echo "check-sanitized.sh: found forbidden content above — fix before committing."
  exit 1
fi

echo "check-sanitized.sh: OK — no forbidden patterns found."
