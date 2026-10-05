#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage:"
  echo "  $0 input.json [output.json]"
  exit 1
fi

in="$1"
out="${2:-${in%.json}_globalAAA.json}"

jq 'to_entries
| map(.value.files |= (map(if (type=="string" and startswith("root://")) then
    ("root://cms-xrd-global.cern.ch//" + (sub("^root://[^/]+/?"; "")))
  else . end)))
| from_entries' "$in" > "$out"

echo "Wrote: $out"
