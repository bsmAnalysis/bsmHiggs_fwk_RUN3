#!/bin/bash

JSON="datasets/$1"
if [ ! -f "$JSON" ]; then
    echo "[ERROR] JSON not found: $JSON" >&2
    exit 1
fi
jq -r 'keys[]' "$JSON" | while read sample; do
  files=$(ls ${sample}_*.root 2>/dev/null)
  if [ -z "$files" ]; then
    echo "[SKIP] $sample: no files"
    continue
  fi

  if [ -f "${sample}.root" ]; then
    echo "[SKIP] $sample: already merged"
    continue
  fi

  echo "[HADD] $sample"
  hadd -f ${sample}.root ${sample}_*.root
done
