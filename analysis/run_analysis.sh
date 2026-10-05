#!/bin/bash
# payload; submit_all.py's launcher extracts folders.
set -e

JOBIDX="$1"
DATASET_JSON="$2"
DATASET_KEY="$3"

if [[ -f x509up ]]; then
    export X509_USER_PROXY="$(pwd -P)/x509up"
fi
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"

OUTFILE="${DATASET_KEY}_${JOBIDX}.root"
BDTFILE="bdt_${DATASET_KEY}_${JOBIDX}.root"

# The submission mode supplies ANALYSIS_DRIVER. Arguments match your original run_analysis.sh,
python "${ANALYSIS_DRIVER:-run_analysis.py}" \
    --job-index "$JOBIDX" \
    --json "$DATASET_JSON" \
    --dataset "$DATASET_KEY" \
    --output "$OUTFILE" \
    --bdt_output "$BDTFILE"
