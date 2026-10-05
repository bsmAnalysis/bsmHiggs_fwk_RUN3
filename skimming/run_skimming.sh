#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: run_skimming.sh JOB_INDEX DATASET_JSON DATASET_KEY" >&2
    exit 64
fi

JOBIDX="$1"
DATASET_JSON="$2"
DATASET_KEY="$3"

if [[ ! "$JOBIDX" =~ ^[0-9]+$ || ! "$DATASET_KEY" =~ ^[A-Za-z0-9_.+-]+$ ]]; then
    echo "Invalid job index or dataset key" >&2
    exit 64
fi
if [[ ! -r skim_job_settings.sh ]]; then
    echo "Missing skim_job_settings.sh; prepare this job with submit_all.py." >&2
    exit 66
fi
source ./skim_job_settings.sh

echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"
echo "Dataset: $DATASET_KEY; input-file index: $JOBIDX"

if [[ -f x509up ]]; then
    export X509_USER_PROXY="$(pwd -P)/x509up"
fi
export EOS_MGM_URL="$SKIM_EOS_SERVER"
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

for archive in "${SKIM_ARCHIVES[@]}"; do
    echo "Extracting $archive"
    tar -xzf "$archive"
done

OUTFILE="${DATASET_KEY}_${JOBIDX}.root"
python3 run_skim_ak4.py \
    --job-index "$JOBIDX" \
    --json "$DATASET_JSON" \
    --dataset "$DATASET_KEY" \
    --output "$OUTFILE" \
    "${SKIM_DRIVER_ARGS[@]}"

if [[ ! -s "$OUTFILE" ]]; then
    echo "Skimming did not create a nonempty output: $OUTFILE" >&2
    exit 1
fi

if [[ "$SKIM_COPY_TO_EOS" == 1 ]]; then
    EOSDIR="${SKIM_BASE_EOS_DIR%/}/${DATASET_KEY}"
    EOSFILE="${SKIM_EOS_SERVER%/}/${EOSDIR}/${OUTFILE}"
    # EOSDIR starts with '/', producing the canonical root://host//eos/... URL.
    xrdfs "$SKIM_EOS_SERVER" mkdir -p "$EOSDIR"
    echo "Copying $OUTFILE to $EOSFILE"
    # A failed copy stops the job; delete the local file only after success.
    xrdcp -f --cksum adler32 "$OUTFILE" "$EOSFILE"
    if [[ "$SKIM_KEEP_LOCAL" == 0 ]]; then
        rm -- "$OUTFILE"
    fi
    echo "EOS output: $EOSFILE"
else
    echo "Local-test output: $(pwd -P)/$OUTFILE"
fi

echo "Job finished"
