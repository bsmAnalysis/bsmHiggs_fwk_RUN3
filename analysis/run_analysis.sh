#!/bin/bash

echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"

JOBIDX=$1
DATASET_JSON=$2
DATASET_KEY=$3

export X509_USER_PROXY=$(realpath x509up)

# Unpack utilities if needed
if [ -f utils.tar.gz ]; then
    echo "Extracting utils.tar.gz..."
    tar -xzf utils.tar.gz
    export PYTHONPATH=$PYTHONPATH:$(pwd)
fi

# Output file names (written in current working directory)
OUTFILE="${DATASET_KEY}_0l_${JOBIDX}.root"
BDTFILE="bdt_${DATASET_KEY}_0l_${JOBIDX}.root"

# Run main analysis
python run_analysis.py \
    --job-index ${JOBIDX} \
    --json ${DATASET_JSON} \
    --dataset ${DATASET_KEY} \
    --output ${OUTFILE} \
    --bdt_output ${BDTFILE}




echo "Job finished for ${OUTFILE} and ${BDTFILE}"

