#!/bin/bash

echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"

JOBIDX=$1
DATASET_JSON=$2

# Use the proxy
export X509_USER_PROXY=$(realpath x509up)

# Unpack utils if exists
if [ -f utils.tar.gz ]; then
    echo "Extracting utils.tar.gz..."
    tar -xzf utils.tar.gz
    export PYTHONPATH=$PYTHONPATH:$(pwd)
fi

# Extract sample name from JSON
SAMPLE=$(python -c "import json; d=json.load(open('${DATASET_JSON}')); print(list(d.keys())[0])")

# Define output file
OUTFILE=${SAMPLE}_ak4_m_${JOBIDX}.root

# Run the processor
python run_analysis.py --job-index ${JOBIDX} --json ${DATASET_JSON} --output ${OUTFILE}
echo "job finished"
