#!/bin/bash

echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"

JOBIDX=$1
DATASET_JSON=$2
DATASET_KEY=$3

export X509_USER_PROXY=$(realpath x509up)

if [ -f utils.tar.gz ]; then
    echo "Extracting utils.tar.gz..."
    tar -xzf utils.tar.gz
    export PYTHONPATH=$PYTHONPATH:$(pwd)
fi

OUTFILE=${DATASET_KEY}_ak4_m_${JOBIDX}.root

python run_analysis.py --job-index ${JOBIDX} --json ${DATASET_JSON} --dataset ${DATASET_KEY} --output ${OUTFILE}

echo "job finished"
