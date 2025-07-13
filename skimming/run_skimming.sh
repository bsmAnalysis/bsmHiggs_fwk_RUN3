#!/bin/bash

echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"

JOBIDX=$1
DATASET_JSON=$2
DATASET_KEY=$3

export X509_USER_PROXY=$(realpath x509up)

OUTFILE=${DATASET_KEY}_${JOBIDX}.root

python run_skim.py --job-index ${JOBIDX} --json ${DATASET_JSON} --dataset ${DATASET_KEY} --output ${OUTFILE}

echo "Copying ${OUTFILE} to EOS..."
xrdcp -f ${OUTFILE} root://eosuser.cern.ch//eos/user/a/ataxeidi/skim/${DATASET_KEY}/${OUTFILE}

rm -f ${OUTFILE}
echo "Removed local copy: ${OUTFILE}"

echo "Job finished"
