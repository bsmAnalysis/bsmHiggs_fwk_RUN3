#!/bin/bash
echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"

JOBIDX=$1
DATASET_JSON=$2

# Use the proxy
export X509_USER_PROXY=$(realpath x509up)

# Extract sample name from the JSON file
SAMPLE=$(python -c "import json; d=json.load(open('${DATASET_JSON}')); print(list(d.keys())[0])")

# Define output file name
OUTFILE=${SAMPLE}_${JOBIDX}.root

# Run inside Singularity
python run_skim.py --job-index ${JOBIDX} --json ${DATASET_JSON} --output ${OUTFILE}

# XRDCP the result to EOS (into a subfolder)
echo "Copying ${OUTFILE} to EOS..."
xrdcp -f ${OUTFILE} root://eosuser.cern.ch//eos/user/a/ataxeidi/prod/${SAMPLE}/${OUTFILE}

# Clean up
rm -f ${OUTFILE}
echo "Removed local copy: ${OUTFILE}"

echo "Job finished"
