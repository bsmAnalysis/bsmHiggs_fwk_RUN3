#!/bin/bash

echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"

JOBIDX=$1
DATASET_JSON=$2
DATASET_KEY=$3

export X509_USER_PROXY=$(realpath x509up)

# Unpack utilities if needed
#if [ -f utils.tar.gz ]; then
#    echo "Extracting utils.tar.gz..."
#    tar -xzf utils.tar.gz
#    export PYTHONPATH=$PYTHONPATH:$(pwd)
#fi

# Unpack XGBoost models
#if [ -f xgb_model.tar.gz ]; then
#    tar -xzf xgb_model.tar.gz
#    export PYTHONPATH=$PYTHONPATH:$(pwd)
#fi

#tar -xzf corrections.tar.gz

# Output file names (written in current working directory)
OUTFILE="${DATASET_KEY}_gen_${JOBIDX}.root"
#BDTFILE="bdt_${DATASET_KEY}_${JOBIDX}.root"

# Run main analysis
python run_gen_haa4b.py \
    --job-index ${JOBIDX} \
    --json ${DATASET_JSON} \
    --dataset ${DATASET_KEY} \
    --output ${OUTFILE} \
    #--bdt_output ${BDTFILE}




#echo "Job finished for ${OUTFILE} and ${BDTFILE}"

