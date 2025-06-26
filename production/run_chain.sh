#!/bin/bash

#set -e

echo "=== Starting job on $(hostname) at $(date)"
echo "=== Working directory: $(pwd)"
ls -lh

# Set job parameters
export JOB_NUMBER=$1
export PROCNAME=$2
export EVENTS_PER_JOB=500
#export SHORTNAME=$(echo "$PROCNAME" | sed 's/_Tune.*//')
#export OUTFILE=${PROCNAME}_${JOB_NUMBER}.root

# Check that input vars are defined
if [[ -z "$JOB_NUMBER" || -z "$PROCNAME" ]]; then
  echo "ERROR: Missing JOB_NUMBER or PROCNAME"
  exit 101
fi

# Load CMS environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc12
############################################
# CMSSW_14_0_21: GEN-SIM -> AODSIM
############################################
echo ">> Setting up CMSSW_14_0_21"
cmsrel CMSSW_14_0_21
cd CMSSW_14_0_21/src
eval `scramv1 runtime -sh`

mkdir -p Configuration/GenProduction/python
cp ../../ZH_ZToAll_HToAATo4B_M-12_TuneCP5_13p6TeV-madgraph_pythia8_cff.py Configuration/GenProduction/python/
cp ../../ZH_ZToAll_HToAATo4B_M-15_TuneCP5_13p6TeV-madgraph_pythia8_cff.py Configuration/GenProduction/python/
cp ../../ZH_ZToAll_HToAATo4B_M-20_TuneCP5_13p6TeV-madgraph_pythia8_cff.py Configuration/GenProduction/python/
cp ../../ZH_ZToAll_HToAATo4B_M-25_TuneCP5_13p6TeV-madgraph_pythia8_cff.py Configuration/GenProduction/python/
scram b -j4 
#[[ ! -f Configuration/GenProduction/python/${PROCNAME}.py ]] && echo "ERROR: Fragment ${PROCNAME}.py not found!" && exit 90
#mkdir -p Configuration/GenProduction/python/
#cp Configuration/GenProduction/python/${PROCNAME}.py Configuration/GenProduction/python/

# Generate GEN-SIM config
cmsDriver.py Configuration/GenProduction/python/${PROCNAME}.py \
  --python_filename step1_${PROCNAME}_GEN_SIM_cfg.py \
  --eventcontent RAWSIM,LHE \
  --datatier GEN-SIM,LHE \
  --conditions 140X_mcRun3_2024_realistic_v26 \
  --beamspot DBrealistic \
  --step LHE,GEN,SIM \
  --geometry DB:Extended \
  --era Run3_2024 \
  --fileout file:step1_${PROCNAME}_GEN_SIM.root \
  --no_exec \
  --mc --nThreads 2 \
  -n ${EVENTS_PER_JOB} || exit 1

# Add random seed
cp ../../inject_rand.py .

python3  inject_rand.py step1_${PROCNAME}_GEN_SIM_cfg.py || exit 2


cmsRun step1_${PROCNAME}_GEN_SIM_cfg.py || exit 4
[[ ! -f step1_${PROCNAME}_GEN_SIM.root ]] && echo "ERROR: step1_GEN_SIM.root not found" && exit 11

# DIGI + HLT
cmsDriver.py step2_${PROCNAME}_DIGI_HLT \
  --filein file:step1_${PROCNAME}_GEN_SIM.root \
  --fileout file:step2_${PROCNAME}_DIGI_HLT.root \
  --mc --eventcontent PREMIXRAW \
  --datatier GEN-SIM-RAW \
  --conditions 140X_mcRun3_2024_realistic_v26 \
  --step DIGI,DATAMIX,L1,DIGI2RAW,HLT:2024v14 \
  --geometry DB:Extended \
  --era Run3_2024 \
  --procModifiers premix_stage2 \
  --pileup_input "dbs:/Neutrino_E-10_gun/RunIIISummer24PrePremix-Premixlib2024_140X_mcRun3_2024_realistic_v26-v1/PREMIX" \
  --datamix PreMix \
  --nThreads 2 \
  --no_exec \
  --python_filename step2_${PROCNAME}_DIGI_HLT_cfg.py \
  -n -1 || exit 5

cmsRun step2_${PROCNAME}_DIGI_HLT_cfg.py || exit 6
[[ ! -f step2_${PROCNAME}_DIGI_HLT.root ]] && echo "ERROR: step2_DIGI_HLT.root not found" && exit 12

# AODSIM
cmsDriver.py step3_${PROCNAME}_AODSIM \
  --filein file:step2_${PROCNAME}_DIGI_HLT.root \
  --fileout file:step3_${PROCNAME}_AODSIM.root \
  --mc --eventcontent AODSIM \
  --datatier AODSIM \
  --conditions 140X_mcRun3_2024_realistic_v26 \
  --step RAW2DIGI,L1Reco,RECO,RECOSIM \
  --geometry DB:Extended \
  --era Run3_2024 \
  --nThreads 2 \
  --no_exec \
  --python_filename step3_${PROCNAME}_AODSIM_cfg.py \
  -n -1 || exit 7
cmsRun step3_${PROCNAME}_AODSIM_cfg.py || exit 8
[[ ! -f step3_${PROCNAME}_AODSIM.root ]] && echo "ERROR: step3_AODSIM.root not found" && exit 13

mv step3_${PROCNAME}_AODSIM.root ../../
cd ../..

############################################
# CMSSW_15_0_5: MINIAOD -> NANOAOD
############################################
echo ">> Setting up CMSSW_15_0_5"
cmsrel CMSSW_15_0_5
cd CMSSW_15_0_5/src
eval `scramv1 runtime -sh`

cp ../../step3_${PROCNAME}_AODSIM.root .

scram b || exit 9

# MINIAOD
cmsDriver.py step4_${PROCNAME}_MINIAOD \
  --filein file:step3_${PROCNAME}_AODSIM.root \
  --fileout file:step4_${PROCNAME}_MINIAOD.root \
  --mc --eventcontent MINIAODSIM \
  --datatier MINIAODSIM \
  --conditions 150X_mcRun3_2024_realistic_v2 \
  --step PAT \
  --geometry DB:Extended \
  --era Run3_2024 \
  --nThreads 2 \
  --no_exec \
  --python_filename step4_${PROCNAME}_MINIAOD_cfg.py \
  -n -1 || exit 10

cmsRun step4_${PROCNAME}_MINIAOD_cfg.py || exit 11
[[ ! -f step4_${PROCNAME}_MINIAOD.root ]] && echo "ERROR: step4_MINIAOD.root not found" && exit 14

# NANOAOD
cmsDriver.py step5_${PROCNAME}_NANO \
  --filein file:step4_${PROCNAME}_MINIAOD.root \
  --fileout file:step5_${PROCNAME}_NANO.root \
  --mc --eventcontent NANOAODSIM \
  --datatier NANOAODSIM \
  --conditions 150X_mcRun3_2024_realistic_v2 \
  --step NANO \
  --scenario pp \
  --era Run3_2024 \
  --nThreads 2 \
  --no_exec \
  --python_filename step5_${PROCNAME}_NANO_cfg.py \
  -n -1 || exit 12

cmsRun step5_${PROCNAME}_NANO_cfg.py || exit 13
[[ ! -f step5_${PROCNAME}_NANO.root ]] && echo "ERROR: step5_NANO.root not found" && exit 15

# Rename and export result
mv step5_${PROCNAME}_NANO.root ${PROCNAME}_${JOB_NUMBER}.root
mv ${PROCNAME}_${JOB_NUMBER}.root  ../../
cd ../..
echo "=== All steps completed successfully"
