# Running the Production

## 1. Clone the repository and enter:

```bash
git clone https://github.com/bsmAnalysis/bsmHiggs_fwk_RUN3.git
cd bsmHiggs_fwk_RUN3/production
```

## 2. Set up CMSSW

```bash
cmssw-el8
cmsrel CMSSW_14_0_21
shopt -s extglob
mv !(CMSSW_14_0_21) CMSSW_14_0_21/src/
cmsenv
scram b -j4
cd ..
exit
```

Make sure `Configuration/GenProduction/python/` contains your fragment(s). Make sure you have them also defined in the chain.jdl : in the transfer_input_files and also in run_chain.sh (see lines 34-37)

## 3. Submit Jobs

Ensure your proxy is valid:

```bash
source MyProxy.sh
```

Then submit jobs:

```bash
condor_submit chain.jdl
```

## Resubmission

Use `resubmit_missing_jobs.py` to detect missing output `.root` files and generate a new `resubmit.jdl`:

```bash
python resubmit_missing_jobs.py
condor_submit resubmit.jdl
```
## hadd the nanoaod output root files 
```bash
cmssw-el8
cmsenv
scram b
```

you can define how many files you want to merge depending on how many events you want each one to contain:
eg: if you want to hadd the first 100 files to have 50k events in that file you can:
```bash
 python3 haddNano.py ZH_ZToAll_HToAATo4B_M-12_TuneCP5_13p6TeV-madgraph_pythia8_cff_.root $(printf "_%d.root " {0..99})
 ```
You can run the merge_files.py script, defining the input/output file names and the no of output files 
##  Physics Chain

Each job runs the full GEN-SIM → DIGI-HLT → AOD → MiniAOD → NANOAOD chain

- **GEN-SIM**: Using `cmsDriver.py` with a fragment and gridpack (external LHE producer).
- **DIGI-HLT**: Includes premixing and simulated HLT.
- **AOD → MiniAOD → NANOAOD**: Follows standard 2024 Run 3 workflows using NANOv15 schema.

##  Input Requirements

- Gridpacks are fetched via CVMFS.
- Pileup premix samples via DBS or XRootD (ensure AAA access is working).
- Fragments must be correctly formatted in `Configuration/GenProduction/python/`.

## Notes

- Output files are written as `${PROCNAME}_${JOBNUM}.root`
- Each job uses a unique seed injected via `inject_rand.py`
  
# Perform skimming to your datasets and save to selected eos path (optional but recommended)
### Prepare the Skimming Configuration
Modify skimming/skim_config.py to select the branches and objects you want to keep.Configure HLT trigger groups if needed.
If you want to change config script, you may need to do some changes to the skim_processor.py, depending on what kind of changes
### define the datasets you want to skim
From inside the skimming/ directory: put the datasets you want to skim inside the dataset folder
### define the eos path you want to save the files
in run_skimming.sh script (line 25)
### skim your selected dataset and save to your eos path
run:
```bash
#to skim all datasets of all processes (in all json files)
python submit_all.py
# to skim the datasets of a selected process
python submit_all.py QCD.json
#to skim a single dataset of a selected process
FILTER_KEY=HT100to200 python submit_all.py QCD.json
```
### Resubmit  if missing files from your generated eos folder:
run:
```bash
python resubmit_skim.py
```

# Running your analysis
### 1. Clone the repository

```bash
git clone https://github.com/bsmAnalysis/HToAATo4B_RUN3_NANOv15.git
cd HToAATo4B_RUN3_NANOv15
```

### 2. Set up your proxy (if running on the grid)

```bash
source init_cms_proxy.sh
```

### 3. Define your analysis processor and import it in `run_analysis.py`

Make sure your processor is in the project and correctly referenced.

### 4. Test locally inside the Coffea Singularity container

```bash
singularity shell /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
python run_analysis.py --job-index ${JOBIDX} --json ${DATASET_JSON} --dataset ${DATASET_KEY} --output ${OUTFILE}
#you define them hardcoded
```
### 5. Submit all jobs

```bash
python submit_all.py
#or of a selected process:
python submit_all.py ZH_HToAATo4B.json  # all datasets inside  ZH_HToAATo4B.json
#or of a selected dataset of a process:
FILTER_KEY=M-15 python submit_all.py ZH_HToAATo4B.json
```
Also update the `FILES_TO_TRANSFER` list inside `submit_all.py` to include your processor. 

### 6. Resubmit failed/missing jobs

```bash
python resubmit_jobs.py
```
### 7. Clean up and move output `.root` files to CMSSW output directory (if you have installed CMSSW as described in production instructions)

```bash
python clean_dir.py
```
All `.root` outputs will be moved to:

```
CMSSW_15_0_5/src/outputs/
```
