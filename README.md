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

# Running  Analysis
# Analysis

Coffea analysis with jobs running in Singularity through HTCondor.

Run these commands from the directory containing `submit_all.py`. Dataset JSON files belong in `datasets/`; quote wildcard patterns.

## Main files

| Files / folder | Purpose |
| --- | --- |
| `run_analysis.py`, `ZH_{0,2}lep_processor_fixedWP.py` | Analysis driver and channel processors |
| `submit_all.py`, `resubmit_missing_jobs.py` | Submission and missing-output recovery |
| `run_analysis.sh`, `run_gen.sh` | Job payloads |
| `run_gen_haa4b.py`, `gen_haa4b_processor.py` | Generator-level studies |
| `run_btag_efficiency.py`, `btag_efficiency_processor.py` | B-tag numerator/denominator production |
| `make_*btag_eff*.py` | Efficiency ROOT maps and JSON conversion |
| `datasets/`, `corrections/`, `utils/`, `xgb_model/` | Input lists, corrections, helpers and BDT models |
| `cmssw/bsmhiggs_fwk/` | Plotting, datacards and Combine workflows |

## Setup

Local tests and Condor jobs use the same Coffea image:

```text
/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux8:0.7.21-fastjet-3.4.0.1
```

List the configured modes:

```bash
python3 submit_all.py --list-modes
```

## Test and submit

Prepare one job without running or submitting it:

```bash
python3 submit_all.py 'ZH-*.json' --mode analysis-2lep --dry-run --max-jobs 1
```

Run one job locally inside Singularity:

```bash
python3 submit_all.py 'ZH-*.json' --mode analysis-2lep --local-test
```

`--local-test` runs exactly one job. Check its logs, exit code and ROOT output before submitting.

Submit all files in the matching JSONs:

```bash
python3 submit_all.py 'ZH-*.json' --mode analysis-2lep
```

Use `--mode analysis-0lep` for 0-lepton jobs. The mode selects the processor in the staged driver copy. Set BDT evaluation/training options in `run_analysis.py` before submission.

Filter dataset keys with a regular expression:

```bash
FILTER_KEY=2E python3 submit_all.py 'DY-4Jets*.json' \
    --mode analysis-2lep --local-test
```

Select a specific input-file index:

```bash
FILTER_KEY='^ZH-ZToAll-HToAATo4B_Par-M-12$' \
python3 submit_all.py 'ZH-ZToAll-HToAATo4B.json' \
    --mode analysis-2lep --job-index 5 --local-test
```

Index `5` is the sixth file. Remove `--local-test` to submit it.

Resource folders are archived automatically. `datasets/`, logs, submission directories and `cmssw/` are excluded. Exclude another folder with `--exclude-dir folder_name`, or add it to `EXCLUDE_DIRS`.

## Outputs and resubmission

Logs go to `out/` and `err/`; ROOT outputs return to the analysis directory. Keep `submissions/` while jobs are queued or running.

Check your jobs and preview missing-output resubmissions:

```bash
condor_q "$USER"
python3 resubmit_missing_jobs.py 'ZH-*.json' \
    --mode analysis-2lep --dry-run
```

Remove `--dry-run` to resubmit. `FILTER_KEY` also works here.

The default completion check uses nonempty analysis outputs; it does not validate ROOT contents or require BDT outputs.

Check for missing jobs before merging with `hadd_root_files_per_json.py`.

## B-tag efficiency maps

Produce numerator/denominator counts → merge counts → build ROOT efficiency maps → convert to correctionlib JSON.

Example for a merged DY count file:

```bash
python3 make_all_btag_eff_ratios_grouped.py \
    out/btag_counts/DYto2E-4Jets_Bin-MLL-50_BTag.root \
    --output-dir out/btag_ratios --strict-names

python3 make_btag_eff_json.py \
    out/btag_ratios/DYto2E-4Jets_Bin-MLL-50_BTag_ratio.root \
    --output corrections/btag_eff_DY_example.json.gz \
    --summary out/btag_eff_DY_example.csv --validate
```

This example contains only DY. Build full maps from the required inputs. `make_btag_eff_ratios_sample_keys.py` covers additional process mappings.

## Plotting and statistical analysis

The code in `cmssw/bsmhiggs_fwk/` provides plotting, datacard utilities and Combine workflows.

Run these tools in the appropriate CMSSW environment. Document the plotting and limit commands in `cmssw/README.md`.
```
have the run_eval=True and is_MVA=False in run_analysis.py (line 45-46)
