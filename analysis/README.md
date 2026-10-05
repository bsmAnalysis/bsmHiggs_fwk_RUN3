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

If your input access needs an X.509 proxy, initialize it on lxplus:

```bash
source init_cms_proxy.sh
voms-proxy-info --file "$PWD/x509up" --all
chmod 600 x509up
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

The code in `cmssw/bsmhiggs_fwk/` provides plotting,  datacard utilities and Combine workflows.

Run these tools in the appropriate CMSSW environment. Document the plotting and limit commands in `cmssw/README.md`.

