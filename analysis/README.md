## Overview

This repository provides a complete analysis workflow for HToAATo4B Run3 NanoAODv15 data using the Coffea framework and CMSConnect.

---

## Documentation

### `run_analysis.py`

Main driver script for running a job on a single file (used in Condor jobs).

- It loads the dataset JSON, selects the job index, runs the processor, and saves a `.root` output file.
- Just need to import your own processor of your analysis.

To run a test in CMSConnect, insert it in Coffea Singularity:

```bash
singularity shell /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
python run_analysis.py --json ZH_HToAATo4B_m20.json --job-index 0 --output ZH_m20_boosted.root
```

### `ZH_ak4_boost_processor.py`

An analysis processor example.

- Implements the `coffea.processor.ProcessorABC` class.
- Handles selection of double-b-tagged AK4 jets using `btagUParTAK4probbb`.
- Separates logic for MC signal (with bb pair matching) and background.
- Fills histograms for UParT scores, kinematics, matching, etc.

---

### `run_analysis.sh`

Shell script that executes a job on the grid.

- Parses job index and dataset JSON.
- Runs the driver script (`run_analysis.py`).

---

### `submit_all.py`

- Loops through all JSON files in `analysis/datasets/`.
- For each dataset:
  - Generates a Condor `.jdl` file for each job.
  - Submits jobs with input files and JSONs (1 job per NanoAOD file).
- Supports Singularity execution via `coffeateam/coffea-dask`.

**You can submit multiple datasets** defined in line 14, e.g.:

```python
json_files = glob.glob("datasets/ZH*.json")
```

### `resubmit_jobs.py`

- Scans the working directory for missing `.root` output files based on the number of input files per dataset.
- Rewrites a `.jdl` file only for the failed/missing jobs.

---

### `init_cms_proxy.sh`

Optional script to initialize and export a VOMS grid proxy to `x509up` (used in job submission).

### `clean_dir.py`

- Moves all `.root` output files to `CMSSW_15_0_5/src/outputs/` (for `hadd` merging).
- Cleans up temporary files copied to root for Condor transfer.

To enter the CMSSW environment inside a container:

```bash
cmssw-el9
cmsenv
```

### `datasets/` directory

Contains `.json` files, one per dataset (automatically generated with the PocketCoffea framework), describing:

- `files`: full list of NanoAOD file paths.
- `metadata`: sample name, cross-section, etc.

If you have skimmed the NanoAOD files and saved them in EOS, you can change the file paths automatically by running:

```bash
python write_json_eos.py --dataset_name.json -- eos_dataset_name.json
```

### `utils/`

Holds helper scripts or functions (e.g., `matching.py`, `jet_id.py`) used by the processor.

These are included in the job tarball and imported dynamically.

---

## 🛠️ Instructions for Use

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
python run_analysis.py --json datasets/ZH_HToAATo4B_m20.json --job-index 0 --output ZH_m20_test.root
```

### 5. Submit all jobs

```bash
python submit_all.py
```

> 🔧 Also update the `FILES_TO_TRANSFER` list inside `submit_all.py` to include your processor.

### 6. Resubmit failed/missing jobs

```bash
python resubmit_jobs.py
```

### 7. Clean up and move output `.root` files to CMSSW output directory

```bash
python clean_dir.py
```

All `.root` outputs will be moved to:

```
CMSSW_15_0_5/src/outputs/
```

