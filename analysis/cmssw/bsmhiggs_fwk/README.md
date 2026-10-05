# Run-3 CMSSW plotting and statistical workflow

This package contains the ROOT plotting, datacard-production and CMS Combine code used for Run-3 
The CMSSW workflow starts from the ROOT histograms produced by the Coffea-based  analysis.

## Repository and CMSSW layout

```text
bsmHiggs_fwk_RUN3/
└── analysis/
    └── cmssw/
        ├── bsmhiggs_fwk/              # source code tracked by Git
        └── CMSSW_14_1_5/              # local CMSSW installation
            └── src/
                ├── HiggsAnalysis/CombinedLimit/
                ├── CombineHarvester/
                └── UserCode/
                    └── bsmhiggs_fwk -> ../../../bsmhiggs_fwk
```

(The symbolic link places the package under `CMSSW_BASE/src/UserCode`, where SCRAM expects it, while the real files remain in the Run-3 Git repository. Editing through either path changes the same files.)

## Software versions

| Component | Version |
| --- | --- |
| Operating system | EL9 |
| SCRAM architecture | `el9_amd64_gcc12` |
| CMSSW | `CMSSW_14_1_5` |
| Combine | `v10.2.1` |
| CombineHarvester | `v3.1.0` |


##  One-time installation on LXPLUS

### Clone the complete Run-3 repository (propably you have already done this)

```bash
git clone https://github.com/bsmAnalysis/bsmHiggs_fwk_RUN3.git
cd bsmHiggs_fwk_RUN3
```

The  package is already included at:

```text
analysis/cmssw/bsmhiggs_fwk
```

### Create the CMSSW release

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12

cd analysis/cmssw
export HAA_PACKAGE_SOURCE="$PWD/bsmhiggs_fwk"

cmsrel CMSSW_14_1_5
cd CMSSW_14_1_5/src
cmsenv
#once before build
git cms-init
```

###  Install Combine
Run from `CMSSW_14_1_5/src`:

```bash
git -c advice.detachedHead=false clone \
    --depth 1 --branch v10.2.1 \
    https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git \
    HiggsAnalysis/CombinedLimit
```

### Install CombineHarvester

```bash
git -c advice.detachedHead=false clone \
    --depth 1 --branch v3.1.0 \
    https://github.com/cms-analysis/CombineHarvester.git \
    CombineHarvester
```

### Link the analysis package into CMSSW & build
Still inside `CMSSW_14_1_5/src`:

```bash
mkdir -p UserCode
ln -s "$HAA_PACKAGE_SOURCE" UserCode/bsmhiggs_fwk
```

```bash
readlink -f UserCode/bsmhiggs_fwk
```
 should resolve to:

```text
.../bsmHiggs_fwk_RUN3/analysis/cmssw/bsmhiggs_fwk
```
```bash
cd "$CMSSW_BASE/src"
scram b clean
scram b -j 8
```

## useful contents

The main locations are:

| Path | Purpose |
| --- | --- |
| `bin/common/runPlotter.cc` | Produce the ROOT inputs used by the limit code |
| `bin/common/computeLimit0l.cc` | Build the zero-lepton statistical model |
| `bin/common/computeLimit2l.cc` | Build the two-lepton statistical model |
| `bin/common/computeLimit_vr.cc` | Build validation-region models |
| `bin/BuildFile.xml` | Declare and compile the CMSSW executables |
| `test/haa4b/submit.sh` | Configure and run the plotter workflow |
 

| `test/haa4b/computeLimit/optimize_haa.py` | Submit limit jobs and make limit plots |

