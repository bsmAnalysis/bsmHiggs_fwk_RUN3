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
cd CMSSW_14_0_21/src
cp -r ../../production/* .
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


