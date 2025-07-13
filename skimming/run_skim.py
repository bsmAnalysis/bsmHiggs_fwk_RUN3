import uproot
import time
import awkward as ak
import warnings
import numpy as np
import os
import json
import argparse
#from coffea.nanoevents import NanoEvents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from skim_processor import NanoAODSkimmer
from skim_config import branches_to_keep, trigger_groups
# Suppress warnings about missing cross-reference indices
from collections.abc import Mapping
warnings.filterwarnings("ignore", message="Missing cross-reference index")
def resolve_for_uproot(array):
    """Fully materialize an awkward array branch, returning a NumPy-compatible structure."""
    if isinstance(array, ak.Array):
        return ak.to_numpy(array)
    return array
# Input NanoAOD json
parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str, required=True, help="Path to JSON file")
parser.add_argument("--job-index", type=int, required=True, help="Index of file to process")
parser.add_argument("--output", type=str, default="skimmed_output.root")
args = parser.parse_args()
with open(args.json) as f:
    all_datasets = json.load(f)

#dataset_name = list(all_datasets.keys())[0]
parser.add_argument("--dataset", type=str, required=True, help="Key in the JSON to process")
...
if args.dataset not in all_datasets:
    raise ValueError(f"Dataset {args.dataset} not found in {args.json}")
dataset_name = args.dataset
dataset = all_datasets[dataset_name]
meta = dataset["metadata"]
files = dataset["files"]

if args.job_index >= len(files):
    raise IndexError(f"Index {args.job_index} out of range: {len(files)}")

file_to_process = files[args.job_index]
sample_name = meta["sample"]
xsec = float(meta["xsec"])
nevts = int(meta["nevents"])

print(f"[INFO] Processing {file_to_process}")
print(f"[INFO] Sample: {sample_name}, xsec={xsec}, nevts={nevts}")

# Adjust the config based on the sample name
include_genpart = any(x in dataset_name for x in ["ZH", "WH"])
include_genttbarid = "TTto" in dataset_name

# Modify the branches to keep
if include_genpart:
    if "GenPart" not in branches_to_keep:
        branches_to_keep["GenPart"] = ["pt", "eta", "phi", "pdgId", "status", "statusFlags", "genPartIdxMother"]

if include_genttbarid:
    if "genTtbarId" not in branches_to_keep:
        branches_to_keep["genTtbarId"] = []  # scalar branch
# Output file
#source /cvmfs/cms.cern.ch/cmsset_default.sh
output_name = args.output
# Load events
events = None
for attempt in range(1, 6):
    try:
        print(f"[INFO] Attempt {attempt} to open NanoAOD file")
        factory = NanoEventsFactory.from_root(
            file_to_process,
            schemaclass=NanoAODSchema,
            uproot_options={"timeout": 200}
        )
        events = factory.events()
        print(f"[INFO] Successfully loaded {len(events)} events.")
        break
    except Exception as e:
        print(f"[WARNING] Attempt {attempt} failed: {e}")
        if attempt == 5:
            print("[ERROR] Giving up after 4 attempts.")
            exit()
        time.sleep(10)

if events is None:
    print("[ERROR] Events could not be loaded. Exiting.")
    exit()

# Initialize processor
'''
branches_to_keep = {
    "Muon": [],
    "Electron": [],
    "Jet": ["pt", "eta", "phi", "hadronFlavour"],
    "FatJet": ["pt", "eta", "phi", "msoftdrop", "tau1", "tau2", "tau3",
               "globalParT3_Xbb", "particleNetMD_Xbb", "particleNetMD_Xcc", "particleNetMD_Xqq", "particleNetMD_QCD",
               "subJetIdx1", "subJetIdx2"],
    "GenPart": ["pt", "eta", "phi", "pdgId", "status", "genPartIdxMother"],
    "MET": ["pt", "phi"],
}
triggers_to_keep = [
    "IsoMu27", "Ele32 WPTight Gsf", "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8", "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
    "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL", "DoubleEle33_CaloIdL_MW", "HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMET120_PFMHT120_IDTight_PFHT60"
]
'''
processor_instance = NanoAODSkimmer(branches_to_keep=branches_to_keep,trigger_groups=trigger_groups,dataset_name= dataset_name)
skimmed_output = processor_instance.process(events)


def deeply_materialize(data):
    """Fully resolve any awkward array data, including lazy VirtualArrays."""
    if isinstance(data, ak.Array):
        # Fully load if it's lazy
        data = ak.materialized(data)
        # Only pack if it's a layout with substructure
        layout = ak.to_layout(data, allow_record=True)
        if hasattr(layout, "to_packed"):
            return layout.to_packed()
        return data  # Already flat like NumpyArray, no packing needed

    elif isinstance(data, Mapping):
        return {k: deeply_materialize(v) for k, v in data.items()}

    elif isinstance(data, list):
        return [deeply_materialize(v) for v in data]

    else:
        return data
materialized_output = deeply_materialize(skimmed_output)
for k, v in materialized_output.items():
    print(f"[DEBUG] {k}: {type(v)}")

with uproot.recreate(output_name,compression=uproot.LZMA(9)) as rootfile:
    rootfile["Events"] = materialized_output

print("[INFO] ROOT file written successfully.")
# Write output
'''
print(f"[INFO] Saving skimmed output to {output_file}")
def deeply_materialize(data):
    """Force load all lazy awkward arrays into memory before writing to ROOT."""
    if isinstance(data, ak.Array):
        # Load virtual content if needed
        layout = ak.to_layout(ata, allow_record=True)
        if layout.purelist_isregular is None:  # Lazy array likely
            data = ak.materialized(data)       # Force full loading
        return ak.to_layout(data, allow_record=True).to_packed()

    elif isinstance(data, Mapping):
        return {k: deeply_materialize(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deeply_materialize(v) for v in data]
    else:
        return data
materialized_output = deeply_materialize(skimmed_output)

with uproot.recreate("skimmed_output.root") as rootfile:
    rootfile["Events"] = materialized_output

def deeply_materialize(data):
    """Force load all awkward structures into fully realized, writable memory arrays."""
    if isinstance(data, ak.Array):
        return ak.to_layout(data, allow_record=True).to_packed()
    elif isinstance(data, Mapping):
        return {k: deeply_materialize(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deeply_materialize(v) for v in data]
    else:
        return data
materialized_output = deeply_materialize(skimmed_output)

if not materialized_output:
    raise RuntimeError("Nothing materialized.")

with uproot.recreate("skimmed_output.root") as rootfile:
    rootfile["Events"] = materialized_output

print("[INFO] ROOT file written successfully.")

materialized_output = {}
for key, array in skimmed_output.items():
    try:
        layout = ak.to_layout(array, allow_record=True)
        materialized_output[key] = layout
    except Exception as e:
        print(f"[WARNING] Skipping {key}: {e}")

if not materialized_output:
    raise RuntimeError("Nothing to write: all branches failed materialization.")
with uproot.recreate("skimmed_output.root") as rootfile:
    rootfile["Events"] = materialized_output

for key, value in skimmed_output.items():
    try:
        resolved = resolve_for_uproot(ak.Array(value))
        materialized_output[key] = resolved
    except Exception as e:
        print(f"[WARNING] Skipping {key}: {e}")

if not materialized_output:
    raise RuntimeError("No writable branches left after materialization.")

# Write to ROOT
with uproot.recreate("skimmed_output.root") as rootfile:
    rootfile["Events"] = materialized_output
# Materialize each array manually

materialized_output = {
    key: ak.to_numpy(ak.Array(value))
    if ak.is_unknown(value) or isinstance(value, ak.highlevel.Array)
    else value
    for key, value in skimmed_output.items()
}
ed_output = {}
for key, value in skimmed_output.items():
    try:
        materialized_output[key] = ak.Array(value)
    except Exception as e:
        print(f"[WARNING] Skipping {key} due to: {e}")
'''
#source /cvmfs/cms.cern.ch/cmsset_default.sh
#xrdcp  output_file  root://eosuser.cern.ch//eos/user/a/ataxeidi/prod/skimmed_output.root
#rm skimmed_output.root
print("[INFO] Skimming complete. Output saved successfully.")
