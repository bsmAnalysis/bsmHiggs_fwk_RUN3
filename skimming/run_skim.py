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

dataset_name = list(all_datasets.keys())[0]
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

print("[INFO] Skimming complete. Output saved successfully.")
