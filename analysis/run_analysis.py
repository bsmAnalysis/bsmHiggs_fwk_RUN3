import sys, hist, uproot, time, warnings, awkward 
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from ZH_2lep_total_processor  import TOTAL_Processor

warnings.filterwarnings("ignore", message="Missing cross-reference index")


import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str, required=True, help="Path to JSON file")
parser.add_argument("--job-index", type=int, required=True)
parser.add_argument("--output", type=str, default="ZH12_ak4_boost.root")
parser.add_argument("--dataset", type=str, required=True, help="Dataset key inside JSON")
args = parser.parse_args()

# Load .json file
with open(args.json) as f:
    all_datasets = json.load(f)
#  assume you have one dataset in the json (else specify key)
#dataset_name = list(all_datasets.keys())[0]
#dataset = all_datasets[dataset_name]
if args.dataset not in all_datasets:
    raise ValueError(f"[ERROR] Dataset {args.dataset} not found in {args.json}")

dataset_name = args.dataset
dataset = all_datasets[dataset_name]
# Get metadata and files
meta = dataset["metadata"]
files = dataset["files"]

# Safe check
if args.job_index >= len(files):
    raise IndexError(f"Index {args.job_index} out of bounds: {len(files)} total files.")

# access the file and meta info
file_to_process = files[args.job_index]
dataset_name = meta["sample"]
xsec = float(meta["xsec"])
nevts = int(meta["nevents"])
lumi=112700
print(f"[INFO] Processing: {file_to_process}")
print(f"[INFO] Sample: {dataset_name}, xsec={xsec}, nevts={nevts}")
file_path = files[args.job_index]
print(f"[INFO] Processing file {args.job_index + 1}/{len(files)} from {dataset_name}")

# --- Load events ---
for attempt in range(1, 6):
    try:
        factory = NanoEventsFactory.from_root(
            file_to_process,
            schemaclass=NanoAODSchema,
            uproot_options={"timeout": 300}
        )
        events = factory.events()
        break
    except Exception as e:
        print(f"[WARNING] Attempt {attempt} failed: {e}")
        if attempt == 6:
            print("[ERROR] Max attempts reached. Skipping file.")
            sys.exit(1)
        time.sleep(10)

# --- Run processor ---
processor_instance = TOTAL_Processor(xsec=xsec, lumi=lumi, nevts=nevts, dataset_name=dataset_name)
output = processor_instance.process(events)

# --- Save output root file---
output_name= args.output
with uproot.recreate(output_name) as rootfile:
    for name, h in output.items():
        if isinstance(h, hist.Hist):
            rootfile[name] = h.to_numpy()

print(f"[INFO] Saved output: {output_name}")
