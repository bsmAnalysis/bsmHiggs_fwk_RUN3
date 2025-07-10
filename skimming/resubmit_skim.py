#!/usr/bin/env python3
import subprocess
import os

dataset_name = "TTto2L2Nu"
import json
with open(f"datasets/{dataset_name}.json") as jf:
    data = json.load(jf)
n_expected_jobs = len(data[dataset_name]["files"])
base_eos_dir = "/eos/user/a/ataxeidi/prod"
eos_dataset_path = f"{base_eos_dir}/{dataset_name}"

print(f"Checking missing jobs for: {dataset_name}")

# Get list of existing .root files
try:
    result = subprocess.run(
        ["xrdfs", "root://eosuser.cern.ch", "ls", eos_dataset_path],
        capture_output=True, text=True, check=True
    )
    files = result.stdout.strip().split("\n")
except subprocess.CalledProcessError:
    print(f"ERROR: Could not list EOS directory {eos_dataset_path}")
    exit()

existing = set()
for f in files:
    if f.endswith(".root") and dataset_name in f:
        try:
            idx = int(f.split("_")[-1].replace(".root", ""))
            existing.add(idx)
        except ValueError:
            continue

expected = set(range(n_expected_jobs))
missing = sorted(expected - existing)
print(f"Found {len(missing)} missing jobs.")

if not missing:
    print("All skim jobs exist.")
    exit()

# Write resubmission JDL
resub_jdl = "resubmit_skim_missing.jdl"
with open(resub_jdl, "w") as f:
    f.write(f"""universe = vanilla
executable = run_skimming.sh
transfer_input_files = run_skim.py, skim_config.py, skim_processor.py, x509up, {dataset_name}.json, run_skimming.sh
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
+SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
+SingularityBindCVMFS = True
+JobFlavour = "workday"
request_cpus = 1
request_memory = 3000
environment = "X509_USER_PROXY=x509up"
X509 = x509up
dataset = {dataset_name}.json

""")
    for idx in missing:
        f.write(f"""arguments = {idx} $(dataset)
output = out/job_$(Cluster)_{idx}.out
error  = err/job_$(Cluster)_{idx}.err
log    = log/job_$(Cluster)_{idx}.log
queue 1

""")

print(f"Created: {resub_jdl}")
print("Submitting...")
subprocess.run(["condor_submit", resub_jdl])
