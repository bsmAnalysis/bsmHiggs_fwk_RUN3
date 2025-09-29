#!/usr/bin/env python3

import os
import json
import glob
import subprocess

def job_outputs_exist(dataset_key, dataset_info, job_idx, outdir="."):
    # Default single-file naming 
    candidates = [f"{dataset_key}_{job_idx}.root"]

    # If this dataset is TT*, also accept flavor-split outputs
    meta = dataset_info.get("metadata", {})
    sample_base = os.path.basename(meta.get("sample", dataset_key)).replace(".root", "").replace("/", "_")
    if sample_base.startswith("TT") or dataset_key.startswith("TT"):
        for flav in ("ttLF", "ttCC", "ttBB"):
            candidates.append(f"{sample_base}_{flav}_{job_idx}.root")

    return any(os.path.exists(os.path.join(outdir, c)) for c in candidates)


OUTPUT_DIR = "."
DATASET_DIR = "datasets"
FILES_TO_TRANSFER = ["run_analysis.py", "VBFH_processor.py", "x509up", "run_analysis.sh", "utils.tar.gz", "xgb_model.tar.gz", "corrections.tar.gz"]

json_files = sorted(glob.glob(f"{DATASET_DIR}/DATA_JetMET*.json"))

for json_path in json_files:
    with open(json_path) as f:
        data = json.load(f)

    dataset_basename = os.path.basename(json_path)  

    for dataset_key, dataset_info in data.items():
        n_jobs = len(dataset_info["files"])
        missing_jobs = []

        for i in range(n_jobs):
            if not job_outputs_exist(dataset_key, dataset_info, i, OUTPUT_DIR):
                missing_jobs.append(i)

        if not missing_jobs:
            print(f"All jobs complete for {dataset_key}")
            continue

        print(f"Found {len(missing_jobs)} missing jobs for {dataset_key}")

        resub_joblist = f"resub_joblist_{dataset_key}.txt"
        with open(resub_joblist, "w") as jf:
            for i in missing_jobs:
                jf.write(f"{i} {dataset_basename} {dataset_key}\n")

        resub_jdl = f"resubmit_{dataset_key}.jdl"
        with open(resub_jdl, "w") as f:
            f.write("universe = vanilla\n")
            f.write("executable = run_analysis.sh\n")
            f.write("arguments = $(jobindex) $(dataset_json) $(dataset_key)\n")
            f.write(f"transfer_input_files = {', '.join(FILES_TO_TRANSFER)}, $(dataset_json)\n")
            f.write("should_transfer_files = YES\n")
            f.write("when_to_transfer_output = ON_EXIT\n")
            f.write("output = out/job_$(Cluster)_$(jobindex)_$(dataset_key).out\n")
            f.write("error  = err/job_$(Cluster)_$(jobindex)_$(dataset_key).err\n")
            f.write("log    = log/job_$(Cluster)_$(jobindex)_$(dataset_key).log\n")
            f.write('+SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"\n')
            f.write("+SingularityBindCVMFS = True\n")
            f.write("+JobFlavour = \"workday\"\n")
            f.write("request_cpus = 1\n")
            f.write("request_memory = 3000\n")
            f.write('environment = "X509_USER_PROXY=x509up"\n')
            f.write("X509 = x509up\n")
            f.write(f"queue jobindex, dataset_json, dataset_key from {resub_joblist}\n")

        subprocess.run(["condor_submit", resub_jdl])
