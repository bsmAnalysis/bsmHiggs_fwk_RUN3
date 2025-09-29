import os
import glob
import json
import subprocess
import sys
import re

DATASET_DIR = "datasets"
FILES_TO_TRANSFER = ["run_analysis.py", "VBFH_processor.py", "x509up", "run_analysis.sh", "utils.tar.gz", "xgb_model.tar.gz", "corrections.tar.gz"]

# Read optional filter
dataset_key_pattern = os.environ.get("FILTER_KEY")

# Allow optional pattern for which files to submit
pattern = sys.argv[1] if len(sys.argv) > 1 else "SingleTop.json"
json_files = sorted(glob.glob(f"{DATASET_DIR}/{pattern}"))

for json_path in json_files:
    with open(json_path) as f:
        data = json.load(f)

    dataset_basename = os.path.basename(json_path)  # e.g., ZH.json

    if not os.path.exists(dataset_basename):
        os.system(f"cp {json_path} {dataset_basename}")

    joblist_file = f"joblist_{dataset_basename}.txt"
    with open(joblist_file, "w") as jf:
        for dataset_key, dataset_info in data.items():
            if dataset_key_pattern and not re.search(dataset_key_pattern, dataset_key):
                continue
            for i in range(len(dataset_info["files"])):
                jf.write(f"{i} {dataset_basename} {dataset_key}\n")

    jdl_file = f"submit_{dataset_basename}.jdl"
    with open(jdl_file, "w") as f:
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
        f.write(f"queue jobindex, dataset_json, dataset_key from {joblist_file}\n")

    print(f"Submitting jobs from: {dataset_basename}")
    subprocess.run(["condor_submit", jdl_file])
