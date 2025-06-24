#!/usr/bin/env python3

import os
import json
import glob
import subprocess

OUTPUT_DIR = "."  # Where the output ROOT files are
DATASET_DIR = "datasets"
FILES_TO_TRANSFER = ["run_test.py", "ak4_boost_m_processor.py", "x509up", "run_analysis.sh", "jet_tight_id.py"]

json_files = sorted(glob.glob(f"{DATASET_DIR}/ZH*.json"))

for json_path in json_files:
    with open(json_path) as f:
        data = json.load(f)

    dataset_basename = os.path.basename(json_path)  # e.g. ZH_HToAATo4B_m20.json
    dataset_name = dataset_basename.replace(".json", "")
    sample_name = list(data.keys())[0]
    n_jobs = len(data[sample_name]["files"])

    # Determine expected output filenames
    missing_jobs = []
    for i in range(n_jobs):
        expected_file = f"{sample_name}_ak4_m_{i}.root"
        if not os.path.exists(os.path.join(OUTPUT_DIR, expected_file)):
            missing_jobs.append(i)

    if not missing_jobs:
        print(f" All jobs complete for {dataset_name}")
        continue

    print(f" Found {len(missing_jobs)} missing jobs for {dataset_name}")

    # Create resubmission joblist
    resub_joblist = f"resub_joblist_{dataset_name}.txt"
    with open(resub_joblist, "w") as jf:
        for i in missing_jobs:
            jf.write(f"{i} {dataset_basename}\n")

    # Write corresponding JDL
    resub_jdl = f"resubmit_{dataset_name}.jdl"
    with open(resub_jdl, "w") as f:
        f.write("universe = vanilla\n")
        f.write("executable = run_analysis.sh\n")
        f.write("arguments = $(jobindex) $(dataset)\n")
        f.write(f"transfer_input_files = {', '.join(FILES_TO_TRANSFER)}, $(dataset)\n")
        f.write("should_transfer_files = YES\n")
        f.write("when_to_transfer_output = ON_EXIT\n")
        f.write("output = out/job_$(Cluster)_$(jobindex)_$(dataset).out\n")
        f.write("error  = err/job_$(Cluster)_$(jobindex)_$(dataset).err\n")
        f.write("log    = log/job_$(Cluster)_$(jobindex)_$(dataset).log\n")
        f.write('+SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"\n')
        f.write("+SingularityBindCVMFS = True\n")
        f.write("+JobFlavour = \"workday\"\n")
        f.write("request_cpus = 1\n")
        f.write("request_memory = 3000\n")
        f.write('environment = "X509_USER_PROXY=x509up"\n')
        f.write("X509 = x509up\n")
        f.write(f"queue jobindex, dataset from {resub_joblist}\n")

    # Submit
    subprocess.run(["condor_submit", resub_jdl])
