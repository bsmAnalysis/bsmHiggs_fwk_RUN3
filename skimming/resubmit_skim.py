#!/usr/bin/env python3
import subprocess
import os
import json
import glob

DATASET_DIR = "datasets"
FILES_TO_TRANSFER = ["run_skim.py", "skim_processor.py", "x509up", "run_skimming.sh", "skim_config.py"]
base_eos_dir = "/eos/user/a/ataxeidi/skim"

json_files = sorted(glob.glob(f"{DATASET_DIR}/*.json"))

for json_path in json_files:
    json_name = os.path.basename(json_path)
    with open(json_path) as jf:
        data = json.load(jf)

    for dataset_key, dataset_info in data.items():
        n_expected_jobs = len(dataset_info["files"])
        eos_dataset_path = f"{base_eos_dir}/{dataset_key}"

        print(f"\nChecking missing jobs for: {dataset_key} (from {json_name})")

        try:
            result = subprocess.run(
                ["xrdfs", "root://eosuser.cern.ch", "ls", eos_dataset_path],
                capture_output=True, text=True, check=True
            )
            files = result.stdout.strip().split("\n")
        except subprocess.CalledProcessError:
            print(f"WARNING: EOS path not found: {eos_dataset_path}")
            files = []

        existing = set()
        for f in files:
            if f.endswith(".root") and dataset_key in f:
                try:
                    idx = int(f.split("_")[-1].replace(".root", ""))
                    existing.add(idx)
                except ValueError:
                    continue

        expected = set(range(n_expected_jobs))
        missing = sorted(expected - existing)
        print(f"Found {len(missing)} missing jobs.")

        if not missing:
            continue

        # Write JDL for missing jobs
        jdl_file = f"resubmit_missing_{dataset_key}.jdl"
        with open(jdl_file, "w") as f:
            f.write("universe = vanilla\n")
            f.write("executable = run_skimming.sh\n")
            f.write(f"transfer_input_files = {', '.join(FILES_TO_TRANSFER)}, {json_name}\n")
            f.write("should_transfer_files = YES\n")
            f.write("when_to_transfer_output = ON_EXIT\n")
            f.write('+SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"\n')
            f.write("+SingularityBindCVMFS = True\n")
            f.write("+JobFlavour = \"workday\"\n")
            f.write("request_cpus = 1\n")
            f.write("request_memory = 3000\n")
            f.write('environment = "X509_USER_PROXY=x509up"\n')
            f.write("X509 = x509up\n\n")

            for idx in missing:
                f.write(f"""arguments = {idx} {json_name} {dataset_key}
output = out/job_$(Cluster)_{idx}_{dataset_key}.out
error  = err/job_$(Cluster)_{idx}_{dataset_key}.err
log    = log/job_$(Cluster)_{idx}_{dataset_key}.log
queue 1

""")
        print(f"Created: {jdl_file}")
        subprocess.run(["condor_submit", jdl_file])
