import os
import glob
import json
import subprocess

DATASET_DIR = "datasets"
FILES_TO_TRANSFER = ["run_analysis.py", "ZH_ak4_boost_processor.py", "x509up", "run_analysis.sh",  "utils.tar.gz",]

# Include all Python files in utils/
#utils_files = glob.glob("utils/*.py")
#FILES_TO_TRANSFER.extend(utils_files)

# Add full relative path for datasets folder — needed for file transfer
json_files = sorted(glob.glob(f"{DATASET_DIR}/ZH*.json"))

for json_path in json_files:
    with open(json_path) as f:
        data = json.load(f)

    sample_name = list(data.keys())[0]
    filelist = data[sample_name]["files"]
    nfiles = len(filelist)

    dataset_basename = os.path.basename(json_path)  # ZH_HToAATo4B_m15.json
    dataset_name = dataset_basename.replace(".json", "")

    # Copy dataset JSON to top-level for Condor transfer (as just the filename)
    if not os.path.exists(dataset_basename):
        os.system(f"cp {json_path} {dataset_basename}")

    # Create job list (maps job index to dataset file)
    joblist_file = f"joblist_{dataset_name}.txt"
    with open(joblist_file, "w") as jf:
        for i in range(nfiles):
            jf.write(f"{i} {dataset_basename}\n")

    # Create the JDL file
    jdl_file = f"submit_{dataset_name}.jdl"
    with open(jdl_file, "w") as f:
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
        f.write(f"queue jobindex, dataset from {joblist_file}\n")

    # Submit the job
    print(f"Submitting {nfiles} jobs for {dataset_name}")
    subprocess.run(["condor_submit", jdl_file])
