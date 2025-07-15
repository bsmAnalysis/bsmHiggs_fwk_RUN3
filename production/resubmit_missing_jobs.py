#!/usr/bin/env python3

import subprocess
import os

# === Configuration ===
EOS_BASE = "/eos/user/a/ataxeidi/prod"
EOS_XRDFS = "root://eosuser.cern.ch"
EVENTS_PER_JOB = 500
TOTAL_JOBS = 500
OUTPUT_DIR = os.getcwd()

# List of processes
PROCESSES = [
    "VBFH_HToAATo4B_M-45_TuneCP5_13p6TeV-madgraph_pythia8_cff",
    "VBFH_HToAATo4B_M-40_TuneCP5_13p6TeV-madgraph_pythia8_cff",
    "VBFH_HToAATo4B_M-35_TuneCP5_13p6TeV-madgraph_pythia8_cff",
    "VBFH_HToAATo4B_M-30_TuneCP5_13p6TeV-madgraph_pythia8_cff",
    "VBFH_HToAATo4B_M-25_TuneCP5_13p6TeV-madgraph_pythia8_cff",
    "VBFH_HToAATo4B_M-20_TuneCP5_13p6TeV-madgraph_pythia8_cff"
]

# JDL Header
JDL_HEADER = """universe = vanilla
executable = run_chain.sh
arguments = $(JOBNUM) $(PROCNAME)

output = out/job_$(Cluster)_$(Process)_$(PROCNAME).out
error  = err/job_$(Cluster)_$(Process)_$(PROCNAME).err
log    = log/job_$(Cluster)_$(Process)_$(PROCNAME).log

x509userproxy = MyProxy
+JobFlavour = "workday"
+ProjectName = "cms.org"
+SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/el8:x86_64"
+SingularityBindCVMFS = True

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files = run_chain.sh, inject_rand.py, \
Configuration/GenProduction/python/VBFH_HToAATo4B_M-45_TuneCP5_13p6TeV-madgraph_pythia8_cff.py, \
Configuration/GenProduction/python/VBFH_HToAATo4B_M-40_TuneCP5_13p6TeV-madgraph_pythia8_cff.py, \
Configuration/GenProduction/python/VBFH_HToAATo4B_M-35_TuneCP5_13p6TeV-madgraph_pythia8_cff.py, \                                              
Configuration/GenProduction/python/VBFH_HToAATo4B_M-30_TuneCP5_13p6TeV-madgraph_pythia8_cff.py, \
Configuration/GenProduction/python/VBFH_HToAATo4B_M-25_TuneCP5_13p6TeV-madgraph_pythia8_cff.py, \
Configuration/GenProduction/python/VBFH_HToAATo4B_M-20_TuneCP5_13p6TeV-madgraph_pythia8_cff.py 


request_cpus = 2
request_memory = 12000

"""

missing_jobs = []

for proc in PROCESSES:
    eos_path = f"{EOS_BASE}/{proc}"
    print(f"\n Checking EOS for: {proc}")

    try:
        result = subprocess.run(
            ["xrdfs", EOS_XRDFS, "ls", eos_path],
            capture_output=True,
            text=True,
            check=True
        )
        files = result.stdout.strip().split("\n")
        existing = set()

        for f in files:
            basename = os.path.basename(f)
            if basename.endswith(".root") and basename.startswith(proc):
                idx = basename.replace(f"{proc}_", "").replace(".root", "")
                if idx.isdigit():
                    existing.add(int(idx))

        expected = set(range(TOTAL_JOBS))
        missing = sorted(expected - existing)

        if missing:
            print(f"Missing: {len(missing)} jobs")
            for idx in missing:
                missing_jobs.append((proc, idx))
        else:
            print("All jobs exist")

    except subprocess.CalledProcessError:
        print(f" Could not access EOS folder: {eos_path}")

# === Write resubmission JDL ===
resub_jdl = "resubmit_missing_jobs.jdl"
if missing_jobs:
    print(f"\n Writing {len(missing_jobs)} missing jobs to {resub_jdl}")

    with open(resub_jdl, "w") as f:
        f.write(JDL_HEADER + "\n")
        f.write("Queue PROCNAME, JOBNUM from (\n")
        for procname, jobnum in missing_jobs:
            f.write(f"{procname} {jobnum}\n")
        f.write(")\n")
    print("Done.")
else:
    print(" No missing jobs found!")
