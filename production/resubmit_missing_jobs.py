import os

# === Config ===
output_dir = os.getcwd()  # or set your output dir explicitly
resubmit_jdl = "resubmit.jdl"

# Your processes and expected job counts
processes = {
    "ZH_ZToAll_HToAATo4B_M-12_TuneCP5_13p6TeV-madgraph_pythia8_cff": 1500,
    "ZH_ZToAll_HToAATo4B_M-15_TuneCP5_13p6TeV-madgraph_pythia8_cff": 1500,
    "ZH_ZToAll_HToAATo4B_M-20_TuneCP5_13p6TeV-madgraph_pythia8_cff": 1500,
    "ZH_ZToAll_HToAATo4B_M-25_TuneCP5_13p6TeV-madgraph_pythia8_cff": 1500,
}

# === JDL Header ===
jdl_header = f"""universe = vanilla
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
transfer_input_files = run_chain.sh, inject_rand.py, \\
Configuration/GenProduction/python/ZH_ZToAll_HToAATo4B_M-12_TuneCP5_13p6TeV-madgraph_pythia8_cff.py, \\
Configuration/GenProduction/python/ZH_ZToAll_HToAATo4B_M-15_TuneCP5_13p6TeV-madgraph_pythia8_cff.py, \\
Configuration/GenProduction/python/ZH_ZToAll_HToAATo4B_M-20_TuneCP5_13p6TeV-madgraph_pythia8_cff.py, \\
Configuration/GenProduction/python/ZH_ZToAll_HToAATo4B_M-25_TuneCP5_13p6TeV-madgraph_pythia8_cff.py

request_cpus = 2
request_memory = 12000

"""

# === Find Missing Jobs ===
missing_jobs = []

for procname, njobs in processes.items():
    for i in range(njobs):
        expected_file = os.path.join(output_dir, f"{procname}_{i}.root")
        if not os.path.isfile(expected_file):
            missing_jobs.append((procname, i))

# === Write JDL ===
with open(resubmit_jdl, "w") as jdl:
    jdl.write(jdl_header + "\n")

    if missing_jobs:
        jdl.write("Queue PROCNAME, JOBNUM from (\n")
        for proc, num in missing_jobs:
            jdl.write(f"{proc} {num}\n")
        jdl.write(")\n")
    else:
        print(" No missing jobs")

print(f"Wrote {resubmit_jdl} with {len(missing_jobs)} missing jobs.")
