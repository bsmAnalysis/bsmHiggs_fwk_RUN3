import sys
import re

if len(sys.argv) != 2:
    print("inject_rand.py <cfg.py>")
    sys.exit(1)

cfg_file = sys.argv[1]
with open(cfg_file, "r") as f:
    lines = f.readlines()

new_lines = []
inserted_top = False
inserted_source = False

for i, line in enumerate(lines):
    new_lines.append(line)

    if not inserted_top and re.match(r'^import|from .* import', line):
        if i+1 < len(lines) and not re.match(r'^import|from .* import', lines[i+1]):
            new_lines.append("""
import os
job_number = int(os.getenv("JOB_NUMBER", "0"))
events_per_job = int(os.getenv("EVENTS_PER_JOB", "500"))
first_event = 1 + job_number * events_per_job
first_lumi = 1 + job_number
""")
            inserted_top = True

    if not inserted_source and "process.source = cms.Source" in line:
        new_lines.append("""process.source.firstEvent = cms.untracked.uint32(first_event)
process.source.firstLuminosityBlock = cms.untracked.uint32(first_lumi)
""")
        inserted_source = True

new_lines.append("""
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()
""")

with open(cfg_file, "w") as f:
    f.writelines(new_lines)
