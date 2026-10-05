import subprocess
import json

dirs = [
    "/eos/cms/store/group/phys_jetmet/ataxeidi/2024/mc/QCD_bbbEnriched_HT1000to1500",
    "/eos/cms/store/group/phys_jetmet/ataxeidi/2024/mc/QCD_bbbEnriched_HT1500to2000",
    "/eos/cms/store/group/phys_jetmet/ataxeidi/2024/mc/QCD_bbbEnriched_HT2000toInf",
    "/eos/cms/store/group/phys_jetmet/ataxeidi/2024/mc/QCD_bbbEnriched_HT300to500",
    "/eos/cms/store/group/phys_jetmet/ataxeidi/2024/mc/QCD_bbbEnriched_HT500to700",
    "/eos/cms/store/group/phys_jetmet/ataxeidi/2024/mc/QCD_bbbEnriched_HT700to1000",
]

datasets = {}

for d in dirs:
    dataset_name = d.rstrip("/").split("/")[-1]

    res = subprocess.run(
        ["xrdfs", "eoscms.cern.ch", "ls", d],
        check=True,
        capture_output=True,
        text=True,
    )

    files = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.endswith(".root"):
            files.append(f"root://eoscms.cern.ch//{line}")

    datasets[dataset_name] = {
        "files": sorted(files)
    }

with open("QCD_bbbEnriched_2024_EOSCMS.json", "w") as f:
    json.dump(datasets, f, indent=2)

print("Wrote QCD_bbbEnriched_2024_EOSCMS.json")
for k, v in datasets.items():
    print(k, len(v["files"]))
