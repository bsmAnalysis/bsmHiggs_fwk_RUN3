import os
import subprocess
import shutil

# Dataset base name
dataset_base = "ZH_ZToAll_HToAATo4B_M-20"
output_dir = dataset_base

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Merge files in batches of 100
for i in range(15):
    start = i * 100
    end = start + 99
    output_filename = f"{dataset_base}_{i}.root"  
    input_files = " ".join(
        f"{dataset_base}_TuneCP5_13p6TeV-madgraph_pythia8_cff_{j}.root"
        for j in range(start, end + 1)
    )

    cmd = f"python3 haddNano.py {output_filename} {input_files}"
    subprocess.run(cmd, shell=True, check=True)

    # Move the merged output to the folder
    destination = os.path.join(output_dir, output_filename)
    os.rename(output_filename, destination)
    print(f"Moved {output_filename} to {output_dir}/")

# cp to eos
eos_path = "/eos/user/a/ataxeidi/prod/" + dataset_base
print(f"\nCopying {output_dir}/ to EOS: {eos_path}\n")

# Make sure eos directory exists
subprocess.run(f"eos mkdir -p {eos_path}", shell=True, check=True)

# Copy each .root file to EOS
for f in os.listdir(output_dir):
    if f.endswith(".root"):
        local_file = os.path.join(output_dir, f)
        eos_file = f"{eos_path}/{f}"
        cmd_copy = f"xrdcp -f {local_file} root://eosuser.cern.ch/{eos_file}"
        print("Copying:", cmd_copy)
        subprocess.run(cmd_copy, shell=True, check=True)

print("\n All files copied to EOS")
