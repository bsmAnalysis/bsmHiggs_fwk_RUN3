import os
import glob
import shutil

# Make sure the output directory exists
output_dir = "CMSSW_15_0_5/src/outputs/"
os.makedirs(output_dir, exist_ok=True)

# Move all .root files to the output directory
root_files = glob.glob("*.root")
for f in root_files:
    try:
        shutil.move(f, os.path.join(output_dir, os.path.basename(f)))
        print(f"Moved {f} → {output_dir}")
    except Exception as e:
        print(f"Failed to move {f}: {e}")

#  to delete
copies = [
    "*.jdl",
    "joblist_*.txt",
    "*.json", 
]

deleted_files = []

for copy in copies:
    for file in glob.glob(copy):
        try:
            os.remove(file)
            deleted_files.append(file)
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

if deleted_files:
    print("Cleaned up the files:")
    for f in deleted_files:
        print(f"  - {f}")
else:
    print("No files for cleanup.")
