import json
import glob

# Step 1: Find all JSON files starting with QCD
json_files = glob.glob("TT*.json")

# Step 2: Merge all into one dict
merged_data = {}
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        merged_data.update(data)

# Step 3: Write to a single merged file
with open("TT.json", "w") as f:
    json.dump(merged_data, f, indent=2)

print(f"Merged {len(json_files)} files into QCD.json")
