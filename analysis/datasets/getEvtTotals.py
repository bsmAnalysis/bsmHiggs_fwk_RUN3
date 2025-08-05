import ROOT
import glob
import os

# ---- Config ----
input_dir = "/eos/user/a/ataxeidi/skim_MC/DYto2Tau-4Jets/"
file_pattern = "DYto2Tau-4Jets_*.root"
output_dir = "evt-counts"
output_summary = os.path.join(output_dir, "summary.txt")
tree_name = "Events"  # Use your actual tree name
branch_to_count = "run"  # Use a known branch in the tree

# ---- Setup ----
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

root_files = glob.glob(os.path.join(input_dir, file_pattern))
if not root_files:
    print("No ROOT files found.")
    exit(1)

# ---- Processing ----
summary = []

for i, root_file in enumerate(root_files):
    print(f"{i+1}/{len(root_files)}: Processing {os.path.basename(root_file)}")

    try:
        tf = ROOT.TFile.Open(root_file)
        if not tf or tf.IsZombie():
            print(f"  Skipping corrupted file: {root_file}")
            continue

        tree = tf.Get(tree_name)
        if not tree:
            print(f"  Tree '{tree_name}' not found in {root_file}")
            tf.Close()
            continue

        n_entries = tree.GetEntries()
        summary.append((os.path.basename(root_file), n_entries))

        out_txt = os.path.join(output_dir, os.path.basename(root_file).replace(".root", ".txt"))
        with open(out_txt, "w") as f:
            f.write(f"{root_file}\n")
            f.write(f"Entries in '{branch_to_count}': {n_entries}\n")

        tf.Close()

    except Exception as e:
        print(f"  Error processing {root_file}: {e}")
        continue

# ---- Write Summary ----
with open(output_summary, "w") as fsum:
    total = sum(x[1] for x in summary)
    for fname, count in summary:
        fsum.write(f"{fname:60}  {count:10}\n")
    fsum.write(f"\nTOTAL EVENTS: {total}\n")

  
