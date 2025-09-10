import ROOT
import glob
import os

# ---- Config ----
input_dir = "/eos/user/a/ataxeidi/NTuples_2024/DYto2E-4Jets_Bin-MLL-50/"
file_pattern = "DYto2E-4Jets_Bin-MLL-50_*.root"
output_dir = "evt-counts"
output_summary = os.path.join(output_dir, "summary.txt")
tree_name = "Meta"  # Use your actual tree name
branch_value = "nEvents"  # Use a known branch in the tree

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
        
        
        df = ROOT.RDataFrame(tree)
        n_entries = int(df.Count().GetValue())
        mean_val = float(df.Mean(branch_value).GetValue()) if n_entries > 0 else 0.0
        sum_val  = float(df.Sum(branch_value).GetValue())  if n_entries > 0 else 0.0
    
        summary.append((os.path.basename(root_file), n_entries, mean_val, sum_val))
        out_txt = os.path.join(output_dir, os.path.basename(root_file).replace(".root", ".txt"))
        with open(out_txt, "w") as f:
            f.write(f"{root_file}\n")
            f.write(f"Entries: {n_entries}\n")
            f.write(f"Mean({branch_value}): {mean_val:.6g}\n")
            f.write(f"Sum({branch_value}):  {sum_val:.6g}\n")

        tf.Close()

    except Exception as e:
        print(f"  Error processing {root_file}: {e}")
        continue

# ---- Write Summary ----
with open(output_summary, "w") as fsum:
    total_entries = sum(x[1] for x in summary)
    # Straight sums of means don't mean much; instead aggregate sums across files
    total_sum = sum(x[3] for x in summary)
    overall_mean = (total_sum / total_entries) if total_entries > 0 else 0.0

    header = f"{'file':60}  {'entries':>10}  {'mean('+branch_value+')':>16}  {'sum('+branch_value+')':>16}\n"
    fsum.write(header)
    fsum.write("-" * (len(header)-1) + "\n")
    for fname, nent, meanv, sumv in summary:
        fsum.write(f"{fname:60}  {nent:10d}  {meanv:16.6g}  {sumv:16.6g}\n")

    fsum.write("\n")
    fsum.write(f"TOTAL ENTRIES: {total_entries}\n")
    fsum.write(f"TOTAL SUM({branch_value}): {total_sum:.0f}\n")
   

