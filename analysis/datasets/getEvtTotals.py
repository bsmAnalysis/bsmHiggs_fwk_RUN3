import ROOT
import glob
import os
import argparse
from collections import defaultdict

# ---------------- CLI arguments ----------------
parser = argparse.ArgumentParser(description="Count events in ROOT files (grouped per dataset)")
parser.add_argument(
    "-i", "--input-dir", required=True,
    help="Input directory pattern (can contain wildcards, e.g. /eos/.../NTuples_2024/*/)"
)
parser.add_argument(
    "-o", "--output-dir", default="evt-counts",
    help="Output directory (default: evt-counts)"
)
parser.add_argument(
    "--per-dataset-files", action="store_true",
    help="Also write summary_<dataset>.txt files with only that dataset's rows"
)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
output_summary = os.path.join(output_dir, "summary.txt")

tree_name = "Meta"
branch_value = "nEvents"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Expand all directories (with wildcards)
dirs = [d for d in glob.glob(input_dir) if os.path.isdir(d)]
if not dirs:
    print("No matching directories found.")
    exit(1)

# Per-dataset storage
rows_by_dataset = defaultdict(list)   # dataset -> list of (file, entries, mean, sum)
totals_by_dataset = defaultdict(lambda: {"entries": 0, "sum": 0.0})

for d in sorted(dirs):
    dataset = os.path.basename(os.path.normpath(d))  # last token
    file_pattern = f"{dataset}_*.root"
    root_files = sorted(glob.glob(os.path.join(d, file_pattern)))
    if not root_files:
        print(f"No ROOT files in {d} with pattern {file_pattern}")
        continue

    print(f"\n[{dataset}] Directory: {d}  |  Files: {len(root_files)}")
    for i, root_file in enumerate(root_files, 1):
        print(f"{i}/{len(root_files)}: Processing {os.path.basename(root_file)}")

        try:
            tf = ROOT.TFile.Open(root_file)
            if not tf or tf.IsZombie():
                print(f"  no file: {root_file}")
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

            fname = os.path.basename(root_file)
            rows_by_dataset[dataset].append((fname, n_entries, mean_val, sum_val))
            totals_by_dataset[dataset]["entries"] += n_entries
            totals_by_dataset[dataset]["sum"]     += sum_val

            # Per-file tiny output (unchanged)
            out_txt = os.path.join(output_dir, fname.replace(".root", ".txt"))
            with open(out_txt, "w") as f:
                f.write(f"{root_file}\n")
                f.write(f"Entries: {n_entries}\n")
                f.write(f"Mean({branch_value}): {mean_val:.6g}\n")
                f.write(f"Sum({branch_value}):  {sum_val:.6g}\n")

            tf.Close()

        except Exception as e:
            print(f"  Error processing {root_file}: {e}")
            continue

# ---- Write combined Summary (grouped per dataset) ----
with open(output_summary, "w") as fsum:
    grand_entries = 0
    grand_sum = 0.0

    for dataset in sorted(rows_by_dataset.keys()):
        rows = sorted(rows_by_dataset[dataset], key=lambda r: r[0])  # sort by filename
        subtotal_entries = totals_by_dataset[dataset]["entries"]
        subtotal_sum = totals_by_dataset[dataset]["sum"]

        fsum.write(f"{dataset}\n")
        header = f"{'file':60}  {'entries':>10}   {'sum('+branch_value+')':>16}\n"
        fsum.write(header)
        fsum.write("-" * (len(header)-1) + "\n")
        for fname, nent, meanv, sumv in rows:
            fsum.write(f"{fname:60}  {nent:10d}  {sumv:16.6g}\n")
        fsum.write("\n")
        fsum.write(f"SUBTOTAL [{dataset}] ENTRIES: {subtotal_entries}\n")
        fsum.write(f"SUBTOTAL [{dataset}] SUM({branch_value}): {subtotal_sum:.0f}\n")
        fsum.write("\n" + "=" * 72 + "\n\n")

        grand_entries += subtotal_entries
        grand_sum += subtotal_sum

    # Grand totals across all datasets
    fsum.write("\nGRAND TOTAL ENTRIES: {}\n".format(grand_entries))
    fsum.write("GRAND TOTAL SUM({}): {:.0f}\n".format(branch_value, grand_sum))

print(f"\nCombined per-dataset summary written to {output_summary}")

# ---- Optional: per-dataset summary files ----
if args.per_dataset_files:
    for dataset in rows_by_dataset:
        per_path = os.path.join(output_dir, f"summary_{dataset}.txt")
        with open(per_path, "w") as f:
            rows = sorted(rows_by_dataset[dataset], key=lambda r: r[0])
            subtotal_entries = totals_by_dataset[dataset]["entries"]
            subtotal_sum = totals_by_dataset[dataset]["sum"]

            f.write(f"{dataset}\n")
            header = f"{'file':60}  {'entries':>10}   {'sum('+branch_value+')':>16}\n"
            f.write(header)
            f.write("-" * (len(header)-1) + "\n")
            for fname, nent, meanv, sumv in rows:
                f.write(f"{fname:60}  {nent:10d}  {sumv:16.6g}\n")
            f.write("\n")
            f.write(f"SUBTOTAL [{dataset}] ENTRIES: {subtotal_entries}\n")
            f.write(f"SUBTOTAL [{dataset}] SUM({branch_value}): {subtotal_sum:.0f}\n")
    print("Per-dataset summaries written (flag --per-dataset-files).")
