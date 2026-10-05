#!/usr/bin/env python3
import ROOT
import glob
import os
import argparse
from collections import defaultdict

# ---------------- CLI arguments ----------------
parser = argparse.ArgumentParser(
    description="Count events in ROOT files using TH1 counters (grouped per dataset): "
                "nevents, nevents_pos, nevents_neg; compute net = pos - neg."
)
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
parser.add_argument(
    "--h-nevents", default="nevents",
    help="Histogram name for total events counter (default: nevents)"
)
parser.add_argument(
    "--h-pos", default="nevents_pos",
    help="Histogram name for positive genWeight counter (default: nevents_pos)"
)
parser.add_argument(
    "--h-neg", default="nevents_neg",
    help="Histogram name for negative genWeight counter (default: nevents_neg)"
)
parser.add_argument(
    "--use-entries", action="store_true",
    help="Use h.GetEntries() instead of bin content (default: use bin content)"
)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
output_summary = os.path.join(output_dir, "summary.txt")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def read_counter(tf, name, use_entries: bool) -> int:
    """Read a 1-bin TH1 counter from the file."""
    h = tf.Get(name)
    if not h:
        return None
    return int(h.GetEntries()) if use_entries else int(h.GetBinContent(1))

# Expand all directories (with wildcards)
dirs = [d for d in glob.glob(input_dir) if os.path.isdir(d)]
if not dirs:
    print("No matching directories found.")
    raise SystemExit(1)

# Per-dataset storage
# dataset -> list of (file, n, npos, nneg, net)
rows_by_dataset = defaultdict(list)
totals_by_dataset = defaultdict(lambda: {"n": 0, "npos": 0, "nneg": 0, "net": 0})

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
                print(f"  Could not open file: {root_file}")
                continue

            n  = read_counter(tf, args.h_nevents, args.use_entries)
            if n is None:
                print(f"  Missing '{args.h_nevents}' in {root_file} (skipping file)")
                tf.Close()
                continue

            # pos/neg may not exist for data -> treat as 0
            npos = read_counter(tf, args.h_pos, args.use_entries)
            nneg = read_counter(tf, args.h_neg, args.use_entries)
            npos = 0 if npos is None else npos
            nneg = 0 if nneg is None else nneg

            net = int(npos - nneg)

            fname = os.path.basename(root_file)
            rows_by_dataset[dataset].append((fname, n, npos, nneg, net))

            totals_by_dataset[dataset]["n"]    += n
            totals_by_dataset[dataset]["npos"] += npos
            totals_by_dataset[dataset]["nneg"] += nneg
            totals_by_dataset[dataset]["net"]  += net

            # Per-file output
            out_txt = os.path.join(output_dir, fname.replace(".root", ".txt"))
            with open(out_txt, "w") as f:
                f.write(f"{root_file}\n")
                mode = "GetEntries" if args.use_entries else "GetBinContent(1)"
                f.write(f"Read mode: {mode}\n")
                f.write(f"{args.h_nevents}: {n}\n")
                f.write(f"{args.h_pos}:    {npos}\n")
                f.write(f"{args.h_neg}:    {nneg}\n")
                f.write(f"net (pos-neg):   {net}\n")

            tf.Close()

        except Exception as e:
            print(f"  Error processing {root_file}: {e}")
            continue

# ---- Write combined Summary (grouped per dataset) ----
with open(output_summary, "w") as fsum:
    grand_n = grand_pos = grand_neg = grand_net = 0

    for dataset in sorted(rows_by_dataset.keys()):
        rows = sorted(rows_by_dataset[dataset], key=lambda r: r[0])
        sub = totals_by_dataset[dataset]

        fsum.write(f"{dataset}\n")
        header = f"{'file':60}  {'n':>12}  {'npos':>12}  {'nneg':>12}  {'net':>12}\n"
        fsum.write(header)
        fsum.write("-" * (len(header) - 1) + "\n")

        for fname, n, npos, nneg, net in rows:
            fsum.write(f"{fname:60}  {n:12d}  {npos:12d}  {nneg:12d}  {net:12d}\n")

        fsum.write("\n")
        fsum.write(f"SUBTOTAL [{dataset}] n:    {sub['n']}\n")
        fsum.write(f"SUBTOTAL [{dataset}] npos: {sub['npos']}\n")
        fsum.write(f"SUBTOTAL [{dataset}] nneg: {sub['nneg']}\n")
        fsum.write(f"SUBTOTAL [{dataset}] net:  {sub['net']}   (npos - nneg)\n")
        fsum.write("\n" + "=" * 72 + "\n\n")

        grand_n   += sub["n"]
        grand_pos += sub["npos"]
        grand_neg += sub["nneg"]
        grand_net += sub["net"]

    fsum.write(f"\nGRAND TOTAL n:    {grand_n}\n")
    fsum.write(f"GRAND TOTAL npos: {grand_pos}\n")
    fsum.write(f"GRAND TOTAL nneg: {grand_neg}\n")
    fsum.write(f"GRAND TOTAL net:  {grand_net}   (npos - nneg)\n")

print(f"\nCombined per-dataset summary written to {output_summary}")

# ---- Optional: per-dataset summary files ----
if args.per_dataset_files:
    for dataset in rows_by_dataset:
        per_path = os.path.join(output_dir, f"summary_{dataset}.txt")
        rows = sorted(rows_by_dataset[dataset], key=lambda r: r[0])
        sub = totals_by_dataset[dataset]

        with open(per_path, "w") as f:
            f.write(f"{dataset}\n")
            header = f"{'file':60}  {'n':>12}  {'npos':>12}  {'nneg':>12}  {'net':>12}\n"
            f.write(header)
            f.write("-" * (len(header) - 1) + "\n")
            for fname, n, npos, nneg, net in rows:
                f.write(f"{fname:60}  {n:12d}  {npos:12d}  {nneg:12d}  {net:12d}\n")

            f.write("\n")
            f.write(f"SUBTOTAL [{dataset}] n:    {sub['n']}\n")
            f.write(f"SUBTOTAL [{dataset}] npos: {sub['npos']}\n")
            f.write(f"SUBTOTAL [{dataset}] nneg: {sub['nneg']}\n")
            f.write(f"SUBTOTAL [{dataset}] net:  {sub['net']}   (npos - nneg)\n")

    print("Per-dataset summaries written (flag --per-dataset-files).")
