#!/usr/bin/env python3
# write L/M/T b-tag efficiency count maps per input root

import uproot
import awkward as ak
import numpy as np
import argparse
import json
import hist

from coffea.nanoevents import NanoEventsFactory, BaseSchema
from btag_efficiency_processor import BTagEfficiencyProcessor


def make_hist2_from_counts(counts, xedges, yedges):
    """Store raw jet counts with Poisson variance in a ROOT-compatible TH2."""
    counts = np.asarray(counts, dtype=np.float64)
    h = hist.Hist(
        hist.axis.Variable(xedges, name="pt", label="p_{T} [GeV]"),
        hist.axis.Variable(yedges, name="abseta", label="|#eta|"),
        storage=hist.storage.Weight(),
    )
    # These maps contain unweighted counts
    h.view().value[...] = counts
    h.view().variance[...] = counts
    return h


parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--job-index", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# Each job processes one file from the selected dataset.
with open(args.json) as f:
    datasets = json.load(f)
file = datasets[args.dataset]["files"][args.job_index]
events = NanoEventsFactory.from_root(
    file,
    treepath="Events",
    schemaclass=BaseSchema,
).events()

# BaseSchema reads flat branches; assemble the collections used by the processor.
events["Muon"] = ak.zip({
    f: events[f"Muon_{f}"] for f in [
        "pt", "eta", "phi", "charge", "tightId",
        "looseId", "mass", "pfRelIso04_all",
    ]
})
events["Electron"] = ak.zip({
    f: events[f"Electron_{f}"] for f in [
        "pt", "eta", "phi", "charge", "cutBased",
        "mass", "pfRelIso03_all", "superclusterEta",
        "mvaIso_WP90", "mvaIso_WP80",
    ]
})
jet_fields = [
    "pt",
    "eta",
    "phi",
    "mass",
    "btagUParTAK4probbb",
    "btagUParTAK4B",
    "hadronFlavour",
]
events["Jet"] = ak.zip({f: events[f"Jet_{f}"] for f in jet_fields})

# UParTAK4B working points for 2024.
WPS = {
    "L": 0.0246,
    "M": 0.1272,
    "T": 0.4648,
}
# Indices match the processor count arrays.
FLAVORS = {"b": 0, "c": 1, "light": 2}
TT_CATEGORIES = ["ttLF", "ttCC", "ttBB"]
# nb4p includes every event with four or more selected truth-b jets.
NBJETS = {"nb0": 0, "nb1": 1, "nb2": 2, "nb3": 3, "nb4p": 4}

# Store all working points and ttbar categories in the same output file.
with uproot.recreate(args.output) as fout:
    if args.dataset.startswith("TTto"):
        categories = TT_CATEGORIES
    else:
        categories = [None]

    for tt_cat in categories:
        for wp_name, wp_value in WPS.items():
            processor = BTagEfficiencyProcessor(
                wp=wp_value,
                tagger="btagUParTAK4B",
                tt_flavor=tt_cat,
            )
            out = processor.process(events)
            prefix = tt_cat if tt_cat is not None else args.dataset

            print("===================================")
            print("prefix:", prefix)
            print("WP:", wp_name)
            print("denom sum:", out["BTagEff_Denom"].sum())
            print("num sum:", out["BTagEff_Num"].sum())
            print("===================================")
            # Write numerator and denominator separately for later merging.
            for flav_name, flav_idx in FLAVORS.items():
                for nb_name, nb_idx in NBJETS.items():
                    denom_2d = out["BTagEff_Denom"][:, :, flav_idx, nb_idx]
                    num_2d = out["BTagEff_Num"][:, :, flav_idx, nb_idx]
                    fout[f"{prefix}_BTagEff_Denom_{flav_name}_{nb_name}_WP{wp_name}"] = (
                        make_hist2_from_counts(denom_2d, out["pt_edges"], out["eta_edges"])
                    )
                    fout[f"{prefix}_BTagEff_Num_{flav_name}_{nb_name}_WP{wp_name}"] = (
                        make_hist2_from_counts(num_2d, out["pt_edges"], out["eta_edges"])
                    )
    print("[DEBUG] keys written:")
    print(fout.keys())
