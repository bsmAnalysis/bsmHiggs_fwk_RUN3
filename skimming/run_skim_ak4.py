# ============================
# file: run_skim.py
# ============================

#!/usr/bin/env python3
import uproot
import time
import awkward as ak
import warnings
import numpy as np
import os
import json
import argparse

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from skim_processor_ak4 import NanoAODSkimmerAK4
from collections.abc import Mapping
from uproot.writing.identify import to_TH1x, to_TAxis
from skim_config import branches_to_keep, trigger_groups, met_filter_flags
warnings.filterwarnings("ignore", message="Missing cross-reference index")


def deeply_materialize(data):
    if isinstance(data, ak.Array):
        data = ak.materialized(data)
        layout = ak.to_layout(data, allow_record=True)
        if hasattr(layout, "to_packed"):
            return layout.to_packed()
        return data
    elif isinstance(data, Mapping):
        return {k: deeply_materialize(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deeply_materialize(v) for v in data]
    else:
        return data


def make_th1_1bin(name, title, count):
    edges = np.array([-0.5, 0.5], dtype=np.float64)
    data = np.array([0.0, float(count), 0.0], dtype=np.float64)  # [uf, bin1, of]
    sumw2 = np.array([0.0, float(count), 0.0], dtype=np.float64)

    h = to_TH1x(
        fName=name,
        fTitle=title,
        data=data,
        fEntries=float(count),
        fTsumw=float(count),
        fTsumw2=float(count),
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fSumw2=sumw2,
        fXaxis=to_TAxis(fName="xaxis", fTitle="", fNbins=1, fXmin=-0.5, fXmax=0.5, fXbins=edges),
    )
    return h


def make_th1_from_counts(name, title, edges, counts):
    """
    Build TH1 with given bin edges and integer counts.
    counts should have length nbins (NOT including under/overflow).
    uproot TH1 wants [uf] + counts + [of]
    """
    edges = np.asarray(edges, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    nbins = len(edges) - 1
    assert len(counts) == nbins

    data = np.zeros(nbins + 2, dtype=np.float64)
    data[1:-1] = counts
    sumw2 = data.copy()

    h = to_TH1x(
        fName=name,
        fTitle=title,
        data=data,
        fEntries=float(np.sum(counts)),
        fTsumw=float(np.sum(counts)),
        fTsumw2=float(np.sum(counts)),
        fTsumwx=0.0,
        fTsumwx2=0.0,
        fSumw2=sumw2,
        fXaxis=to_TAxis(
            fName="xaxis",
            fTitle="",
            fNbins=nbins,
            fXmin=float(edges[0]),
            fXmax=float(edges[-1]),
            fXbins=edges,
        ),
    )
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Path to datasets JSON file")
    parser.add_argument("--job-index", type=int, required=True, help="Index of file to process")
    parser.add_argument("--output", type=str, default="skimmed_output.root")
    parser.add_argument("--dataset", type=str, required=True, help="Key in the JSON to process")
    parser.add_argument("--corrections-dir", type=str, default=None)
    parser.add_argument("--golden-json-dir", type=str, default=None)
    parser.add_argument("--do-jer", action="store_true", help="Enable JER smearing + propagation")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-event", type=int, default=0)
    args = parser.parse_args()

    with open(args.json) as f:
        all_datasets = json.load(f)

    dataset_name = args.dataset
    dataset = all_datasets[dataset_name]
    files = dataset["files"]

    if args.job_index >= len(files):
        raise IndexError(f"Index {args.job_index} out of range: {len(files)}")

    file_to_process = files[args.job_index]
    print(f"[INFO] Processing: {file_to_process}")
    print(f"[INFO] Dataset: {dataset_name}")
     # Adjust the config based on the sample name                                                                                                                         
    include_genpart = any(x in dataset_name for x in ["ZH-ZToAll-HToAATo4B", "WH_WToAll_HToAATo4B","VBFH_HToAATo4B","TTH-TTToAll_HToAATo4B","GluGluH-01J_HToAATo4B"])
    include_genttbarid = "TTto" in dataset_name
    
    # Modify the branches to keep                                                                                                                                        
    if include_genpart:
        if "GenPart" not in branches_to_keep:
            branches_to_keep["GenPart"] = ["pt", "eta", "phi","mass", "pdgId", "statusFlags", "genPartIdxMother"]
    if include_genttbarid:
        if "genTtbarId" not in branches_to_keep:
            branches_to_keep["genTtbarId"] = []  # scalar branch   
    # Load NanoAOD
    events = None
    for attempt in range(1, 6):
        try:
            print(f"[INFO] Attempt {attempt} to open NanoAOD file")
            factory = NanoEventsFactory.from_root(
                file_to_process,
                schemaclass=NanoAODSchema,
                uproot_options={"timeout": 300},
            )
            events = factory.events()
            print(f"[INFO] Loaded {int(len(events))} events")
            print("[INFO] NanoAOD opened successfully")
            break
        except Exception as e:
            print(f"[WARNING] Attempt {attempt} failed: {e}")
            if attempt == 5:
                raise
            time.sleep(15)

    # import your config
    

    corrections_dir = args.corrections_dir or os.path.join(os.path.dirname(__file__), "corrections")
    golden_json_dir = args.golden_json_dir or os.path.join(os.path.dirname(__file__), "golden_json")

    processor_instance = NanoAODSkimmerAK4(
        branches_to_keep=branches_to_keep,
        trigger_groups=trigger_groups,
        met_filter_flags=met_filter_flags,
        dataset_name=dataset_name,
        corrections_dir=corrections_dir,
        golden_json_dir=golden_json_dir,
        do_jer=args.do_jer,
        debug=args.debug,
        debug_event_index=args.debug_event,
    )

    result = processor_instance.process(events)

    events_dict = deeply_materialize(result.get("Events", {}))
    counters = result.get("MetaCounters", {})
    meta_hists = result.get("MetaHists", {})

    # compression for faster reads
    #compression = uproot.LZ4(4)

    with uproot.recreate(args.output, compression=uproot.LZMA(9)) as rootfile:
        if events_dict:
            rootfile["Events"] = events_dict

        # Counters (hadd-safe)
        if counters:
            rootfile["nevents"] = make_th1_1bin("nevents", ";nevents;nevents", counters.get("nevents", 0))
            rootfile["nevents_pos"] = make_th1_1bin("nevents_pos", ";nevents_pos;nevents_pos", counters.get("nevents_pos", 0))
            rootfile["nevents_neg"] = make_th1_1bin("nevents_neg", ";nevents_neg;nevents_neg", counters.get("nevents_neg", 0))

        # Debug histos (optional)
        if meta_hists:
            edges_jet = meta_hists["edges_jet"]
            edges_met = meta_hists["edges_met"]

            rootfile["hJetPt_nano"] = make_th1_from_counts("hJetPt_nano", ";Jet pT (Nano);count", edges_jet, meta_hists["hJetPt_nano"])
            rootfile["hJetPt_jec"] = make_th1_from_counts("hJetPt_jec", ";Jet pT (JEC);count", edges_jet, meta_hists["hJetPt_jec"])
            rootfile["hJetPt_jecjer"] = make_th1_from_counts("hJetPt_jecjer", ";Jet pT (JEC+JER);count", edges_jet, meta_hists["hJetPt_jecjer"])

            rootfile["hMet_raw"] = make_th1_from_counts("hMet_raw", ";RawPuppiMET pT;count", edges_met, meta_hists["hMet_raw"])
            rootfile["hMet_t1_jec"] = make_th1_from_counts("hMet_t1_jec", ";Type-1 MET (JEC) pT;count", edges_met, meta_hists["hMet_t1_jec"])
            #rootfile["hMet_t1_jecjer"] = make_th1_from_counts("hMet_t1_jecjer", ";Type-1 MET (JEC+JER) pT;count", edges_met, meta_hists["hMet_t1_jecjer"])

        n_out = int(len(events_dict["event"])) if (events_dict and "event" in events_dict) else 0

    print(f"[INFO] Wrote ROOT file: {args.output}")
    print(f"[INFO] Counters: {counters}")


if __name__ == "__main__":
    main()
