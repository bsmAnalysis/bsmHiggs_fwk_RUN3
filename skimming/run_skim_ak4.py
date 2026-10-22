"""Nominal-only AK4 skim; legacy Coffea 0.7 / Awkward 1 / Uproot 4.

Default: >=2 central tight-lepton-veto candidates without a pT cut,
nominal JEC+JER on MC jets, and JEC-only Type-1 MET with replay inputs.
All up/down calculations move to compact_jet_variations.py in analysis.
"""
import argparse
import copy
import json
import os
import time
import warnings
from collections.abc import Mapping

import awkward as ak
import coffea
import numpy as np
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from uproot.writing.identify import to_TH1x, to_TAxis

from skim_processor_ak4_upart import NanoAODSkimmerAK4
from skim_config import branches_to_keep, trigger_groups, met_filter_flags

warnings.filterwarnings("ignore", message="Missing cross-reference index")


def deeply_materialize(data):
    """Keep packed high-level arrays for the ROOT writer."""
    if isinstance(data, ak.Array):
        if hasattr(ak, "materialized"):
            data = ak.materialized(data)
        if hasattr(ak, "to_packed"):
            return ak.to_packed(data)
        return ak.packed(data)
    if isinstance(data, Mapping):
        return {key: deeply_materialize(value) for key, value in data.items()}
    if isinstance(data, list):
        return [deeply_materialize(value) for value in data]
    return data


def make_th1_1bin(name, title, count):
    edges = np.array([-0.5, 0.5], dtype=np.float64)
    data = np.array([0.0, float(count), 0.0], dtype=np.float64)
    return to_TH1x(
        fName=name, fTitle=title, data=data, fEntries=float(count),
        fTsumw=float(count), fTsumw2=float(count), fTsumwx=0.0,
        fTsumwx2=0.0, fSumw2=data.copy(),
        fXaxis=to_TAxis(fName="xaxis", fTitle="", fNbins=1,
                      fXmin=-0.5, fXmax=0.5, fXbins=edges),
    )


def make_th1_from_counts(name, title, edges, counts):
    edges = np.asarray(edges, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    nbins = len(edges) - 1
    if len(counts) != nbins:
        raise ValueError(f"Histogram {name}: {len(counts)} counts for {nbins} bins")
    data = np.zeros(nbins + 2, dtype=np.float64)
    data[1:-1] = counts
    return to_TH1x(
        fName=name, fTitle=title, data=data, fEntries=float(np.sum(counts)),
        fTsumw=float(np.sum(counts)), fTsumw2=float(np.sum(counts)),
        fTsumwx=0.0, fTsumwx2=0.0, fSumw2=data.copy(),
        fXaxis=to_TAxis(fName="xaxis", fTitle="", fNbins=nbins,
                      fXmin=float(edges[0]), fXmax=float(edges[-1]), fXbins=edges),
    )


def write_output(output_path, result):
    events_dict = deeply_materialize(result.get("Events", {}))
    counters = result.get("MetaCounters", {})
    meta_hists = result.get("MetaHists", {})
    with uproot.recreate(output_path, compression=uproot.LZMA(9)) as rootfile:
        if result.get("MetaSystematics"):
            rootfile["JERCSystematics"] = json.dumps(result["MetaSystematics"], sort_keys=True)
        if events_dict:
            # In Uproot 4 this writes a TTree with flattened Jet_* / PuppiMET_*.
            rootfile["Events"] = events_dict
        for name in ("nevents", "nevents_pos", "nevents_neg"):
            if counters:
                rootfile[name] = make_th1_1bin(name, f";{name};{name}", counters.get(name, 0))
        if meta_hists:
            for name, title in (
                ("hJetPt_nano", "Jet pT (Nano)"),
                ("hJetPt_jec", "Jet pT (JEC)"),
                ("hJetPt_jecjer", "Jet pT (JEC+JER)"),
            ):
                rootfile[name] = make_th1_from_counts(
                    name, f";{title};count", meta_hists["edges_jet"], meta_hists[name]
                )
            for name, title in (
                ("hMet_raw", "Input NanoAOD PuppiMET pT"),
                ("hMet_t1_jec", "Type-1 MET (JEC) pT"),
            ):
                rootfile[name] = make_th1_from_counts(
                    name, f";{title};count", meta_hists["edges_met"], meta_hists[name]
                )
    return len(events_dict["event"]) if events_dict else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="Dataset JSON path")
    parser.add_argument("--job-index", type=int, required=True, help="Input file index")
    parser.add_argument("--output", default="skimmed_output.root")
    parser.add_argument("--dataset", required=True, help="Dataset key in the JSON")
    parser.add_argument("--corrections-dir", default=None)
    parser.add_argument("--golden-json-dir", default=None)
    jer_options = parser.add_mutually_exclusive_group()
    jer_options.add_argument("--do-jer", dest="do_jer", action="store_true",
                             help="Apply nominal MC jet JER (already the default); MET stays JEC-only")
    jer_options.add_argument("--no-jer", dest="do_jer", action="store_false",
                             help="Explicitly produce JEC-only MC jets")
    parser.set_defaults(do_jer=True)
    parser.add_argument("--jet-selection-mode", choices=("fixed", "standard", "union"), default="fixed",
                        help="Fixed central/JetID candidates without pT cut (default); legacy standard/union available")
    parser.add_argument("--jet-systematics-storage", choices=("nominal", "compact", "full"), default="nominal",
                        help="Nominal quantities plus jet/MET replay inputs (default); legacy compact/full available")
    parser.add_argument("--met-jes-selection", choices=("nominal", "varied"), default="varied",
                        help="Record the later MET JES membership convention: varied (default) or fixed nominal")
    parser.add_argument("--jes-scheme", choices=("reduced", "full", "total"), default="reduced",
                        help="11 Regrouped sources (default), 27 full sources, or Total only (single-NP use requires JME approval)")
    parser.add_argument("--list-jes-sources", action="store_true",
                        help="Validate correction inputs and print selected JES sources without opening NanoAOD")
    parser.add_argument("--expected-jerc-sha256", default=None,
                        help="Optional checksum guard for regJet_jerc.json.gz")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-event", type=int, default=0)
    return parser.parse_args()


def require_files(paths):
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Required job inputs are missing: " + ", ".join(missing))


def main():
    args = parse_args()
    with open(args.json) as source:
        all_datasets = json.load(source)
    if args.dataset not in all_datasets:
        raise KeyError(f"Dataset {args.dataset!r} is absent from {args.json}")
    files = all_datasets[args.dataset]["files"]
    if not 0 <= args.job_index < len(files):
        raise IndexError(f"Index {args.job_index} outside [0, {len(files)})")
    file_to_process = files[args.job_index]
    project_dir = os.path.dirname(os.path.abspath(__file__))
    corrections_dir = args.corrections_dir or os.path.join(project_dir, "corrections")
    golden_json_dir = args.golden_json_dir or os.path.join(project_dir, "golden_json")
    require_files([os.path.join(corrections_dir, "regJet_jerc.json.gz"),
                   os.path.join(corrections_dir, "jetvetomaps.json.gz")])
    if args.list_jes_sources:
        inspector = NanoAODSkimmerAK4(
            branches_to_keep=branches_to_keep, trigger_groups=trigger_groups,
            met_filter_flags=met_filter_flags, dataset_name=args.dataset,
            corrections_dir=corrections_dir, golden_json_dir=golden_json_dir,
            do_jer=args.do_jer, jes_scheme=args.jes_scheme,
            met_jes_selection=args.met_jes_selection,
            jet_selection_mode=args.jet_selection_mode,
            jet_systematics_storage=args.jet_systematics_storage,
        )
        inspector._ensure_jerc_loaded()
        if args.expected_jerc_sha256 and inspector._jerc_sha256 != args.expected_jerc_sha256.lower():
            raise RuntimeError("JERC payload SHA256 does not match --expected-jerc-sha256")
        print(json.dumps(inspector._systematics_metadata(), indent=2, sort_keys=True))
        return
    print(f"[INFO] Versions: coffea={coffea.__version__}, awkward={ak.__version__}, uproot={uproot.__version__}")
    if int(uproot.__version__.split(".")[0]) != 4:
        raise RuntimeError("This launcher uses the Uproot 4 dictionary-to-TTree writer; use your legacy container")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Processing: {file_to_process}")
    if args.jet_systematics_storage == "nominal" and args.jet_selection_mode != "fixed":
        raise ValueError("Nominal-only output requires --jet-selection-mode fixed (no pT cut)")
    if args.jet_selection_mode == "fixed":
        print(f"[INFO] Jet selection: >=2 tight-lepton-veto candidates with |eta|<{NanoAODSkimmerAK4.JET_ETA_MAX:g}; NO pT cut")
    else:
        print(f"[INFO] Jet selection: {args.jet_selection_mode}, threshold {NanoAODSkimmerAK4.JET_PT_MIN:g} GeV")
    print(f"[INFO] MET JES membership: {args.met_jes_selection}; JER -> MET disabled")
    print(f"[INFO] JES scheme for later reconstruction: {args.jes_scheme}")
    print(f"[INFO] Jet systematic storage: {args.jet_systematics_storage}")
    for attempt in range(1, 6):
        try:
            print(f"[INFO] Attempt {attempt} to open NanoAOD file")
            factory = NanoEventsFactory.from_root(
                file_to_process, treepath="Events", schemaclass=NanoAODSchema,
                uproot_options={"timeout": 300},
            )
            events = factory.events()
            print(f"[INFO] Loaded {len(events)} events")
            break
        except Exception as exc:
            print(f"[WARNING] Attempt {attempt} failed: {exc}")
            if attempt == 5:
                raise
            time.sleep(15)
    is_data = not hasattr(events, "genWeight")
    if is_data:
        require_files([os.path.join(golden_json_dir, "Cert_Collisions2024_378981_386951_Golden.json")])
    elif args.do_jer:
        require_files([os.path.join(corrections_dir, "jer_smear.json.gz")])
    print(f"[INFO] MC jet JER enabled: {args.do_jer and not is_data}")
    dataset_branches = copy.deepcopy(branches_to_keep)
    include_genpart = any(label in args.dataset for label in (
        "ZH-ZToAll-HToAATo4B", "WH_WToAll_HToAATo4B", "VBFH_HToAATo4B",
        "TTH-TTToAll_HToAATo4B", "GluGluH-01J_HToAATo4B",
    ))
    if include_genpart:
        dataset_branches.setdefault("GenPart", ["pt", "eta", "phi", "mass", "pdgId",
                                               "statusFlags", "genPartIdxMother"])
    if "TTto" in args.dataset:
        dataset_branches.setdefault("genTtbarId", [])
    processor_instance = NanoAODSkimmerAK4(
        branches_to_keep=dataset_branches, trigger_groups=trigger_groups,
        met_filter_flags=met_filter_flags, dataset_name=args.dataset,
        corrections_dir=corrections_dir, golden_json_dir=golden_json_dir,
        do_jer=args.do_jer, jet_selection_mode=args.jet_selection_mode,
        met_jes_selection=args.met_jes_selection,
        jes_scheme=args.jes_scheme,
        jet_systematics_storage=args.jet_systematics_storage,
        debug=args.debug, debug_event_index=args.debug_event,
    )
    processor_instance._ensure_jerc_loaded()
    if args.expected_jerc_sha256 and processor_instance._jerc_sha256 != args.expected_jerc_sha256.lower():
        raise RuntimeError("JERC payload SHA256 does not match --expected-jerc-sha256")
    result = processor_instance.process(events)
    n_out = write_output(args.output, result)
    print(f"[INFO] Wrote ROOT file: {args.output}; accepted events: {n_out}")
    print(f"[INFO] Counters: {result.get('MetaCounters', {})}")


if __name__ == "__main__":
    main()
