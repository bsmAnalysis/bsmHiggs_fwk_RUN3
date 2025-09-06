import sys, hist, uproot, time, warnings, os 
from coffea.nanoevents import NanoEventsFactory, BaseSchema
from Wh_processor import Wh_Processor
import awkward as ak
import numpy as np
import json
import hist as _hist
import inspect
import argparse
from uproot.writing import identify as upid
from array import array
warnings.filterwarnings("ignore", message="Missing cross-reference index")

# ------------------ Uproot THx writers with Sumw2 ------------------

def _is_num_axis(ax):
    return isinstance(ax, (_hist.axis.Regular, _hist.axis.Variable))

#----------------------------------------------------------------------------------------------------------------------------------------------

def _call_to_TH1x(name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2, sumw2, xaxis, yaxis, zaxis):
    """
    Call identify.to_TH1x with the correct signature for the installed uproot version.
    """
    sig = inspect.signature(upid.to_TH1x)
    params = list(sig.parameters.keys())
    if len(params) > 0 and params[0] == "classname":
        return upid.to_TH1x(
            "TH1D", name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2,
            sumw2, xaxis, yaxis, zaxis
        )
    elif "classname" in sig.parameters:
        return upid.to_TH1x(
            name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2,
            sumw2, xaxis, yaxis, zaxis, classname="TH1D"
        )
    else:
        return upid.to_TH1x(
            name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2,
            sumw2, xaxis, yaxis, zaxis
        )

#----------------------------------------------------------------------------------------------------------------------------------------------

def _call_to_TH2x(name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2, fTsumwy, fTsumwy2, sumw2, xaxis, yaxis, zaxis):
    sig = inspect.signature(upid.to_TH2x)
    params = list(sig.parameters.keys())
    if len(params) > 0 and params[0] == "classname":
        return upid.to_TH2x(
            "TH2D", name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2,
            fTsumwy, fTsumwy2, sumw2, xaxis, yaxis, zaxis
        )
    elif "classname" in sig.parameters:
        return upid.to_TH2x(
            name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2,
            fTsumwy, fTsumwy2, sumw2, xaxis, yaxis, zaxis, classname="TH2D"
        )
    else:
        return upid.to_TH2x(
            name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2,
            fTsumwy, fTsumwy2, sumw2, xaxis, yaxis, zaxis
        )

#----------------------------------------------------------------------------------------------------------------------------------------------

def write_hist_uproot_sumw2(rootfile, fullpath, h):
    """
    Write 1D/2D numeric-axes boost-hist 'hist.Hist' to ROOT TH1/TH2 with Sumw2.
    Falls back to plain numpy if uproot signature mismatches or axes are categorical.
    """
    try:
        # Skip truly empty
        if float(np.sum(h.values())) == 0.0:
            return

        # 1D numeric
        if h.ndim == 1 and _is_num_axis(h.axes[0]):
            counts, xedges = h.to_numpy()
            vari = h.variances()
            nb = len(xedges) - 1

            data = np.zeros(nb + 2, dtype=np.float64); data[1:-1] = counts
            sumw2 = np.zeros(nb + 2, dtype=np.float64); sumw2[1:-1] = (vari if vari is not None else counts)

            xcent   = 0.5 * (xedges[:-1] + xedges[1:])
            fEntries = float((counts**2).sum() / max(sumw2[1:-1].sum(), 1e-12))
            fTsumw   = float(counts.sum())
            fTsumw2  = float(sumw2[1:-1].sum())
            fTsumwx  = float((counts * xcent).sum())
            fTsumwx2 = float((counts * xcent * xcent).sum())

            xaxis = upid.to_TAxis("xaxis", "xaxis", nb, float(xedges[0]), float(xedges[-1]), xedges.astype(np.float64))
            yaxis = upid.to_TAxis("yaxis", "yaxis", 0, 0.0, 0.0, None)
            zaxis = upid.to_TAxis("zaxis", "zaxis", 0, 0.0, 0.0, None)

            th1 = _call_to_TH1x(fullpath, fullpath, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2, sumw2, xaxis, yaxis, zaxis)
            rootfile[fullpath] = th1
            return

        if h.ndim == 2 and _is_num_axis(h.axes[0]) and _is_num_axis(h.axes[1]):
            counts, xedges, yedges = h.to_numpy()
            vari = h.variances()
            nx, ny = counts.shape

            data = np.zeros((nx + 2, ny + 2), dtype=np.float64); data[1:-1, 1:-1] = counts
            sumw2 = np.zeros((nx + 2, ny + 2), dtype=np.float64); sumw2[1:-1, 1:-1] = (vari if vari is not None else counts)

            xcent = 0.5 * (xedges[:-1] + xedges[1:])
            ycent = 0.5 * (yedges[:-1] + yedges[1:])
            fEntries = float((counts**2).sum() / max(sumw2[1:-1,1:-1].sum(), 1e-12))
            fTsumw   = float(counts.sum())
            fTsumw2  = float(sumw2[1:-1,1:-1].sum())
            fTsumwx  = float((counts * xcent[:, None]).sum())
            fTsumwx2 = float((counts * (xcent[:, None] ** 2)).sum())
            fTsumwy  = float((counts * ycent[None, :]).sum())
            fTsumwy2 = float((counts * (ycent[None, :] ** 2)).sum())

            xaxis = upid.to_TAxis("xaxis", "xaxis", nx, float(xedges[0]), float(xedges[-1]), xedges.astype(np.float64))
            yaxis = upid.to_TAxis("yaxis", "yaxis", ny, float(yedges[0]), float(yedges[-1]), yedges.astype(np.float64))
            zaxis = upid.to_TAxis("zaxis", "zaxis", 0, 0.0, 0.0, None)

            th2 = _call_to_TH2x(fullpath, fullpath, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2, fTsumwy, fTsumwy2, sumw2, xaxis, yaxis, zaxis)
            rootfile[fullpath] = th2
            return

        # Fallback: categorical axes or anything unexpected → write plain numpy 
        rootfile[fullpath] = h.to_numpy()

    except Exception as e:
        print(f"[WARN] TH writer failed for {fullpath}: {e} — writing plain numpy without sumw2")
        rootfile[fullpath] = h.to_numpy()

#----------------------------------------------------------------------------------------------------------------------------------------------

# --- Argument parser --- #
parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str, required=True, help="Path to JSON file")
parser.add_argument("--job-index", type=int, required=True)
parser.add_argument("--output", type=str, required=True, help="Histogram output ROOT file")
parser.add_argument("--dataset", type=str, required=True, help="Dataset key inside JSON")
parser.add_argument("--bdt_output", type=str, default=None, help="Optional: output file for BDT trees")
args = parser.parse_args()

# --- Load dataset info --- #
with open(args.json) as f:
    all_datasets = json.load(f)
    
if args.dataset not in all_datasets:
    raise ValueError(f"[ERROR] Dataset '{args.dataset}' not found in {args.json}")

dataset = all_datasets[args.dataset]
meta = dataset["metadata"]
files = dataset["files"]

# Safe check
if args.job_index >= len(files):
    raise IndexError(f"[ERROR] job-index {args.job_index} is out of range (0 - {len(files)-1})")

file_to_process = files[args.job_index]
dataset_name = meta["sample"]
nevts   = int(meta["nevents"])
isMC    = meta["isMC"].lower() == "true"
lumi    = 108960  
isMVA   = False
runEval = True
isQCD   = False

print(f"[INFO] Processing file {args.job_index+1}/{len(files)}: {file_to_process}")
if isMC:
    xsec = float(meta["xsec"])
    print(f"[INFO] Sample: {dataset_name} (xsec={xsec}, nevts={nevts})")
else:
    xsec = 1.0
    print(f"[INFO] Sample: {dataset_name} (xsec=1.0, nevts={nevts})")

# --- Load NanoAOD events --- #
for attempt in range(1, 6):
    try:
        #factory = NanoEventsFactory.from_root(
        #    file_to_process,
        #    schemaclass=NanoAODSchema,
        #    uproot_options={"timeout": 300}
        #)
        #events = factory.events()
        
        events = NanoEventsFactory.from_root(file_to_process,
                                             treepath="Events", 
                                             schemaclass=BaseSchema,
                                             uproot_options={"timeout": 300}
                                             ).events()
        
        break
    except Exception as e:
        print(f"[WARNING] Attempt {attempt} failed: {e}")
        if attempt == 5:
            print("[ERROR] Max attempts reached. Skipping file.")
            sys.exit(1)
            time.sleep(10)
            
# --- Rebuild Muon ---
events["Muon"] = ak.zip({f: events[f"Muon_{f}"] for f in [
    "pt","eta","phi","charge","tightId","looseId","mass","pfRelIso04_all"
]})
# --- Rebuild Electron ---
events["Electron"] = ak.zip({f: events[f"Electron_{f}"] for f in [
    "pt","eta","phi","charge","cutBased","mass","pfRelIso03_all",
    "seedGain","r9","superclusterEta","mvaIso_WP90"
]})

# --- Rebuild Jet ---
jet_fields = [
    "pt","eta","phi","mass","rawFactor","area","pt_genMatched",
    "btagUParTAK4probbb","btagUParTAK4B","passJetIdTightLepVeto",
    "pt_regressed"
]
if isMC:
    jet_fields += ["hadronFlavour", "partonFlavour"]

events["Jet"] = ak.zip({f: events[f"Jet_{f}"] for f in jet_fields})

# --- Rebuild PuppiMET ---
events["PuppiMET"] = ak.zip({f: events[f"PuppiMET_{f}"] for f in ["pt", "phi"]})

# --- Pileup (PU) info ---
if isMC:
    events["Pileup"] = ak.zip({f: events[f"Pileup_{f}"] for f in ["nPU", "nTrueInt"]})

# --- PV info ---
if not dataset_name.startswith("QCD"):
    events["PV"] = ak.zip({f: events[f"PV_{f}"] for f in ["npvsGood", "npvs"]})


#  Split TTbar samples to tt+bb tt+cc tt+qq
#  way to split found at: https://github.com/cms-sw/cmssw/blob/master/TopQuarkAnalysis/TopTools/plugins/GenTtbarCategorizer.cc

if dataset_name.startswith("TT") and not isMVA:
    print("[INFO] TTbar sample detected splitting into ttLF, ttCC, ttBB")
    gen_id = events.genTtbarId

    masks = {
        "ttLF": (gen_id % 100 == 0),
        "ttCC": (gen_id % 100 >= 41) & (gen_id % 100 <= 45),
        "ttBB": (gen_id % 100 >= 51) & (gen_id % 100 <= 55),
    }

    job_suffix = os.path.basename(args.output).split("_")[-1]

    for flavor, mask in masks.items():
        events_flavor = events[mask]
        n_flavor = len(events_flavor)
        print(f"[INFO] Processing flavor: {flavor} (nEvents: {n_flavor})")

        if n_flavor == 0:
            print(f"[INFO] No events found for {flavor} — skipping.")
            continue

        processor_instance = Wh_Processor(
            xsec=xsec,
            nevts=nevts,
            isMC=isMC,
            dataset_name=dataset_name,
            isQCD=isQCD,
            isMVA=False,
            runEval=runEval,
        )
        output = processor_instance.process(events_flavor)

        sample_base = os.path.basename(dataset_name).replace(".root", "").replace("/", "_")
        out_name = f"{sample_base}_{flavor}_{job_suffix}"

        with uproot.recreate(out_name) as rootfile:
            for name, h in output.items():
                if not isinstance(h, hist.Hist):
                    continue
                if np.sum(h.values()) == 0:
                    continue
        
                if "gen:" in name:
                    uproot_name = f"gen/{name}"
                elif "_boosted" in name:
                    uproot_name = f"boosted/{name}"
                elif "_resolved" in name:
                    uproot_name = f"resolved/{name}"
                else:
                    uproot_name = name
        
                write_hist_uproot_sumw2(rootfile, uproot_name, h)
               
else:
    # --- Normal (non-TTbar) processing --- #
    if isMC:
        xsec = float(meta["xsec"])
    else:
        xsec = 1.0
        
    processor_instance = Wh_Processor(
           xsec=xsec,
           nevts=nevts,
           isMC=isMC,
           dataset_name=dataset_name,
           isQCD=isQCD,
           isMVA=isMVA,
           runEval=runEval
       )

    output = processor_instance.process(events)

    # --- Save output root file--- #
    out_name = args.output
    with uproot.recreate(out_name) as rootfile:
        for name, h in output.items():
            if not isinstance(h, hist.Hist):
                continue
            if np.sum(h.values()) == 0:
                continue
    
            if "gen:" in name:
                uproot_name = f"gen/{name}"
            elif "_boosted" in name:
                uproot_name = f"boosted/{name}"
            elif "_resolved" in name:
                uproot_name = f"resolved/{name}"
            else:
                uproot_name = name
    
            write_hist_uproot_sumw2(rootfile, uproot_name, h)

    print(f"[INFO] Wrote ROOT histograms with Sumw2 to {out_name}")
                                                        
    # --- Save BDT trees --- #
    bdt_output_name = args.bdt_output or f"bdt_{os.path.basename(args.output)}"
    tree_data = output.get("trees", None)

    # fallback in case trees not injected into output
    if not tree_data and hasattr(processor_instance, "_trees"):
        tree_data = processor_instance._trees

    if tree_data:
        with uproot.recreate(bdt_output_name) as bdtfile:
            for regime, tree_dict in tree_data.items():
                if not tree_dict:
                    # Define expected variables per regime
                    if regime == "boosted":
                        keys = processor_instance.bdt_eval_boosted.var_list
                    elif regime == "resolved":
                        keys = processor_instance.bdt_eval_resolved.var_list
                    else:
                        keys = ["dummy"]
                        
                    # Write empty tree with correct structure
                    dummy_tree = {k: np.array([], dtype=np.float64) for k in keys}
                    print(f"[INFO] Writing empty tree for regime '{regime}' to maintain hadd compatibility.")
                    bdtfile[regime] = dummy_tree
                    continue
                
                bdtfile[regime] = tree_dict
        print(f"[INFO] Saved BDT training trees in: {bdt_output_name}")
    else:
        print("[WARNING] No BDT trees found — nothing was written to tree output file.")


