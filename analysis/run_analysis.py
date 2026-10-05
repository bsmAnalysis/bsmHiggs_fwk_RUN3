import sys, hist, uproot, time, warnings, os
from coffea.nanoevents import NanoEventsFactory, BaseSchema
from ZH_2lep_processor_fixedWP import TOTAL_Processor
import awkward as ak
import numpy as np
import json
import hist as _histC
from hist import Hist as HistType
import inspect
import argparse
from uproot.writing import identify as upid

warnings.filterwarnings("ignore", message="Missing cross-reference index")

def _is_num_axis(ax):
    return isinstance(
        ax,
        (
            _histC.axis.Regular,
            _histC.axis.Variable,
            _histC.axis.Integer,
        ),
    )

def _call_to_TH1x(name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2, sumw2, xaxis, yaxis, zaxis):
    """
    Call identify.to_TH1x with the correct signature for the installed uproot version.
    Some versions are to_TH1x(name, title, data, ... [, classname=])
    Others are     to_TH1x(classname, name, title, data, ...)
    """
    sig = inspect.signature(upid.to_TH1x)
    params = list(sig.parameters.keys())
    if len(params) > 0 and params[0] == "classname":
        # old-style: classname first
        return upid.to_TH1x(
            "TH1D", name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2,
            sumw2, xaxis, yaxis, zaxis
        )
    elif "classname" in sig.parameters:
        # classname exists but not first → pass by keyword
        return upid.to_TH1x(
            name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2,
            sumw2, xaxis, yaxis, zaxis, classname="TH1D"
        )
    else:
        # no classname parameter
        return upid.to_TH1x(
            name, title, data, fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2,
            sumw2, xaxis, yaxis, zaxis
        )

#----------------------------------------------------------------------------------------------------------------------------------------------
def make_hist2_from_values(values, variances, xedges, yedges, xname, yname, xlabel=None, ylabel=None):
    values = np.asarray(values, dtype=np.float64)

    if variances is None:
        variances = values
    variances = np.asarray(variances, dtype=np.float64)

    h2 = hist.Hist(
        hist.axis.Variable(np.asarray(xedges, dtype=np.float64), name=xname, label=xlabel or xname),
        hist.axis.Variable(np.asarray(yedges, dtype=np.float64), name=yname, label=ylabel or yname),
        storage=hist.storage.Weight(),
    )

    h2.view().value[...] = values
    h2.view().variance[...] = variances

    return h2

def make_hist2_dbtag(counts, edges):
    h = hist.Hist(
        hist.axis.Variable(edges, name="lead", label="leading UParTAK4probbb"),
        hist.axis.Variable(edges, name="sublead", label="subleading UParTAK4probbb"),
        storage=hist.storage.Weight(),
    )

    h.view().value[...] = np.asarray(counts, dtype=np.float64)
    h.view().variance[...] = np.asarray(counts, dtype=np.float64)

    return h

def _call_to_TH2x(fullpath, data_flat,
                  fEntries, fTsumw, fTsumw2, fTsumwx, fTsumwx2, fTsumwxy, fTsumwy, fTsumwy2,
                  sumw2_flat, xaxis, yaxis, zaxis):
    """
    Call identify.to_TH2x with the correct signature (varies across uproot versions).
    Always passes x/y edges via TAxis and includes the cross-term fTsumwxy.
    """
    sig = inspect.signature(upid.to_TH2x)
    params = set(sig.parameters.keys())

    # Newer uproot: named fields (fName/fTitle/…)
    if {"fName","fTitle","data","fEntries","fTsumw","fTsumw2","fTsumwx","fTsumwx2",
        "fTsumwxy","fTsumwy","fTsumwy2","fSumw2","fXaxis","fYaxis","fZaxis"}.issubset(params):
        return upid.to_TH2x(
            fName=str(fullpath),
            fTitle=str(fullpath),
            data=data_flat,
            fEntries=fEntries,
            fTsumw=fTsumw,     fTsumw2=fTsumw2,
            fTsumwx=fTsumwx,   fTsumwx2=fTsumwx2,
            fTsumwxy=fTsumwxy,
            fTsumwy=fTsumwy,   fTsumwy2=fTsumwy2,
            fSumw2=sumw2_flat,
            fXaxis=xaxis, fYaxis=yaxis, fZaxis=zaxis,
        )

    # Mid uproot: keyword with classname
    if "classname" in params:
        return upid.to_TH2x(
            name=str(fullpath), title=str(fullpath),
            data=data_flat,
            fEntries=fEntries,
            fTsumw=fTsumw,     fTsumw2=fTsumw2,
            fTsumwx=fTsumwx,   fTsumwx2=fTsumwx2,
            fTsumwxy=fTsumwxy,
            fTsumwy=fTsumwy,   fTsumwy2=fTsumwy2,
            sumw2=sumw2_flat,
            xaxis=xaxis, yaxis=yaxis, zaxis=zaxis,
            classname="TH2D",
        )

    # Old uproot: positional args
    return upid.to_TH2x(
        "TH2D", str(fullpath), str(fullpath),
        data_flat,
        fEntries, fTsumw, fTsumw2,
        fTsumwx, fTsumwx2, fTsumwxy,
        fTsumwy, fTsumwy2,
        sumw2_flat, xaxis, yaxis, zaxis,
    )

#----------------------------------------------------------------------------------------------------------------------------------------------

def write_hist_uproot_sumw2(rootfile, fullpath, h):
    if not isinstance(h, HistType):
        raise TypeError(
            f"write_hist_uproot_sumw2 expected hist.Hist for '{fullpath}', "
            f"got {type(h).__name__}"
        )

    try:
        values_check = np.asarray(
            h.values(flow=False),
            dtype=np.float64,
        )
        if not np.any(values_check != 0.0):
            return

        # ------------------------------------------------------------
        # 1D numeric
        # ------------------------------------------------------------
       
        if h.ndim == 1 and _is_num_axis(h.axes[0]):

            ax = h.axes[0]
            xedges = np.asarray(ax.edges, dtype=np.float64)
            nb = len(xedges) - 1
            
            counts_noflow = h.values(flow=False)
            vari_noflow = h.variances(flow=False)
            
            counts_flow = h.values(flow=True)
            vari_flow = h.variances(flow=True)
            
            data = np.zeros(nb + 2, dtype=np.float64)
            sumw2 = np.zeros(nb + 2, dtype=np.float64)
            
            # If hist has flow bins, counts_flow should already be nb+2.
            # Otherwise keep normal bins only.
            if len(counts_flow) == nb + 2:
                data[:] = counts_flow
                if vari_flow is not None:
                    sumw2[:] = vari_flow
                else:
                    sumw2[:] = counts_flow
            else:
                data[1:-1] = counts_noflow
                if vari_noflow is not None:
                    sumw2[1:-1] = vari_noflow
                else:
                    sumw2[1:-1] = counts_noflow
                    
            xcent = 0.5 * (xedges[:-1] + xedges[1:])

            # ROOT bookkeeping. For integrals, data/sumw2 are what matter.
            fEntries = float((data.sum() ** 2) / max(sumw2.sum(), 1e-12))
            fTsumw = float(data.sum())
            fTsumw2 = float(sumw2.sum())
            fTsumwx = float((data[1:-1] * xcent).sum())
            fTsumwx2 = float((data[1:-1] * xcent * xcent).sum())
            
            xaxis = upid.to_TAxis(
                "xaxis", "xaxis", nb,
                float(xedges[0]), float(xedges[-1]),
                xedges.astype(np.float64),
            )
            yaxis = upid.to_TAxis("yaxis", "yaxis", 0, 0.0, 0.0, None)
            zaxis = upid.to_TAxis("zaxis", "zaxis", 0, 0.0, 0.0, None)
            
            rootfile[fullpath] = _call_to_TH1x(
                fullpath,
                fullpath,
                data,
                fEntries,
                fTsumw,
                fTsumw2,
                fTsumwx,
                fTsumwx2,
                sumw2,
                xaxis,
                yaxis,
                zaxis,
            )
            return
        # ------------------------------------------------------------
        # 1D string/category eventflows
        # ------------------------------------------------------------
       
        if h.ndim == 1 and isinstance(h.axes[0], _histC.axis.StrCategory):
            ax = h.axes[0]
            final_labels = list(ax)
            counts = np.asarray(h.values(), dtype=np.float64)

            vari = h.variances()
            if vari is None:
                vari = counts.copy()
            else:
                vari = np.asarray(vari, dtype=np.float64)
            
            nb = len(final_labels)
            xedges = np.arange(nb + 1, dtype=np.float64)
            data = np.zeros(nb + 2, dtype=np.float64)
            data[1:-1] = counts

            sumw2 = np.zeros(nb + 2, dtype=np.float64)
            sumw2[1:-1] = vari if vari is not None else counts

            xcent = 0.5 * (xedges[:-1] + xedges[1:])

            fEntries = float((counts.sum() ** 2) / max(sumw2[1:-1].sum(), 1e-12))
            fTsumw = float(counts.sum())
            fTsumw2 = float(sumw2[1:-1].sum())
            fTsumwx = float((counts * xcent).sum())
            fTsumwx2 = float((counts * xcent * xcent).sum())

            xaxis = upid.to_TAxis(
                "xaxis", "xaxis", nb,
                float(xedges[0]), float(xedges[-1]),
                xedges,
            )
            for i, lab in enumerate(final_labels, start=1):
                xaxis.member("fLabels").append(upid.to_TObjString(lab))
            yaxis = upid.to_TAxis("yaxis", "yaxis", 0, 0.0, 0.0, None)
            zaxis = upid.to_TAxis("zaxis", "zaxis", 0, 0.0, 0.0, None)

            rootfile[fullpath] = _call_to_TH1x(
                fullpath,
                fullpath,
                data,
                fEntries,
                fTsumw,
                fTsumw2,
                fTsumwx,
                fTsumwx2,
                sumw2,
                xaxis,
                yaxis,
                zaxis,
            )
            return

        # ------------------------------------------------------------
        # 2D cut_index × numeric: BDT/cut-shape histograms
        # ------------------------------------------------------------
        if (
            h.ndim == 2
            and getattr(h.axes[0], "name", "") == "cut_index"
            and _is_num_axis(h.axes[1])
        ):
            counts = h.values()
            vari = h.variances()
            nx, ny = counts.shape

            xedges = np.arange(nx + 1, dtype=np.float64)

            ax1 = h.axes[1]
            yedges = np.asarray(ax1.edges, dtype=np.float64)

            data2 = np.zeros((nx + 2, ny + 2), dtype=np.float64)
            sumw2 = np.zeros((nx + 2, ny + 2), dtype=np.float64)

            data2[1:-1, 1:-1] = counts
            sumw2[1:-1, 1:-1] = vari if vari is not None else counts

            data_flat = np.asfortranarray(data2).ravel(order="F")
            sumw2_flat = np.asfortranarray(sumw2).ravel(order="F")

            xcent = 0.5 * (xedges[:-1] + xedges[1:])
            ycent = 0.5 * (yedges[:-1] + yedges[1:])

            fEntries = float(
                (counts.sum() ** 2) / max(sumw2[1:-1, 1:-1].sum(), 1e-12)
            )
            fTsumw = float(counts.sum())
            fTsumw2 = float(sumw2[1:-1, 1:-1].sum())
            fTsumwx = float((counts * xcent[:, None]).sum())
            fTsumwx2 = float((counts * xcent[:, None] ** 2).sum())
            fTsumwy = float((counts * ycent[None, :]).sum())
            fTsumwy2 = float((counts * ycent[None, :] ** 2).sum())
            fTsumwxy = float((counts * xcent[:, None] * ycent[None, :]).sum())

            xaxis = upid.to_TAxis(
                "xaxis", "cut_index", nx,
                float(xedges[0]), float(xedges[-1]),
                xedges,
            )
            yaxis = upid.to_TAxis(
                "yaxis",
                getattr(ax1, "label", ax1.name),
                ny,
                float(yedges[0]),
                float(yedges[-1]),
                yedges,
            )
            zaxis = upid.to_TAxis("zaxis", "zaxis", 0, 0.0, 0.0, None)

            rootfile[fullpath] = _call_to_TH2x(
                fullpath,
                data_flat,
                fEntries,
                fTsumw,
                fTsumw2,
                fTsumwx,
                fTsumwx2,
                fTsumwxy,
                fTsumwy,
                fTsumwy2,
                sumw2_flat,
                xaxis,
                yaxis,
                zaxis,
            )
            return

        # ------------------------------------------------------------
        # 2D numeric × numeric: normal COLZ-style TH2
        # ------------------------------------------------------------
        if h.ndim == 2 and _is_num_axis(h.axes[0]) and _is_num_axis(h.axes[1]):
            values = h.values(flow=False)
            variances = h.variances(flow=False)

            ax0 = h.axes[0]
            ax1 = h.axes[1]

            h2 = make_hist2_from_values(
                values,
                variances,
                np.asarray(ax0.edges, dtype=np.float64),
                np.asarray(ax1.edges, dtype=np.float64),
                ax0.name or "x",
                ax1.name or "y",
                getattr(ax0, "label", ax0.name or "x"),
                getattr(ax1, "label", ax1.name or "y"),
            )

            rootfile[fullpath] = h2
            return

        if "_shapes_" in fullpath:
            raise RuntimeError(
                f"[ERROR] Unsupported axes for {fullpath}: "
                f"ndim={h.ndim}, axes={[type(ax).__name__ for ax in h.axes]}"
            )

        print(f"[WARN] Unsupported histogram axes for {fullpath}; writing fallback")
        rootfile[fullpath] = h.to_numpy()

    except Exception as e:
        if "_shapes_" in fullpath:
            raise RuntimeError(
                f"[ERROR] TH writer failed for {fullpath}. "
                f"Shape histograms must be written as TH2D. Original error: {e}"
            )

        print(f"[WARN] TH writer failed for {fullpath}: {e}; writing fallback")
        rootfile[fullpath] = h.to_numpy()
parser = argparse.ArgumentParser()
parser.add_argument("--json", type=str, required=True)
parser.add_argument("--job-index", type=int, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--bdt_output", type=str, default=None)

args = parser.parse_args()

with open(args.json) as f:
    all_datasets = json.load(f)

if args.dataset not in all_datasets:
    raise ValueError(f"[ERROR] Dataset '{args.dataset}' not found in {args.json}")

dataset = all_datasets[args.dataset]
meta = dataset["metadata"]
files = dataset["files"]

if args.job_index >= len(files):
    raise IndexError(f"[ERROR] job-index {args.job_index} is out of range 0-{len(files)-1}")

file_to_process = files[args.job_index]
dataset_name = meta["sample"]
nevts = int(meta["nevents"])
isMC = meta["isMC"].lower() == "true"

isMVA = False
run_eval = True
isDBstudy = False
#isDBstudy = args.db_study
print(f"[INFO] Processing file {args.job_index+1}/{len(files)}: {file_to_process}")

xsec = float(meta["xsec"]) if isMC else 1.0
print(f"[INFO] Sample: {dataset_name} xsec={xsec} nevts={nevts}")

for attempt in range(1, 6):
    try:
        events = NanoEventsFactory.from_root(
            file_to_process,
            treepath="Events",
            schemaclass=BaseSchema,
            uproot_options={"timeout": 600},
        ).events()
        break
    except Exception as e:
        print(f"[WARNING] Attempt {attempt} failed: {e}")
        if attempt == 5:
            print("[ERROR] Max attempts reached. Skipping file.")
            sys.exit(1)
        time.sleep(10)

events["Muon"] = ak.zip({f: events[f"Muon_{f}"] for f in [
    "pt", "eta", "phi", "charge", "tightId", "looseId",
    "mass", "pfRelIso04_all",
]})

events["Electron"] = ak.zip({f: events[f"Electron_{f}"] for f in [
    "pt", "eta", "phi", "charge", "cutBased", "mass",
    "pfRelIso03_all", "superclusterEta", "mvaIso_WP90", "mvaIso_WP80",
]})

jet_fields = [
    "pt", "eta", "phi", "mass",
    "btagUParTAK4probbb", "btagUParTAK4B",
]
if isMC:
    jet_fields += ["hadronFlavour"]

events["Jet"] = ak.zip({f: events[f"Jet_{f}"] for f in jet_fields})

events["PuppiMET"] = ak.zip({f: events[f"PuppiMET_{f}"] for f in ["pt", "phi"]})

if isMC:
    events["Pileup"] = ak.zip({f: events[f"Pileup_{f}"] for f in ["nTrueInt"]})

events["PV"] = ak.zip({f: events[f"PV_{f}"] for f in ["npvsGood"]})

processor_instance = TOTAL_Processor(
    xsec=xsec,
    nevts=nevts,
    isMC=isMC,
    dataset_name=dataset_name,
    isMVA=isMVA,
    run_eval=run_eval,
)

result = processor_instance.process(events)

sample_base = os.path.basename(dataset_name).replace(".root", "").replace("/", "_")
job_suffix = os.path.basename(args.output).split("_")[-1]

# ============================================================
# Histogram writing
#
# In DB-study mode, skip histogram ROOT writing and write only
# the DB-study tree below.
#
# If you want both histograms and DB trees in the same job,
# change this condition to: if True:
# ============================================================

if not isDBstudy:
    if isinstance(result, dict) and all(k in result for k in ["ttBB", "ttCC", "ttLF"]):
        print("[INFO] Writing TTbar flavor-split outputs")

        for flavor, output in result.items():
            out_name = f"{sample_base}_{flavor}_{job_suffix}"

            with uproot.recreate(out_name) as rootfile:
                for name, h in output.items():

                    if name == "LeadSubleadDBTag2D":
                        rootfile[name] = make_hist2_dbtag(
                            h,
                            output["dbtag_edges"],
                        )
                        continue

                    if name == "dbtag_edges":
                        continue

                    if not isinstance(h, hist.Hist):
                        continue

                    values_check = np.asarray(
                        h.values(flow=False),
                        dtype=np.float64,
                    )
                    if not np.any(values_check != 0.0):
                        continue

                    write_hist_uproot_sumw2(rootfile, name, h)

            print(f"[INFO] Wrote {flavor}: {out_name}")
   
    else:
        out_name = args.output
      
        with uproot.recreate(out_name) as rootfile:
            for name, h in result.items():

                if name == "LeadSubleadDBTag2D":
                    rootfile[name] = make_hist2_dbtag(
                        h,
                        result["dbtag_edges"],
                    )
                    continue

                if name == "dbtag_edges":
                    continue

                if not isinstance(h, hist.Hist):
                    continue

                values_check = np.asarray(
                    h.values(flow=False),
                    dtype=np.float64,
                )
                if not np.any(values_check != 0.0):
                    continue

                write_hist_uproot_sumw2(rootfile, name, h)
        print(f"[INFO] Wrote ROOT histograms with Sumw2 to {out_name}")

else:
    print("[INFO] isDBstudy=True: skipping histogram ROOT writing")


# ============================================================
# DB-tag WP study tree writing
#
# This is intentionally OUTSIDE the TTbar flavor-split block.
# Therefore TTbar DB-study trees are written as one unsplit tree,
# not as ttBB/ttCC/ttLF.
# ============================================================

# ============================================================
# BDT tree writing
# ============================================================

if isMVA:
    bdt_output_name = args.bdt_output or f"bdt_{os.path.basename(args.output)}"

    tree_data = result.get("trees", None) if isinstance(result, dict) else None

    if not tree_data and hasattr(processor_instance, "_trees"):
        tree_data = processor_instance._trees

    if tree_data:
        with uproot.recreate(bdt_output_name) as bdtfile:
            for regime, tree_dict in tree_data.items():

                if regime.startswith("boosted"):
                    keys = processor_instance.tree_schema_boosted
                elif regime.startswith("resolved"):
                    keys = processor_instance.tree_schema_resolved
                else:
                    keys = ["dummy"]

                if not tree_dict:
                    clean_tree = {
                        k: np.array([], dtype=np.float64)
                        for k in keys
                    }
                    print(f"[INFO] Writing empty tree for regime '{regime}'")
                else:
                    clean_tree = {
                        k: np.asarray(v, dtype=np.float64)
                        for k, v in tree_dict.items()
                    }

                    n_entries = len(next(iter(clean_tree.values()))) if clean_tree else 0

                    for k in keys:
                        if k not in clean_tree:
                            clean_tree[k] = np.zeros(n_entries, dtype=np.float64)

                bdtfile[regime] = clean_tree

        print(f"[INFO] Saved BDT training trees in: {bdt_output_name}")

    else:
        print("[WARNING] No BDT trees found — nothing was written to tree output")
