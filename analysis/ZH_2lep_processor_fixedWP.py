import awkward as ak
import numpy as np
import uproot
import math
import os
import json
import argparse
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods import vector
from coffea import processor
from coffea.analysis_tools import Weights
import coffea.util
import hist
from hist import Hist, axis as hax
import itertools
import boost_histogram as bh
from boost_histogram import storage
from collections import defaultdict
from collections import Counter
from utils.xgb_tools import XGBHelper
import correctionlib
import gzip
from utils.deltas_array import (
    delta_r,
    clean_by_dr,
    delta_phi,
    delta_eta
)
from utils.variables_def import (
    min_dm_bb_bb,
    dr_bb_bb_avg,
    m_bbj,
    dr_bb_avg,
    higgs_kin
)
# UParTAK4B working points for 2024.
BTAG_WPS = {
    "L": 0.0246,
    "M": 0.1272,
    "T": 0.4648,
    }
FIXED_BTAG_WP = "T"
FIXED_BTAG_THRESHOLD = BTAG_WPS[FIXED_BTAG_WP]


def make_vector(obj):
    return ak.zip({
        "pt": obj.pt,
        "eta": obj.eta,
        "phi": obj.phi,
        "mass": obj.mass
    }, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)


def make_regressed_vector(jets):
    return ak.zip({
        "pt": jets.pt_regressed,
        "eta": jets.eta,
        "phi": jets.phi,
        "mass": jets.mass
    }, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)


def make_vector_met(met):
    return ak.zip({
        "pt": met.pt,
        "phi": met.phi,
        "eta": ak.zeros_like(met.pt),
        "mass": ak.zeros_like(met.pt)
    }, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)


def delta_eta_vec(a, b):
    return np.abs(a.eta - b.eta)


def delta_phi_raw(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi

MA_GEN_VALUES = (12.0, 15.0, 20.0, 25.0, 30.0)


def reduced_masses(higgs_vec, lead_vec, sub_vec, ma_vals=MA_GEN_VALUES):
    out = {}
    mH = higgs_vec.mass
    m1 = lead_vec.mass
    m2 = sub_vec.mass
    for ma in ma_vals:
        ma1_red = m1 - ma
        ma2_red = m2 - ma
        mH_red = mH - 125 - ma1_red - ma2_red   # mH-125 - m1 - m2 + 2*ma
        out[int(ma)] = (mH_red, ma1_red, ma2_red)
    return out


# Keep the same bin order in every chunk, including bins with no selected events.
EVENTFLOW_BINS = {
    "ee_eventflow_SR_resolved": (
        ">=2lep OSSF", "trigger", "mll", ">=3jets",
        ">=3jets & >=2bjets", ">=3jets & >=3bjets",
        ">=3jets & >=3bjets & <2dbjets",
    ),
    "mumu_eventflow_SR_resolved": (
        ">=2lep OSSF", "trigger", "mll", ">=3jets",
        ">=3jets & >=2bjets", ">=3jets & >=3bjets",
        ">=3jets & >=3bjets & <2dbjets",
    ),
    "emu_eventflow_TTCR_resolved": (
        ">=2lep OSDF", "trigger", ">=3jets",
        ">=3jets & >=2bjets", ">=3jets & >=3bjets",
        ">=3jets & >=3bjets & <2dbjets",
    ),
    "ee_eventflow_SR1_boosted": (
        ">=2lep OSSF", "trigger", "mll", ">=2jets",
        ">=2jets & >=1dbjet", ">=2jets & >=2dbjets",
    ),
    "mumu_eventflow_SR1_boosted": (
        ">=2lep OSSF", "trigger", "mll", ">=2jets",
        ">=2jets & >=1dbjet", ">=2jets & >=2dbjets",
    ),
    "emu_eventflow_TTCR1_boosted": (
        ">=2lep OSDF", "trigger", ">=2jets",
        ">=2jets & >=1dbjet", ">=2jets & >=2dbjets",
    ),
}


def fill_eventflow(output, name, cut, mask, w_all):
    mask = np.asarray(mask, dtype=bool)

    if not np.any(mask):
        return

    selected_weights = np.asarray(
        w_all,
        dtype=np.float64,
    )[mask]

    output[name].fill(
        cut=np.full(
            len(selected_weights),
            cut,
            dtype=object,
        ),
        weight=selected_weights,
    )


def z_window_mask(leptons, base_mask, zlo=75.0, zhi=105.0):
    out = np.zeros(len(base_mask), dtype=bool)
    if not np.any(base_mask):
        return out

    pairs = leptons[base_mask][:, :2]
    mll = (make_vector(pairs[:, 0]) + make_vector(pairs[:, 1])).mass
    idx = np.where(base_mask)[0]
    out[idx] = ak.to_numpy((mll > zlo) & (mll < zhi))
    return out


def map_hadron_flavor_to_eff_flavor(hadflav):
    """Map hadronFlavour to the efficiency indices: b=0, c=1, light=2."""
    return np.where(
        hadflav == 5, 0,
        np.where(
            hadflav == 4, 1,
            2
        )
    )


def _clip_nextafter(x, lo, hi):
    lo2 = np.nextafter(lo, 1.0)
    hi2 = np.nextafter(hi, -1.0)
    x = ak.where(x < lo2, lo2, x)
    x = ak.where(x > hi2, hi2, x)
    return x


def _unflatten_like(flat, counts):
    return ak.unflatten(ak.Array(flat), counts)

# Lazy booking for 1D and 2D histograms.


def _to_numpy_flat(x):
    try:
        if isinstance(x, (ak.Array, ak.Record)):
            return ak.to_numpy(ak.flatten(x))
    except Exception:
        pass
    return np.asarray(x)


class _AutoHist:

    def __init__(self, parent_dict, name, parent_proc=None):
        self._parent = parent_dict
        self._name = name
        self._hist = None
        self._proc = parent_proc  # Reference to the processor binning.
        self.systematics_labels = [""]

    def _axis_for_numeric(self, key, arr):

        k = key.lower()

        if k in {"btag_sf", "btag_sf_resolved"}:
            return hax.Regular(
                61, 0., 3.,
                name=key,
                label="b-tag SF (resolved)",
                underflow=True,
                overflow=True,
            )
        if k == "n_btruth_jets":
            return hax.Integer(
                0,
                8,
                name=key,
                label="Selected truth-b jets",
                underflow=False,
                overflow=True,
            )

        if k == "n_ctruth_jets":
            return hax.Integer(
                0,
                6,
                name=key,
                label="Selected truth-c jets",
                underflow=False,
                overflow=True,
            )

        if k == "n_lighttruth_jets":
            return hax.Integer(
                0,
                10,
                name=key,
                label="Selected truth-light jets",
                underflow=False,
                overflow=True,
            )
        # Explicit binning for analysis variables.

        if k == "h_mass":
            return hax.Regular(50, 0.0, 1000.0,
                               name=key, label=key,
                               underflow=True, overflow=True)

        if k == "h_pt":
            return hax.Regular(50, 0.0, 500.0,
                               name=key, label=key,
                               underflow=True, overflow=True)
        if k == "z_pt":
            return hax.Regular(50, 0.0, 500.0,
                               name=key, label=key,
                               underflow=True, overflow=True)

        if k == "ht":
            return hax.Regular(50, 0.0, 1000.0,
                               name=key, label=key,
                               underflow=True, overflow=True)

        if k == "mbbj":
            return hax.Regular(50, 0.0, 1000.0,
                               name=key, label=key,
                               underflow=True, overflow=True)

        if k in {"eta"}:
            return hax.Regular(30, -5.0, 5.0, name=key, label=key,underflow=True, overflow=True )
        if k in {"phi", "met_phi", "puppimet_phi"}:
            return hax.Regular(16, -math.pi, math.pi, name=key, label=key,underflow=True, overflow=True)
        if k.startswith("dphi") or k in {"dphi"}:
            return hax.Regular(16, 0.0, math.pi, name=key, label=key,underflow=True, overflow=True)
        if k.startswith("dr") or k in {"dr","dR"}:
            return hax.Regular(15, 0.0, 5.0, name=key, label=key,underflow=True, overflow=True)
        if k.startswith("deta") or k in {"deta"}:
            return hax.Regular(10, 0.0, 6.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"bdt"}:
            edges = getattr(self._proc, "bdt_edges", None)
            if edges is None:
                return hax.Regular(50, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)
            return hax.Variable(edges, name=key, label=key,underflow=True, overflow=True)
        if k in {"score", "btag_score"}:
            return hax.Regular(25, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)

        if k in {"btag"}:
            return hax.Regular(25, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"n", "n_jets", "n_bjets", "n_untag"}:
            return hax.Regular(10, 0.0, 10., name=key, label=key,underflow=True, overflow=True)
        if k in {"npv"}:
            return hax.Regular(100, 0.0, 100., name=key, label=key,underflow=True, overflow=True)
        if k.startswith("btag"):
            return hax.Regular(50, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"gentop_pt", "top_pt", "gen_top_pt", "genTop_pt"}:
            return hax.Regular(100, 0.0, 2000.0, name=key, label="gen top pT [GeV]",underflow=True, overflow=True)
        if k in {"lep_pt", "lep0_pt", "lep1_pt"}:
            return hax.Regular(10, 0.0, 200.0, name=key, label="gen top pT [GeV]",underflow=True, overflow=True)
        if k in {"bjet_pt", "lead_bjet_pt", "sublead_bjet_pt"}:
            return hax.Regular(
                10, 0.0, 200.0,
                name=key,
                label=key,
                underflow=True,
                overflow=True,
            )
        if k in {"met", "met_pt", "puppimet_pt"}:
            return hax.Regular(20, 0.0, 400.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"ht"}:
            return hax.Regular(40, 0.0, 1000.0, name=key, label=key,underflow=True, overflow=True)
        if k in { "pt_h","pt_b1","pt_b2","z_pt","ll_pt","pt_ll","pt_z","pt_pretrig","pt_posttrig"}:
            return hax.Regular(25, 0.0, 500.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"h_m_red","m_h_red"}:
            return hax.Regular(6, -100., 200.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"a1_m_red","m_a1_red","a2_m_red","m_a2_red"}:
            return hax.Regular(4, -20., 20.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"bjet_pt", "lead_bjet_pt", "sublead_bjet_pt"}:
            return hax.Regular(
                10, 0.0, 200.0,
                name=key,
                label=key,
                underflow=True,
                overflow=True,
            )

        if k in {"dbjet_pt"}:
            return hax.Regular(
                10, 0.0, 200.0,
                name=key,
                label=key,
                underflow=True,
                overflow=True,
            )

        if k in {"dbjet_mass"}:
            return hax.Regular(
                20, 0.0, 100.0,
                name=key,
                label=key,
                underflow=True,
                overflow=True,
            )

        if k in {"h_pt","pt_h"}:
            return hax.Regular(25, 0.0, 500.0, name=key, label=key,underflow=True, overflow=True)
        if k in { "m_h","h_m","mbbj" ,"m_bbj"}:
            return hax.Regular(40, 0.0, 1000.0, name=key, label=key,underflow=True, overflow=True)
        if k in { "z_m", "ll_m","m_z", "m_ll"}:
            return hax.Regular(20, 70, 110.0, name=key, label=key,underflow=True, overflow=True)
        if "ratio" in k:
            return hax.Regular(25, -0.0, 5.0, name=key, label=key,underflow=True, overflow=True)
        if "dm" in k:
            return hax.Regular(16, 0.0, 160.0, name=key, label=key,underflow=True, overflow=True)

        # For other variables, choose the range from the first fill.
        a = _to_numpy_flat(arr)
        a = a[np.isfinite(a)]
        if a.size == 0:
            # Use a unit range when no finite values are available.
            return hax.Regular(50, 0.0, 1.0, name=key, label=key)

        lo, hi = np.min(a), np.max(a)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, 1.0
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        return hax.Regular(50, float(lo - pad), float(hi + pad), name=key, label=key)

    def _axes_from_kwargs(self, kwargs):
        # Each argument except weight defines an axis.
        dims = [(k, v) for k, v in kwargs.items() if k != "weight"]
        if not dims:
            return [hax.Regular(1, 0.0, 1.0, name="unit", label="unit")]

        axes = []
        for key, val in dims:
            # Use common axes for cut labels and scan indices.
            if key == "cut":
                labels = getattr(self._proc, "fixed_cut_axes", {}).get(self._name, None)

                if labels is None:
                    raise RuntimeError(
                        f"[HIST AXIS ERROR] Histogram {self._name} uses cut axis "
                        "but was not pre-booked with fixed labels."
                    )

                axes.append(
                    hax.StrCategory(
                        labels,
                        name="cut",
                        label="cut",
                        growth=False,
                    )
                )
                continue

            if key == "cut_index":
                ncuts = len(getattr(self._proc, "optim_Cuts1_bdt", []))
                ncuts = max(ncuts, 1)
                axes.append(hax.IntCategory(list(range(ncuts)), name="cut_index", label="cut index", growth=False))
                continue

            # Inspect the flattened array to choose the axis type.
            v = _to_numpy_flat(val)

            # Allow new categories for string and object arrays.
            if v.dtype.kind in {"U", "S", "O"}:
                axes.append(hax.StrCategory([], name=key, label=key, growth=True))
                continue

            # Keep Boolean categories fixed across chunks.
            if v.dtype.kind == "b":
                axes.append(hax.IntCategory([0, 1], name=key, label=key, growth=False))
                continue

            # Use the numeric binning defined above.
            axes.append(self._axis_for_numeric(key, v))

        return axes

    def _ensure_hist(self, **kwargs):
        if self._hist is not None:
            return self._hist

        axes = self._axes_from_kwargs(kwargs)
        self._hist = Hist(*axes, storage=storage.Weight())
        # Replace the proxy with the booked histogram.
        self._parent[self._name] = self._hist
        return self._hist

    def fill(self, **kwargs):
        h = self._ensure_hist(**kwargs)
        return h.fill(**kwargs)

    def copy(self):
        # Return an empty histogram until the first fill.
        return Hist(hax.Regular(1, 0.0, 1.0, name="unit", label="unit"), storage=storage.Weight())


class AutoHistDict(dict):
    """Book each histogram on its first fill and cache it in the dictionary."""

    def __init__(self, parent_proc=None):
        super().__init__()
        self._proc = parent_proc

    def __getitem__(self, key):
        if key not in self:
            super().__setitem__(key, _AutoHist(self, key, parent_proc=self._proc))
        return super().__getitem__(key)

    def spawn_accumulator(self):
        return AutoHistDict(parent_proc=self._proc)


class TOTAL_Processor(processor.ProcessorABC):

    def __init__(
        self,
        xsec=1.0,
        nevts=1.0,
        isMC=True,
        dataset_name=None,
        isMVA=False,
        run_eval=True,
    ):
        self.xsec = xsec
        self.nevts = nevts
        self.isMC = isMC
        self.isMVA = isMVA
        self.run_eval = run_eval
        self.dataset_name = dataset_name
        self.dbtag_edges = np.linspace(0.0, 1.0, 101)
        self._trees = {regime: defaultdict(list) for regime in ["boosted_SR1", "resolved"]} if isMVA else None
        self.fixed_cut_axes = {}
        self._histograms_phys = AutoHistDict(parent_proc=self)
        self._histograms_bdt = AutoHistDict(parent_proc=self)

        BDT_BOOSTED_FEATURES = [
            "H_mass",
            "H_pt",
            "HT",
            "puppimet_pt",
            "dr_bb_bb_ave",
            "dm_bb_bb_min",
            "dphi_HZ",
            "dr_ll",
            "n_jets",
            "lep1_pt"
        ]

        BDT_RESOLVED_FEATURES = [
            "H_mass",
            "H_pt",
            "HT",
            "puppimet_pt",
            "dr_bb_ave",
            "dm_bb_bb_min",
            "dphi_HZ",
            "dr_ll",
            "mbbj",
            "n_jets",
            "lep1_pt"
        ]

        self.tree_schema_resolved = [
            "H_mass",
            "H_pt",
            "HT",
            "puppimet_pt",
            "dr_bb_ave",
            "dm_bb_bb_min",
            "dphi_HZ",
            "n_untag",
            "dr_ll",
            "mbbj",
            "n_bjets",
            "lep0_pt",
            "lep1_pt",
            "weight",
        ]

        self.tree_schema_boosted = [
            "H_mass",
            "H_pt",
            "HT",
            "puppimet_pt",
            "dr_bb_bb_ave",
            "dm_bb_bb_min",
            "dphi_HZ",
            "dphi_untag_Z",
            "n_untag",
            "dr_ll",
            "n_bjets",
            "lep0_pt",
            "lep1_pt",
            "weight",
        ]

        # Load the boosted and resolved BDT models.
        self.bdt_eval_boosted = XGBHelper(
            os.path.join("xgb_model", "bdt_model_boosted.json"),
            BDT_BOOSTED_FEATURES
        )

        self.bdt_eval_resolved = XGBHelper(
            os.path.join("xgb_model", "bdt_model_resolved.json"),
            BDT_RESOLVED_FEATURES
        )

        self.bdt_edges = np.linspace(0.0, 1.0, 51 )
        self.optim_Cuts1_bdt = self.bdt_edges[:-1].tolist()

        self.systematics_labels = [""]

        nvarsToInclude = len(self.systematics_labels)
        nCuts = len(self.optim_Cuts1_bdt)

        HERE = os.path.dirname(__file__)
        CORR_DIR = os.path.join(HERE, "corrections")

        # Pileup weights.
        self.pu_json_path = os.path.join(CORR_DIR, "puWeights_BCDEFGHI.json.gz")
        self._pu_corr = None

        if os.path.exists(self.pu_json_path):
            cset = correctionlib.CorrectionSet.from_file(self.pu_json_path)
            key = "Collisions24_BCDEFGHI_goldenJSON"
            self._pu_corr = cset[key]
            print(f"[PU] Loaded {key} from {self.pu_json_path}")
        else:
            print(f"[PU] Missing PU json: {self.pu_json_path}")

        # Fixed-WP b-tagging scale factors.
        self._btag_sf = None
        self.btag_json = os.path.join(CORR_DIR, "btag_merged_2024_final.json.gz")

        if os.path.exists(self.btag_json):
            cset_btag = correctionlib.CorrectionSet.from_file(self.btag_json)
            self._btag_sf = cset_btag["UParTAK4_merged"]
            print("[BTV] Loaded UParTAK4_merged")
        else:
            print(f"[BTV] Missing b-tag SF JSON: {self.btag_json}")

        # B-tagging efficiencies for the 2-lepton selection.
        self._btag_eff = None
        self._btag_eff_process = None

        if self.isMC:
            if self.dataset_name is None:
                raise RuntimeError(
                    "dataset_name is required for b-tag efficiencies"
                )

            sample = str(self.dataset_name)
            sample_no_year = (
                sample[:-5]
                if sample.endswith("_2024")
                else sample
            )

            tt_efficiency_files = {
                "TTto2L2Nu":
                    "btag_eff_2lep_TTto2L2Nu_2024.json.gz",

                "TTtoLNu":
                    "btag_eff_2lep_TTtoLNu2Q_2024.json.gz",

                "TTtoLNu2Q":
                    "btag_eff_2lep_TTtoLNu2Q_2024.json.gz",

                "TTto4Q":
                    "btag_eff_2lep_TTto4Q_2024.json.gz",
            }

            if sample_no_year in tt_efficiency_files:
                eff_name = tt_efficiency_files[sample_no_year]
                self._btag_eff_process = None
            else:
                if sample.lower().startswith("ttto"):
                    raise KeyError(
                        "No TT efficiency JSON configured for "
                        f"dataset={sample!r}"
                    )

                eff_name = (
                    "btag_eff_2lep_2024_nontt.json.gz"
                )

                # Non-ttbar efficiency keys use metadata["sample"].
                self._btag_eff_process = sample

            eff_json = os.path.join(
                CORR_DIR,
                eff_name,
            )

            if not os.path.exists(eff_json):
                raise FileNotFoundError(
                    f"Missing b-tag efficiency JSON: {eff_json}"
                )

            eff_cset = correctionlib.CorrectionSet.from_file(
                eff_json
            )

            if "btag_eff" not in eff_cset:
                raise KeyError(
                    f"Missing correction 'btag_eff' in {eff_json}"
                )

            self._btag_eff = eff_cset["btag_eff"]

            expected_eff_inputs = [
                "process",
                "wp",
                "flavour",
                "nbjets",
                "pt",
                "abseta",
            ]

            actual_eff_inputs = [
                item.name
                for item in self._btag_eff.inputs
            ]

            if actual_eff_inputs != expected_eff_inputs:
                raise RuntimeError(
                    "Unexpected efficiency inputs: "
                    f"{actual_eff_inputs}; "
                    f"expected={expected_eff_inputs}"
                )

            # Check that each required process key can be evaluated.
            probe_processes = (
                ("ttLF", "ttCC", "ttBB")
                if sample_no_year in tt_efficiency_files
                else (self._btag_eff_process,)
            )

            for process_key in probe_processes:
                try:
                    probe = self._btag_eff.evaluate(
                        process_key,
                        FIXED_BTAG_WP,2, 0, 50.0,0.5,)
                except Exception as exc:
                    raise KeyError(
                        "Missing efficiency process key "
                        f"{process_key!r} in {eff_json}"
                    ) from exc

                if not np.isfinite(probe):
                    raise RuntimeError(
                        "Non-finite efficiency probe for "
                        f"{process_key!r}"
                    )

            print(
                f"[BTV] Loaded grouped efficiency: {eff_json}; "
                f"dataset={sample}; "
                f"process={self._btag_eff_process}"
            )

    @property
    def histograms(self):
        # Combine the physics and BDT histograms.
        out = AutoHistDict(parent_proc=self)

        out.update(self._histograms_phys)
        out.update(self._histograms_bdt)
        return out

    def add_tree_entry(self, regime, data_dict):
        if not self._trees or regime not in self._trees:
            return

        for key, val in data_dict.items():
            val = np.asarray(val)
            self._trees[regime][key].extend(val.tolist())

    def compat_tree_variables(self, tree_dict):
        """Convert tree branches to float64 arrays for consistent ROOT merging."""
        for key in tree_dict:
            tree_dict[key] = np.asarray(tree_dict[key], dtype=np.float64)

    def _validate_tree_dict(self, regime, tree_dict):
        """Check branch lengths and finite values; warn about mostly zero branches."""

        if not tree_dict:
            print(f"[TREE CHECK] {regime}: EMPTY tree")
            return

        lengths = {k: len(v) for k, v in tree_dict.items()}

        # All branches must have the same number of entries.
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise RuntimeError(
                f"[TREE ERROR] {regime}: Branch length mismatch:\n{lengths}"
            )

        n_entries = list(unique_lengths)[0]

        print(f"[TREE CHECK] {regime}: entries = {n_entries}")

        for k, v in tree_dict.items():
            arr = np.asarray(v)

            if np.any(~np.isfinite(arr)):
                raise RuntimeError(f"[TREE ERROR] {regime}:{k} contains NaN or inf")

            # Warn about branches dominated by zeros.
            if n_entries > 0:
                zero_fraction = np.mean(arr == 0.0)
                if zero_fraction > 0.95:
                    print(
                        f"[WARNING] {regime}:{k} has {zero_fraction*100:.1f}% zeros"
                    )

    def build_vars_resolved(
        self,
        leptons_sel,  # Selected leptons.
        single_jets_sel,  # Selected, cleaned jets.
        single_bjets_sel,  # Tagged subset of single_jets_sel.
        single_untag_sel,  # Untagged subset of single_jets_sel.
        PuppiMET_sel,  # MET for the selected events.
    ):
        # Reconstruct the Z from the leading lepton pair.
        leps = leptons_sel[:, :2]
        l0, l1 = leps[:, 0], leps[:, 1]
        l0v, l1v = make_vector(l0), make_vector(l1)
        Z = l0v + l1v

        sj = single_jets_sel
        sbj = single_bjets_sel
        sut = single_untag_sel

        v_sj = make_vector(sj)
        v_sbj = make_vector(sbj)

        mH, ptH, _, _ = higgs_kin(v_sbj, v_sj)

        H = ak.zip(
            {"pt": ptH, "eta": ak.zeros_like(ptH), "phi": ak.zeros_like(ptH), "mass": mH},
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        return {
            "H_mass": ak.to_numpy(mH),
            "H_pt":   ak.to_numpy(ptH),
            "HT":     ak.to_numpy(ak.sum(sj.pt, axis=1)),
            "puppimet_pt": ak.to_numpy(PuppiMET_sel.pt),
            "puppimet_phi": ak.to_numpy(PuppiMET_sel.phi),
            "dr_bb_ave": ak.to_numpy(dr_bb_avg(v_sbj)),
            "dm_bb_bb_min": ak.to_numpy(min_dm_bb_bb(v_sbj, all_jets=v_sj)),

            "dphi_HZ": ak.to_numpy(np.abs(H.delta_phi(Z))),
            "n_untag": ak.to_numpy(ak.num(sut)),

            "dr_ll":   ak.to_numpy(l0v.delta_r(l1v)),
            "mbbj":    ak.to_numpy(m_bbj(v_sbj, all_jets=v_sj)),

            "n_jets":  ak.to_numpy(ak.num(sj)),
            "n_bjets": ak.to_numpy(ak.num(sbj)),
            "lep0_pt": ak.to_numpy(l0.pt),
            "lep1_pt": ak.to_numpy(l1.pt),
            "Z_pt":    ak.to_numpy(Z.pt),
        }

    def build_vars_boosted(
        self,
        leptons_sel,  # Selected leptons.
        double_jets_sel,  # Selected, cleaned jets.
        double_bjets_sel,  # Tagged subset of double_jets_sel.
        double_untag_sel,  # Untagged subset of double_jets_sel.
        PuppiMET_sel,  # MET for the selected events.
    ):
        leps = leptons_sel[:, :2]
        l0, l1 = leps[:, 0], leps[:, 1]
        l0v, l1v = make_vector(l0), make_vector(l1)
        Z = l0v + l1v

        dj = double_jets_sel
        dbj = double_bjets_sel
        dut = double_untag_sel

        bb = dbj[:, :2]
        b1, b2 = bb[:, 0], bb[:, 1]
        b1v, b2v = make_vector(b1), make_vector(b2)
        H = b1v + b2v

        dm = np.abs(b1.mass - b2.mass)

        return {
            "H_mass": ak.to_numpy(H.mass),
            "H_pt":   ak.to_numpy(H.pt),
            "HT":     ak.to_numpy(ak.sum(dj.pt, axis=1)),
            "puppimet_pt": ak.to_numpy(PuppiMET_sel.pt),
            "puppimet_phi": ak.to_numpy(PuppiMET_sel.phi),
            "dr_bb_bb_ave": ak.to_numpy(b1v.delta_r(b2v)),
            "dm_bb_bb_min": ak.to_numpy(dm),

            "dphi_HZ": ak.to_numpy(np.abs(H.delta_phi(Z))),

            "dphi_untag_Z": ak.to_numpy(
                ak.fill_none(
                    ak.max(np.abs(make_vector(dut).delta_phi(Z)), axis=1),
                    0.0
                )
            ),

            "n_untag": ak.to_numpy(ak.num(dut)),
            "dr_ll":   ak.to_numpy(l0v.delta_r(l1v)),

            "n_jets":  ak.to_numpy(ak.num(dj)),
            "n_bjets": ak.to_numpy(ak.num(dbj)),
            "lep0_pt": ak.to_numpy(l0.pt),
            "lep1_pt": ak.to_numpy(l1.pt),
            "Z_pt":    ak.to_numpy(Z.pt),
        }

    def _as_np(self, x):
        return ak.to_numpy(x) if isinstance(x, ak.Array) else np.asarray(x)

    def eval_btag_sf_fixedWP_resolved(
        self,
        jets,
        eff_process,
        wp_name=FIXED_BTAG_WP,
        wp_value=FIXED_BTAG_THRESHOLD,
        syst='central',
    ):
        if self._btag_sf is None or self._btag_eff is None:
            return ak.ones_like(
                ak.num(jets),
                dtype=np.float64,
            )

        counts = ak.num(jets)

        nbtruthb = ak.sum(np.abs(jets.hadronFlavour) == 5,axis=1,)
        nbtruthb = ak.where(nbtruthb >= 4, 4, nbtruthb,)
        nbtruthb_perjet = ak.broadcast_arrays(nbtruthb,jets.pt,)[0]
        flat = ak.flatten(jets)

        pt = ak.to_numpy(flat.pt).astype(np.float64)
        eta = ak.to_numpy(
            np.abs(flat.eta)
        ).astype(np.float64)
        hadf = ak.to_numpy(
            flat.hadronFlavour
        ).astype(np.int32)
        score = ak.to_numpy(
            flat.btagUParTAK4B
        ).astype(np.float64)
        nbjet = ak.to_numpy(
            ak.flatten(nbtruthb_perjet)
        ).astype(np.int32)

        if len(pt) == 0:
            return ak.ones_like(
                counts,
                dtype=np.float64,
            )

        eff_flav = map_hadron_flavor_to_eff_flavor(hadf)

        valid = (
            np.isfinite(pt)
            & np.isfinite(eta)
            & np.isfinite(score)
            & (pt >= 20.0)
            & (eta < 2.5)
            & np.isin(np.abs(hadf), [0, 4, 5])
            & np.isin(nbjet, [0, 1, 2, 3, 4])
        )

        pt_eval = np.clip(
            pt,
            20.0,
            np.nextafter(1000.0, -np.inf),
        )

        eta_eval = np.clip(
            eta,
            0.0,
            np.nextafter(2.5, -np.inf),
        )

        sf = np.ones(len(pt), dtype=np.float64)
        eff = np.zeros(len(pt), dtype=np.float64)

        sf[valid] = self._btag_sf.evaluate(
            syst,
            wp_name,
            hadf[valid],
            eta_eval[valid],
            pt_eval[valid],
        )

        eff[valid] = self._btag_eff.evaluate(
            eff_process,
            wp_name,
            eff_flav[valid],
            nbjet[valid],
            pt_eval[valid],
            eta_eval[valid],
        )

        bad = (
            valid
            & (
                ~np.isfinite(sf)
                | ~np.isfinite(eff)
                | (sf <= 0.0)
                | (eff < 0.0)
                | (eff > 1.0)
            )
        )

        if np.any(bad):
            raise RuntimeError(
                f"Invalid fixed-WP inputs for "
                f"process={eff_process}, WP={wp_name}"
            )

        tagged = valid & (score >= wp_value)
        untagged = valid & (score < wp_value)

        jet_weight = np.ones(
            len(pt),
            dtype=np.float64,
        )

        jet_weight[tagged] = sf[tagged]

        numerator = 1.0 - sf * eff
        denominator = 1.0 - eff

        good_untagged = (
            untagged
            & np.isfinite(numerator)
            & np.isfinite(denominator)
            & (numerator > 0.0)
            & (denominator > 0.0)
        )

        bad_untagged = untagged & ~good_untagged

        jet_weight[good_untagged] = (
            numerator[good_untagged]
            / denominator[good_untagged]
        )

        if np.any(bad_untagged):
            jet_weight[bad_untagged] = 1.0

            print(
                f"[BTV WARNING] Fixed-{wp_name} untagged "
                "factors set to unity: "
                f"{np.count_nonzero(bad_untagged)}/"
                f"{np.count_nonzero(untagged)}"
            )

        if (
            np.any(~np.isfinite(jet_weight))
            or np.any(jet_weight <= 0.0)
        ):
            raise RuntimeError(
                f"Invalid fixed-{wp_name} per-jet weights"
            )

        return ak.prod(
            ak.unflatten(
                jet_weight,
                counts,
            ),
            axis=1,
        )

    def eval_and_fill_bdt(self, *, channel, region, regime, vals, weight):
        if regime == "resolved":
            bdt_eval = self.bdt_eval_resolved
        elif regime == "boosted":
            bdt_eval = self.bdt_eval_boosted
        else:
            raise ValueError(regime)

        inputs = {v: np.asarray(vals[v], dtype=np.float64)
                for v in bdt_eval.var_list}

        score = np.ravel(bdt_eval.eval(inputs))

        # Inclusive BDT score.
        self._histograms_bdt[
            f"{channel}_{region}_bdt_{regime}"
        ].fill(bdt=score, weight=weight)

        # Distributions at each BDT threshold.
        for i, cut in enumerate(self.optim_Cuts1_bdt):
            sel = score > cut
            if not np.any(sel):
                continue

            self._histograms_bdt[
                f"{channel}_{region}_bdt_shapes_{regime}"
            ].fill(
                cut_index=i,
                bdt=score[sel],
                weight=weight[sel],
            )

    def process(self, events):
        try:
            events = events.eager_compute_divisions()
        except Exception:

            pass

        try:
            n = len(events)
        except TypeError:
            events = events.compute()

        n = len(events) # Rebuild the weights for the selected events.
        weights = Weights(n, storeIndividual=True)

        # Event weights and flavour-split outputs.

        # Inclusive output for samples other than ttbar.
        output = AutoHistDict(parent_proc=self)

        output_ttBB = None
        output_ttCC = None
        output_ttLF = None

        is_ttbar_sample = (
            self.isMC
            and self.dataset_name is not None
            and self.dataset_name.startswith("TTto")
            and "genTtbarId" in events.fields
        )

        if is_ttbar_sample:
            output_ttBB = AutoHistDict(parent_proc=self)
            output_ttCC = AutoHistDict(parent_proc=self)
            output_ttLF = AutoHistDict(parent_proc=self)

        def make_empty_dbtag2d():
            return np.zeros(
                (len(self.dbtag_edges) - 1, len(self.dbtag_edges) - 1),
                dtype=np.float64,
            )

        if is_ttbar_sample:
            output_ttBB["LeadSubleadDBTag2D"] = make_empty_dbtag2d()
            output_ttCC["LeadSubleadDBTag2D"] = make_empty_dbtag2d()
            output_ttLF["LeadSubleadDBTag2D"] = make_empty_dbtag2d()

            output_ttBB["dbtag_edges"] = self.dbtag_edges
            output_ttCC["dbtag_edges"] = self.dbtag_edges
            output_ttLF["dbtag_edges"] = self.dbtag_edges
        else:
            output["LeadSubleadDBTag2D"] = make_empty_dbtag2d()
            output["dbtag_edges"] = self.dbtag_edges
        if self.isMC:

            norm = (self.xsec / self.nevts)
            weight_array = np.ones(n,dtype="float64") * norm
            weights.add("norm", weight_array)
            # Top-pT reweighting for ttbar.
            is_ttbar = ((self.dataset_name is not None)and any(tag in self.dataset_name.lower() for tag in ["ttto"]))

            if is_ttbar and ("topptWeight" in events.fields):
                toppt_w = ak.to_numpy(ak.fill_none(events.topptWeight, 1.0)).astype("float64")
                toppt_w = np.clip(toppt_w, 0.0, 10.0)
                weights.add("toppt", toppt_w)

            # Weights before pileup reweighting.
            w_bef_pu = weights.weight()

            if self._pu_corr is not None:
                npu = ak.to_numpy(events.Pileup.nTrueInt).astype(np.float64)
                npu = np.clip(npu, 0, 99)
                pu_w = self._pu_corr.evaluate(npu, "nominal")
                pu_w = np.clip(pu_w, 0.0, 1000.0)
                # Apply pileup reweighting.
                weights.add("pileup", pu_w)
        else:
            weights.add("ones", np.ones(n, dtype="float64"))
            w_bef_pu = weights.weight()
        w_all = weights.weight()

        # Split ttbar events by heavy-flavour content.
        tt_masks = None

        if (
            self.isMC
            and self.dataset_name is not None
            and self.dataset_name.startswith("TTto")
            and "genTtbarId" in events.fields
        ):
            gen_id = ak.to_numpy(events.genTtbarId)

            tt_masks = {
                "ttLF": (gen_id % 100 < 41),
                "ttCC": (gen_id % 100 >= 41) & (gen_id % 100 <= 45),
                "ttBB": (gen_id % 100 >= 51) & (gen_id % 100 <= 55),
            }

        # Histogram filling with the same ttbar flavour masks.
        def fill_hist_auto(name, mask, weight, **kwargs):

            if not np.any(mask):
                return

            if tt_masks is None:
                output[name].fill(
                    **{k: v[mask] for k, v in kwargs.items()},
                    weight=weight[mask]
                )
            else:
                for flavor, fmask in tt_masks.items():

                    mask_flav = mask & fmask
                    if not np.any(mask_flav):
                        continue

                    if flavor == "ttBB":
                        out_dict = output_ttBB
                    elif flavor == "ttCC":
                        out_dict = output_ttCC
                    else:
                        out_dict = output_ttLF

                    out_dict[name].fill(
                        **{k: v[mask_flav] for k, v in kwargs.items()},
                        weight=weight[mask_flav],
                    )

        def fill_lead_sublead_dbtag2d_auto(mask, jets_clean, weight):
            if not np.any(mask):
                return

            if len(jets_clean) == 0:
                return
            jets_sorted = jets_clean[
                ak.argsort(jets_clean.btagUParTAK4probbb, axis=-1, ascending=False)
            ]

            jets_sorted = jets_sorted[ak.num(jets_sorted) >= 2]

            lead = ak.to_numpy(jets_sorted[:, 0].btagUParTAK4probbb)
            sub = ak.to_numpy(jets_sorted[:, 1].btagUParTAK4probbb)

            idx = np.where(mask)[0]
            w = weight[idx]

            hist2d, _, _ = np.histogram2d(
                lead,
                sub,
                bins=[self.dbtag_edges, self.dbtag_edges],
                weights=w,
            )

            if tt_masks is None:
                output["LeadSubleadDBTag2D"] += hist2d
            else:
                for flavor, fmask in tt_masks.items():
                    local = fmask[idx]
                    if not np.any(local):
                        continue

                    hist2d_flav, _, _ = np.histogram2d(
                        lead[local],
                        sub[local],
                        bins=[self.dbtag_edges, self.dbtag_edges],
                        weights=w[local],
                    )

                    if flavor == "ttBB":
                        output_ttBB["LeadSubleadDBTag2D"] += hist2d_flav
                    elif flavor == "ttCC":
                        output_ttCC["LeadSubleadDBTag2D"] += hist2d_flav
                    else:
                        output_ttLF["LeadSubleadDBTag2D"] += hist2d_flav

        # Cutflow filling with the same ttbar flavour masks.
        def fill_eventflow_auto(name, cut, mask, weight):

            if tt_masks is None:
                fill_eventflow(output, name, cut, mask, weight)
            else:
                for flavor, fmask in tt_masks.items():

                    mask_flav = mask & fmask
                    if not np.any(mask_flav):
                        continue

                    if flavor == "ttBB":
                        out_dict = output_ttBB
                    elif flavor == "ttCC":
                        out_dict = output_ttCC
                    else:
                        out_dict = output_ttLF

                    fill_eventflow(out_dict, name, cut, mask_flav, weight)

        def book_eventflow_bins(name, labels):
            labels = list(labels)

            if name in self.fixed_cut_axes:
                if self.fixed_cut_axes[name] != labels:
                    raise RuntimeError(
                        f"[CATEGORY LABEL ERROR] {name} already booked with different labels.\n"
                        f"old = {self.fixed_cut_axes[name]}\n"
                        f"new = {labels}"
                    )
            else:
                self.fixed_cut_axes[name] = labels

            # Book the cutflow axis in each output, including the ttbar flavours.
            target_dicts = [output]

            if tt_masks is not None:
                target_dicts += [output_ttBB, output_ttCC, output_ttLF]

            for out_dict in target_dicts:
                if out_dict is None:
                    continue

                for label in labels:
                    out_dict[name].fill(
                        cut=np.array([label]),
                        weight=np.array([0.0], dtype=np.float64),
                    )

        book_eventflow_bins(
            "ttbar_flavour_counts",
            ["parent_total", "ttBB", "ttCC", "ttLF"]
        )

        def fill_category_flow_auto(name, labels_masks, weight):
            """Fill the jet and b-tag categories using the ttbar flavour masks."""
            for label, mask in labels_masks:
                fill_eventflow_auto(name, label, mask, weight)

        for flow_name, flow_labels in EVENTFLOW_BINS.items():
            book_eventflow_bins(flow_name, flow_labels)

        # Fill the BDT cut scan, including the ttbar flavour outputs.

        if tt_masks is None:

            # Inclusive sample.
            for i, cut in enumerate(self.optim_Cuts1_bdt):
                output["all_optim_cut"].fill(cut_index=i, weight=cut)

            for label in self.systematics_labels:
                output["all_optim_systs"].fill(syst=label, weight=1)

        else:

            for flavor, fmask in tt_masks.items():

                if flavor == "ttBB":
                    out_dict = output_ttBB
                elif flavor == "ttCC":
                    out_dict = output_ttCC
                else:
                    out_dict = output_ttLF

                # Fill each ttbar flavour separately.
                for i, cut in enumerate(self.optim_Cuts1_bdt):
                    out_dict["all_optim_cut"].fill(cut_index=i, weight=cut)

                for label in self.systematics_labels:
                    out_dict["all_optim_systs"].fill(syst=label, weight=1)

        # Configure leptons and jets.
        muons = events.Muon[(events.Muon.pt > 10) & (np.abs(events.Muon.eta) < 2.4) & (events.Muon.tightId>0.5) & (events.Muon.pfRelIso04_all < 0.15) ]
        electrons = events.Electron[(events.Electron.pt > 15) & (np.abs(events.Electron.superclusterEta) < 2.5) & ((np.abs(events.Electron.superclusterEta) < 1.4442) | (np.abs(events.Electron.superclusterEta) > 1.566)) & (events.Electron.mvaIso_WP80>0) & (events.Electron.pfRelIso03_all < 0.15)]
        muons = ak.with_field(muons, "mu", "lepton_type")
        electrons = ak.with_field(electrons, "e", "lepton_type")
        leptons = ak.concatenate([muons, electrons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
        n_leptons = ak.num(leptons)
        PuppiMETCorr = events.PuppiMET

        # Jet selection.
        jets = events.Jet
        # Select jets before lepton cleaning.
        jets_base = jets[
            (jets.pt > 20) &
            (np.abs(jets.eta) < 2.5)
        ]

        # Clean jets against all loose leptons.
        jets_base_cc_allleps = clean_by_dr(jets_base, leptons, 0.4)

        w_resolved_nosf = w_all

        def build_resolved_permask(mask):

            n_events = len(mask)

            if not np.any(mask):
                empty = ak.Array([[]] * n_events)
                return empty, empty, empty, np.zeros(n_events), np.zeros(n_events)

            jets_clean = jets_base_cc_allleps[mask]
            jets_clean = jets_clean[
                ak.argsort(jets_clean.btagUParTAK4B, axis=-1, ascending=False)
            ]

            tagged = (jets_clean.btagUParTAK4B >= FIXED_BTAG_THRESHOLD)

            bjets = jets_clean[tagged]
            untag = jets_clean[~tagged]
            # Evaluate the selected-event arrays.
            nj_small = ak.to_numpy(ak.num(jets_clean))
            nb_small = ak.to_numpy(ak.num(bjets))

            # Restore the full event indexing.
            nj_full = np.zeros(n_events, dtype=np.int32)
            nb_full = np.zeros(n_events, dtype=np.int32)

            idx = np.where(mask)[0]
            nj_full[idx] = nj_small
            nb_full[idx] = nb_small

            return jets_clean, bjets, untag, nj_full, nb_full

        def count_bjets_wp(jets_clean, wp_value, n_events, base_mask):
            nb_small = ak.to_numpy(
                ak.num(jets_clean[jets_clean.btagUParTAK4B >= wp_value])
            )

            nb_full = np.zeros(n_events, dtype=np.int32)
            idx = np.where(base_mask)[0]
            nb_full[idx] = nb_small

            return nb_full

        def compute_btag_sf_fixed_full_auto(jets_ge3, mask_ge3j):
            sf_full = np.ones(
                len(events),
                dtype=np.float64,
            )

            mask_ge3j = np.asarray(
                mask_ge3j,
                dtype=bool,
            )

            if not self.isMC or not np.any(mask_ge3j):
                return sf_full

            idx = np.where(mask_ge3j)[0]

            if len(jets_ge3) != len(idx):
                raise RuntimeError(
                    "Fixed-WP mask/jets length mismatch"
                )

            # Non-ttbar sample.
            if tt_masks is None:
                sf_full[idx] = ak.to_numpy(
                    self.eval_btag_sf_fixedWP_resolved(
                        jets_ge3,
                        eff_process=self._btag_eff_process,
                    )
                )

                return sf_full

            # TT sample: use ttBB, ttCC or ttLF efficiency maps.
            covered = np.zeros(
                len(idx),
                dtype=bool,
            )

            for flavour in ("ttBB", "ttCC", "ttLF"):
                local_mask = np.asarray(
                    tt_masks[flavour],
                    dtype=bool,
                )[mask_ge3j]

                if not np.any(local_mask):
                    continue

                if np.any(covered & local_mask):
                    raise RuntimeError(
                        "Overlapping ttbar flavour masks"
                    )

                covered |= local_mask

                sf_full[idx[local_mask]] = ak.to_numpy(
                    self.eval_btag_sf_fixedWP_resolved(
                        jets_ge3[local_mask],
                        eff_process=flavour,
                    )
                )

            if not np.all(covered):
                raise RuntimeError(
                    "Some selected ttbar events are not classified "
                    "as ttBB/ttCC/ttLF"
                )

            return sf_full

        DBTAG_WP = 0.12

        def fill_boosted_jet_dbtag_categories(channel, region, base_mask, jets_clean):
            if not np.any(base_mask):
                return

            idx = np.where(base_mask)[0]

            nj_small = ak.to_numpy(ak.num(jets_clean))
            ndb_small = ak.to_numpy(
                ak.num(jets_clean[jets_clean.btagUParTAK4probbb >= DBTAG_WP])
            )

            nj_full = np.zeros(len(events), dtype=np.int32)
            ndb_full = np.zeros(len(events), dtype=np.int32)

            nj_full[idx] = nj_small
            ndb_full[idx] = ndb_small

            labels_masks = [
                ("eq2j_eq1db",     base_mask & (nj_full == 2) & (ndb_full == 1)),
                ("eq2j_eq2db",     base_mask & (nj_full == 2) & (ndb_full == 2)),

                ("eq3j_eq1db",     base_mask & (nj_full == 3) & (ndb_full == 1)),

                ("eq3j_eq2db",     base_mask & (nj_full == 3) & (ndb_full == 2)),
                ("eq3j_eq3db",     base_mask & (nj_full == 3) & (ndb_full == 3)),
                ("eq4j_eq1db",     base_mask & (nj_full == 4) & (ndb_full == 1)),
                ("eq4j_eq2db",     base_mask & (nj_full == 4) & (ndb_full == 2)),
                ("eq4j_eq3db",     base_mask & (nj_full == 4) & (ndb_full == 3)),
                ("eq4j_eq4db",     base_mask & (nj_full == 4) & (ndb_full == 4)),
                 ("geq5j_eq1db",     base_mask & (nj_full >= 5) & (ndb_full == 1)),
                 ("geq5j_eq2db",     base_mask & (nj_full >= 5) & (ndb_full == 2)),
                 ("geq5j_eq3db",     base_mask & (nj_full >= 5) & (ndb_full == 3)),
                ("geq5j_eq4db",     base_mask & (nj_full >= 5) & (ndb_full == 4)),
                 ("geq5j_eq5db",     base_mask & (nj_full >= 5) & (ndb_full >= 5)),
            ]

            labels_boosted = [label for label, _ in labels_masks]

            name_boost = f"{channel}_{region}_jet_dbtag_categories_boosted"

            book_eventflow_bins(name_boost, labels_boosted)

            fill_category_flow_auto(
                name_boost,
                labels_masks,
                w_all,
            )

        def fill_fixedwp_prebtag_validation(
            channel,
            region,
            mask_ge3j,
            jets_ge3,
            njets_full,
            nbjets_full,
            btag_sf_full,
            w_nosf,
            w_sf,
        ):
            #Compare no-SF and with-SF weights before the b-jet requirement.Both distributions use the same events with at least three jets.
            
            mask_ge3j = np.asarray(
                mask_ge3j,
                dtype=bool,
            )

            njets_full = np.asarray(
                njets_full,
                dtype=np.int32,
            )

            nbjets_full = np.asarray(
                nbjets_full,
                dtype=np.int32,
            )

            btag_sf_full = np.asarray(
                btag_sf_full,
                dtype=np.float64,
            )

            w_nosf = np.asarray(
                w_nosf,
                dtype=np.float64,
            )

            w_sf = np.asarray(
                w_sf,
                dtype=np.float64,
            )

            n_events = len(events)

            for name, values in (
                ("mask_ge3j", mask_ge3j),
                ("njets_full", njets_full),
                ("nbjets_full", nbjets_full),
                ("btag_sf_full", btag_sf_full),
                ("w_nosf", w_nosf),
                ("w_sf", w_sf),
            ):
                if len(values) != n_events:
                    raise RuntimeError(
                        f"{channel}/{region}: {name} has "
                        f"length {len(values)}, expected {n_events}"
                    )

            if not np.any(mask_ge3j):
                return

            idx = np.where(mask_ge3j)[0]

            if len(jets_ge3) != len(idx):
                raise RuntimeError(
                    f"{channel}/{region}: jets/mask mismatch: "
                    f"len(jets_ge3)={len(jets_ge3)}, "
                    f"selected events={len(idx)}"
                )

            # Keep HT indexed by the full event collection.
            ht_full = np.zeros(
                n_events,
                dtype=np.float64,
            )

            ht_full[idx] = ak.to_numpy(
                ak.sum(
                    jets_ge3.pt,
                    axis=1,
                )
            )

            # Check the relation between the no-SF and with-SF weights.
            expected_sf_weight = (
                w_nosf[mask_ge3j]
                * btag_sf_full[mask_ge3j]
            )

            if not np.allclose(
                w_sf[mask_ge3j],
                expected_sf_weight,
                rtol=1e-12,
                atol=1e-14,
            ):
                raise RuntimeError(
                    f"{channel}/{region}: fixed-WP weight is not "
                    "w_nosf * btag_sf"
                )

            if np.any(~np.isfinite(btag_sf_full[mask_ge3j])):
                raise RuntimeError(
                    f"{channel}/{region}: non-finite fixed-WP SF"
                )

            if np.any(btag_sf_full[mask_ge3j] <= 0.0):
                raise RuntimeError(
                    f"{channel}/{region}: non-positive fixed-WP SF"
                )

            wp_tag = f"fixed{FIXED_BTAG_WP}WP"

            # Use the same event mask for both weight choices.
            for suffix, weight in (
                ("nosf", w_nosf),
                ("withSF", w_sf),
            ):
                fill_hist_auto(
                    f"{channel}_{region}_ge3j_"
                    f"njets_{wp_tag}_{suffix}_resolved",
                    mask_ge3j,
                    weight,
                    n_jets=njets_full,
                )

                fill_hist_auto(
                    f"{channel}_{region}_ge3j_"
                    f"nbjets_{wp_tag}_{suffix}_resolved",
                    mask_ge3j,
                    weight,
                    n_bjets=nbjets_full,
                )

                fill_hist_auto(
                    f"{channel}_{region}_ge3j_"
                    f"HT_{wp_tag}_{suffix}_resolved",
                    mask_ge3j,
                    weight,
                    HT=ht_full,
                )

                fill_hist_auto(
                    f"{channel}_{region}_ge3j_"
                    f"nbjets_vs_njets_{wp_tag}_{suffix}_resolved",
                    mask_ge3j,
                    weight,
                    n_jets=njets_full,
                    n_bjets=nbjets_full,
                )

            # Weight the SF distribution with the nominal no-SF weights.
            fill_hist_auto(
                f"{channel}_{region}_ge3j_"
                f"event_btagSF_{wp_tag}_resolved",
                mask_ge3j,
                w_nosf,
                btag_sf=btag_sf_full,
            )

            fill_hist_auto(
                f"{channel}_{region}_ge3j_"
                f"event_btagSF_vs_njets_{wp_tag}_resolved",
                mask_ge3j,
                w_nosf,
                n_jets=njets_full,
                btag_sf=btag_sf_full,
            )

            fill_hist_auto(
                f"{channel}_{region}_ge3j_"
                f"event_btagSF_vs_nbjets_{wp_tag}_resolved",
                mask_ge3j,
                w_nosf,
                n_bjets=nbjets_full,
                btag_sf=btag_sf_full,
            )

        def fill_fixedwp_categories(
            channel,
            region,
            base_mask,
            njets_full,
            nbjets_full,
            ndb_full,
            w_nosf,
            w_sf,
        ):
            base_mask = np.asarray(base_mask, dtype=bool)
            njets_full = np.asarray(njets_full, dtype=np.int32)
            nbjets_full = np.asarray(nbjets_full, dtype=np.int32)
            ndb_full = np.asarray(ndb_full, dtype=np.int32)

            pre_veto_mask = base_mask

            resolved_mask = (
                base_mask
                & (ndb_full < 2)
            )

            wp_tag = f"fixed{FIXED_BTAG_WP}WP"

            for selection, mask in (
                ("preBoostVeto", pre_veto_mask),
                ("resolved", resolved_mask),
            ):
                for suffix, weight in (
                    ("nosf", w_nosf),
                    ("withSF", w_sf),
                ):
                    fill_hist_auto(
                        f"{channel}_{region}_"
                        f"nbjets_vs_njets_{wp_tag}_"
                        f"{selection}_{suffix}",
                        mask,
                        weight,
                        n_jets=njets_full,
                        n_bjets=nbjets_full,
                    )

            category_mask = (
                resolved_mask
                & (njets_full >= 3)
            )

            labels = [
                "eq0b",
                "eq1b",
                "eq2b",
                "eq3b",
                "eq4b",
                "ge5b",
            ]

            categories = [
                ("eq0b", category_mask & (nbjets_full == 0)),
                ("eq1b", category_mask & (nbjets_full == 1)),
                ("eq2b", category_mask & (nbjets_full == 2)),
                ("eq3b", category_mask & (nbjets_full == 3)),
                ("eq4b", category_mask & (nbjets_full == 4)),
                ("ge5b", category_mask & (nbjets_full >= 5)),
            ]

            no_sf_name = (
                f"{channel}_{region}_ge3j_"
                f"nb_categories_{wp_tag}_nosf"
            )

            with_sf_name = (
                f"{channel}_{region}_ge3j_"
                f"nb_categories_{wp_tag}_withSF"
            )

            book_eventflow_bins(no_sf_name, labels)
            book_eventflow_bins(with_sf_name, labels)

            for label, category in categories:
                fill_eventflow_auto(
                    no_sf_name,
                    label,
                    category,
                    w_nosf,
                )

                fill_eventflow_auto(
                    with_sf_name,
                    label,
                    category,
                    w_sf,
                )

        def build_boosted_permask(mask):

            n_events = len(mask)

            if not np.any(mask):
                empty = ak.Array([[]] * n_events)
                return empty, empty, empty, np.zeros(n_events), np.zeros(n_events)

            jets_clean = jets_base_cc_allleps[mask]

            jets_clean = jets_clean[
                ak.argsort(jets_clean.btagUParTAK4probbb, axis=-1, ascending=False)
            ]

            bjets = jets_clean[jets_clean.btagUParTAK4probbb >= 0.12]
            untag = jets_clean[jets_clean.btagUParTAK4probbb <  0.12]

            nj_small = ak.to_numpy(ak.num(jets_clean))
            nb_small = ak.to_numpy(ak.num(bjets))

            nj_full = np.zeros(n_events, dtype=np.int32)
            nb_full = np.zeros(n_events, dtype=np.int32)

            idx = np.where(mask)[0]
            nj_full[idx] = nj_small
            nb_full[idx] = nb_small

            return jets_clean, bjets, untag, nj_full, nb_full

        def make_resolved_btag_masks_full(mask_ge3j, jets_ge3, ndb_full):
            out = {}

            for wp_name in BTAG_WPS:
                out[wp_name] = {
                    "ge3b_preVeto": np.zeros(len(events), dtype=bool),
                    "eq2b_preVeto": np.zeros(len(events), dtype=bool),
                    "ge3b_boostVeto": np.zeros(len(events), dtype=bool),
                    "eq2b_boostVeto": np.zeros(len(events), dtype=bool),
                }

            if not np.any(mask_ge3j):
                return out

            if "btagUParTAK4B" not in ak.fields(jets_ge3):
                return out

            idx = np.where(mask_ge3j)[0]
            ndb_small = ndb_full[idx]

            for wp_name, wp_value in BTAG_WPS.items():
                nb_small = ak.to_numpy(
                    ak.sum(jets_ge3.btagUParTAK4B >= wp_value, axis=1)
                )

                out[wp_name]["ge3b_preVeto"][idx] = nb_small >= 3
                out[wp_name]["eq2b_preVeto"][idx] = nb_small == 2
                out[wp_name]["ge3b_boostVeto"][idx] = (nb_small >= 3) & (ndb_small < 2)
                out[wp_name]["eq2b_boostVeto"][idx] = (nb_small == 2) & (ndb_small < 2)

            return out

        def fill_boosted_predbtag_scores(channel, region, mask_ge2j, jets_boost_ge2):
            """Fill the two highest double-b scores before the tag requirement.

            The events already pass the two-jet requirement.
            """
            if not np.any(mask_ge2j):
                return

            jets_sorted = jets_boost_ge2[
                ak.argsort(jets_boost_ge2.btagUParTAK4probbb, axis=-1, ascending=False)
            ]

            idx = np.where(mask_ge2j)[0]

            lead_dbtag = np.zeros(len(events), dtype=np.float64)
            sub_dbtag = np.zeros(len(events), dtype=np.float64)

            lead_dbtag[idx] = ak.to_numpy(
                ak.fill_none(jets_sorted[:, 0].btagUParTAK4probbb, 0.0)
            )
            sub_dbtag[idx] = ak.to_numpy(
                ak.fill_none(jets_sorted[:, 1].btagUParTAK4probbb, 0.0)
            )

            fill_hist_auto(
                f"{channel}_{region}_ge2j_lead_UParTAK4probbb_boosted",
                mask_ge2j,
                w_all,
                btag=lead_dbtag,
            )

            fill_hist_auto(
                f"{channel}_{region}_ge2j_sublead_UParTAK4probbb_boosted",
                mask_ge2j,
                w_all,
                btag=sub_dbtag,
            )

        def fill_boosted_predbtag_validation(
            channel,
            region,
            mask_ge2j,
            jets_boost_ge2,
            nj_full,
            nb_full,
        ):
            if not np.any(mask_ge2j):
                return

            ht_full = np.zeros(len(events), dtype=np.float64)
            idx = np.where(mask_ge2j)[0]

            ht_full[idx] = ak.to_numpy(ak.sum(jets_boost_ge2.pt, axis=1))

            fill_hist_auto(
                f"{channel}_{region}_ge2j_njets_predbtag_boosted",
                mask_ge2j,
                w_all,
                n_jets=nj_full,
            )

            fill_hist_auto(
                f"{channel}_{region}_ge2j_ndbjets_predbtag_boosted",
                mask_ge2j,
                w_all,
                n_bjets=nb_full,
            )

            fill_hist_auto(
                f"{channel}_{region}_ge2j_HT_predbtag_boosted",
                mask_ge2j,
                w_all,
                HT=ht_full,
            )

        def fill_boosted_dbjet_kinematics(channel, region, mask, dbjets, sort_by):
            if not np.any(mask):
                return

            if sort_by == "pt":
                jets_sorted = dbjets[
                    ak.argsort(dbjets.pt, axis=-1, ascending=False)
                ]
            elif sort_by == "btag":
                jets_sorted = dbjets[
                    ak.argsort(dbjets.btagUParTAK4probbb, axis=-1, ascending=False)
                ]
            else:
                raise ValueError(f"Unknown sort_by: {sort_by}")

            jets_sorted = jets_sorted[ak.num(jets_sorted) >= 2]

            if len(jets_sorted) == 0:
                return

            idx = np.where(mask)[0]

            lead = jets_sorted[:, 0]
            sub = jets_sorted[:, 1]

            lead_pt = np.zeros(len(events), dtype=np.float64)
            sub_pt = np.zeros(len(events), dtype=np.float64)
            lead_mass = np.zeros(len(events), dtype=np.float64)
            sub_mass = np.zeros(len(events), dtype=np.float64)

            lead_pt[idx] = ak.to_numpy(lead.pt)
            sub_pt[idx] = ak.to_numpy(sub.pt)
            lead_mass[idx] = ak.to_numpy(lead.mass)
            sub_mass[idx] = ak.to_numpy(sub.mass)

            tag = f"sortedBy{sort_by}"

            fill_hist_auto(
                f"{channel}_{region}_lead_dbjet_pt_{tag}_boosted",
                mask,
                w_all,
                dbjet_pt=lead_pt,
            )

            fill_hist_auto(
                f"{channel}_{region}_sublead_dbjet_pt_{tag}_boosted",
                mask,
                w_all,
                dbjet_pt=sub_pt,
            )

            fill_hist_auto(
                f"{channel}_{region}_lead_dbjet_mass_{tag}_boosted",
                mask,
                w_all,
                dbjet_mass=lead_mass,
            )

            fill_hist_auto(
                f"{channel}_{region}_sublead_dbjet_mass_{tag}_boosted",
                mask,
                w_all,
                dbjet_mass=sub_mass,
            )
        # Select opposite-sign, same-flavour lepton pairs.
        # Offline lepton-pT thresholds.
        PT_E_LEAD = 25.0
        PT_E_SUB = 15.0
        PT_MU_LEAD = 20.0
        PT_MU_SUB = 10.0

        # Require at least two leptons.
        has_2lep = ak.num(leptons) >= 2
        if not np.any(ak.to_numpy(has_2lep)):
            return output

        # Build pairs only in events with at least two leptons.
        leps_ge2 = leptons[has_2lep]
        lead2 = leps_ge2[:, :2]

        sf_small = ak.fill_none(lead2[:, 0].lepton_type == lead2[:, 1].lepton_type, False)
        os_small = ak.fill_none(lead2[:, 0].charge * lead2[:, 1].charge == -1,       False)

        # Identify ee and mumu pairs.
        ee_pair_small = ak.fill_none((lead2[:, 0].lepton_type == "e")  & (lead2[:, 1].lepton_type == "e"),  False)
        mumu_pair_small = ak.fill_none((lead2[:, 0].lepton_type == "mu") & (lead2[:, 1].lepton_type == "mu"), False)

        # Apply the channel-specific pT thresholds to this pair.
        pt_ee_small = ak.fill_none((lead2[:, 0].pt > PT_E_LEAD)  & (lead2[:, 1].pt > PT_E_SUB),   False)
        pt_mumu_small = ak.fill_none((lead2[:, 0].pt > PT_MU_LEAD) & (lead2[:, 1].pt > PT_MU_SUB),  False)

        step1_small = sf_small & os_small & ((ee_pair_small & pt_ee_small) | (mumu_pair_small & pt_mumu_small))

        # Restore the masks to the full event collection.
        mask_step1 = np.zeros(len(events), dtype=bool)
        mask_ee = np.zeros(len(events), dtype=bool)
        mask_mumu = np.zeros(len(events), dtype=bool)

        idx_has2 = np.where(ak.to_numpy(has_2lep))[0]
        mask_step1[idx_has2] = ak.to_numpy(step1_small)

        # Channel masks include the opposite-sign and pT requirements.
        mask_ee[idx_has2] = ak.to_numpy(ee_pair_small   & pt_ee_small   & sf_small & os_small)
        mask_mumu[idx_has2] = ak.to_numpy(mumu_pair_small & pt_mumu_small & sf_small & os_small)
        mask_ll = mask_ee | mask_mumu
        # Build the opposite-sign e-mu seed for the ttbar control region.
        # Use the same leading pair for the e-mu selection.

        # Require different flavours.
        df_small = ak.fill_none(lead2[:, 0].lepton_type != lead2[:, 1].lepton_type, False)
        os_df_small = ak.fill_none(lead2[:, 0].charge * lead2[:, 1].charge == -1, False)

        l0 = lead2[:, 0]
        l1 = lead2[:, 1]

        # Apply the electron and muon pT thresholds.
        pt_l_ok = (
            (((l0.lepton_type == "e")  & (l0.pt > PT_E_LEAD)) &  ((l1.lepton_type == "mu") & (l1.pt > PT_E_SUB))) |
            (((l0.lepton_type == "mu") & (l0.pt > PT_E_LEAD))& ((l1.lepton_type == "e")  & (l1.pt > PT_E_SUB)))
        )

        pt_df_small = pt_l_ok

        step1_df_small = df_small & os_df_small & pt_df_small

        # Restore the e-mu mask to the full event collection.
        mask_df = np.zeros(len(events), dtype=bool)
        mask_df[idx_has2] = ak.to_numpy(step1_df_small)
        # Leading and subleading lepton pT before the trigger.

        lep0_pt_full = np.zeros(len(events))
        lep1_pt_full = np.zeros(len(events))

        if np.any(has_2lep):
            pairs = leptons[has_2lep][:, :2]
            idx = np.where(ak.to_numpy(has_2lep))[0]

            lep0_pt_full[idx] = ak.to_numpy(pairs[:, 0].pt)
            lep1_pt_full[idx] = ak.to_numpy(pairs[:, 1].pt)

        fill_hist_auto(
            "ee_lep0_pt_preTrig",
            mask_ee,
            w_all,
            lep0_pt=lep0_pt_full,
        )

        fill_hist_auto(
            "ee_lep1_pt_preTrig",
            mask_ee,
            w_all,
            lep1_pt=lep1_pt_full,
        )

        fill_hist_auto(
            "mumu_lep0_pt_preTrig",
            mask_mumu,
            w_all,
            lep0_pt=lep0_pt_full,
        )

        fill_hist_auto(
            "mumu_lep1_pt_preTrig",
            mask_mumu,
            w_all,
            lep1_pt=lep1_pt_full,
        )

        fill_hist_auto(
            "emu_lep0_pt_preTrig",
            mask_df,
            w_all,
            lep0_pt=lep0_pt_full,
        )

        fill_hist_auto(
            "emu_lep1_pt_preTrig",
            mask_df,
            w_all,
            lep1_pt=lep1_pt_full,
        )

        fill_eventflow_auto("ee_eventflow_SR1_boosted",  '>=2lep OSSF', mask_ee,   w_all)
        fill_eventflow_auto("mumu_eventflow_SR1_boosted",'>=2lep OSSF', mask_mumu, w_all)

        fill_eventflow_auto("ee_eventflow_SR_resolved", '>=2lep OSSF', mask_ee,   w_resolved_nosf)
        fill_eventflow_auto("mumu_eventflow_SR_resolved",'>=2lep OSSF', mask_mumu, w_resolved_nosf)

        fill_eventflow_auto("emu_eventflow_TTCR1_boosted",'>=2lep OSDF', mask_df,   w_all)

        fill_eventflow_auto("emu_eventflow_TTCR_resolved",'>=2lep OSDF', mask_df,   w_resolved_nosf)

        # Apply the channel triggers.

        # Trigger bit positions in the skimmed trigger word.
        MUMU_BM = (1 << 0)  # Dimuon.
        MU_BM = (1 << 1)  # Single muon.
        EE_BM = (1 << 2)  # Dielectron.
        E_BM = (1 << 3)  # Single electron.
        EMU_BM = (1 << 4)  # MuonEG.

        # Keep the trigger word as a NumPy array to match the masks.
        if "trigger_type" in events.fields :
            trig_word = ak.to_numpy(events.trigger_type).astype(np.int64)
        else:
            trig_word = np.zeros(len(events), dtype=np.int64)

        trg_mumu = (trig_word & MUMU_BM) != 0
        trg_mu = (trig_word & MU_BM)   != 0
        trg_ee = (trig_word & EE_BM)   != 0
        trg_e = (trig_word & E_BM)    != 0
        trg_emu = (trig_word & EMU_BM)  != 0

        # Match each channel to its trigger.
        pass_trig_ee = mask_ee   & (trg_ee)
        pass_trig_ee_mc = mask_ee   & (trg_ee)
        pass_trig_mumu = mask_mumu & ( trg_mumu)
        pass_trig_df = mask_df   & (trg_emu )

        # Assign data events to their primary dataset.
        if not self.isMC:

            dname = self.dataset_name.lower()

            isMuonEG = dname.startswith("muoneg")
            isEGamma = dname.startswith("egamma")
            isMuon = dname.startswith("muon") and not isMuonEG

            mask_step_trig_ee = np.zeros(len(events), dtype=bool)
            mask_step_trig_mumu = np.zeros(len(events), dtype=bool)
            mask_step_trig_df = np.zeros(len(events), dtype=bool)

            if isMuon:
                mask_step_trig_mumu = pass_trig_mumu

            elif isEGamma:
                mask_step_trig_ee = pass_trig_ee

            elif isMuonEG:
                mask_step_trig_df = pass_trig_df

        else:
            # MC has no primary-dataset ownership restriction.
            mask_step_trig_ee = pass_trig_ee_mc
            mask_step_trig_mumu = pass_trig_mumu
            mask_step_trig_df = pass_trig_df
        mask_step_trig_ll = mask_step_trig_ee | mask_step_trig_mumu

        fill_eventflow_auto("ee_eventflow_SR1_boosted",  "trigger", mask_step_trig_ee,   w_all)
        fill_eventflow_auto("mumu_eventflow_SR1_boosted","trigger", mask_step_trig_mumu, w_all)

        fill_eventflow_auto("ee_eventflow_SR_resolved", "trigger", mask_step_trig_ee,   w_resolved_nosf)
        fill_eventflow_auto("mumu_eventflow_SR_resolved","trigger", mask_step_trig_mumu, w_resolved_nosf)

        fill_eventflow_auto("emu_eventflow_TTCR1_boosted","trigger", mask_step_trig_df,   w_all)
        fill_eventflow_auto("emu_eventflow_TTCR_resolved","trigger", mask_step_trig_df,   w_resolved_nosf)
        # Leading and subleading lepton pT after the trigger.

        fill_hist_auto(
            "mumu_lep0_pt_postTrig",
            mask_step_trig_mumu,
            w_all,
            lep0_pt=lep0_pt_full,
                )

        fill_hist_auto(
            "mumu_lep1_pt_postTrig",
            mask_step_trig_mumu,
            w_all,
            lep1_pt=lep1_pt_full,
        )

        fill_hist_auto(
            "emu_lep0_pt_postTrig",
            mask_step_trig_df,
            w_all,
            lep0_pt=lep0_pt_full,
        )

        fill_hist_auto(
            "emu_lep1_pt_postTrig",
            mask_step_trig_df,
            w_all,
            lep1_pt=lep1_pt_full,
        )

        fill_hist_auto(
        "ee_lep0_pt_postTrig",
        mask_step_trig_ee,
        w_all,
        lep0_pt=lep0_pt_full,
        )

        fill_hist_auto(
            "ee_lep1_pt_postTrig",
            mask_step_trig_ee,
            w_all,
            lep1_pt=lep1_pt_full,
        )

        # TTCR jet multiplicity after the trigger, without a Z-mass cut.

        sj_emu, sbj_emu, sut_emu, nj_emu, nb_emu = build_resolved_permask(mask_step_trig_df)

        fill_hist_auto(
            "emu_TTCR_njets_pre3_resolved",
            mask_step_trig_df,
            w_resolved_nosf,
            n_jets=nj_emu,
        )

        # Z-mass window.
        ZLO, ZHI = 75.0, 105.0
        mask_z_ee = z_window_mask(leptons, mask_step_trig_ee)
        mask_z_mumu = z_window_mask(leptons, mask_step_trig_mumu)

        mask_z_ll = mask_z_ee | mask_z_mumu

        # Reconstruct mll before applying the Z-mass window.
        mask_ll_trig = mask_step_trig_ee | mask_step_trig_mumu

        if np.any(mask_ll_trig):
            pairs = leptons[mask_ll_trig][:, :2]
            mll_all = (make_vector(pairs[:, 0]) + make_vector(pairs[:, 1])).mass

            idx = np.where(mask_ll_trig)[0]
            mll_full = np.zeros(len(events))
            mll_full[idx] = ak.to_numpy(mll_all)
        else:
            mll_full = np.zeros(len(events))
        # Resolved mll distributions before the Z-mass window.
        fill_hist_auto(
            "ee_SR_mll_preZ_resolved",
            mask_step_trig_ee,
            w_resolved_nosf,
            m_ll=mll_full,
        )

        fill_hist_auto(
            "mumu_SR_mll_preZ_resolved",
            mask_step_trig_mumu,
            w_resolved_nosf,
            m_ll=mll_full,
        )

        # Jet multiplicity after the Z window and before the resolved jet cut.
        # Dielectron signal region.
        # Resolved selection.
        sj_ee, sbj_ee, sut_ee, nj_ee, nb_ee = build_resolved_permask(mask_z_ee)
        mask_res_jets2_ee = mask_z_ee & (nj_ee >= 2)
        mask_res_jets_ee = mask_z_ee & (nj_ee >= 3)
        mask_2res_2btag_ee = mask_res_jets2_ee & (nb_ee == 2)
        mask_res_2btag_ee = mask_res_jets_ee & (nb_ee >= 2)
        # Boosted selection.
        sj_b_ee, sbj_b_ee, sut_b_ee, nj_b_ee, nb_b_ee = build_boosted_permask(mask_z_ee)
        mask_boosted_jets_ee = mask_z_ee & (nj_b_ee >= 2)
        mask_boosted_btag_ee = mask_boosted_jets_ee & (nb_b_ee >= 2)

        fill_hist_auto(
        "ee_SR_njets_pre3_resolved",
        mask_z_ee,
        w_resolved_nosf,
        n_jets=nj_ee,
        )
        # Dimuon signal region.
        sj_mumu, sbj_mumu, sut_mumu, nj_mumu, nb_mumu = build_resolved_permask(mask_z_mumu)
        dj_mumu, dbj_mumu, dut_mumu, njb_mumu, nbb_mumu = build_boosted_permask(mask_z_mumu)

        # The e-mu TTCR has no Z-mass window.
        sj_emu, sbj_emu, sut_emu, nj_emu, nb_emu = build_resolved_permask(mask_step_trig_df)
        dj_emu, dbj_emu, dut_emu, njb_emu, nbb_emu = build_boosted_permask(mask_step_trig_df)

        mask_boost_ge2j_ee = mask_z_ee & (nj_b_ee >= 2)
        mask_boost_ge2j_mumu = mask_z_mumu & (njb_mumu >= 2)
        mask_boost_ge2j_emu = mask_step_trig_df & (njb_emu >= 2)

        # Count the jet and one-double-b-tag steps before the final boosted cut.
        mask_boost_ge1db_ee = mask_boost_ge2j_ee & (nb_b_ee >= 1)
        mask_boost_ge1db_mumu = mask_boost_ge2j_mumu & (nbb_mumu >= 1)
        mask_boost_ge1db_emu = mask_boost_ge2j_emu & (nbb_emu >= 1)

        for flow_name, mask_ge2j, mask_ge1db in (
            ("ee_eventflow_SR1_boosted", mask_boost_ge2j_ee, mask_boost_ge1db_ee),
            ("mumu_eventflow_SR1_boosted", mask_boost_ge2j_mumu, mask_boost_ge1db_mumu),
            ("emu_eventflow_TTCR1_boosted", mask_boost_ge2j_emu, mask_boost_ge1db_emu),
        ):
            fill_eventflow_auto(flow_name, ">=2jets", mask_ge2j, w_all)
            fill_eventflow_auto(flow_name, ">=2jets & >=1dbjet", mask_ge1db, w_all)

        jets_boost_ee_ge2 = sj_b_ee[ak.num(sj_b_ee) >= 2]
        jets_boost_mumu_ge2 = dj_mumu[ak.num(dj_mumu) >= 2]
        jets_boost_emu_ge2 = dj_emu[ak.num(dj_emu) >= 2]

        mask_ll_ge2j_dbtag = mask_boost_ge2j_ee | mask_boost_ge2j_mumu

        jets_ll_ge2j_dbtag = ak.concatenate(
            [jets_boost_ee_ge2, jets_boost_mumu_ge2],
            axis=0,
        )

        fill_lead_sublead_dbtag2d_auto(
            mask_ll_ge2j_dbtag,
            jets_ll_ge2j_dbtag,
            w_all,
        )
        fill_boosted_predbtag_scores(
            "ee", "SR", mask_boost_ge2j_ee, jets_boost_ee_ge2
        )

        fill_boosted_predbtag_scores(
            "mumu", "SR", mask_boost_ge2j_mumu, jets_boost_mumu_ge2
        )

        fill_boosted_predbtag_scores(
            "emu", "TTCR", mask_boost_ge2j_emu, jets_boost_emu_ge2
        )
        fill_boosted_predbtag_validation(
            "ee", "SR",
            mask_boost_ge2j_ee,
            jets_boost_ee_ge2,
            nj_b_ee,
            nb_b_ee,
        )

        fill_boosted_predbtag_validation(
            "mumu", "SR",
            mask_boost_ge2j_mumu,
            jets_boost_mumu_ge2,
            njb_mumu,
            nbb_mumu,
        )

        fill_boosted_predbtag_validation(
            "emu", "TTCR",
            mask_boost_ge2j_emu,
            jets_boost_emu_ge2,
            njb_emu,
            nbb_emu,
        )
        # Boosted jet and double-b-tag multiplicity histograms.
        # Use the selected DBTAG_WP and the nominal event weights.

        fill_boosted_jet_dbtag_categories(
            "ee",
            "SR",
            mask_boost_ge2j_ee,
            jets_boost_ee_ge2,
        )

        fill_boosted_jet_dbtag_categories(
            "mumu",
            "SR",
            mask_boost_ge2j_mumu,
            jets_boost_mumu_ge2,
        )

        fill_boosted_jet_dbtag_categories(
            "emu",
            "TTCR",
            mask_boost_ge2j_emu,
            jets_boost_emu_ge2,
        )

        fill_eventflow_auto("ee_eventflow_SR_resolved", 'mll', mask_z_ee, w_resolved_nosf)
        fill_eventflow_auto("mumu_eventflow_SR_resolved", 'mll', mask_z_mumu, w_resolved_nosf)

        fill_eventflow_auto("ee_eventflow_SR1_boosted", 'mll', mask_z_ee, w_all)
        fill_eventflow_auto("mumu_eventflow_SR1_boosted", 'mll', mask_z_mumu, w_all)

        mask_res_jets_ee = mask_z_ee   & (nj_ee >= 3)
        mask_res_jets_mumu = mask_z_mumu & (nj_mumu >= 3)

        mask_res_ex3or4jets_ee = mask_z_ee   & ((nj_ee   == 3) |(nj_ee == 4) )
        mask_res_ex3or4jets_mumu = mask_z_mumu & ((nj_mumu == 3) | (nj_mumu == 4))

        mask_res_ex3or4jets_and2b_ee = mask_res_ex3or4jets_ee   & (nb_ee   == 2)
        mask_res_ex3or4jets_and2b_mumu = mask_res_ex3or4jets_mumu   & (nb_mumu   == 2)

        mask_res_3or4b_ee = mask_res_ex3or4jets_ee    & ((nb_ee   == 3) | (nb_ee   == 4))
        mask_res_3or4b_mumu = mask_res_ex3or4jets_mumu & ((nb_mumu   == 3) | (nb_mumu   == 4))

        mask_res_leq3j_3or4b_ee = mask_res_jets_ee    & ((nb_ee   == 3) | (nb_ee   == 4))
        mask_res_leq3j_3or4b_mumu = mask_res_jets_mumu & ((nb_mumu   == 3) | (nb_mumu   == 4))

        mask_res_2b_ee = mask_res_jets_ee   & (nb_ee   >= 2)
        mask_res_2b_mumu = mask_res_jets_mumu & (nb_mumu >= 2)

        mask_res_jets_emu = mask_step_trig_df & (nj_emu   >= 3)
        mask_res_ex3or4jets_emu = mask_step_trig_df   & ((nj_emu   == 3) |(nj_emu == 4) )
        mask_res_3or4b_emu = mask_res_ex3or4jets_emu   & ((nb_emu   == 3)|(nb_emu   == 4))
        mask_res_leq3j_3or4b_emu = mask_res_jets_emu    & ((nb_emu   == 3)|(nb_emu   == 4))

        # Cleaned jets after the resolved jet-multiplicity cut.

        sj_ee_ge3, _, _, _, _ = build_resolved_permask(mask_res_jets_ee)
        sj_mumu_ge3, _, _, _, _ = build_resolved_permask(mask_res_jets_mumu)
        sj_emu_ge3, _, _, _, _ = build_resolved_permask(mask_res_jets_emu)
        # build_resolved_permask() already returns the fixed-WP b-jet counts.
        nb_fixed_ee = nb_ee
        nb_fixed_mumu = nb_mumu
        nb_fixed_emu = nb_emu

        # ee/mumu signal region: >=3 fixed-WP b jets.
        mask_res_btag_ee = (
            mask_res_jets_ee
            & (nb_fixed_ee >= 3)
        )

        mask_res_btag_mumu = (
            mask_res_jets_mumu
            & (nb_fixed_mumu >= 3)
        )

        # emu ttbar control region: >=3 fixed-WP b jets.
        mask_ttcr_res_ge3b = (
            mask_res_jets_emu
            & (nb_fixed_emu >= 3)
        )

        # Z control regions: exactly two fixed-WP b jets.
        mask_crz_ee = (
            mask_res_jets_ee
            & (nb_fixed_ee == 2)
        )

        mask_crz_mumu = (
            mask_res_jets_mumu
            & (nb_fixed_mumu == 2)
        )

        btag_sf_fixed_ee = compute_btag_sf_fixed_full_auto(
            sj_ee_ge3,
            mask_res_jets_ee,
        )

        btag_sf_fixed_mumu = compute_btag_sf_fixed_full_auto(
            sj_mumu_ge3,
            mask_res_jets_mumu,
        )

        btag_sf_fixed_emu = compute_btag_sf_fixed_full_auto(
            sj_emu_ge3,
            mask_res_jets_emu,
        )

        w_resolved_sf_ee = (
            w_resolved_nosf
            * btag_sf_fixed_ee
        )

        w_resolved_sf_mumu = (
            w_resolved_nosf
            * btag_sf_fixed_mumu
        )

        w_resolved_sf_emu = (
            w_resolved_nosf
            * btag_sf_fixed_emu
        )

        fill_fixedwp_prebtag_validation(
            channel="ee",
            region="SR",
            mask_ge3j=mask_res_jets_ee,
            jets_ge3=sj_ee_ge3,
            njets_full=nj_ee,
            nbjets_full=nb_fixed_ee,
            btag_sf_full=btag_sf_fixed_ee,
            w_nosf=w_resolved_nosf,
            w_sf=w_resolved_sf_ee,
        )

        fill_fixedwp_prebtag_validation(
            channel="mumu",
            region="SR",
            mask_ge3j=mask_res_jets_mumu,
            jets_ge3=sj_mumu_ge3,
            njets_full=nj_mumu,
            nbjets_full=nb_fixed_mumu,
            btag_sf_full=btag_sf_fixed_mumu,
            w_nosf=w_resolved_nosf,
            w_sf=w_resolved_sf_mumu,
        )

        fill_fixedwp_prebtag_validation(
            channel="emu",
            region="TTCR",
            mask_ge3j=mask_res_jets_emu,
            jets_ge3=sj_emu_ge3,
            njets_full=nj_emu,
            nbjets_full=nb_fixed_emu,
            btag_sf_full=btag_sf_fixed_emu,
            w_nosf=w_resolved_nosf,
            w_sf=w_resolved_sf_emu,
        )
        fill_fixedwp_categories(
            channel="ee",
            region="SR",
            base_mask=mask_res_jets_ee,
            njets_full=nj_ee,
            nbjets_full=nb_fixed_ee,
            ndb_full=nb_b_ee,
            w_nosf=w_resolved_nosf,
            w_sf=w_resolved_sf_ee,
        )

        fill_fixedwp_categories(
            channel="mumu",
            region="SR",
            base_mask=mask_res_jets_mumu,
            njets_full=nj_mumu,
            nbjets_full=nb_fixed_mumu,
            ndb_full=nbb_mumu,
            w_nosf=w_resolved_nosf,
            w_sf=w_resolved_sf_mumu,
        )

        fill_fixedwp_categories(
            channel="emu",
            region="TTCR",
            base_mask=mask_res_jets_emu,
            njets_full=nj_emu,
            nbjets_full=nb_fixed_emu,
            ndb_full=nbb_emu,
            w_nosf=w_resolved_nosf,
            w_sf=w_resolved_sf_emu,
        )
        fill_hist_auto(
            f"ee_SR_event_btagSF_fixed{FIXED_BTAG_WP}WP_resolved",
            mask_res_jets_ee,
            w_resolved_nosf,
            btag_sf=btag_sf_fixed_ee,
        )

        fill_hist_auto(
            f"mumu_SR_event_btagSF_fixed{FIXED_BTAG_WP}WP_resolved",
            mask_res_jets_mumu,
            w_resolved_nosf,
            btag_sf=btag_sf_fixed_mumu,
        )

        fill_hist_auto(
            f"emu_TTCR_event_btagSF_fixed{FIXED_BTAG_WP}WP_resolved",
            mask_res_jets_emu,
            w_resolved_nosf,
            btag_sf=btag_sf_fixed_emu,
        )
        # Jet versus double-b-tagged jet multiplicity.
        fill_hist_auto("ee_n_Dbjets_vs_njets_boosted", mask_z_ee, w_all,n_jets=nj_b_ee, n_bjets=nb_b_ee)
        fill_hist_auto("mumu_n_Dbjets_vs_njets_boosted", mask_z_mumu, w_all,n_jets=njb_mumu, n_bjets=nbb_mumu)
        fill_hist_auto("emu_n_Dbjets_vs_njets_boosted",  mask_step_trig_df, w_all,n_jets=njb_emu, n_bjets=nbb_emu)

        lead_dbtag = np.zeros(len(events))
        sub_dbtag = np.zeros(len(events))
        mask_SR1_boosted_ee = mask_step_trig_ee   & mask_z_ee   & (nj_b_ee  >= 2) & (nb_b_ee  >= 2)
        mask_SR1_boosted_mumu = mask_step_trig_mumu & mask_z_mumu & (njb_mumu >= 2) & (nbb_mumu >= 2)

        mask_SR1_boosted_ll = mask_SR1_boosted_ee | mask_SR1_boosted_mumu

        lead_dbtag = np.zeros(len(events))
        sub_dbtag = np.zeros(len(events))

        # Dielectron boosted signal region.
        dj_sr1_ee, dbj_sr1_ee, _, _, _ = build_boosted_permask(mask_SR1_boosted_ee)

        idx_ee = np.where(mask_SR1_boosted_ee)[0]

        if len(idx_ee) > 0:
            bb_ee = dbj_sr1_ee[:, :2]   # The arrays are already restricted to selected events.
            lead_dbtag[idx_ee] = ak.to_numpy(bb_ee[:, 0].btagUParTAK4probbb)
            sub_dbtag[idx_ee] = ak.to_numpy(bb_ee[:, 1].btagUParTAK4probbb)

        # Dimuon boosted signal region.
        dj_sr1_mm, dbj_sr1_mm, _, _, _ = build_boosted_permask(mask_SR1_boosted_mumu)

        idx_mm = np.where(mask_SR1_boosted_mumu)[0]

        if len(idx_mm) > 0:
            bb_mm = dbj_sr1_mm[:, :2]
            lead_dbtag[idx_mm] = ak.to_numpy(bb_mm[:, 0].btagUParTAK4probbb)
            sub_dbtag[idx_mm] = ak.to_numpy(bb_mm[:, 1].btagUParTAK4probbb)

        fill_eventflow_auto( "ee_eventflow_SR1_boosted", '>=2jets & >=2dbjets', mask_SR1_boosted_ee, w_all)

        fill_eventflow_auto("mumu_eventflow_SR1_boosted", '>=2jets & >=2dbjets', mask_SR1_boosted_mumu, w_all)

        # MVA TREES (combined ee+μμ, SR only, no padding)
        '''
        if self.isMVA:
            # resolved SR (combined)
            mask = mask_SR_resolved_ll
            sj, sbj, sut, nj, nb = build_resolved_permask(mask)
            # ---------- resolved SR (correct weight = w_resolved) ----------
            
            vals = self.build_vars_resolved(
                leptons[mask],
                sj,
                sbj,
                sut,
                PuppiMETCorr[mask],
            )

            vals["weight"] = w_resolved_nosf[mask]
            # Ensure all schema branches exist
            for key in self.tree_schema_resolved:
                if key not in vals:
                    vals[key] = np.zeros(np.sum(mask), dtype=np.float64)

            if np.any(mask):
                w = w_resolved_nosf[mask]
            else:
                w = np.array([], dtype=np.float64)

            vals["weight"] = w

            self.compat_tree_variables(vals)
            self.add_tree_entry("resolved", vals)
            # ========================================================
            # BOOSTED SR1
            # ========================================================
            mask = mask_SR1_boosted_ll
            dj, dbj, dut, _, _ = build_boosted_permask(mask)

            vals = self.build_vars_boosted(
                leptons[mask],
                dj,
                dbj,
                dut,
                PuppiMETCorr[mask],
            )
            vals["weight"] = w_all[mask]

            n_entries = int(np.sum(mask))
            for key in self.tree_schema_boosted:
                if key not in vals:
                    vals[key] = np.zeros(n_entries, dtype=np.float64)

            if n_entries > 0:
                vals["weight"] = w_all[mask]
            else:
                vals["weight"] = np.array([], dtype=np.float64)

            self.compat_tree_variables(vals)
            self.add_tree_entry("boosted_SR1", vals)

        '''

        # Opposite-sign e-mu TTCR after the trigger, without a Z-mass cut.
        mask_ttcr_res_prebtag = mask_step_trig_df & (nj_emu >= 3)
        mask_ttcr_res_ge3j = mask_step_trig_df & (nj_emu >= 3)

        mask_ttcr_boost_prebtag = mask_step_trig_df & (njb_emu >= 2)

        # Keep the resolved selection orthogonal to the boosted selection.
        # Veto events with two or more selected double-b-tagged jets.

        mask_res_btag_lt2db_ee = (
            mask_res_btag_ee &
            (nb_b_ee < 2)
        )

        mask_res_btag_lt2db_mumu = (
            mask_res_btag_mumu &
            (nbb_mumu < 2)
        )

        mask_crz_lt2db_ee = (
            mask_crz_ee &
            (nb_b_ee < 2)
        )

        mask_crz_lt2db_mumu = (
            mask_crz_mumu &
            (nbb_mumu < 2)
        )

        mask_ttcr_res_ge3b_lt2db = (
            mask_ttcr_res_ge3b &
            (nbb_emu < 2)
        )

        # Resolved jet and b-tag steps use each channel's fixed-WP SF weight.
        # Count >=3b before applying the veto against the boosted selection.
        for flow_name, mask_ge3j, nbjets_full, mask_ge3b, mask_lt2db, weight in (
            (
                "ee_eventflow_SR_resolved", mask_res_jets_ee, nb_fixed_ee,
                mask_res_btag_ee, mask_res_btag_lt2db_ee, w_resolved_sf_ee,
            ),
            (
                "mumu_eventflow_SR_resolved", mask_res_jets_mumu, nb_fixed_mumu,
                mask_res_btag_mumu, mask_res_btag_lt2db_mumu, w_resolved_sf_mumu,
            ),
            (
                "emu_eventflow_TTCR_resolved", mask_res_jets_emu, nb_fixed_emu,
                mask_ttcr_res_ge3b, mask_ttcr_res_ge3b_lt2db, w_resolved_sf_emu,
            ),
        ):
            fill_eventflow_auto(flow_name, ">=3jets", mask_ge3j, weight)
            fill_eventflow_auto(
                flow_name, ">=3jets & >=2bjets", mask_ge3j & (nbjets_full >= 2), weight,
            )
            fill_eventflow_auto(flow_name, ">=3jets & >=3bjets", mask_ge3b, weight)
            fill_eventflow_auto(
                flow_name, ">=3jets & >=3bjets & <2dbjets", mask_lt2db, weight,
            )
        # Compare Z-control-region distributions with and without the SF.

        fill_hist_auto(
            "emu_TTCR_prebtag_njets_boosted",
            mask_ttcr_boost_prebtag,
            w_all,
            n_jets=njb_emu,
        )

        fill_hist_auto(
            "emu_TTCR_prebtag_nbjets_boosted",
            mask_ttcr_boost_prebtag,
            w_all,
            n_bjets=nbb_emu,
        )

        # Double-b-tag working points for the boosted control region.
        DBTAG_LOOSE = 0.12   # Baseline SR1 and TTCR1 threshold.
        DBTAG_TIGHT = 0.38

        mask_df_base = mask_step_trig_df  & (njb_emu >= 2)

        mask_ttcr1 = mask_step_trig_df & (njb_emu >= 2) & (nbb_emu >= 2)
        lead_dbtag = np.zeros(len(events))
        sub_dbtag = np.zeros(len(events))

        dj_ttcr, dbj_ttcr, _, _, _ = build_boosted_permask(mask_ttcr1)

        idx = np.where(mask_ttcr1)[0]

        if len(idx) > 0:
            bb = dbj_ttcr[:, :2]
            lead_dbtag[idx] = ak.to_numpy(bb[:, 0].btagUParTAK4probbb)
            sub_dbtag[idx] = ak.to_numpy(bb[:, 1].btagUParTAK4probbb)
        # Boosted jet kinematics.

        fill_boosted_dbjet_kinematics(
            "ee",
            "A_SR_3b",
            mask_SR1_boosted_ee,
            dbj_sr1_ee,
            "pt",
        )

        fill_boosted_dbjet_kinematics(
            "ee",
            "A_SR_3b",
            mask_SR1_boosted_ee,
            dbj_sr1_ee,
            "btag",
        )

        fill_boosted_dbjet_kinematics(
            "mumu",
            "A_SR_3b",
            mask_SR1_boosted_mumu,
            dbj_sr1_mm,
            "pt",
        )

        fill_boosted_dbjet_kinematics(
            "mumu",
            "A_SR_3b",
            mask_SR1_boosted_mumu,
            dbj_sr1_mm,
            "btag",
        )

        fill_boosted_dbjet_kinematics(
           "emu",
            "A_CR_3b",
           mask_ttcr1,
           dbj_ttcr,
           "pt",
        )

        fill_boosted_dbjet_kinematics(
            "emu",
            "A_CR_3b",
            mask_ttcr1,
            dbj_ttcr,
            "btag",
        )
        # Resolved TTCR: at least three jets and three fixed-WP b jets.

        fill_eventflow_auto( "emu_eventflow_TTCR1_boosted", '>=2jets & >=2dbjets',
               mask_ttcr1, w_all)

        REGIONS = {}

        REGIONS[("ee",   "A_SR_3b", "resolved")] = mask_res_btag_lt2db_ee
        REGIONS[("mumu", "A_SR_3b", "resolved")] = mask_res_btag_lt2db_mumu

        REGIONS[("ee",   "A_CR_3b", "resolved")] = mask_crz_lt2db_ee
        REGIONS[("mumu", "A_CR_3b", "resolved")] = mask_crz_lt2db_mumu

        REGIONS[("emu",  "A_CR_3b", "resolved")] = mask_ttcr_res_ge3b_lt2db

        REGIONS[("ee",   "A_SR_3b", "boosted")] = mask_SR1_boosted_ee
        REGIONS[("mumu", "A_SR_3b", "boosted")] = mask_SR1_boosted_mumu
        REGIONS[("emu",  "A_CR_3b", "boosted")] = mask_ttcr1

        def get_output_for_flavor(flavor):
            if flavor == "ttBB":
                return output_ttBB
            if flavor == "ttCC":
                return output_ttCC
            if flavor == "ttLF":
                return output_ttLF
            return output

        def fill_bdt_to_dict(out_dict, channel, region, regime, vals, weight):
            if not self.run_eval:
                return

            bdt_eval = self.bdt_eval_resolved if regime == "resolved" else self.bdt_eval_boosted

            inputs = {
                v: np.asarray(vals[v], dtype=np.float64)
                for v in bdt_eval.var_list
            }

            score = np.ravel(bdt_eval.eval(inputs))

            out_dict[f"{channel}_{region}_bdt_{regime}"].fill(
                bdt=score,
                weight=weight,
            )

            for i, cut in enumerate(self.optim_Cuts1_bdt):
                sel = score > cut
                if not np.any(sel):
                    continue

                out_dict[f"{channel}_{region}_bdt_shapes_{regime}"].fill(
                    cut_index=np.full(np.sum(sel), i, dtype=np.int64),
                    bdt=score[sel],
                    weight=weight[sel],
                )

        def fill_region_to_dict(
            out_dict,
            channel,
            region,
            regime,
            mask,
            weight_nosf_full,
            weight_sf_full,
        ):
            if not np.any(mask):
                return

            if regime == "resolved":

                sj, sbj, sut, _, _ = build_resolved_permask(mask)

                vals = self.build_vars_resolved(
                    leptons[mask],
                    sj,
                    sbj,
                    sut,
                    PuppiMETCorr[mask],
                )

            elif regime == "boosted":
                dj, dbj, dut, _, _ = build_boosted_permask(mask)

                vals = self.build_vars_boosted(
                    leptons[mask],
                    dj,
                    dbj,
                    dut,
                    PuppiMETCorr[mask],
                )

            else:
                raise ValueError(regime)

            w_nosf = np.asarray(weight_nosf_full)[mask]
            w_sf = np.asarray(weight_sf_full)[mask]
            # Final distributions with the region-specific weights.
            # Resolved regions use the fixed-WP SF; boosted regions use w_all.
            for var, arr in vals.items():
                out_dict[
                    f"{channel}_{region}_{var}_{regime}"
                ].fill(
                    **{var: arr},
                    weight=w_sf,
                )

            # Resolved jet and b-jet multiplicities with both weight choices.
            if regime == "resolved":
                for var in ("n_jets", "n_bjets"):
                    arr = vals[var]

                    # Weights without the b-tagging SF.
                    out_dict[
                        f"{channel}_{region}_{var}_nosf_{regime}"
                    ].fill(
                        **{var: arr},
                        weight=w_nosf,
                    )

                    # Weights with the fixed-WP b-tagging SF.
                    out_dict[
                        f"{channel}_{region}_{var}_{FIXED_BTAG_WP}WP_{regime}"
                    ].fill(
                        **{var: arr},
                        weight=w_sf,
                    )

            # Fill the final BDT distributions.
            fill_bdt_to_dict(
                out_dict,
                channel,
                region,
                regime,
                vals,
                w_sf,
            )

        def region_weights(channel, regime):
            if regime == "boosted":
                return w_all, w_all

            if channel == "ee":
                return (
                    w_resolved_nosf,
                    w_resolved_sf_ee,
                )

            if channel == "mumu":
                return (
                    w_resolved_nosf,
                    w_resolved_sf_mumu,
                )

            if channel == "emu":
                return (
                    w_resolved_nosf,
                    w_resolved_sf_emu,
                )

            raise ValueError(
                f"Unknown channel/regime: "
                f"{channel}, {regime}"
            )

        # Fill the final analysis regions.
        for (channel, region, regime), mask in REGIONS.items():
            (
                weight_nosf_full,
                weight_sf_full,
            ) = region_weights(channel, regime)

            if tt_masks is None:
                fill_region_to_dict(
                    out_dict=output,
                    channel=channel,
                    region=region,
                    regime=regime,
                    mask=mask,
                    weight_nosf_full=weight_nosf_full,
                    weight_sf_full=weight_sf_full,
                )

            else:
                for flavour, flavour_mask in tt_masks.items():
                    mask_flavour = (
                        mask
                        & np.asarray(flavour_mask, dtype=bool)
                    )

                    if not np.any(mask_flavour):
                        continue

                    fill_region_to_dict(
                        out_dict=get_output_for_flavor(flavour),
                        channel=channel,
                        region=region,
                        regime=regime,
                        mask=mask_flavour,
                        weight_nosf_full=weight_nosf_full,
                        weight_sf_full=weight_sf_full,
                    )
        # Check the accumulated training trees once per chunk.
        if self.isMVA and self._trees is not None:
            for regime, tree_dict in self._trees.items():
                self._validate_tree_dict(regime, tree_dict)

        if is_ttbar_sample:

            return {
                "ttBB": output_ttBB,
                "ttCC": output_ttCC,
                "ttLF": output_ttLF,
            }

        else:
            # Combine the T histograms.
            output.update(self._histograms_phys)
            output.update(self._histograms_bdt)
            return output

    def postprocess(self, accumulator):
        return accumulator
