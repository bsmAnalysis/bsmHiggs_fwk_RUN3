import awkward as ak
import numpy as np
import uproot
import math
import os
import json
import argparse
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
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

BTAG_WPS = {
    "L": 0.0246,
    "M": 0.1272,
    "T": 0.4648,
}
# Resolved b-tag working point.

FIXED_BTAG_WP = "M"
FIXED_BTAG_THRESHOLD = BTAG_WPS[FIXED_BTAG_WP]

from utils.deltas_array import (
    delta_r,
    clean_by_dr,
    delta_phi,
    delta_eta,
)


def delta_eta_vec(a, b):
    return np.abs(a.eta - b.eta)


def delta_phi_raw(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi

from utils.variables_def import (
    min_dm_bb_bb,
    dr_bb_bb_avg,
    min_dm_doubleb_bb,
    dr_doubleb_bb,
    m_bbj,
    dr_bb_avg,
    higgs_kin
)


def make_vector(obj):
    return ak.zip({
        "pt": obj.pt,
        "eta": obj.eta,
        "phi": obj.phi,
        "mass": obj.mass
    }, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)

MA_GEN_VALUES = (12.0, 15.0, 20.0, 25.0, 30.0)


def reduced_masses(higgs_vec, lead_vec, sub_vec, ma_vals=MA_GEN_VALUES):
    out = {}
    mH = higgs_vec.mass
    m1 = lead_vec.mass
    m2 = sub_vec.mass
    for ma in ma_vals:
        ma1_red = m1 - ma
        ma2_red = m2 - ma
        mH_red = mH - 125 - ma1_red - ma2_red   # mH -125- m1 - m2 + 2*ma
        out[int(ma)] = (mH_red, ma1_red, ma2_red)
    return out

# Fixed binning for the ABCD boundary scan.

ABCD_MASS_EDGES = np.arange(
    0.0,
    1025.0,
    25.0,
    dtype=np.float64,
)

# Use 0.05-rad bins
ABCD_DPHI_EDGES = np.concatenate([
    np.arange(0.0, 3.101, 0.05, dtype=np.float64),
    np.array([np.pi], dtype=np.float64),
])

# Book these labels before filling so empty bins keep the same order in every file.
EVENTFLOW_BINS = {
    "eventflow_SR_resolved": (
        "veto", "trigger", "met", ">=3jets", ">=3jets & >=2bjets",
        ">=3jets & >=3bjets", ">=3jets & >=3bjets & <2dbjets",
        "A_mH", "A_dphi",
    ),
    "eventflow_SR_boosted": (
        "veto", "trigger", "met", ">=2jets", ">=2jets & >=1dbjet",
        ">=2jets & >=2dbjets", "A_mH", "A_dphi",
    ),
    "eventflow_QCDCR_resolved": ("A", "B", "C", "D", "Astar", "Bstar"),
    "eventflow_QCDCR_boosted": ("A", "B", "C", "D", "Astar", "Bstar"),
    "eventflow_TTCR_resolved": (
        "1lep", "trigger", "met", ">=3jets", ">=3jets & >=2bjets",
        ">=3jets & >=3bjets", ">=3jets & >=3bjets & <2dbjets", "mH",
    ),
    "eventflow_TTCR_boosted": (
        "1lep", "trigger", "met", ">=2jets", ">=2jets & >=1dbjet",
        ">=2jets & >=2dbjets", "mH",
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


def make_vector_met(met):
    return ak.zip({
        "pt": met.pt,
        "phi": met.phi,
        "eta": ak.zeros_like(met.pt),
        "mass": ak.zeros_like(met.pt)
    },with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)

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
                20, 0.4, 2.4,
                name=key,
                label="b-tag SF (resolved)",
                underflow=True,
                overflow=True,
            )

        # Explicit binning for analysis variables.
        if k == "ndoubleb":
            return hax.Regular(
                8, 0.0, 8,
                name=key,
                label=key,
                underflow=True,
                overflow=True)

        if k == "nsingleb":
            return hax.Regular(
                8, 0.0, 8,
                name=key,
                label=key,
                underflow=True,
                overflow=True)

        if k == "h_mass":
            return hax.Regular(40, 0.0, 1000.0,
                               name=key, label=key,
                               underflow=True, overflow=True)

        if k == "h_pt":
            return hax.Regular(20, 0.0, 500.0,
                               name=key, label=key,
                               underflow=True, overflow=True)
        if k == "z_pt":
            return hax.Regular(50, 0.0, 500.0,
                               name=key, label=key,
                               underflow=True, overflow=True)

        if k == "ht":
            return hax.Regular(40, 0.0, 1000.0,
                               name=key, label=key,
                               underflow=True, overflow=True)

        if k == "mbbj":
            return hax.Regular(50, 0.0, 1000.0,
                               name=key, label=key,
                               underflow=True, overflow=True)

        if k in {"eta"}:
            return hax.Regular(30, -5.0, 5.0, name=key, label=key,underflow=True, overflow=True )
        if k in {"phi"}:
            return hax.Regular(32, -math.pi, math.pi, name=key, label=key,underflow=True, overflow=True)
        if k.startswith("dphi") or k in {"dphi"}:
            return hax.Regular(32, 0.0, math.pi, name=key, label=key,underflow=True, overflow=True)
        if k.startswith("dr") or k in {"dr","dR"}:
            return hax.Regular(25, 0.0, 5.0, name=key, label=key,underflow=True, overflow=True)
        if k.startswith("deta") or k in {"deta"}:
            return hax.Regular(30, 0.0, 6.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"bdt"}:
            edges = getattr(self._proc, "bdt_edges", None)
            if edges is None:
                return hax.Regular(50, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)
            return hax.Variable(edges, name=key, label=key,underflow=True, overflow=True)
        if k in {"score", "btag_score"}:
            return hax.Regular(25, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)
        if "mass_red" in k:
            return hax.Regular(30, -100.0, 200.0, name=key, label=key,
                       underflow=True, overflow=True)
        if k in {"btag"}:
            return hax.Regular(25, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"n", "n_jets", "n_bjets", "n_untag"}:
            return hax.Regular(10, 0.0, 10., name=key, label=key,underflow=True, overflow=True)
        if k in {"nlighttruth", "nctruth", "nbtruth"}:
            # Use the same multiplicity axis when merging chunks.
            return hax.Regular(20, 0.0, 20.0, name=key, label=key,
                               underflow=True, overflow=True)
        if k in {"npv"}:
            return hax.Regular(100, 0.0, 100., name=key, label=key,underflow=True, overflow=True)
        if k.startswith("btag"):
            return hax.Regular(50, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"gentop_pt", "top_pt", "gen_top_pt", "genTop_pt"}:
            return hax.Regular(100, 0.0, 2000.0, name=key, label="gen top pT [GeV]",underflow=True, overflow=True)
        if k in {"lep_pt", "lep0_pt", "lep1_pt"}:
            return hax.Regular(40, 0.0, 200.0, name=key, label="gen top pT [GeV]",underflow=True, overflow=True)

        if k in {"met", "met_pt", "puppimet_pt"}:
            return hax.Regular(20, 0.0, 500.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"ht"}:
            return hax.Regular(40, 0.0, 1000.0, name=key, label=key,underflow=True, overflow=True)
        if k in { "pt_h","pt_b1","pt_b2","z_pt","ll_pt","pt_ll","pt_z","pt_pretrig","pt_posttrig"}:
            return hax.Regular(20, 0.0, 500.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"h_m_red","m_h_red"}:
            return hax.Regular(30, -100., 200.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"a1_m_red","m_a1_red","a2_m_red","m_a2_red"}:
            return hax.Regular(4, -20., 20.0, name=key, label=key,underflow=True, overflow=True)

        if k in {"h_pt","pt_h"}:
            return hax.Regular(20, 0.0, 500.0, name=key, label=key,underflow=True, overflow=True)
        if k in { "m_h","h_m","mbbj" ,"m_bbj"}:
            return hax.Regular(40, 0.0, 1000.0, name=key, label=key,underflow=True, overflow=True)
        if k in { "z_m", "ll_m","m_z", "m_ll"}:
            return hax.Regular(20, 70, 110.0, name=key, label=key,underflow=True, overflow=True)
        if "ratio" in k:
            return hax.Regular(25, -0.0, 5.0, name=key, label=key,underflow=True, overflow=True)
        if "dm" in k:
            return hax.Regular(16, 0.0, 160.0, name=key, label=key,underflow=True, overflow=True)
        # B-tagging scores.
        if "btag" in k and not k.endswith("_pt"):
            return hax.Regular(50, 0.0, 1.0, name=key, label=key,
                               underflow=True, overflow=True)

        # Transverse-momentum variables.
        if (
                k.endswith("_pt")
                or k.endswith("_pt_pt")
                or k in {
                    "h_pt", "ht", "puppimet_pt", "mbbj",
                "bj_max_btag_pt", "bj_2nd_btag_pt",
                }
        ):
            hi = 1000.0
            if k == "puppimet_pt":
                hi = 1000.0
            if k == "ht":
                hi = 1000.0
            return hax.Regular(50, 0.0, hi, name=key, label=key,
                       underflow=True, overflow=True)

        # Pseudorapidity.
        if k.endswith("_eta") or k == "eta":
            return hax.Regular(25, -5.0, 5.0, name=key, label=key,
                       underflow=True, overflow=True)

        # Azimuth.
        if k.endswith("_phi") or k == "phi":
            return hax.Regular(32, -math.pi, math.pi, name=key, label=key,
                               underflow=True, overflow=True)

        # Mass variables.
        if k.endswith("_mass") or k in {"h_mass", "m_h", "h_m"}:
            return hax.Regular(50, 0.0, 1000.0, name=key, label=key,
                               underflow=True, overflow=True)

        # Azimuthal separations.
        if k.startswith("dphi") or "_dphi" in k:
            return hax.Regular(32, 0.0, math.pi, name=key, label=key,
                               underflow=True, overflow=True)

        # Angular distances.
        if k.startswith("dr") or "_dr" in k:
            return hax.Regular(20, 0.0, 5.0, name=key, label=key,
                               underflow=True, overflow=True)

        # Mass differences.
        if k.startswith("dm") or "_dm" in k:
            return hax.Regular(50, 0.0, 300.0, name=key, label=key,
                               underflow=True, overflow=True)
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
                axes.append(hax.StrCategory([], name="cut", label="cut", growth=True))
                continue
            if key == "cut_index":
                ncuts = len(getattr(self._proc, "optim_Cuts1_bdt", []))
                axes.append(
                    hax.Regular(
                        ncuts, -0.5, ncuts - 0.5,
                        name="cut_index",
                        label="BDT cut index",
                        underflow=False,
                        overflow=False,
                    )
                )
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


def book_abcd_scan_hist(out_dict, name):
    
   #Use 25-GeV mass bins up to 1000 GeV and 0.05-rad angular bins,with the last angular bin ending at pi.
    if name not in out_dict or isinstance(out_dict.get(name), _AutoHist):

        out_dict[name] = Hist(
            hax.Variable(
                ABCD_MASS_EDGES,
                name="H_mass",
                label="mH [GeV]",
                underflow=True,
                overflow=True,
            ),
            hax.Variable(
                ABCD_DPHI_EDGES,
                name="dphi",
                label="min dphi(b,MET)",
                underflow=True,
                overflow=True,
            ),
            storage=storage.Weight(),
        )

    return out_dict[name]


class TOTAL_Processor(processor.ProcessorABC):

    def __init__(
        self,
        xsec=1.0,
        nevts=1.0,
        isMC=True,
        dataset_name=None,
        isMVA=False,
        runQCD=False,
        run_eval=False,
    ):
        self.xsec = xsec
        self.nevts = nevts
        self.isMC = isMC
        self.isMVA = isMVA
        self.runQCD = runQCD
        self.run_eval = run_eval
        self.dataset_name = dataset_name

        self._histograms_phys = AutoHistDict(parent_proc=self)
        self._histograms_bdt = AutoHistDict(parent_proc=self)

        if isMVA:
            if runQCD:
                tree_names = [
                    "boosted_B", "boosted_C", "boosted_D",
                "resolved_B", "resolved_C", "resolved_D",
                ]
            else:
                tree_names = [
                    "boosted_A",
                    "resolved_A",
                ]

            self._trees = {name: defaultdict(list) for name in tree_names}
        else:
            self._trees = None

        BDT_BOOSTED_FEATURES = [
            "HT",
            "H_eta",
            "H_mass",
            "H_pt",
            "bj_2nd_pt_pt",
            "dm_bb_bb_min",
            "dphi_bb_MET_min",
            "dr_bb_bb_ave",
            "n_jets",
            "puppimet_pt",
        ]
        BDT_RESOLVED_FEATURES = [
            "HT",
            "H_mass",
            "H_pt",
            "bj_2nd_pt_pt",
            "dm_bb_bb_min",
            "dphi_b_MET_min",
            "dr_bb_ave",
            "mbbj",
            "n_jets",
            "puppimet_pt",
        ]
        self.tree_schema_resolved = [
            "H_mass",
            "H_pt",
            "H_eta",
            "H_phi",
            "HT",
            "puppimet_pt",
            "dphi_b_MET_min",
            "dphi_J_MET_min",
            "dphi_H_MET",
            "dr_bb_ave",
            "dm_bb_bb_min",
            "mbbj",
            "n_jets",
            "n_bjets",
            "bj_max_pt_pt",
            "bj_max_pt_eta",
            "bj_max_pt_phi",

            "bj_2nd_pt_pt",

            "weight",
        ]

        self.tree_schema_boosted = [
            "H_mass",
            "H_pt",
            "H_eta",
            "H_phi",
            "HT",
            "puppimet_pt",
            "dphi_H_MET",
            "dphi_bb_MET_min",
            "dr_bb_bb_ave",
            "dm_bb_bb_min",
            "n_jets",
            "n_bjets",
            "bj_max_pt_pt",
            "bj_max_pt_eta",
            "bj_max_pt_phi",
            "bj_2nd_pt_pt",
            "weight",
        ]
        # Load the boosted and resolved BDT models.
        self.bdt_eval_boosted = XGBHelper(
            os.path.join("xgb_model", "bdt_model_boosted_A.json"),
            BDT_BOOSTED_FEATURES
        )

        self.bdt_eval_resolved = XGBHelper(
            os.path.join("xgb_model", "bdt_model_resolved_A.json"),
            BDT_RESOLVED_FEATURES
        )

        self.bdt_edges = np.linspace(0.0, 1.0, 51)
        self.optim_Cuts1_bdt = self.bdt_edges[:-1].tolist()

        self.systematics_labels = [""]

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

        # Fixed-WP scale factors and efficiency maps.

        self._btag_sf = None
        self._btag_eff = None

        if self.isMC:
            self.btag_json = os.path.join(
                CORR_DIR,
                "btag_merged_2024_final.json.gz",
            )

            sample_lower = (self.dataset_name or "").lower()

            # Select the efficiency map for this sample.
            if sample_lower.startswith("ttto2l2nu"):
                eff_name = "btag_eff_TTto2L2Nu_2024.json.gz"

            elif (
                sample_lower.startswith("tttolnu2q")
                or sample_lower.startswith("tttolnu")
            ):
                eff_name = "btag_eff_TTtoLNu2Q_2024.json.gz"

            elif sample_lower.startswith("ttto4q"):
                eff_name = "btag_eff_TTto4Q_2024.json.gz"

            elif (
                    sample_lower.startswith("zto2q-4jets_bin-ht-")
                    or sample_lower.startswith("zto2q-4jets-bin-ht-")
                ):
                eff_name = "btag_eff_Zto2Q_2024.json.gz"
            elif (
                sample_lower == "ttg-1jets_bin-ptg-200"
                or sample_lower == "ttg-1jets-bin-ptg-200"
            ):
                eff_name = "btag_eff_TTG200_2024.json.gz"
            elif (
                sample_lower.startswith("qcdb-4jets_bin-ht-")
                or sample_lower.startswith("qcdb-4jets-bin-ht-")
            ):
                eff_name = "btag_eff_QCDB_2024.json.gz"

            else:
                eff_name = "btag_eff_0lep_2024_nontt.json.gz"

            self.btag_eff_json = os.path.join(CORR_DIR, eff_name)
            # Check the efficiency-map JSON.
            for label, path in (
                ("b-tag SF", self.btag_json),
                ("b-tag efficiency", self.btag_eff_json),

            ):
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Missing {label} JSON for dataset "
                        f"{self.dataset_name!r}: {path}"
                    )

            self._btag_sf = correctionlib.CorrectionSet.from_file(
                self.btag_json
            )["UParTAK4_merged"]

            self._btag_eff = correctionlib.CorrectionSet.from_file(
                self.btag_eff_json
            )["btag_eff"]

    @staticmethod
    def _strip_year(name):
        return name[:-5] if name.endswith("_2024") else name

    def _is_ttbar(self):
        return (self.dataset_name or "").lower().startswith("ttto")

    def _is_qcd_sample(self):
        sample = (self.dataset_name or "").lower()
        return sample.startswith("qcd") or "qcd" in sample

    def btag_eff_process_key(self, tt_flavor=None):
        """Choose the process key used by the loaded efficiency map."""

        sample = self.dataset_name

        if sample is None:
            raise RuntimeError(
                "dataset_name is required for efficiency lookup"
            )

        if self._is_ttbar():
            if tt_flavor not in {"ttBB", "ttCC", "ttLF"}:
                raise ValueError(
                    f"TT sample requires tt flavor, got {tt_flavor!r}"
                )
            return tt_flavor

        stripped = self._strip_year(sample)

        aliases = {
            "TTG-1Jets_Bin-PTG-100":
                "TTG-1Jets-Bin-PTG-100",

            "TTG-1Jets_Bin-PTG-200":
                "TTG-1Jets-Bin-PTG-200",

            "TTG-1Jets-Bin-PTG-200":
                "TTG-1Jets-Bin-PTG-200",
        }

        if stripped.startswith("Zto2Q-4Jets_Bin-HT-"):
            primary = stripped.replace(
                "Zto2Q-4Jets_Bin-HT-",
                "Zto2Q-4Jets-Bin-HT-",
                1,
            )

        elif stripped.startswith("QCDB-4Jets_Bin-HT-"):
            primary = stripped.replace(
                "QCDB-4Jets_Bin-HT-",
                "QCDB-4Jets-Bin-HT-",
                1,
            )

        else:
            primary = aliases.get(stripped, stripped)
        # Candidate process keys in the efficiency map.
        candidates = list(dict.fromkeys([
            primary,
            aliases.get(stripped),
            sample,
            stripped,
            f"{stripped}_2024",
        ]))

        candidates = [
            candidate for candidate in candidates
            if candidate is not None
        ]

        failures = []

        for candidate in candidates:
            try:
                value = self._btag_eff.evaluate(
                    candidate,FIXED_BTAG_WP, 2, 0, 50.0, 0.5,)

                if np.isfinite(value):
                    print(f"[BTag efficiency] dataset={sample}, "f"process={candidate}")
                    return candidate

            except Exception as exc:
                failures.append(
                    f"{candidate}: {type(exc).__name__}: {exc}"
                )

        raise KeyError(
            f"No efficiency map for dataset={sample!r}; "
            f"attempted={candidates}; failures={failures}"
        )

    @staticmethod
    def selected_truth_flavour_counts(jets):
        hadron_flavour = np.abs(jets.hadronFlavour)
        nlight = ak.to_numpy(ak.sum(hadron_flavour == 0, axis=1)).astype(np.float64)
        nc = ak.to_numpy(ak.sum(hadron_flavour == 4, axis=1)).astype(np.float64)
        nb = ak.to_numpy(ak.sum(hadron_flavour == 5, axis=1)).astype(np.float64)
        njets = ak.to_numpy(ak.num(jets)).astype(np.int64)
        counted = nlight.astype(np.int64) + nc.astype(np.int64) + nb.astype(np.int64)
        if not np.array_equal(counted, njets):
            bad = np.where(counted != njets)[0]
            raise RuntimeError(
                "Selected jets were not fully classified as hadronFlavour "
                f"0/4/5; first bad indices={bad[:10].tolist()}"
            )
        return nlight, nc, nb

    def eval_btag_sf_fixedWP_resolved(
        self,
        jets,
        eff_process,
        wp_name=FIXED_BTAG_WP,
        wp_value=FIXED_BTAG_THRESHOLD,
        syst='central',
    ):
        #Compute the fixed-WP event b-tagging weight. Jets at or above the threshold are tagged; jets below it are untagged.
      
        if self._btag_sf is None or self._btag_eff is None:
            return ak.ones_like(
                ak.num(jets),
                dtype=np.float64,
            )

        counts = ak.num(jets)

        # Truth-b multiplicity used by the efficiency maps.
        nb_event = ak.sum(
            np.abs(jets.hadronFlavour) == 5,
            axis=1,
        )

        nb_event = ak.where(
            nb_event >= 4,
            4,
            nb_event,
        )

        nb_perjet = ak.broadcast_arrays(
            nb_event,
            jets.pt,
        )[0]

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
        nb_bin = ak.to_numpy(
            ak.flatten(nb_perjet)
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
            & np.isin(nb_bin, [0, 1, 2, 3, 4])
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
            nb_bin[valid],
            pt_eval[valid],
            eta_eval[valid],
        )

        bad_inputs = (
            valid
            & (
                ~np.isfinite(sf)
                | ~np.isfinite(eff)
                | (sf <= 0.0)
                | (eff < 0.0)
                | (eff > 1.0)
            )
        )

        if np.any(bad_inputs):
            raise RuntimeError(
                f"Invalid fixed-WP SF/efficiency values for "
                f"process={eff_process}, WP={wp_name}"
            )

        tagged = valid & (score >= wp_value)
        untagged = valid & (score < wp_value)

        jet_weight = np.ones(
            len(pt),
            dtype=np.float64,
        )

        # Tagged-jet contribution.
        jet_weight[tagged] = sf[tagged]

        # Untagged-jet contribution.
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
            numerator[good_untagged]/ denominator[good_untagged]
        )

        if np.any(bad_untagged):
            jet_weight[bad_untagged] = 1.0

            print(
                f"[BTV WARNING] Fixed-WP {wp_name} untagged "
                "factors set to unity: "
                f"{np.count_nonzero(bad_untagged)}/"
                f"{np.count_nonzero(untagged)}"
            )

        if (
            np.any(~np.isfinite(jet_weight))
            or np.any(jet_weight <= 0.0)
        ):
            raise RuntimeError(
                f"Invalid fixed-WP {wp_name} per-jet factors"
            )

        return ak.prod(
            ak.unflatten(
                jet_weight,
                counts,
            ),
            axis=1,
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
       #Convert tree branches to float64 arrays for consistent ROOT merging
        for key in tree_dict:
            tree_dict[key] = np.asarray(tree_dict[key], dtype=np.float64)

    def _validate_tree_dict(self, regime, tree_dict):
        #Check branch lengths and finite values; warn about mostly zero branches

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

    def fill_mva_tree_0lep(
        self,
        tree_name,
        mask,
        regime,
        met_all,
        w_all,
        build_boosted_0lep_permask,
        build_resolved_0lep_permask,
    ):
        n_entries = int(np.sum(mask))

        if regime == "boosted":
            schema = self.tree_schema_boosted

            if n_entries > 0:
                dj, dbj, dut, _, _ = build_boosted_0lep_permask(mask)

                vals = self.build_vars_boosted(
                    dj,
                    dbj,
                    dut,
                    met_all[mask],
                )
            else:
                vals = {}

        elif regime == "resolved":
            schema = self.tree_schema_resolved

            if n_entries > 0:
                sj, sbj, sut, _, _ = build_resolved_0lep_permask(mask)

                vals = self.build_vars_resolved(
                    sj,
                    sbj,
                    sut,
                    met_all[mask],
                )
            else:
                vals = {}

        else:
            raise ValueError(f"Unknown regime: {regime}")

        if n_entries > 0:
            vals["weight"] = np.asarray(w_all[mask], dtype=np.float64)
        else:
            vals["weight"] = np.array([], dtype=np.float64)

        for key in schema:
            if key not in vals:
                vals[key] = np.zeros(n_entries, dtype=np.float64)

        for key, arr in vals.items():
            if len(arr) != n_entries:
                raise ValueError(
                    f"[TREE ERROR] {tree_name}: branch {key} has len={len(arr)}, expected {n_entries}"
                )

        self.compat_tree_variables(vals)
        self.add_tree_entry(tree_name, vals)

    def build_vars_resolved(
        self,
        single_jets_sel,
        single_bjets_sel,
        single_untag_sel,
        PuppiMET_sel,
    ):
        sj = single_jets_sel
        sbj = single_bjets_sel
        sut = single_untag_sel

        bjets_pt_sort = sbj[
            ak.argsort(sbj.pt, axis=-1, ascending=False)
        ]

        jets_pt_sort = sj[
            ak.argsort(sj.pt, axis=-1, ascending=False)
        ]

        v_sj = make_vector(sj)
        v_sbj = make_vector(sbj)

        mH, ptH, phiH, etaH = higgs_kin(v_sbj, v_sj)

        H = ak.zip(
            {"pt": ptH, "eta": etaH, "phi": phiH, "mass": mH},
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        metv = make_vector_met(PuppiMET_sel)

        dphi_b_met_min = ak.to_numpy(
            ak.min(
                np.abs(delta_phi_raw(sbj.phi, PuppiMET_sel.phi)),
                axis=1,
                initial=999,
                mask_identity=False,
            )
        )

        dphi_j_met_min = ak.to_numpy(
            ak.min(
                np.abs(delta_phi_raw(sj.phi, PuppiMET_sel.phi)),
                axis=1,
                initial=999,
                mask_identity=False,
            )
        )

        return {
            "H_mass": ak.to_numpy(mH),
            "H_pt":   ak.to_numpy(ptH),
            "H_eta":  ak.to_numpy(etaH),
            "H_phi":  ak.to_numpy(phiH),

            "HT": ak.to_numpy(ak.sum(sj.pt, axis=1)),
            "puppimet_pt": ak.to_numpy(PuppiMET_sel.pt),
            "puppimet_phi": ak.to_numpy(PuppiMET_sel.phi),

            "dphi_b_MET_min": dphi_b_met_min,
            "dphi_J_MET_min": dphi_j_met_min,
            "dphi_H_MET": ak.to_numpy(np.abs(H.delta_phi(metv))),

            "dr_bb_ave": ak.to_numpy(dr_bb_avg(v_sbj)),
            "dm_bb_bb_min": ak.to_numpy(min_dm_bb_bb(v_sbj, all_jets=v_sj)),
            "mbbj": ak.to_numpy(m_bbj(v_sbj, all_jets=v_sj)),

            "n_jets":  ak.to_numpy(ak.num(sj)),
            "n_bjets": ak.to_numpy(ak.num(sbj)),

            "bj_max_pt_pt":  ak.to_numpy(bjets_pt_sort[:, 0].pt),
            "bj_max_pt_eta": ak.to_numpy(bjets_pt_sort[:, 0].eta),
            "bj_max_pt_phi": ak.to_numpy(bjets_pt_sort[:, 0].phi),

            "bj_2nd_pt_pt": ak.to_numpy(bjets_pt_sort[:, 1].pt),

        }

    def build_vars_boosted(self, double_jets_sel, double_bjets_sel, double_untag_sel, PuppiMET_sel):
        dj = double_jets_sel
        dbj = double_bjets_sel
        dut = double_untag_sel

        bb = dbj[:, :2]
        b1, b2 = bb[:, 0], bb[:, 1]

        b1v = make_vector(b1)
        b2v = make_vector(b2)
        H = b1v + b2v
        red_masses = reduced_masses(H, b1v, b2v)
        metv = make_vector_met(PuppiMET_sel)

        dphi_b1_met = np.abs(b1v.delta_phi(metv))
        dphi_b2_met = np.abs(b2v.delta_phi(metv))
        dphi_bb_met_min = np.minimum(
            ak.to_numpy(dphi_b1_met),
            ak.to_numpy(dphi_b2_met),
        )

        dm = np.abs(b1.mass - b2.mass)

        dbjets_pt_sort = dbj[
            ak.argsort(dbj.pt, axis=-1, ascending=False)
        ]

        jets_pt_sort = dj[
            ak.argsort(dj.pt, axis=-1, ascending=False)
        ]

        out = {
            "H_mass": ak.to_numpy(H.mass),
            "H_pt":   ak.to_numpy(H.pt),
            "H_eta":  ak.to_numpy(H.eta),
            "H_phi":  ak.to_numpy(H.phi),

            "HT": ak.to_numpy(ak.sum(dj.pt, axis=1)),
            "puppimet_pt": ak.to_numpy(PuppiMET_sel.pt),
            "puppimet_phi": ak.to_numpy(PuppiMET_sel.phi),

            "dphi_H_MET": ak.to_numpy(np.abs(H.delta_phi(metv))),

            "dphi_bb_MET_min": dphi_bb_met_min,

            "dr_bb_bb_ave": ak.to_numpy(b1v.delta_r(b2v)),
            "dm_bb_bb_min": ak.to_numpy(dm),

            "n_jets":  ak.to_numpy(ak.num(dj)),
            "n_bjets": ak.to_numpy(ak.num(dbj)),

            "bj_max_pt_pt":  ak.to_numpy(dbjets_pt_sort[:, 0].pt),
            "bj_max_pt_eta": ak.to_numpy(dbjets_pt_sort[:, 0].eta),
            "bj_max_pt_phi": ak.to_numpy(dbjets_pt_sort[:, 0].phi),

            "bj_2nd_pt_pt":  ak.to_numpy(dbjets_pt_sort[:, 1].pt),

        }

        return out

    def process(self, events):
        try:
            events = events.eager_compute_divisions()
        except Exception:
            pass

        try:
            n = len(events)
        except TypeError:
            events = events.compute()
            n = len(events)

        weights = Weights(n, storeIndividual=True)
        # Event weights and flavour-split outputs.
        # Inclusive output for samples other than ttbar.

        output = AutoHistDict(parent_proc=self)

        output_ttBB = None
        output_ttCC = None
        output_ttLF = None

        is_ttbar_sample = (
            self.isMC and (not self.isMVA or self.run_eval)
            and self.dataset_name is not None
            and self.dataset_name.startswith("TTto")
            and "genTtbarId" in events.fields
        )

        if is_ttbar_sample:
            output_ttBB = AutoHistDict(parent_proc=self)
            output_ttCC = AutoHistDict(parent_proc=self)
            output_ttLF = AutoHistDict(parent_proc=self)

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
                pu_w = np.clip(pu_w, 0.0, 10.0)
                # Apply pileup reweighting.
                weights.add("pileup", pu_w)
        else:
            weights.add("ones", np.ones(n, dtype="float64"))
            w_bef_pu = weights.weight()

        w_all = weights.weight()
        # Split ttbar events by heavy-flavour content.
        tt_masks = None

        if is_ttbar_sample:
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

        def book_category_bins(name, labels):
            #Book the same axis in all outputs, including ttbar flavours
            labels = list(labels)

            target_outputs = [output]

            if tt_masks is not None:
                target_outputs.extend([
                    output_ttBB,
                    output_ttCC,
                    output_ttLF,
                ])

            for out_dict in target_outputs:
                if out_dict is None:
                    continue

                for label in labels:
                    out_dict[name].fill(
                        cut=np.array([label], dtype=object),
                        weight=np.array([0.0], dtype=np.float64),
                    )

        for flow_name, flow_labels in EVENTFLOW_BINS.items():
            book_category_bins(flow_name, flow_labels)

        def get_output_for_flavor(flavor):
            if flavor == "ttBB":
                return output_ttBB
            elif flavor == "ttCC":
                return output_ttCC
            elif flavor == "ttLF":
                return output_ttLF
            return output

        def fill_abcd_scan_auto(name, mask, weight, H_mass, dphi):
            #Fill the ABCD scan plane with the ttbar flavour splitting

            mask = np.asarray(mask, dtype=bool)

            if not np.any(mask):
                return

            def fill_one(out_dict, local_mask):

                if not np.any(local_mask):
                    return

                h2 = book_abcd_scan_hist(
                    out_dict,
                    name,
                )

                h2.fill(
                    H_mass=np.asarray(H_mass[local_mask], dtype=np.float64),
                    dphi=np.asarray(dphi[local_mask], dtype=np.float64),
                    weight=np.asarray(weight[local_mask], dtype=np.float64),
                )

            if tt_masks is None:

                fill_one(
                    output,
                    mask,
                )

            else:

                for flavor, fmask in tt_masks.items():

                    mask_flav = mask & fmask

                    if not np.any(mask_flav):
                        continue

                    fill_one(
                        get_output_for_flavor(flavor),
                        mask_flav,
                    )

        def compute_btag_sf_fixed_full_auto(jets_selected, event_mask):
            sf_full = np.ones(
                len(events),
                dtype=np.float64,
            )

            if not self.isMC or not np.any(event_mask):
                return sf_full

            idx = np.where(event_mask)[0]

            if tt_masks is None:
                eff_process = self.btag_eff_process_key()

                sf_small = self.eval_btag_sf_fixedWP_resolved(
                    jets_selected,
                    eff_process=eff_process,
                )

                sf_full[idx] = ak.to_numpy(sf_small)
                return sf_full

            # TT samples: evaluate ttLF, ttCC and ttBB separately.
            for flavour in ("ttLF", "ttCC", "ttBB"):
                flavour_mask_full = (
                    event_mask
                    & np.asarray(tt_masks[flavour], dtype=bool)
                )

                if not np.any(flavour_mask_full):
                    continue

                local_mask = np.asarray(
                    tt_masks[flavour],
                    dtype=bool,
                )[event_mask]

                jets_flavour = jets_selected[local_mask]

                sf_small = self.eval_btag_sf_fixedWP_resolved(
                    jets_flavour,
                    eff_process=flavour,
                )

                flavour_idx = np.where(flavour_mask_full)[0]
                sf_full[flavour_idx] = ak.to_numpy(sf_small)

            return sf_full

        def sr_name(region, var, regime):
            return f"veto_{region}_SR_3b_{var}_{regime}"

        def cr_name(region, var, regime):
            return f"lep1_{region}_CR_3b_{var}_{regime}"

        def book_bdt_shapes(out_dict, name):
            nCuts = len(self.optim_Cuts1_bdt)

            if name not in out_dict or isinstance(out_dict.get(name), _AutoHist):
                out_dict[name] = (
                    hist.Hist.new
                    .IntCategory(range(nCuts), name="cut_index")
                    .Variable(self.bdt_edges, name="bdt")
                    .Weight()
                )
            return out_dict[name]

        def book_cut_shape(out_dict, name, var_name, axis_kind, axis_args):
            nCuts = len(self.optim_Cuts1_bdt)

            if name not in out_dict or isinstance(out_dict.get(name), _AutoHist):

                builder = hist.Hist.new.IntCategory(
                    range(nCuts),
                    name="cut_index",
                )

                if axis_kind == "variable":
                    builder = builder.Variable(
                        axis_args,
                        name=var_name,
                    )
                elif axis_kind == "regular":
                    nbins, lo, hi = axis_args
                    builder = builder.Reg(
                        nbins,
                        lo,
                        hi,
                        name=var_name,
                    )
                else:
                    raise ValueError(f"Unknown axis kind: {axis_kind}")

                out_dict[name] = builder.Weight()

            return out_dict[name]

        def eval_and_fill_bdt_0lep(channel, region, regime, mask, vals_full, weight):

            if not self.run_eval:
                return

            if not np.any(mask):
                return

            bdt_eval = self.bdt_eval_boosted if regime == "boosted" else self.bdt_eval_resolved

            def fill_one(out_dict, local_mask):

                if not np.any(local_mask):
                    return

                inputs = {
                    v: np.asarray(vals_full[v][local_mask], dtype=np.float64)
                    for v in bdt_eval.var_list
                }

                score = np.ravel(bdt_eval.eval(inputs))
                w = np.asarray(weight[local_mask], dtype=np.float64)

                if len(score) == 0:
                    return

                # Inclusive BDT score.

                prefix = channel if region == "" else f"{channel}_{region}"

                out_dict[
                    f"{prefix}_bdt_{regime}"
                ].fill(bdt=score,
                    weight=w,
                )
                # Distributions at each BDT threshold.
                h2 = book_bdt_shapes(
                    out_dict,
                    f"{prefix}_bdt_shapes_{regime}",
                )

                for i, cut in enumerate(self.optim_Cuts1_bdt):
                    sel = score > cut
                    if not np.any(sel):
                        continue

                    h2.fill(
                        cut_index=np.full(np.sum(sel), i, dtype=np.int64),
                        bdt=score[sel],
                        weight=w[sel],
                    )
                shape_specs = {

                    "H_mass": {
                        "hist_suffix": "H_mass_shapes",
                        "var_name": "H_mass",
                        "values": np.asarray(vals_full["H_mass"][local_mask], dtype=np.float64),
                        "axis_kind": "regular",
                        "axis_args": (40, 0.0, 1000.0),
                    },
                    "H_pt": {
                        "hist_suffix": "H_pt_shapes",
                        "var_name": "H_pt",
                        "values": np.asarray(vals_full["H_pt"][local_mask], dtype=np.float64),
                        "axis_kind": "regular",
                        "axis_args": (20, 0.0, 500.0),
                    },
                        "HT": {
                        "hist_suffix": "HT_shapes",
                        "var_name": "HT",
                        "values": np.asarray(vals_full["HT"][local_mask], dtype=np.float64),
                        "axis_kind": "regular",
                        "axis_args": (40, 0.0, 1000.0),
                    },
                    "puppimet_pt": {
                        "hist_suffix": "met_shapes",
                        "var_name": "puppimet_pt",
                        "values": np.asarray(vals_full["puppimet_pt"][local_mask], dtype=np.float64),
                        "axis_kind": "regular",
                        "axis_args": (20, 0.0, 500.0),
                    },
                    "n_jets": {
                        "hist_suffix": "n_jets_shapes",
                        "var_name": "n_jets",
                        "values": np.asarray(vals_full["n_jets"][local_mask], dtype=np.float64),
                        "axis_kind": "regular",
                        "axis_args": (10, 0.0, 10.0),
                    },
                }

                if regime == "boosted":

                    shape_specs["dr_bb_bb_ave"] = {
                        "hist_suffix": "dr_bb_shapes",
                        "var_name": "dr_bb_bb_ave",
                        "values": np.asarray(vals_full["dr_bb_bb_ave"][local_mask], dtype=np.float64),
                        "axis_kind": "regular",
                        "axis_args": (20, 0.0, 5),
                    }

                if regime == "resolved":

                    shape_specs["dr_bb_ave"] = {
                        "hist_suffix": "dr_bb_shapes",
                        "var_name": "dr_bb_ave",
                        "values": np.asarray(vals_full["dr_bb_ave"][local_mask], dtype=np.float64),
                        "axis_kind": "regular",
                        "axis_args": (20, 0.0, 5),
                    }
                for _, spec in shape_specs.items():
                    h2 = book_cut_shape(
                        out_dict,
                        f"{prefix}_{spec['hist_suffix']}_{regime}",
                        spec["var_name"],
                        spec["axis_kind"],
                        spec["axis_args"],
                    )

                    values = spec["values"]

                    for i, cut in enumerate(self.optim_Cuts1_bdt):
                        sel = score > cut
                        if not np.any(sel):
                            continue

                        h2.fill(
                            cut_index=np.full(np.sum(sel), i, dtype=np.int64),
                            **{spec["var_name"]: values[sel]},
                            weight=w[sel],
                        )
            if tt_masks is None:
                fill_one(output, mask)
            else:
                for flavor, fmask in tt_masks.items():
                    mask_flav = mask & fmask
                    fill_one(get_output_for_flavor(flavor), mask_flav)
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
        muons = events.Muon[(events.Muon.pt > 10)   & (np.abs(events.Muon.eta) < 2.4)  & (events.Muon.looseId) & (events.Muon.pfRelIso04_all < 0.25)]
        electrons = events.Electron[(events.Electron.pt > 10) & (np.abs(events.Electron.eta) < 2.5) &  (events.Electron.mvaIso_WP90 )]
        muons = ak.with_field(muons, "mu", "lepton_type")
        electrons = ak.with_field(electrons, "e", "lepton_type")
        leptons = ak.concatenate([muons, electrons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
        n_leptons = ak.num(leptons)
        jets_all = events.Jet
        met_all = events.PuppiMET
        # Jet selection.
        goodJet = (
            (jets_all.pt > 20)
            & (np.abs(jets_all.eta) < 2.4)

        )

        single_jets = jets_all[goodJet]
        double_jets = jets_all[goodJet]

        single_jets = clean_by_dr(single_jets, leptons, 0.4)

        # Rank jets by decreasing UParTAK4B score.
        single_jets = single_jets[
            ak.argsort(
                single_jets.btagUParTAK4B,
                axis=-1,
                ascending=False,
            )
        ]

        n_single_jets = ak.num(single_jets)

        double_jets = clean_by_dr(double_jets, leptons, 0.4)
        double_jets = double_jets[ak.argsort(double_jets.btagUParTAK4probbb, axis=-1, ascending=False)]

        n_double_jets = ak.num(double_jets)
        double_bjets = double_jets[double_jets.btagUParTAK4probbb > 0.12]
        double_untag_jets = double_jets[double_jets.btagUParTAK4probbb < 0.12]
        double_bjets = double_bjets[ak.argsort(double_bjets.btagUParTAK4probbb,  axis=-1, ascending=False)]
        double_untag_jets = double_untag_jets[ak.argsort(double_untag_jets.pt, axis=-1, ascending=False)]
        n_double_bjets = ak.num(double_bjets)

        def build_boosted_0lep_permask(mask):
            n_events = len(mask)
            mask = np.asarray(mask, dtype=bool)

            if not np.any(mask):
                empty = ak.Array([])
                zeros = np.zeros(n_events, dtype=np.int32)
                return empty, empty, empty, zeros, zeros

            jets_sorted = double_jets[mask]
            jets_sorted = jets_sorted[
                ak.argsort(
                    jets_sorted.btagUParTAK4probbb,
                    axis=-1,
                    ascending=False,
                )
            ]

            tagged = (
                jets_sorted.btagUParTAK4probbb >= 0.12
            )

            bjets = jets_sorted[tagged]
            untag = jets_sorted[~tagged]

            nj_small = ak.to_numpy(
                ak.num(jets_sorted)
            ).astype(np.int32)

            nb_small = ak.to_numpy(
                ak.num(bjets)
            ).astype(np.int32)

            nj_full = np.zeros(n_events, dtype=np.int32)
            nb_full = np.zeros(n_events, dtype=np.int32)

            idx = np.where(mask)[0]
            nj_full[idx] = nj_small
            nb_full[idx] = nb_small

            return (
                jets_sorted,
                bjets,
                untag,
                nj_full,
                nb_full,
            )

        def build_resolved_0lep_permask(mask):
            n_events = len(mask)

            if not np.any(mask):
                empty = ak.Array([])
                zeros = np.zeros(n_events, dtype=np.int32)
                return empty, empty, empty, zeros, zeros

            jets_masked = single_jets[mask]

            jets_sorted = jets_masked[
                ak.argsort(
                    jets_masked.btagUParTAK4B,
                    axis=-1,
                    ascending=False,
                )
            ]

            score = jets_sorted.btagUParTAK4B

            tagged_mask = (score >= FIXED_BTAG_THRESHOLD)
            bjets = jets_sorted[tagged_mask]
            untag = jets_sorted[~tagged_mask]

            nj_small = ak.to_numpy(
                ak.num(jets_sorted)
            ).astype(np.int32)

            nb_small = ak.to_numpy(
                ak.num(bjets)
            ).astype(np.int32)

            nj_full = np.zeros(n_events, dtype=np.int32)
            nb_full = np.zeros(n_events, dtype=np.int32)

            idx = np.where(mask)[0]
            nj_full[idx] = nj_small
            nb_full[idx] = nb_small

            # Jagged objects are restricted to the selected events.
            # Multiplicity arrays keep the full event indexing.
            return jets_sorted, bjets, untag, nj_full, nb_full

        def fill_fixedwp_category_histograms(
            channel,
            region,
            base_mask,
            njets_full,
            nbjets_full,
            n_double_bjets_full,
            w_nosf,
            w_sf,
        ):
            #Compare jet and b-jet categories with and without the b-tagging SF
            base_mask = np.asarray(base_mask, dtype=bool)
            njets_full = np.asarray(njets_full, dtype=np.int32)
            nbjets_full = np.asarray(nbjets_full, dtype=np.int32)
            n_double_bjets_full = np.asarray(
                n_double_bjets_full,
                dtype=np.int32,
            )

            if len(base_mask) != len(events):
                raise RuntimeError(
                    "Fixed-WP category base mask has wrong length"
                )

            if (
                len(njets_full) != len(events)
                or len(nbjets_full) != len(events)
                or len(n_double_bjets_full) != len(events)
            ):
                raise RuntimeError(
                    "Fixed-WP category arrays have wrong length"
                )

            # Before removing events that enter the boosted selection.
            pre_veto_mask = base_mask

            # Final resolved phase space.
            resolved_mask = (
                base_mask
                & (n_double_bjets_full < 2)
            )

            wp_tag = f"fixed{FIXED_BTAG_WP}WP"

            for selection_name, selection_mask in (
                ("preBoostVeto", pre_veto_mask),
                ("resolved", resolved_mask),
            ):
                for suffix, weight in (
                    ("nosf", w_nosf),
                    (f"{wp_tag}SF", w_sf),
                ):
                    fill_hist_auto(
                        f"{channel}_{region}_"
                        f"nbjets_vs_njets_{selection_name}_{suffix}",
                        selection_mask,
                        weight,
                        n_jets=njets_full,
                        n_bjets=nbjets_full,
                    )

            # Inclusive fixed-WP b-tag multiplicity categories.
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
                f"nb_categories_{wp_tag}_resolved_nosf"
            )

            with_sf_name = (
                f"{channel}_{region}_ge3j_"
                f"nb_categories_{wp_tag}_resolved_withSF"
            )

            book_category_bins(no_sf_name, labels)
            book_category_bins(with_sf_name, labels)

            for label, mask_category in categories:
                fill_eventflow_auto(
                    no_sf_name,
                    label,
                    mask_category,
                    w_nosf,
                )

                fill_eventflow_auto(
                    with_sf_name,
                    label,
                    mask_category,
                    w_sf,
                )

        # Zero-lepton signal-region selection.
        mask_0lep = ak.to_numpy(n_leptons == 0)
        fill_eventflow_auto("eventflow_SR_boosted", "veto", mask_0lep, w_all)

        fill_eventflow_auto("eventflow_SR_resolved", "veto", mask_0lep, w_all)

        # MET triggers from the skimmed trigger word.
        if "trigger_type" in events.fields:
            trig_word = ak.to_numpy(events.trigger_type).astype(np.int64)
        else:
            trig_word = np.zeros(len(events), dtype=np.int64)

        met1_trig = (trig_word & (1 << 5)) != 0
        met2_trig = (trig_word & (1 << 6)) != 0

        met_trig = met1_trig | met2_trig
        mask_trig_MET = mask_0lep & met_trig

        # Assign data events to their primary dataset.
        if not self.isMC:

            dname = self.dataset_name.lower()
            isJetMET = dname.startswith("jetmet")
            mask_step_trig_met = np.zeros(len(events), dtype=bool)
            if isJetMET:
                mask_step_trig_met = mask_trig_MET

        else:
            # MC has no primary-dataset ownership restriction.
            mask_step_trig_met = mask_trig_MET
        fill_eventflow_auto("eventflow_SR_boosted",     "trigger", mask_step_trig_met, w_all)

        fill_eventflow_auto("eventflow_SR_resolved",    "trigger", mask_step_trig_met, w_all)

        # Require MET above 150 GeV.
        mask_step_met = ak.to_numpy(met_all.pt > 150.0)
        mask0 = mask_step_trig_met & mask_step_met
        fill_eventflow_auto("eventflow_SR_boosted", "met", mask0, w_all)
        fill_eventflow_auto("eventflow_SR_resolved", "met", mask0, w_all)

        # Build boosted jets after the lepton veto, trigger and MET cut.
        dj_0lep, dbj_0lep, dut_0lep, nj_0lep, nb_0lep = build_boosted_0lep_permask(mask0)

        # Jet multiplicity before the two-jet requirement.
        fill_hist_auto(
            "SR_njets_double_pre2j_boosted",
            mask0,
            w_all,
            n_jets=nj_0lep,
        )

        # Require at least two jets in the boosted collection.
        mask_step2j = mask0 & (nj_0lep >= 2)

        fill_eventflow_auto("eventflow_SR_boosted", ">=2jets", mask_step2j, w_all)

        mask_boosted_1db = mask_step2j & (nb_0lep >= 1)
        fill_eventflow_auto(
            "eventflow_SR_boosted", ">=2jets & >=1dbjet", mask_boosted_1db, w_all,
        )

        fill_hist_auto(
            "SR_nbjets_double_pre2b_boosted",
            mask_step2j,
            w_all,
            n_bjets=nb_0lep,
        )

        # Leading and subleading double-b scores before the tag requirement.

        lead_double_btag_full = np.zeros(len(events), dtype=np.float32)
        sublead_double_btag_full = np.zeros(len(events), dtype=np.float32)

        if np.any(mask_step2j):
            dj_after2j, dbj_after2j, dut_after2j, nj_after2j, nb_after2j = build_boosted_0lep_permask(mask_step2j)

            idx_after2j = np.where(mask_step2j)[0]

            lead_double_btag_full[idx_after2j] = ak.to_numpy(dj_after2j[:, 0].btagUParTAK4probbb)
            sublead_double_btag_full[idx_after2j] = ak.to_numpy(dj_after2j[:, 1].btagUParTAK4probbb)

        fill_hist_auto(
            "SR_lead_double_btag_after2j_boosted",
            mask_step2j,
            w_all,
            lead_double_btag=lead_double_btag_full,
        )

        fill_hist_auto(
            "SR_sublead_double_btag_after2j_boosted",
            mask_step2j,
            w_all,
            sublead_double_btag=sublead_double_btag_full,
        )
        H_mass_boosted_full = np.zeros(len(events), dtype=np.float32)
        min_dphi_jmet_boosted_full = np.zeros(len(events), dtype=np.float32)
        # Require at least two double-b-tagged jets.
        mask_boosted_2b = mask_step2j & (nb_0lep >= 2)

        fill_eventflow_auto(
            "eventflow_SR_boosted", ">=2jets & >=2dbjets", mask_boosted_2b, w_all,
        )

        # Kinematics after the double-b-tag requirement.
        # Build the selected variables once.
        BOOSTED_VARS_FULL = {}
        BOOSTED_VAR_NAMES = [
            "H_mass", "H_pt", "H_eta", "H_phi",
            "HT", "puppimet_pt", "puppimet_phi",
            "dphi_H_MET",  "dphi_bb_MET_min",
            "dr_bb_bb_ave", "dm_bb_bb_min",
             "n_jets", "n_bjets",
            "bj_max_pt_pt", "bj_max_pt_eta", "bj_max_pt_phi",
            "bj_2nd_pt_pt"

        ]

        BOOSTED_VARS_FULL = {
            v: np.zeros(len(events), dtype=np.float32)
            for v in BOOSTED_VAR_NAMES
        }
        if np.any(mask_boosted_2b):
            idx_2b = np.where(mask_boosted_2b)[0]

            dj_2b, dbj_2b, dut_2b, nj_2b, nb_2b = build_boosted_0lep_permask(mask_boosted_2b)

            vars_boosted = self.build_vars_boosted(
                dj_2b,
                dbj_2b,
                dut_2b,
                met_all[mask_boosted_2b],
            )

            for vname, arr_small in vars_boosted.items():
                BOOSTED_VARS_FULL[vname][idx_2b] = ak.to_numpy(arr_small)

        for vname, arr_full in BOOSTED_VARS_FULL.items():
            fill_hist_auto(
                f"veto_SR_boosted_{vname}_after2b",
                mask_boosted_2b,
                w_all,
                **{vname: arr_full},
            )

        H_mass = BOOSTED_VARS_FULL["H_mass"]
        dphi_bb = BOOSTED_VARS_FULL["dphi_bb_MET_min"]

        fill_hist_auto(
            "veto_SR_boosted_ABCD_plane_after2b",
            mask_boosted_2b,
            w_all,
            H_mass=H_mass,
            dphi_bb_MET_min=dphi_bb,
        )
        # Fill the boosted ABCD plane for the boundary scan.

        fill_abcd_scan_auto(
            "veto_SR_boosted_ABCD_scan_after2b",
            mask_boosted_2b,
            w_all,
            H_mass,
            dphi_bb,
        )
        # Boosted ABCD and validation regions.
        mass_A = (H_mass >= 75.0)  & (H_mass < 175.0)
        mass_Astar = ((H_mass >= 175.0) & (H_mass < 225.0) )
        mass_CD = ((H_mass >= 225.0) & (H_mass < 325.0) )

        dphi_high = dphi_bb > 1.0
        dphi_low = dphi_bb <= 0.5

        veto_SR_A_boosted = mask_boosted_2b & mass_A     & dphi_high
        veto_SR_B_boosted = mask_boosted_2b & mass_A     & dphi_low

        veto_VR_Astar_boosted = mask_boosted_2b & mass_Astar & dphi_high
        veto_VR_Bstar_boosted = mask_boosted_2b & mass_Astar & dphi_low

        veto_SR_C_boosted = mask_boosted_2b & mass_CD    & dphi_high
        veto_SR_D_boosted = mask_boosted_2b & mass_CD    & dphi_low

        mask_boosted_A_mass = mask_boosted_2b & mass_A
        fill_eventflow_auto("eventflow_SR_boosted", "A_mH", mask_boosted_A_mass, w_all)
        fill_eventflow_auto("eventflow_SR_boosted", "A_dphi", veto_SR_A_boosted, w_all)

        BOOSTED_REGIONS = {
            "A": veto_SR_A_boosted,
            "B": veto_SR_B_boosted,
            "Astar": veto_VR_Astar_boosted,
             "Bstar": veto_VR_Bstar_boosted,
            "C": veto_SR_C_boosted,
            "D": veto_SR_D_boosted,
        }

        for region_name in EVENTFLOW_BINS["eventflow_QCDCR_boosted"]:
            fill_eventflow_auto(
                "eventflow_QCDCR_boosted", region_name,
                BOOSTED_REGIONS[region_name], w_all,
            )

        def region_tag(region_name):
            if region_name in ["A", "B", "C", "D"]:
                return f"{region_name}_SR"
            if region_name == "Astar":
                return "Astar_VR"
            if region_name == "Bstar":
                return "Bstar_VR"
            return region_name
        for region_name, region_mask in BOOSTED_REGIONS.items():
            eval_and_fill_bdt_0lep(
                channel= f"veto_{region_name}_SR_3b",
                region= "",
                regime="boosted",
                mask=region_mask,
                vals_full=BOOSTED_VARS_FULL,
                weight=w_all,
            )
        for region_name, region_mask in BOOSTED_REGIONS.items():
            for vname, arr_full in BOOSTED_VARS_FULL.items():
                fill_hist_auto(
                    sr_name(region_name, vname, "boosted"),
                    region_mask,
                    w_all,
                    **{vname: arr_full},
                )
            fill_hist_auto(
                sr_name(region_name, "nbjets_vs_njets", "boosted"),
                region_mask,
                w_all,
                n_jets=BOOSTED_VARS_FULL["n_jets"],
                n_bjets=BOOSTED_VARS_FULL["n_bjets"],
            )

        # Resolved zero-lepton selection.
        # Start from the lepton veto, MET triggers and MET above 150 GeV.
        # Require at least three jets and three fixed-WP b jets.
        # Define ABCD regions in H mass and the minimum |dphi(b, MET)|.

        sj_0lep, sbj_0lep, sut_0lep, nj_res_0lep, nb_res_0lep = build_resolved_0lep_permask(mask0)

        # Jet multiplicity before the three-jet requirement.
        fill_hist_auto(
            "SR_njets_single_pre3j_resolved",
            mask0,
            w_all,
            n_jets=nj_res_0lep,
        )

        # Require at least three jets in the resolved collection.
        mask_resolved_3j = mask0 & (nj_res_0lep >= 3)
        sj_veto_ge3 = single_jets[mask_resolved_3j]

        w_resolved_nosf = np.asarray(w_all, dtype=np.float64)
        btag_sf_fixed_veto = compute_btag_sf_fixed_full_auto(
            sj_veto_ge3,
            mask_resolved_3j,
        )

        w_resolved_sf_veto = ( w_resolved_nosf * btag_sf_fixed_veto )

        # build_resolved_0lep_permask() returns the fixed-WP b-jet counts.
        nb_fixed_veto = nb_res_0lep
        fill_hist_auto( f"veto_ge3j_event_btagSF_"
        f"{FIXED_BTAG_WP}WP_resolved",mask_resolved_3j,w_resolved_nosf,btag_sf=btag_sf_fixed_veto,)

        fill_fixedwp_category_histograms(
            channel="veto",
            region="SR",
            base_mask=mask_resolved_3j,
            njets_full=nj_res_0lep,
            nbjets_full=nb_fixed_veto,
            n_double_bjets_full=nb_0lep,
            w_nosf=w_resolved_nosf,
            w_sf=w_resolved_sf_veto,
        )
        fill_eventflow_auto(
            "eventflow_SR_resolved", ">=3jets", mask_resolved_3j, w_resolved_sf_veto,
        )

        mask_resolved_2b = mask_resolved_3j & (nb_fixed_veto >= 2)
        fill_eventflow_auto(
            "eventflow_SR_resolved", ">=3jets & >=2bjets",
            mask_resolved_2b, w_resolved_sf_veto,
        )

        # B-jet multiplicity after the jet cut and before the b-tag cut.
        fill_hist_auto(
            "SR_nbjets_single_pre3b_resolved",
            mask_resolved_3j,
            w_resolved_sf_veto,
            n_bjets=nb_res_0lep
        )

        mask_resolved_3b_preVeto = (
            mask_resolved_3j
            & (nb_fixed_veto >= 3)
        )
        fill_eventflow_auto(
            "eventflow_SR_resolved", ">=3jets & >=3bjets",
            mask_resolved_3b_preVeto, w_resolved_sf_veto,
        )

        # Remove events that enter the boosted selection only after counting b jets.
        mask_resolved_3b = (
            mask_resolved_3b_preVeto
            & (nb_0lep < 2)
        )
        fill_hist_auto(
            "check_resolved_preVeto_nDoubleB",
            mask_resolved_3b_preVeto,
            w_resolved_sf_veto,
            n_bjets=nb_0lep
        )

        fill_hist_auto(
            "check_resolved_preVeto_nSingleB_vs_nDoubleB",
            mask_resolved_3b_preVeto,
            w_resolved_sf_veto,
            n_jets=nb_res_0lep,
            n_bjets=nb_0lep
        )

        fill_eventflow_auto(
            "eventflow_SR_resolved", ">=3jets & >=3bjets & <2dbjets",
            mask_resolved_3b, w_resolved_sf_veto,
        )

        # Resolved kinematics after the b-tag requirement.

        RESOLVED_VAR_NAMES = [
            "H_mass", "H_pt", "H_eta", "H_phi",
            "HT", "puppimet_pt", "puppimet_phi",
            "dphi_b_MET_min", "dphi_J_MET_min", "dphi_H_MET",
            "dr_bb_ave", "dm_bb_bb_min", "mbbj",
            "n_jets", "n_bjets",
            "bj_max_pt_pt", "bj_max_pt_eta", "bj_max_pt_phi",
            "bj_2nd_pt_pt",
        ]

        def build_resolved_vars_for_mask(base_mask):
            RES_VARS = {
                v: np.zeros(len(events), dtype=np.float32)
                for v in RESOLVED_VAR_NAMES
            }

            if np.any(base_mask):
                idx = np.where(base_mask)[0]

                sj_sel, sbj_sel, sut_sel, _, _ = build_resolved_0lep_permask(base_mask)

                vals = self.build_vars_resolved(
                    sj_sel,
                    sbj_sel,
                    sut_sel,
                    met_all[base_mask],
                )

                for vname, arr_small in vals.items():
                    RES_VARS[vname][idx] = ak.to_numpy(arr_small)

            return RES_VARS

        RESOLVED_VARS_FULL = build_resolved_vars_for_mask(mask_resolved_3b)

        H_mass_resolved = RESOLVED_VARS_FULL["H_mass"]
        dphi_b_resolved = RESOLVED_VARS_FULL["dphi_b_MET_min"]

        mass_A = (H_mass_resolved >= 75.0)  & (H_mass_resolved < 225.0)
        mass_Astar = (H_mass_resolved >= 225.0) & (H_mass_resolved < 300.0)
        mass_CD = (H_mass_resolved >= 300.0) & (H_mass_resolved < 500.0)

        dphi_high = dphi_b_resolved > 1.0
        dphi_low = dphi_b_resolved <= 0.5

        veto_SR_A_resolved = mask_resolved_3b & mass_A     & dphi_high
        veto_SR_B_resolved = mask_resolved_3b & mass_A     & dphi_low
        veto_VR_Astar_resolved = mask_resolved_3b & mass_Astar & dphi_high
        veto_VR_Bstar_resolved = mask_resolved_3b & mass_Astar & dphi_low
        veto_SR_C_resolved = mask_resolved_3b & mass_CD    & dphi_high
        veto_SR_D_resolved = mask_resolved_3b & mass_CD    & dphi_low

        RESOLVED_REGIONS = {
            "A": veto_SR_A_resolved,
            "B": veto_SR_B_resolved,
            "Astar": veto_VR_Astar_resolved,
            "Bstar": veto_VR_Bstar_resolved,
            "C": veto_SR_C_resolved,
            "D": veto_SR_D_resolved,
        }
        mask_resolved_A_mass = mask_resolved_3b & mass_A
        fill_eventflow_auto(
            "eventflow_SR_resolved", "A_mH", mask_resolved_A_mass, w_resolved_sf_veto,
        )
        fill_eventflow_auto(
            "eventflow_SR_resolved", "A_dphi", veto_SR_A_resolved, w_resolved_sf_veto,
        )

        for region_name in EVENTFLOW_BINS["eventflow_QCDCR_resolved"]:
            fill_eventflow_auto(
                "eventflow_QCDCR_resolved", region_name,
                RESOLVED_REGIONS[region_name], w_resolved_sf_veto,
            )

        fill_hist_auto(
            "veto_SR_resolved_ABCD_plane_after3b",
            mask_resolved_3b,
            w_resolved_sf_veto,
            H_mass=H_mass_resolved,
            dphi_b_MET_min=dphi_b_resolved,
        )
        # Fill the resolved ABCD plane for the boundary scan.

        fill_abcd_scan_auto(
            "veto_SR_resolved_ABCD_scan_after3b",
            mask_resolved_3b,
            w_resolved_sf_veto,
            H_mass_resolved,
            dphi_b_resolved,
        )
        for region_name, region_mask in RESOLVED_REGIONS.items():
            eval_and_fill_bdt_0lep(
                channel= f"veto_{region_name}_SR_3b",
                region= "",
                regime="resolved",
                mask=region_mask,
                vals_full=RESOLVED_VARS_FULL,
                weight=w_resolved_sf_veto,
            )
        for region_name, region_mask in RESOLVED_REGIONS.items():
            for vname, arr_full in RESOLVED_VARS_FULL.items():
                fill_hist_auto(
                  sr_name(region_name, vname, "resolved"),
                    region_mask,
                    w_resolved_sf_veto,
                    **{vname: arr_full},
                )
            fill_hist_auto(
                sr_name(region_name, "nbjets_vs_njets", "resolved"),
                region_mask,
                w_resolved_sf_veto,
                n_jets=RESOLVED_VARS_FULL["n_jets"],
                n_bjets=RESOLVED_VARS_FULL["n_bjets"],
            )

            # Compare the final-region weights using identical event masks.
            for suffix, comparison_weight in (
                ("nobtagSF", w_resolved_nosf),
                (f"fixed{FIXED_BTAG_WP}WPSF", w_resolved_sf_veto,),
            ):
                for variable in ("n_jets", "n_bjets", "HT"):
                    fill_hist_auto(
                        sr_name(
                            region_name,
                            f"{variable}_{suffix}",
                            "resolved",
                        ),
                        region_mask,
                        comparison_weight,
                        **{variable: RESOLVED_VARS_FULL[variable]},
                    )

        # Training trees for the zero-lepton ABCD regions.

        if self.isMVA:
            mass_A_res = (H_mass_resolved >= 75.0)  & (H_mass_resolved < 225.0)
            mass_Astar_res = (H_mass_resolved >= 225.0) & (H_mass_resolved < 300.0)
            mass_CD_res = (H_mass_resolved >= 300.0) & (H_mass_resolved < 500.0)

            dphi_high_resolved = dphi_b_resolved > 1.0
            dphi_low_resolved = dphi_b_resolved <= 0.5

            veto_SR_A_resolved = mask_resolved_3b & mass_A_res & dphi_high_resolved
            veto_SR_B_resolved = mask_resolved_3b & mass_A_res  & dphi_low_resolved

            veto_VR_Astar_resolved = mask_resolved_3b & mass_Astar_res & dphi_high_resolved
            veto_VR_Bstar_resolved = mask_resolved_3b & mass_Astar_res & dphi_low_resolved

            veto_SR_C_resolved = mask_resolved_3b & mass_CD_res    & dphi_high_resolved
            veto_SR_D_resolved = mask_resolved_3b & mass_CD_res    & dphi_low_resolved

            RESOLVED_REGIONS = {
                "A": veto_SR_A_resolved,
                "B": veto_SR_B_resolved,
                "Astar": veto_VR_Astar_resolved,
                "Bstar": veto_VR_Bstar_resolved,
                "C": veto_SR_C_resolved,
                "D": veto_SR_D_resolved,
            }

            if self.runQCD:
                BOOSTED_TREE_REGIONS = {
                    "boosted_B": veto_SR_B_boosted,
                    "boosted_C": veto_SR_C_boosted,
                    "boosted_D": veto_SR_D_boosted,
                }

                RESOLVED_TREE_REGIONS = {
                    "resolved_B": veto_SR_B_resolved,
                    "resolved_C": veto_SR_C_resolved,
                    "resolved_D": veto_SR_D_resolved,
                }

            else:
                BOOSTED_TREE_REGIONS = {
                    "boosted_A": veto_SR_A_boosted,
                }

                RESOLVED_TREE_REGIONS = {
                    "resolved_A": veto_SR_A_resolved,
                }

            for tree_name, tree_mask in BOOSTED_TREE_REGIONS.items():
                self.fill_mva_tree_0lep(
                    tree_name,
                    tree_mask,
                    "boosted",
                    met_all,
                    w_all,
                    build_boosted_0lep_permask,
                    build_resolved_0lep_permask,
                )

            for tree_name, tree_mask in RESOLVED_TREE_REGIONS.items():
                self.fill_mva_tree_0lep(
                    tree_name,
                    tree_mask,
                    "resolved",
                    met_all,
                    w_resolved_sf_veto,
                    build_boosted_0lep_permask,
                    build_resolved_0lep_permask,
                )

        # Single-lepton ttbar control region.

        # Require one electron or muon, its trigger and MET above 150 GeV.

        has_1lep = ak.to_numpy(n_leptons == 1)

        mask_1lep_pt = np.zeros(len(events), dtype=bool)
        mask_1e = np.zeros(len(events), dtype=bool)
        mask_1mu = np.zeros(len(events), dtype=bool)

        if np.any(has_1lep):
            idx_1lep = np.where(has_1lep)[0]

            lep0 = leptons[has_1lep][:, 0]

            is_e = ak.to_numpy(lep0.lepton_type == "e")
            is_mu = ak.to_numpy(lep0.lepton_type == "mu")

            pt_ok = ak.to_numpy(
                ((lep0.lepton_type == "e")  & (lep0.pt > 35.0)) |
                ((lep0.lepton_type == "mu") & (lep0.pt > 25.0))
            )

            mask_1lep_pt[idx_1lep] = pt_ok
            mask_1e[idx_1lep] = is_e & pt_ok
            mask_1mu[idx_1lep] = is_mu & pt_ok

        fill_eventflow_auto("eventflow_TTCR_resolved", "1lep", mask_1lep_pt, w_all)
        # Single-lepton triggers.

        if "trigger_type" in events.fields:
            trig_word = ak.to_numpy(events.trigger_type).astype(np.int64)
        else:
            trig_word = np.zeros(len(events), dtype=np.int64)

        MU_BM = (1 << 1)
        E_BM = (1 << 3)

        trg_mu = (trig_word & MU_BM) != 0
        trg_e = (trig_word & E_BM)  != 0

        pass_trig_1mu = mask_1mu & trg_mu
        pass_trig_1e = mask_1e  & trg_e

        mask_ttcr_trig = pass_trig_1mu | pass_trig_1e
        if not self.isMC:
            dname = self.dataset_name.lower()

            isEGamma = dname.startswith("egamma")
            isMuon = dname.startswith("muon") and not dname.startswith("muoneg")

            mask_step_trig_ttcr = np.zeros(len(events), dtype=bool)

            if isMuon:
                mask_step_trig_ttcr = pass_trig_1mu

            elif isEGamma:
                mask_step_trig_ttcr = pass_trig_1e

        else:
            mask_step_trig_ttcr = mask_ttcr_trig

        fill_eventflow_auto("eventflow_TTCR_resolved", "trigger", mask_step_trig_ttcr, w_all)

        mask_ttcr_met = mask_step_trig_ttcr & ak.to_numpy(met_all.pt > 150.0)
        fill_eventflow_auto("eventflow_TTCR_resolved", "met", mask_ttcr_met, w_all)

        fill_eventflow_auto("eventflow_TTCR_boosted", "1lep", mask_1lep_pt, w_all)
        fill_eventflow_auto("eventflow_TTCR_boosted", "trigger", mask_step_trig_ttcr, w_all)
        fill_eventflow_auto("eventflow_TTCR_boosted", "met", mask_ttcr_met, w_all)
        # Boosted TTCR.
        # Require at least two jets and two double-b-tagged jets.

        dj_ttcr, dbj_ttcr, dut_ttcr, nj_ttcr_boo, nb_ttcr_boo = build_boosted_0lep_permask(mask_ttcr_met)

        fill_hist_auto(
            "TTCR_njets_double_pre2j_boosted",
            mask_ttcr_met,
            w_all,
            n_jets=nj_ttcr_boo,
        )

        mask_ttcr_boosted_2j = mask_ttcr_met & (nj_ttcr_boo >= 2)

        fill_eventflow_auto("eventflow_TTCR_boosted", ">=2jets", mask_ttcr_boosted_2j, w_all)

        mask_ttcr_boosted_1db = mask_ttcr_boosted_2j & (nb_ttcr_boo >= 1)
        fill_eventflow_auto(
            "eventflow_TTCR_boosted", ">=2jets & >=1dbjet", mask_ttcr_boosted_1db, w_all,
        )

        fill_hist_auto(
            "TTCR_nbjets_double_pre2b_boosted",
            mask_ttcr_boosted_2j,
            w_all,
            n_bjets=nb_ttcr_boo,
        )

        mask_ttcr_boosted_2b = mask_ttcr_boosted_2j & (nb_ttcr_boo >= 2)

        fill_eventflow_auto(
            "eventflow_TTCR_boosted", ">=2jets & >=2dbjets", mask_ttcr_boosted_2b, w_all,
        )
        # Resolved TTCR.
        # Require at least three jets and three fixed-WP b jets.

        sj_ttcr, sbj_ttcr, sut_ttcr, nj_ttcr_res, nb_ttcr_res = build_resolved_0lep_permask(mask_ttcr_met)

        fill_hist_auto(
            "TTCR_njets_single_pre3j_resolved",
            mask_ttcr_met,
            w_all,
            n_jets=nj_ttcr_res,
        )
        mask_ttcr_resolved_ex3j = mask_ttcr_met & (nj_ttcr_res == 3)
        mask_ttcr_resolved_3j = mask_ttcr_met & (nj_ttcr_res >= 3)
        sj_lep1_ge3 = single_jets[mask_ttcr_resolved_3j]

        # The one-lepton TTCR currently uses w_all, without a resolved b-tag SF.
        w_resolved_lep1 = np.asarray(
            w_all,
            dtype=np.float64,
        )

        fill_eventflow_auto(
            "eventflow_TTCR_resolved", ">=3jets", mask_ttcr_resolved_3j, w_resolved_lep1,
        )

        mask_ttcr_resolved_2b = mask_ttcr_resolved_3j & (nb_ttcr_res >= 2)
        fill_eventflow_auto(
            "eventflow_TTCR_resolved", ">=3jets & >=2bjets",
            mask_ttcr_resolved_2b, w_resolved_lep1,
        )

        mask_ttcr_resolved_3b_preVeto = mask_ttcr_resolved_3j & (nb_ttcr_res >= 3)
        fill_eventflow_auto(
            "eventflow_TTCR_resolved", ">=3jets & >=3bjets",
            mask_ttcr_resolved_3b_preVeto, w_resolved_lep1,
        )

        fill_hist_auto(
            "TTCR_nbjets_single_pre3b_resolved",
            mask_ttcr_resolved_3j,
            w_resolved_lep1,
            n_bjets=nb_ttcr_res,
        )

        mask_ttcr_resolved_3b = (mask_ttcr_resolved_3j & (nb_ttcr_res >= 3)& (nb_ttcr_boo < 2))

        fill_eventflow_auto(
            "eventflow_TTCR_resolved",
            ">=3jets & >=3bjets & <2dbjets",
            mask_ttcr_resolved_3b,
            w_resolved_lep1,
        )
        if np.any(mask_ttcr_boosted_2b):
            dj_ttcr_2b, dbj_ttcr_2b, dut_ttcr_2b, _, _ = build_boosted_0lep_permask(mask_ttcr_boosted_2b)

            vals_ttcr_boosted = self.build_vars_boosted(
                dj_ttcr_2b,
                dbj_ttcr_2b,
                dut_ttcr_2b,
                met_all[mask_ttcr_boosted_2b],
            )

            TTCR_BOOSTED_FULL = {
                v: np.zeros(len(events), dtype=np.float32)
                for v in BOOSTED_VAR_NAMES
            }

            idx = np.where(mask_ttcr_boosted_2b)[0]

            for vname, arr in vals_ttcr_boosted.items():
                TTCR_BOOSTED_FULL[vname][idx] = ak.to_numpy(arr)

            fill_hist_auto(
                "lep1_TTCR_H_mass_after2db_boosted",
                mask_ttcr_boosted_2b,
                w_all,
                H_mass=TTCR_BOOSTED_FULL["H_mass"],
            )

            fill_hist_auto(
                "lep1_TTCR_dphi_bb_MET_min_after2db_boosted",
                mask_ttcr_boosted_2b,
                w_all,
                dphi_bb_MET_min=TTCR_BOOSTED_FULL["dphi_bb_MET_min"],
            )
            H_mass_ttcr_boosted = TTCR_BOOSTED_FULL["H_mass"]

            mass_A_ttcr_boosted = (
            (H_mass_ttcr_boosted <= 75.0)
            | (H_mass_ttcr_boosted >= 175.0)
        )

            mask_ttcr_boosted_A = mask_ttcr_boosted_2b & mass_A_ttcr_boosted

            fill_eventflow_auto(
                "eventflow_TTCR_boosted",
                "mH",
                mask_ttcr_boosted_A,
                w_all,
            )

            for vname, arr_full in TTCR_BOOSTED_FULL.items():
                fill_hist_auto(
                    cr_name("A", vname, "boosted"),
                    mask_ttcr_boosted_A,
                    w_all,
                    **{vname: arr_full},
                )

            eval_and_fill_bdt_0lep(
                channel="lep1_A_CR_3b",
                region="",
                regime="boosted",
                mask=mask_ttcr_boosted_A,
                vals_full=TTCR_BOOSTED_FULL,
                weight=w_all,
            )

        if np.any(mask_ttcr_resolved_3b):

            TTCR_RESOLVED_FULL = build_resolved_vars_for_mask(mask_ttcr_resolved_3b)

            H_mass_ttcr_resolved = TTCR_RESOLVED_FULL["H_mass"]
            fill_hist_auto(
                "lep1_TTCR_H_mass_after3b_resolved",
                mask_ttcr_resolved_3b,
                w_resolved_lep1,
                H_mass=TTCR_RESOLVED_FULL["H_mass"],
            )

            fill_hist_auto(
                "lep1_TTCR_dphi_b_MET_min_after3b_resolved",
                mask_ttcr_resolved_3b,
                w_resolved_lep1,
                dphi_b_MET_min=TTCR_RESOLVED_FULL["dphi_b_MET_min"],
            )
            mass_A_ttcr_resolved = (
                (H_mass_ttcr_resolved <= 75.0)
                | (H_mass_ttcr_resolved >= 225.0)
            )

            mask_ttcr_resolved_A = mask_ttcr_resolved_3b & mass_A_ttcr_resolved

            fill_eventflow_auto(
                "eventflow_TTCR_resolved",
                "mH",
                mask_ttcr_resolved_A,
                w_resolved_lep1,
            )

            for vname, arr_full in TTCR_RESOLVED_FULL.items():
                fill_hist_auto(
                    cr_name("A", vname, "resolved"),
                    mask_ttcr_resolved_A,
                    w_resolved_lep1,
                    **{vname: arr_full},
                )

            for variable in ("n_jets", "n_bjets", "HT"):
                fill_hist_auto(
                    cr_name(
                        "A",
                        f"{variable}_nobtagSF",
                        "resolved",
                    ),
                    mask_ttcr_resolved_A,
                    w_resolved_lep1,
                    **{
                        variable:
                            TTCR_RESOLVED_FULL[variable]
                    },
                )

            eval_and_fill_bdt_0lep(
                channel="lep1_A_CR_3b",
                region="",
                regime="resolved",
                mask=mask_ttcr_resolved_A,
                vals_full=TTCR_RESOLVED_FULL,
                weight=w_resolved_lep1,
            )

        if self.isMVA and self._trees is not None:
            for regime, tree_dict in self._trees.items():
                self._validate_tree_dict(regime, tree_dict)

        if self.isMVA:
            output["trees"] = self._trees

        if tt_masks is not None:
            return {
                "ttBB": output_ttBB,
                "ttCC": output_ttCC,
                "ttLF": output_ttLF,
            }
        return output

    def postprocess(self, accumulator):

        return accumulator
