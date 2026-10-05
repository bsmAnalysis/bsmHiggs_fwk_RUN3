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

#----------------------------------------------------------------------------------------------------------------------------------------------
def _is_ak(x):
    return isinstance(x, (ak.Array, ak.Record))

#----------------------------------------------------------------------------------------------------------------------------------------------
def _ptphi_to_pxpy(pt, phi):
    return pt * np.cos(phi), pt * np.sin(phi)

#----------------------------------------------------------------------------------------------------------------------------------------------
def _pxpy_to_ptphi(px, py):
    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)
    return pt, phi

#----------------------------------------------------------------------------------------------------------------------------------------------
def _deltaR2(eta1, phi1, eta2, phi2):
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    deta = eta1 - eta2
    return deta*deta + dphi*dphi

#----------------------------------------------------------------------------------------------------------------------------------------------
def _mask_lepton_overlap(jets, leptons, dr=0.4):
    """
    Per jet: True if ΔR(jet, ANY lepton) > dr.
    Works even when an event has 0 leptons or 0 jets (vacuous truth).
    """
    if leptons is None:
        return ak.ones_like(jets.pt, dtype=bool)
    
    # Build all jet–lepton pairs per event: shape [evt, njet, nlep]
    pairs = ak.cartesian({"j": jets, "l": leptons}, axis=1, nested=True)
    
    # ΔR^2
    dphi = np.arctan2(np.sin(pairs.j.phi - pairs.l.phi), np.cos(pairs.j.phi - pairs.l.phi))
    deta = pairs.j.eta - pairs.l.eta
    dr2 = dphi * dphi + deta * deta
    
    # Keep jets that are farther than dr from ALL leptons.
    return ak.all(dr2 > (dr * dr), axis=2)


def _btag_wp_threshold(self, working_point="M"):
    """Return discriminator threshold for the given WP from the JSON; safe fallback if missing."""
    if self._btag_wp_vals is not None:
        try:
            return float(self._btag_wp_vals.evaluate(str(working_point)))
        except Exception:
            pass
        
    defaults = {"L": 0.0246, "M": 0.1272, "T": 0.4648, "XT": 0.6298, "XXT": 0.9739}
    return defaults.get(str(working_point), defaults["M"])

#----------------------------------------------------------------------------------------------------------------------------------------------

def btag_sf_perjet(self, jets, working_point="M", systematic="central"):
    if (self._btag_sf_node is None) or (jets is None) or (len(jets) == 0):
        return ak.ones_like(jets.pt, dtype=float)

    pt     = jets.pt
    abseta = abs(jets.eta)
    flav   = ak.values_astype(getattr(jets, "hadronFlavour", ak.zeros_like(jets.pt)), np.int32)

    counts = ak.num(pt, axis=1)
    try:
        sf_flat = _eval_corr_vectorized(
            self._btag_sf_node,
            systematic=str(systematic),
            working_point=str(working_point),
            flavor=ak.to_numpy(ak.flatten(flav)),
            abseta=ak.to_numpy(ak.flatten(abseta)),
            pt=ak.to_numpy(ak.flatten(pt)),
        )
        sf_flat = np.asarray(sf_flat, dtype=float)
    except Exception:
        f_flat  = ak.to_numpy(ak.flatten(flav))
        a_flat  = ak.to_numpy(ak.flatten(abseta))
        p_flat  = ak.to_numpy(ak.flatten(pt))
        sf_flat = np.ones_like(p_flat, dtype=float)
        mask_b  = (f_flat == 5)
        if np.any(mask_b):
            try:
                sf_flat[mask_b] = _eval_corr_vectorized(
                    self._btag_sf_node,
                    systematic=str(systematic),
                    working_point=str(working_point),
                    flavor=f_flat[mask_b],
                    abseta=a_flat[mask_b],
                    pt=p_flat[mask_b],
                )
            except Exception:
                pass

    return _unflatten_like(sf_flat, counts)

#----------------------------------------------------------------------------------------------------------------------------------------------

def btag_event_weight_tagged_only(self, jets, working_point="T", score_field="btagUParTAK4B", systematic="central"):
    """
    Event b-tag weight (simple): product of SFs **for jets that pass the WP**.
    """
    if jets is None or len(jets) == 0:
        return np.ones(ak.num(jets, axis=1), dtype=float)

    thr = self._btag_wp_threshold(working_point)
    tagged = jets[getattr(jets, score_field) >= thr]

    sfs_tagged = self.btag_sf_perjet(tagged, working_point=working_point, systematic=systematic)
    w = ak.prod(ak.fill_none(sfs_tagged, 1.0), axis=1)
    return ak.to_numpy(w)

#----------------------------------------------------------------------------------------------------------------------------------------------

def btag_event_weight_full(self, jets, effs_mc, working_point="T", score_field="btagUParTAK4B", systematic="central"):
    """
    Full fixed-WP weight (needs MC eff per jet):
      if tagged:      SF
      else untagged: (1 - SF*ε)/(1 - ε)
    """
    if jets is None or len(jets) == 0 or effs_mc is None:
        return np.ones(ak.num(jets, axis=1), dtype=float)

    thr = self._btag_wp_threshold(working_point)
    tagged_mask = (getattr(jets, score_field) >= thr)

    sfs = self.btag_sf_perjet(jets, working_point=working_point, systematic=systematic)
    eff = ak.where(effs_mc < 1e-6, 1e-6, ak.where(effs_mc > 1 - 1e-6, 1 - 1e-6, effs_mc))

    perjet = ak.where(tagged_mask, sfs, (1.0 - sfs * eff) / (1.0 - eff))
    w = ak.prod(ak.fill_none(perjet, 1.0), axis=1)
    
    return ak.to_numpy(w)
#----------------------------------------------------------------------------------------------------------------------------------------------
def _eval_corr_vectorized(corr, **arrays):
    """
    Vectorized correctionlib eval for awkward/numpy.
    - Strings stay scalars (e.g. systematic='nom').
    - Awkward detection uses isinstance(..., (ak.Array, ak.Record)).
    """
    if corr is None:
        return None
    
    names = [i.name for i in corr.inputs]
    
    def _is_ak(x):
        return isinstance(x, (ak.Array, ak.Record))
    
    ref_len = None
    for v in arrays.values():
        if isinstance(v, (str, bytes, np.str_)):
            continue
        try:
            if _is_ak(v):
                ref_len = len(ak.flatten(v))
                break
            v_np = np.asarray(v)
            if v_np.ndim > 0:
                ref_len = v_np.size
                break
        except Exception:
            pass
    
    if ref_len is None:
        ref_len = 1
        
    vals = []
    for n in names:
        v = arrays[n]
        if isinstance(v, (str, bytes, np.str_)):
            vals.append(str(v))
            continue
            
        v_np = None
        try:
            v_np = np.asarray(v)
            if v_np.ndim == 0 and v_np.dtype.kind in ("S", "U", "O"):
                vals.append(v_np.item() if hasattr(v_np, "item") else str(v_np))
                continue
        except Exception:
            pass
            
        if _is_ak(v):
            vv = ak.to_numpy(ak.flatten(v))
        else:
            if v_np is None:
                v_np = np.asarray(v)
            vv = np.repeat(v_np, ref_len) if v_np.ndim == 0 else v_np.ravel()
        vals.append(vv)
        
    return corr.evaluate(*vals)

#----------------------------------------------------------------------------------------------------------------------------------------------
def _rng_normal_like(objs, seed=12345, size=None):
    """
    Deterministic N(0,1) per (event, object-index). Works for jagged arrays.
    Requires fields: objs.event and objs.pt
    Returns a flat numpy float64 array of the same flattened length as objs.
    """
    # Per-object keys
    evt = ak.broadcast_arrays(objs.event, objs.pt)[0]
    idx = ak.local_index(objs.pt, axis=1)
    
    e = ak.to_numpy(ak.flatten(evt)).astype(np.uint64)
    j = ak.to_numpy(ak.flatten(idx)).astype(np.uint64)
    s = np.uint64(seed)
    
    def _mix(x):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = x
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = z ^ (z >> np.uint64(31))
        return z
        
    h1 = _mix(e ^ (j + s))
    h2 = _mix(e ^ (j + s + np.uint64(0xA5A5A5A5A5A5A5A5)))
    
    inv_2_53 = np.float64(1.0 / (1 << 53))
    u1 = ((h1 >> np.uint64(11)).astype(np.float64)) * inv_2_53
    u2 = ((h2 >> np.uint64(11)).astype(np.float64)) * inv_2_53
    
    # Avoid exact 0 or 1 to be safe for log
    eps = np.finfo(np.float64).tiny
    onem = np.nextafter(1.0, 0.0)
    u1 = np.clip(u1, eps, onem)
    u2 = np.clip(u2, eps, onem)
    
    # Box–Muller
    r = np.sqrt(-2.0 * np.log(u1))
    theta = np.float64(2.0 * np.pi) * u2
    z = r * np.cos(theta)
    return z

#----------------------------------------------------------------------------------------------------------------------------------------------
def _clip_nextafter(x, lo, hi):
    lo2 = np.nextafter(lo, 1.0)
    hi2 = np.nextafter(hi, -1.0)
    x = ak.where(x < lo2, lo2, x)
    x = ak.where(x > hi2, hi2, x)
    return x

#----------------------------------------------------------------------------------------------------------------------------------------------
def _unflatten_like(flat, counts):
    return ak.unflatten(ak.Array(flat), counts)

#----------------------------------------------------------------------------------------------------------------------------------------------
def _lookup_th2_vals(x, y, xedges, yedges, vals):
    # Accept ak.Array or numpy, return 1D numpy array
    x_np = np.asarray(ak.to_numpy(x), dtype=float).ravel()
    y_np = np.asarray(ak.to_numpy(y), dtype=float).ravel()
    
    nx, ny = len(xedges) - 1, len(yedges) - 1
    
    x_min = np.nextafter(xedges[0], xedges[1]); x_max = np.nextafter(xedges[-1], xedges[0])
    y_min = np.nextafter(yedges[0], yedges[1]); y_max = np.nextafter(yedges[-1], yedges[0])
    
    x_np = np.clip(x_np, x_min, x_max)
    y_np = np.clip(y_np, y_min, y_max)
    
    ix = np.clip(np.digitize(x_np, xedges, right=False) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y_np, yedges, right=False) - 1, 0, ny - 1)
    
    if vals.shape == (ny, nx):
        out = vals[iy, ix]
    elif vals.shape == (nx, ny):
        out = vals[ix, iy]
    else:
        raise RuntimeError(f"Bad TH2 shape {vals.shape}")
    
    return out

# =========================================================
# Auto-booking histograms with hist.Hist (1D & 2D)
# =========================================================
def _to_numpy_flat(x):
    try:
        if isinstance(x, (ak.Array, ak.Record)):
            return ak.to_numpy(ak.flatten(x))
    except Exception:
        pass
    return np.asarray(x)

class _AutoHist:
    """
    Proxy for a lazily-created hist.Hist.
    On the first .fill(...) it books the proper axes from the kwargs and stores a real Hist into the parent dict.
    After that, calls are forwarded.
    """
    def __init__(self, parent_dict, name, parent_proc=None):
        self._parent = parent_dict
        self._name = name
        self._hist = None
        self._proc = parent_proc  # to access things like self.bdt_edges
        self.systematics_labels = [""]  
        

    def _axis_for_numeric(self, key, arr):
        # Presets by variable name (case-insensitive)
        k = key.lower()
        
        # Fixed common physics ranges
        if k in {"eta"}:
            return hax.Regular(60, -5.0, 5.0, name=key, label=key,underflow=True, overflow=True )
        if k in {"phi"}:
            return hax.Regular(64, -math.pi, math.pi, name=key, label=key,underflow=True, overflow=True)
        if k.startswith("dphi") or k in {"dphi"}:
            return hax.Regular(64, 0.0, math.pi, name=key, label=key,underflow=True, overflow=True)
        if k.startswith("dr") or k in {"dr","dR"}:
            return hax.Regular(50, 0.0, 5.0, name=key, label=key,underflow=True, overflow=True)
        if k.startswith("deta") or k in {"deta"}:
            return hax.Regular(30, 0.0, 6.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"bdt"}:
            # Variable axis using your configured bin edges
            edges = getattr(self._proc, "bdt_edges", None)
            if edges is None:
                # fallback if not available
                return hax.Regular(50, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)
            return hax.Variable(edges, name=key, label=key,underflow=True, overflow=True)
        if k in {"score", "btag_score"}:
        # fixed binning for classifier / btag scores
            return hax.Regular(50, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)

        if k in {"btag"}:
            return hax.Regular(50, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)

        if k in {"n", "n_jets", "n_bjets","n_untag"}:
            # integer counting variables
            # let it grow so unknown categories are allowed
            #return hax.IntCategory([], name=key, label=key, growth=True)
            return hax.IntCategory(list(range(0, 14)), name=key, label=key, growth=False, overflow=True)
        if k.startswith("btag"):
            return hax.Regular(50, 0.0, 1.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"met", "met_pt", "puppimet_pt"}:
            return hax.Regular(60, 0.0, 600.0, name=key, label=key,underflow=True, overflow=True)
        if k in {"ht"}:
            return hax.Regular(50, 0.0, 1500.0, name=key, label=key,underflow=True, overflow=True)
        if k in { "pt_b1","pt_b2","z_pt","ll_pt","pt_ll","pt_z","pt_pretrig","pt_posttrig"}:
            return hax.Regular(50, 0.0, 500.0, name=key, label=key,underflow=True, overflow=True)

        if k in {"h_pt","pt_h"}:
            return hax.Regular(100, 0.0, 1000.0, name=key, label=key,underflow=True, overflow=True)
        if k in { "m_h","h_m","mbbj" ,"m_bbj"}:
            return hax.Regular(50, 0.0, 1000.0, name=key, label=key,underflow=True, overflow=True)
        if k in { "z_m", "ll_m","m_z", "m_ll"}:
            return hax.Regular(5, 70, 120.0, name=key, label=key,underflow=True, overflow=True)
        if "ratio" in k:
            return hax.Regular(50, -0.0, 5.0, name=key, label=key,underflow=True, overflow=True)
        if "dm" in k:
            return hax.Regular(50, 0.0, 200.0, name=key, label=key,underflow=True, overflow=True)
            
        # If no preset: derive from data on the first fill
        a = _to_numpy_flat(arr)
        a = a[np.isfinite(a)]
        if a.size == 0:
            # robust default
            return hax.Regular(50, 0.0, 1.0, name=key, label=key)
            
        lo, hi = np.min(a), np.max(a)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, 1.0
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        return hax.Regular(50, float(lo - pad), float(hi + pad), name=key, label=key)
        
    def _axes_from_kwargs(self, kwargs):
        # Weight key is optional; everything else is a dimension
        dims = [(k, v) for k, v in kwargs.items() if k != "weight"]
        if not dims:
            # degenerate scalar counter
            return [hax.Regular(1, 0.0, 1.0, name="unit", label="unit")]
            
        axes = []
        for key, val in dims:
            v = _to_numpy_flat(val)
            
            # Decide category vs numeric
            if key == "cut":
                axes.append(hax.StrCategory([], name="cut", label="cut",  growth=True))
                continue
            if key == "cut_index":
                axes.append(hax.IntCategory([], name="cut_index", label="cut index", growth=True))
                continue
                
            # String/object arrays -> StrCategory
            if v.dtype.kind in {"U", "S"}:
                axes.append(hax.StrCategory([], name=key, label=key, growth=True))
                continue
                
            # Numeric
            axes.append(self._axis_for_numeric(key, v))
            
        return axes
        
    def _ensure_hist(self, **kwargs):
        if self._hist is not None:
            return self._hist
            
        axes = self._axes_from_kwargs(kwargs)
        self._hist = Hist(*axes, storage=storage.Weight())
        # store into the parent dict for future access
        self._parent[self._name] = self._hist
        return self._hist
        
    def fill(self, **kwargs):
        h = self._ensure_hist(**kwargs)
        # hist.Hist accepts awkward or numpy; keep as-is
        return h.fill(**kwargs)
        
    # useful when the processor prepares an accumulator
    def copy(self):
        # before booking, behave like an empty (no-op) hist in the accumulator
        return Hist(hax.Regular(1, 0.0, 1.0, name="unit", label="unit"), storage=storage.Weight())

class AutoHistDict(dict):
    """
    Dict that returns a lazy _AutoHist proxy; on first .fill it materializes a real Hist and replaces the proxy.
    """
    def __init__(self, parent_proc=None):
        super().__init__()
        self._proc = parent_proc
        
    def __getitem__(self, key):
        if key not in self:
            super().__setitem__(key, _AutoHist(self, key, parent_proc=self._proc))
        return super().__getitem__(key)
        
    # When the processor wants a fresh output accumulator
    def spawn_accumulator(self):
        return AutoHistDict(parent_proc=self._proc)

#----------------------------------------------------------------------------------------------------------------------------------------------
class TOTAL_Processor(processor.ProcessorABC):
    def __init__(self, xsec=1.0, nevts=1.0, isMC=True, dataset_name=None, isMVA=True,  run_eval=False):
        self.xsec = xsec
        self.nevts = nevts
        self.isMC = isMC
        self.isMVA = isMVA
        self.run_eval = run_eval
        #self.verbose = verbose
        self.dataset_name = dataset_name
        
        self._trees = {regime: defaultdict(list) for regime in ["boosted", "resolved"]} if isMVA else None
        self._histograms = AutoHistDict(parent_proc=self)
        
        self.bdt_eval_boosted = XGBHelper(os.path.join("xgb_model", "bdt_model_boosted.json"), 
                                         ["H_mass", "H_pt","H_eta", "Z_pt", "HT", "pt_ratio","puppimet_pt", "btag_min", "dr_bb_bb_ave" , 
                                          "dm_bb_bb_min","dphi_HZ","deta_HZ","dr_HZ", "dphi_untag_Z", "n_jets","n_untag","dr_ll"])
        
        self.bdt_eval_resolved = XGBHelper(os.path.join("xgb_model", "bdt_model_resolved.json"),
                                          ["H_mass", "H_pt","H_eta" ,"Z_pt", "HT", "pt_ratio","puppimet_pt", "btag_min", "dr_bb_bb_ave" , 
                                           "dr_bb_ave","dphi_HZ","deta_HZ","dr_HZ", "dphi_untag_Z", "n_jets","n_untag","dr_ll","mbbj" ])
        
        self.bdt_edges = np.linspace(0.0, 1.0, 51 + 1)
        self.optim_Cuts1_bdt = self.bdt_edges[:-1].tolist()
        
        self.systematics_labels = [""]  # Replace with actual systematic names later
        nvarsToInclude = len(self.systematics_labels)
        nCuts = len(self.optim_Cuts1_bdt)
        
        HERE = os.path.dirname(__file__)
        CORR_DIR = os.path.join(HERE, "corrections")
        
        #--- Electron energy and smearing corrections (2024) ---#
        self.egm_json_path = os.path.join(CORR_DIR, "electronSS_EtDependent_V1.json.gz")
        self._egm_scale = None
        self._egm_smear = None
        
        if os.path.exists(self.egm_json_path):
            try:
                egm_cset = correctionlib.CorrectionSet.from_file(self.egm_json_path)
                
                def _take(cset, name):
                    try:
                        return cset[name]
                    except Exception:
                        return None
                
                # DATA vs MC will be used at evaluation time
                self._egm_scale = _take(egm_cset, "EGMScale_Compound_Ele_2024")
                self._egm_smear = _take(egm_cset, "EGMSmearAndSyst_ElePTsplit_2024")
                
                if (self._egm_scale is None) and (self._egm_smear is None):
                    print(f"[ANA:EGM] No expected keys in {self.egm_json_path}. Available: {list(egm_cset.keys())}")
                else:
                    print("[ANA:EGM] Loaded electron scale/smear JSON.")
            except Exception as e:
                print(f"[ANA:EGM] Failed to load: {e}")
        else:
            print("[ANA:EGM] electronSS_EtDependent.json.gz not found; skipping EGM energy corrections.")
            
        #--- Electron ID Tight SFs (2024 combined egamma) ---#
        self.ele_id_root = os.path.join(CORR_DIR, "merged_EGamma_SF2D_Tight.root")
        self._ele_id_edges_x = self._ele_id_edges_y = self._ele_id_vals = None
        
        try:
            with uproot.open(self.ele_id_root) as f:
                h = f["EGamma_SF2D"]
                self._ele_id_edges_x = h.axes[0].edges()
                self._ele_id_edges_y = h.axes[1].edges()
                self._ele_id_vals = h.values()
        except Exception as e:
            print(f"[ANA:ElectronID] Failed to load TH2: {e}")
            
        #--- HLT Ele30 TightID (2023D proxy) / Need to change later (expected at the beginning of September) ---#
        self.ele_hlt_root = os.path.join(CORR_DIR, "egammaEffi.txt_EGM2D.root")
        self._ele_hlt_edges_x = self._ele_hlt_edges_y = None
        self._ele_hlt_effD = self._ele_hlt_effM = None
        
        try:
            with uproot.open(self.ele_hlt_root) as f:
                keys = {k.split(";")[0] for k in f.keys()}
                if "EGamma_EffData2D" in keys and "EGamma_EffMC2D" in keys:
                    hD = f["EGamma_EffData2D"]; hM = f["EGamma_EffMC2D"]
                    self._ele_hlt_edges_x = hD.axes[0].edges()
                    self._ele_hlt_edges_y = hD.axes[1].edges()
                    self._ele_hlt_effD = hD.values()
                    self._ele_hlt_effM = hM.values()
                else:
                    print("[ANA:EleHLT] No eff histos found; Ele HLT SFs disabled.")
        except Exception as e:
            print(f"[ANA:EleHLT] Failed to load HLT eff: {e}")
            
        #--- Muon ID Tight and Iso SFs (2023D) ---#
        self.mu_idiso_json = os.path.join(CORR_DIR, "ScaleFactors_Muon_Z_ID_ISO_2023_BPix_schemaV2.json")
        self._mu_id = self._mu_iso = None
        
        try:
            mu_cset = correctionlib.CorrectionSet.from_file(self.mu_idiso_json)
            self._mu_id = mu_cset["NUM_TightID_DEN_TrackerMuons"]
            self._mu_iso = mu_cset["NUM_TightPFIso_DEN_TightID"]
            print("[ANA:Muon] Loaded ID+ISO corrections.")
        except Exception as e:
            print(f"[ANA:Muon] Failed to load ID/ISO JSON: {e}")
            
        #--- HLT IsoMu24 TightID (2023D proxy) ---#
        self.mu_hlt_json = os.path.join(CORR_DIR, "muon_Z.json")
        self._mu_hlt = None
        
        try:
            hlt_cset = correctionlib.CorrectionSet.from_file(self.mu_hlt_json)
            self._mu_hlt = hlt_cset["NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight"]
            print("[ANA:MuonHLT] Loaded HLT IsoMu24 corrections.")
        except Exception as e:
            print(f"[ANA:MuonHLT] Failed to load HLT JSON: {e}")
            
        # ====================== #
        # JERC (JEC + JER)       #
        # ====================== #
        self.jerc_json_path = os.path.join(CORR_DIR, "jet_jerc.json.gz")
        self._jec_L1 = self._jec_L2 = self._jec_L3 = self._jec_residual = None
        self._jer_sf = self._jer_res = None
        self._jes_unc_total = None
        
        if os.path.exists(self.jerc_json_path):
            try:
                jerc = correctionlib.CorrectionSet.from_file(self.jerc_json_path)
                
                # --- pick the first key that matches these substrings
                def _pick(*parts):
                    for k in jerc:
                        if all(p in k for p in parts):
                            return jerc[k], k
                    return None, None
                    
                # AK4 PFPuppi chain (Summer24Prompt24*)
                if self.isMC:
                    self._jec_L1, key_L1 = _pick("L1FastJet", "AK4PFPuppi", "MC")
                    self._jec_L2, key_L2 = _pick("L2Relative", "AK4PFPuppi", "MC")
                    self._jec_L3, key_L3 = _pick("L3Absolute", "AK4PFPuppi", "MC")
                    key_resid = None
                else:
                    self._jec_L1, key_L1 = _pick("L1FastJet", "AK4PFPuppi", "DATA")
                    self._jec_L2, key_L2 = _pick("L2Relative", "AK4PFPuppi", "DATA")
                    self._jec_L3, key_L3 = _pick("L3Absolute", "AK4PFPuppi", "DATA")
                    # Please note that for 2024, we have Run-based L2L3Residual corrections
                    self._jec_residual, key_resid = _pick("L2L3Residual", "AK4PFPuppi", "DATA")
                    
                print(f"[ANA:JEC] L1={key_L1} L2={key_L2} L3={key_L3} Residual={key_resid}")
                
                def _get(name_fragment):
                    for k in jerc:
                        if all(s in k for s in name_fragment.split()):
                            return jerc[k], k
                    return None, None
                    
                # JER pieces
                self._jer_res, jer_res_key = _get("Summer23BPixPrompt23_RunD_JRV1_MC PtResolution AK4PFPuppi")
                self._jer_sf, jer_sf_key = _get("Summer23BPixPrompt23_RunD_JRV1_MC ScaleFactor AK4PFPuppi")
                print("[ANA:JER] res:", jer_res_key, " sf:", jer_sf_key)
                
                self._jes_unc_total, jes_key = _pick("MC", "Total", "AK4PFPuppi")
                if self._jes_unc_total is None:
                    self._jes_unc_total, jes_key = _pick("Total", "AK4PFPuppi")
                if self._jes_unc_total is not None:
                    print(f"[ANA:JES] Loaded uncertainty: {jes_key}")
                else:
                    print("[ANA:JES] Total uncertainty not found (will skip JES up/down).")
                    
            except Exception as e:
                print(f"[ANA:JERC] Failed to load jet_jerc.json: {e}")
        else:
            print("[ANA:JERC] jet_jerc.json not found; skipping JEC/JER.")
            
        # ====================== #
        #    b-tagging (UParT)   #
        # ====================== #
        self.btag_json_path = os.path.join(CORR_DIR, "btagging_preliminary.json")
        self._btag_wp_vals = None      
        self._btag_sf_node = None     

        if os.path.exists(self.btag_json_path):
            try:
                btag_cset = correctionlib.CorrectionSet.from_file(self.btag_json_path)
        
                def _grab(cset, name):
                    try: return cset[name]
                    except Exception: return None
        
                self._btag_wp_vals = _grab(btag_cset, "UParTAK4_wp_values")
                self._btag_sf_node = _grab(btag_cset, "UParTAK4_kinfit")

                if self._btag_wp_vals is None or self._btag_sf_node is None:
                    print(f"[ANA:BTAG] Missing nodes in {self.btag_json_path}. Keys: {list(btag_cset.keys())}")


                else:
                    print("[ANA:BTAG] Loaded UParTAK4 WP + kinfit SFs.")
            except Exception as e:
                print(f"[ANA:BTAG] Failed to load {self.btag_json_path}: {e}")
        else:
            print("[ANA:BTAG] btagging_preliminary.json not found; skipping b-tag SFs.")


        TOTAL_Processor._btag_wp_threshold             = _btag_wp_threshold
        TOTAL_Processor.btag_sf_perjet                 = btag_sf_perjet
        TOTAL_Processor.btag_event_weight_tagged_only  = btag_event_weight_tagged_only
        TOTAL_Processor.btag_event_weight_full         = btag_event_weight_full
    







    @property
    def histograms(self):
        return self._histograms
        
    def add_tree_entry(self, regime, data_dict):
        if not self._trees or regime not in self._trees:
            return
            
        for key, val in data_dict.items():
            val = np.asarray(val)
            self._trees[regime][key].extend(val.tolist())
            
    def compat_tree_variables(self, tree_dict):
        '''Ensure all output branches are float64 numpy arrays for hadd compatibility.'''
        for key in tree_dict:
            tree_dict[key] = np.asarray(tree_dict[key], dtype=np.float64)
            
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
          
        ###################################
        # DETECTOR LEVEL ANALYSIS         #
        ###################################    

        
        weights = Weights(n)
        
        if self.isMC:
           
            norm = (self.xsec / self.nevts)
            weight_array = np.ones(n,dtype="float64") * norm
            weights.add("norm", weight_array)
        else:
            weights.add("ones", np.ones(n, dtype="float64"))
            
        output = self._histograms.spawn_accumulator()
      
        # =============================== #
        # STEP 2 : EGM energy corrections #
        # =============================== #
        ele_all = events.Electron
        
        if "event" not in ak.fields(ele_all):
            ele_all = ak.with_field(ele_all, ak.broadcast_arrays(events.event, ele_all.pt)[0], "event")
            
        scEta    = ele_all.superclusterEta
        absScEta = np.abs(scEta)
        
        if (not self.isMC) and (self._egm_scale is not None):
            # DATA: multiplicative scale
            run_e = ak.broadcast_arrays(events.run, ele_all.pt)[0]
            counts = ak.num(ele_all.pt, axis=1)
        
            scale_flat = self._egm_scale.evaluate(
                "scale",
                ak.to_numpy(ak.flatten(run_e)),
                ak.to_numpy(ak.flatten(scEta)),
                ak.to_numpy(ak.flatten(ele_all.r9)),
                ak.to_numpy(ak.flatten(absScEta)),
                ak.to_numpy(ak.flatten(ele_all.pt)),
                ak.to_numpy(ak.flatten(ele_all.seedGain)),
            )
            scale = _unflatten_like(scale_flat, counts)
        
            _val = ele_all.pt * scale
            ele_corr_pt = ak.values_astype(ak.where(_val > 0.0, _val, 0.0), "float32")
            ElectronCorr = ak.with_field(ele_all, ele_corr_pt, "pt")
        
        elif self.isMC and (self._egm_smear is not None):
            # MC: Gaussian smearing width from JSON
            counts = ak.num(ele_all.pt, axis=1)
        
            smear_width_flat = self._egm_smear.evaluate(
                "smear",
                ak.to_numpy(ak.flatten(ele_all.pt)),
                ak.to_numpy(ak.flatten(ele_all.r9)),
                ak.to_numpy(ak.flatten(absScEta)),
            )
            smear_width = _unflatten_like(smear_width_flat, counts)
        
            n = _unflatten_like(_rng_normal_like(ele_all), counts)  # deterministic per electron
            _val = ele_all.pt * (1.0 + smear_width * n)
            ele_corr_pt = ak.values_astype(ak.where(_val > 0.0, _val, 0.0), "float32")
            ElectronCorr = ak.with_field(ele_all, ele_corr_pt, "pt")
        
        else:
            ElectronCorr = ele_all  # no EGM available



        # ================================== #
        # STEP 3 : JEC + JER for AK4 PFPuppi #
        # ================================== #
        jets_in = events.Jet
        
        if "event" not in ak.fields(jets_in):
            jets_in = ak.with_field(jets_in, ak.broadcast_arrays(events.event, jets_in.pt)[0], "event")
            
            rawFactor = getattr(jets_in, "rawFactor", ak.zeros_like(jets_in.pt))
            pt_raw    = jets_in.pt   * (1.0 - rawFactor)
            mass_raw  = jets_in.mass * (1.0 - rawFactor)
            
            rho_evt = events.fixedGridRhoFastjetAll
            counts  = ak.num(pt_raw, axis=1)
            rho     = ak.broadcast_arrays(rho_evt, pt_raw)[0]
            
            pt_step = pt_raw
            
        # L1
        if self._jec_L1 is not None:
            c = self._jec_L1
            corr_flat = _eval_corr_vectorized(
                c, JetA=jets_in.area, JetEta=jets_in.eta, JetPt=pt_step, Rho=rho, JetPhi=jets_in.phi
            )
            if corr_flat is not None:
                cfac = _unflatten_like(corr_flat, counts)
                pt_step = pt_step * cfac
                
        # L2
        if self._jec_L2 is not None:
            c = self._jec_L2
            corr_flat = _eval_corr_vectorized(
                c, JetA=jets_in.area, JetEta=jets_in.eta, JetPt=pt_step, Rho=rho, JetPhi=jets_in.phi
                )
            if corr_flat is not None:
                cfac = _unflatten_like(corr_flat, counts)
                pt_step = pt_step * cfac
                
        # L3
        if self._jec_L3 is not None:
            c = self._jec_L3
            corr_flat = _eval_corr_vectorized(
                c, JetA=jets_in.area, JetEta=jets_in.eta, JetPt=pt_step, Rho=rho, JetPhi=jets_in.phi
        )
            if corr_flat is not None:
                cfac = _unflatten_like(corr_flat, counts)
                pt_step = pt_step * cfac
                    
        # Data-only residual
        if (not self.isMC) and (self._jec_residual is not None):
            c = self._jec_residual
            run_b = ak.broadcast_arrays(events.run, pt_raw)[0]
            kwargs = {
                "JetA": jets_in.area, "JetEta": jets_in.eta, "JetPhi": jets_in.phi, "Rho": rho,
                "JetPt": pt_raw, "JetPtRaw": pt_raw, "PtRaw": pt_raw,
                "run": run_b, "Run": run_b, "RunNumber": run_b,
            }
            corr_flat = _eval_corr_vectorized(c, **kwargs)
            if corr_flat is not None:
                cfac    = _unflatten_like(corr_flat, counts)
                pt_step = pt_step * cfac

        # JEC factor and mass
        jec_factor = ak.where(pt_raw > 0, pt_step / pt_raw, 1.0)
        jec_factor = ak.values_astype(_clip_nextafter(jec_factor, 0.0, np.inf), "float32")
        pt_jec     = pt_step
        mass_jec   = mass_raw * jec_factor
        
        # ---- JER using stored Jet_pt_genMatched ----
        if self.isMC and (self._jer_sf is not None):
            # JEC-level inputs
            pt  = pt_jec
            eta = jets_in.eta
            
            # Stored match: <=0 means unmatched
            pt_gen  = ak.fill_none(getattr(jets_in, "pt_genMatched", ak.zeros_like(pt)), 0.0)
            has_gen = ak.values_astype(pt_gen > 0.0, bool)
            
            # SF (nom)
            sf_nom_flat = _eval_corr_vectorized(self._jer_sf, JetEta=eta, JetPt=pt, systematic="nom")
            sf_nom = _unflatten_like(sf_nom_flat, counts) if sf_nom_flat is not None else ak.ones_like(pt)
            
            # Resolution at JEC kinematics
            if self._jer_res is not None:
                res_flat = _eval_corr_vectorized(self._jer_res, JetEta=eta, Rho=rho, JetPt=pt)
                res = _unflatten_like(res_flat, counts)
            else:
                res = ak.zeros_like(pt)

            # Start from un-smeared
            pt_corr = pt
            
            # Matched: hybrid
            _tmp = pt_gen + sf_nom * (pt - pt_gen)
            pt_matched = ak.where(_tmp > 0.0, _tmp, 0.0)
            pt_corr = ak.where(has_gen, pt_matched, pt_corr)
            
            # Unmatched: stochastic (deterministic per-jet)
            n = _unflatten_like(_rng_normal_like(jets_in), counts)
            sigma = res * np.sqrt(np.maximum(sf_nom**2 - 1.0, 0.0))
            smear_factor = 1.0 + sigma * n
            pt_corr = ak.where(~has_gen, pt * smear_factor, pt_corr)

            # Propagate mass
            jer_factor = ak.where(pt > 0, pt_corr / pt, 1.0)
            mass_corr  = mass_jec * jer_factor
        else:
            pt_corr  = pt_jec
            mass_corr = mass_jec

        jets = ak.with_field(jets_in, ak.values_astype(pt_corr,  "float32"), "pt")
        jets = ak.with_field(jets,    ak.values_astype(mass_corr,"float32"), "mass")
        jets = ak.with_field(jets,    jec_factor, "jecFactor")
        if self.isMC:
            jerFactor = ak.values_astype(ak.where(pt_jec > 0, pt_corr / pt_jec, 1.0), "float32")
        else:
            jerFactor = ak.values_astype(ak.ones_like(pt_jec), "float32")
        jets = ak.with_field(jets, jerFactor, "jerFactor")

        # ================== #
        # STEP 4: Type-1 MET #
        # ================== #
        met_in = events.PuppiMET
        met_px, met_py = _ptphi_to_pxpy(met_in.pt, met_in.phi)
        
        # reuse jets_in for MET building
        rawFactor = getattr(jets_in, "rawFactor", ak.zeros_like(jets_in.pt))
        pt_raw    = jets_in.pt * (1.0 - rawFactor)
        
        # be robust to missing passJetIdTight (fall back to LepVeto if needed)
        _pass_tight = getattr(jets_in, "passJetIdTightLepVeto", None)
        if _pass_tight is None:
            pass_tight_bool = ak.ones_like(jets_in.pt, dtype=bool)
        else:
            pass_tight_bool = ak.values_astype(_pass_tight, bool)

        jet_for_met = (
            (pt_raw > 15.0)
            & (np.abs(jets_in.eta) < 4.8)
            & pass_tight_bool
            & ak.values_astype(jets_in.passJetIdTightLepVeto, bool)
        )
        
        # Δp from JEC
        dpt_jec = ak.where(jet_for_met, (pt_jec - pt_raw), 0.0)
        dpx_jec = ak.sum(dpt_jec * np.cos(jets_in.phi), axis=1)
        dpy_jec = ak.sum(dpt_jec * np.sin(jets_in.phi), axis=1)
        
        # Δp from JER (MC only)
        if self.isMC:
            dpt_jer = ak.where(jet_for_met, (pt_corr - pt_jec), 0.0)
            dpx_jer = ak.sum(dpt_jer * np.cos(jets_in.phi), axis=1)
            dpy_jer = ak.sum(dpt_jer * np.sin(jets_in.phi), axis=1)
        else:
            dpx_jer = ak.zeros_like(dpx_jec)
            dpy_jer = ak.zeros_like(dpy_jec)

        # apply to MET
        met_px_corr = met_px - (dpx_jec + dpx_jer)
        met_py_corr = met_py - (dpy_jec + dpy_jer)
        met_pt_corr, met_phi_corr = _pxpy_to_ptphi(met_px_corr, met_py_corr)

        PuppiMETCorr = ak.zip(
            {"pt": ak.values_astype(met_pt_corr, "float32"),
             "phi": ak.values_astype(met_phi_corr, "float32")},
            with_name="MET",
        )

        if self.isMC:
            el0 = ele_all
            el1 = ElectronCorr
            dpx_el = ak.sum((el1.pt - el0.pt) * np.cos(el0.phi), axis=1)
            dpy_el = ak.sum((el1.pt - el0.pt) * np.sin(el0.phi), axis=1)
            met_px_corr = met_px_corr - dpx_el
            met_py_corr = met_py_corr - dpy_el
            met_pt_corr, met_phi_corr = _pxpy_to_ptphi(met_px_corr, met_py_corr)
            PuppiMETCorr = ak.with_field(PuppiMETCorr, ak.values_astype(met_pt_corr, "float32"), "pt")
            PuppiMETCorr = ak.with_field(PuppiMETCorr, ak.values_astype(met_phi_corr, "float32"), "phi")
            
            # stash the systematics
            systs = {"jets": {}, "met": {}, "weights": {}}
       
            # ============================== #
            # STEP 5 : JER up/down (MC only) #
            # ============================== #
            if self.isMC and (self._jer_sf is not None):
                jets_in   = events.Jet
                rawFactor = getattr(jets_in, "rawFactor", ak.zeros_like(jets_in.pt))
                pt_raw    = jets_in.pt * (1.0 - rawFactor)
                
                # inputs at JEC level
                eta    = jets_in.eta
                pt     = pt_jec
                rho    = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, pt)[0]
                counts = ak.num(pt, axis=1)
                
                # gen matching via stored branch (<=0 means "unmatched")
                pt_gen  = ak.fill_none(getattr(jets_in, "pt_genMatched", ak.zeros_like(pt)), 0.0)
                has_gen = ak.values_astype(pt_gen > 0.0, bool)
                
                # SF up/down
                sf_up   = _unflatten_like(_eval_corr_vectorized(self._jer_sf, JetEta=eta, JetPt=pt, systematic="up"),   counts)
                sf_down = _unflatten_like(_eval_corr_vectorized(self._jer_sf, JetEta=eta, JetPt=pt, systematic="down"), counts)
                
                # resolution at JEC kinematics
                if self._jer_res is not None:
                    res = _unflatten_like(_eval_corr_vectorized(self._jer_res, JetEta=eta, Rho=rho, JetPt=pt), counts)
                else:
                    res = ak.zeros_like(pt)

                # deterministic RNG per (event, jet)
                if "event" not in ak.fields(jets_in):
                    jets_in = ak.with_field(jets_in, ak.broadcast_arrays(events.event, jets_in.pt)[0], "event")
                n = _unflatten_like(_rng_normal_like(jets_in), counts)

                def _smear(pt_in, pt_gen_in, has_gen_in, sf_in, res_in, n_in):
                    # matched: hybrid; unmatched: stochastic
                    _tmp = pt_gen_in + sf_in * (pt_in - pt_gen_in)
                    pt_m = ak.where(_tmp > 0.0, _tmp, 0.0)
                    sigma = res_in * np.sqrt(np.maximum(sf_in**2 - 1.0, 0.0))
                    pt_u = pt_in * (1.0 + sigma * n_in)
                    return ak.where(has_gen_in, pt_m, pt_u)
                
                pt_jerUp   = _smear(pt, pt_gen, has_gen, sf_up,   res, n)
                pt_jerDown = _smear(pt, pt_gen, has_gen, sf_down, res, n)
                
                # propagate to mass from JEC mass
                mass_jerUp   = mass_jec * ak.where(pt_jec > 0, pt_jerUp   / pt_jec, 1.0)
                mass_jerDown = mass_jec * ak.where(pt_jec > 0, pt_jerDown / pt_jec, 1.0)
                
                jets_jerUp   = ak.with_field(ak.with_field(jets_in, ak.values_astype(pt_jerUp,   "float32"), "pt"),
                                             ak.values_astype(mass_jerUp,   "float32"), "mass")
                jets_jerDown = ak.with_field(ak.with_field(jets_in, ak.values_astype(pt_jerDown, "float32"), "pt"),
                                             ak.values_astype(mass_jerDown, "float32"), "mass")

                # MET variants (Type-1 with JEC+JER), include electron shift (MC)
                def _met_variant(pt_step_var, pt_final_var):
                    jet_for_met = (
                        (pt_raw > 15.0)
                        & (np.abs(jets_in.eta) < 4.8)
                        #& ak.values_astype(jets_in.passJetIdTight, bool)
                        & ak.values_astype(jets_in.passJetIdTightLepVeto, bool)
                    )
                    dpt_jec = ak.where(jet_for_met, pt_step_var  - pt_raw, 0.0)
                    dpt_jer = ak.where(jet_for_met, pt_final_var - pt_step_var, 0.0)
                    dpx = ak.sum((dpt_jec + dpt_jer) * np.cos(jets_in.phi), axis=1)
                    dpy = ak.sum((dpt_jec + dpt_jer) * np.sin(jets_in.phi), axis=1)
                    
                    px = met_px - dpx   # met_px/met_py from Step 4
                    py = met_py - dpy

                    # electron MET shift (MC)
                    el0, el1 = ele_all, ElectronCorr
                    dpx_el = ak.sum((el1.pt - el0.pt) * np.cos(el0.phi), axis=1)
                    dpy_el = ak.sum((el1.pt - el0.pt) * np.sin(el0.phi), axis=1)
                    px = px - ak.values_astype(dpx_el, "float64")
                    py = py - ak.values_astype(dpy_el, "float64")
                    
                    ptv, phiv = _pxpy_to_ptphi(px, py)
                    return ak.zip({"pt": ak.values_astype(ptv, "float32"),
                                   "phi": ak.values_astype(phiv, "float32")}, with_name="MET")

                PuppiMETCorr_jerUp   = _met_variant(pt_jec, pt_jerUp)
                PuppiMETCorr_jerDown = _met_variant(pt_jec, pt_jerDown)
                
                systs["jets"]["_jerup"]   = jets_jerUp
                systs["jets"]["_jerdown"] = jets_jerDown
                systs["met"]["_jerup"]    = PuppiMETCorr_jerUp
                systs["met"]["_jerdown"]  = PuppiMETCorr_jerDown
            
            # ============================ #
            # STEP 6 : JES "Total" up/down # 
            # ============================ #
            if self._jes_unc_total is not None:
                jets_in   = events.Jet
                rawFactor = getattr(jets_in, "rawFactor", ak.zeros_like(jets_in.pt))
                pt_raw    = jets_in.pt * (1.0 - rawFactor)
                mass_raw  = jets_in.mass * (1.0 - rawFactor)  
                counts    = ak.num(pt_jec, axis=1)            
            
                # fractional uncertainty u(eta, pt_jec)
                unc_flat = _eval_corr_vectorized(self._jes_unc_total, JetEta=jets_in.eta, JetPt=pt_jec)
                unc = _unflatten_like(unc_flat, counts)
            
                # reuse nominal JER ratio (so JES varies only the JEC stage)
                jer_ratio = ak.where(pt_jec > 0, pt_corr / pt_jec, 1.0)
            
                # vary at JEC stage, then apply same JER ratio
                pt_step_jesUp   = pt_jec * (1.0 + unc)
                pt_step_jesDown = pt_jec * (1.0 - unc)
            
                pt_jesUp   = pt_step_jesUp   * jer_ratio
                pt_jesDown = pt_step_jesDown * jer_ratio
            
                # propagate mass with same pT ratio w.r.t. *raw*
                mass_jesUp   = ak.values_astype(mass_raw * ak.where(pt_raw > 0, pt_jesUp   / pt_raw, 1.0), "float32")
                mass_jesDown = ak.values_astype(mass_raw * ak.where(pt_raw > 0, pt_jesDown / pt_raw, 1.0), "float32")
            
                jets_jesUp   = ak.with_field(jets_in, ak.values_astype(pt_jesUp,   "float32"), "pt")
                jets_jesUp   = ak.with_field(jets_jesUp,   mass_jesUp,   "mass")
                jets_jesDown = ak.with_field(jets_in, ak.values_astype(pt_jesDown, "float32"), "pt")
                jets_jesDown = ak.with_field(jets_jesDown, mass_jesDown, "mass")
            
                # MET variant helper (recompute jet_for_met locally)
                def _met_variant(pt_step_var, pt_final_var):
                    jet_for_met = (
                        (pt_raw > 15.0)
                        & (np.abs(jets_in.eta) < 4.8)
                        #& ak.values_astype(jets_in.passJetIdTight, bool)
                        & ak.values_astype(jets_in.passJetIdTightLepVeto, bool)
                    )
                    dpt_jec = ak.where(jet_for_met, pt_step_var  - pt_raw, 0.0)
                    dpt_jer = ak.where(jet_for_met, pt_final_var - pt_step_var, 0.0)
            
                    dpx = ak.sum((dpt_jec + dpt_jer) * np.cos(jets_in.phi), axis=1)
                    dpy = ak.sum((dpt_jec + dpt_jer) * np.sin(jets_in.phi), axis=1)
            
                    px = met_px - dpx      
                    py = met_py - dpy
            
                    if self.isMC:
                        el0, el1 = ele_all, ElectronCorr
                        dpx_el = ak.sum((el1.pt - el0.pt) * np.cos(el0.phi), axis=1)
                        dpy_el = ak.sum((el1.pt - el0.pt) * np.sin(el0.phi), axis=1)
                        
                        px = px - ak.values_astype(dpx_el, "float64")
                        py = py - ak.values_astype(dpy_el, "float64")
            
                    ptv, phiv = _pxpy_to_ptphi(px, py)
                    return ak.zip({"pt": ak.values_astype(ptv, "float32"),
                                   "phi": ak.values_astype(phiv, "float32")}, with_name="MET")
            
                PuppiMETCorr_jesUp   = _met_variant(pt_step_jesUp,   pt_jesUp)
                PuppiMETCorr_jesDown = _met_variant(pt_step_jesDown, pt_jesDown)
                
                systs["jets"]["_jesup"]   = jets_jesUp
                systs["jets"]["_jesdown"] = jets_jesDown
                systs["met"]["_jesup"]    = PuppiMETCorr_jesUp
                systs["met"]["_jesdown"]  = PuppiMETCorr_jesDown
            
                
            # ============================================ #
            # STEP 7 : Unclustered MET (_umetup/_umetdown) #
            # ============================================ #
            pm_fields = set(ak.fields(events.PuppiMET))
            need = {"ptUnclusteredUp", "phiUnclusteredUp", "ptUnclusteredDown", "phiUnclusteredDown"}
            
            if need.issubset(pm_fields):
                # baseline: your corrected MET
                px_corr, py_corr = _ptphi_to_pxpy(PuppiMETCorr.pt, PuppiMETCorr.phi)
            
                # deltas from Nano (raw PuppiMET → UnclusteredUp/Down)
                px0,  py0  = _ptphi_to_pxpy(events.PuppiMET.pt,                 events.PuppiMET.phi)
                pxUp, pyUp = _ptphi_to_pxpy(events.PuppiMET.ptUnclusteredUp,   events.PuppiMET.phiUnclusteredUp)
                pxDn, pyDn = _ptphi_to_pxpy(events.PuppiMET.ptUnclusteredDown, events.PuppiMET.phiUnclusteredDown)
            
                dpx_up = pxUp - px0
                dpy_up = pyUp - py0
                dpx_dn = pxDn - px0
                dpy_dn = pyDn - py0
            
                # apply those deltas on top of your corrected MET
                pt_up, phi_up = _pxpy_to_ptphi(px_corr + dpx_up, py_corr + dpy_up)
                pt_dn, phi_dn = _pxpy_to_ptphi(px_corr + dpx_dn, py_corr + dpy_dn)
            
                PuppiMETCorr_umetUp = ak.with_field(PuppiMETCorr, ak.values_astype(pt_up,  "float32"), "pt")
                PuppiMETCorr_umetUp = ak.with_field(PuppiMETCorr_umetUp, ak.values_astype(phi_up, "float32"), "phi")
            
                PuppiMETCorr_umetDown = ak.with_field(PuppiMETCorr, ak.values_astype(pt_dn,  "float32"), "pt")
                PuppiMETCorr_umetDown = ak.with_field(PuppiMETCorr_umetDown, ak.values_astype(phi_dn, "float32"), "phi")
            
                systs["met"]["_umetup"]   = PuppiMETCorr_umetUp
                systs["met"]["_umetdown"] = PuppiMETCorr_umetDown
        ######################################################
        # S T A R T   T H E   A N A L Y S I S               #
        ######################################################
        
        # STEP0: Raw events
        output["eventflow_boosted"].fill(cut="raw", weight=np.sum(weights.weight()))
        output["eventflow_resolved"].fill(cut="raw", weight=np.sum(weights.weight()))
        output["ee_eventflow_boosted"].fill(cut="raw", weight=np.sum(weights.weight()))
        output["ee_eventflow_resolved"].fill(cut="raw", weight=np.sum(weights.weight()))
        output["mumu_eventflow_boosted"].fill(cut="raw", weight=np.sum(weights.weight()))
        output["mumu_eventflow_resolved"].fill(cut="raw", weight=np.sum(weights.weight()))
        
        # ========== Object Configuration ========== #
        muons = events.Muon[(events.Muon.pt > 10) & (np.abs(events.Muon.eta) < 2.4) & events.Muon.tightId & (events.Muon.pfRelIso04_all < 0.15)]
        electrons = ElectronCorr[(ElectronCorr.pt > 15) & (np.abs(ElectronCorr.eta) < 2.5) & (ElectronCorr.cutBased >= 4) & (ElectronCorr.pfRelIso03_all < 0.15)]
        
        muons = ak.with_field(muons, "mu", "lepton_type")
        electrons = ak.with_field(electrons, "e", "lepton_type")
        leptons = ak.concatenate([muons, electrons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
        n_leptons = ak.num(leptons)
        
        # ----- jets -----
        goodJet = (
            (jets.pt > 20) &
            (np.abs(jets.eta) < 2.5) &
            jets.passJetIdTightLepVeto
        )
        
        single_jets = jets[goodJet]
        double_jets = jets[goodJet]
        
        single_jets = clean_by_dr(single_jets, leptons, 0.4)
        double_jets = clean_by_dr(double_jets, leptons, 0.4)
        
        single_jets = single_jets[ak.argsort(single_jets.btagUParTAK4B, axis=-1, ascending=False)]
        double_jets = double_jets[ak.argsort(double_jets.btagUParTAK4probbb, axis=-1, ascending=False)]
        
        n_single_jets = ak.num(single_jets)
        single_bjets = single_jets[single_jets.btagUParTAK4B >= 0.4648]
        single_untag_jets = single_jets[single_jets.btagUParTAK4B < 0.4648]
        single_untag_jets = single_untag_jets[ak.argsort(single_untag_jets.btagUParTAK4B, axis=-1, ascending=True)]
        n_single_bjets = ak.num(single_bjets)
        
        double_bjets = double_jets[double_jets.btagUParTAK4probbb >= 0.14]
        double_untag_jets = double_jets[double_jets.btagUParTAK4probbb < 0.14]
        double_untag_jets = double_untag_jets[ak.argsort(double_untag_jets.pt, axis=-1, ascending=False)]
        n_double_jets = ak.num(double_jets)
        n_double_bjets = ak.num(double_bjets)
         # ===================== #
        # b-tag efficiencies ε  #
        # ===================== #
        # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/#scale-factor-recommendations-for-event-reweighting
        
        if self.isMC and (self._btag_sf_node is not None):
            # --- choose the jet collection and tagger/WP you want to correct --- #
            jets_for_btag = single_jets                 
            score_field   = "btagUParTAK4B"             
            wp_name       = "T"                         
            thr           = float(self._btag_wp_threshold(wp_name))
        
            pt     = ak.to_numpy(ak.flatten(jets_for_btag.pt))
            abseta = ak.to_numpy(ak.flatten(np.abs(jets_for_btag.eta)))
            flav   = ak.to_numpy(ak.flatten(getattr(jets_for_btag, "hadronFlavour", ak.zeros_like(jets_for_btag.pt))))
            passed = ak.to_numpy(ak.flatten(getattr(jets_for_btag, score_field) >= thr))
        
            
            pt_edges  = np.array([20.0, 30.0, 50.0, 70.0, 100.0, 140.0, 200.0, 300.0, 600.0], dtype=float)
            eta_edges = np.array([0.0, 2.5], dtype=float)
        
            nx, ny = len(pt_edges) - 1, len(eta_edges) - 1
        
            def _eff_table_for_flav(target_flav):
                # digitize into bins
                ix = np.clip(np.digitize(pt, pt_edges, right=False) - 1, 0, nx - 1)
                iy = np.clip(np.digitize(abseta, eta_edges, right=False) - 1, 0, ny - 1)
                mask = (flav == target_flav)
                # accumulators
                den = np.zeros((ny, nx), dtype=np.int64)
                num = np.zeros((ny, nx), dtype=np.int64)
                if mask.any():
                    np.add.at(den, (iy[mask], ix[mask]), 1)
                    np.add.at(num, (iy[mask], ix[mask]), passed[mask].astype(np.int64))
                # raw ε
                with np.errstate(divide="ignore", invalid="ignore"):
                    eff = num / np.maximum(den, 1)  # avoids div by 0
                # handle empty bins
                if mask.any():
                    global_rate = float(np.mean(passed[mask]))
                else:
                    global_rate = 0.0
                eff[den == 0] = global_rate
                # clip to sane range
                eff = np.clip(eff, 1e-6, 1 - 1e-6)
                return eff
        
            eff_b   = _eff_table_for_flav(5)
            eff_c   = _eff_table_for_flav(4)
            eff_udg = _eff_table_for_flav(0)   
        
            # --- lookup ε for each jet --- #
            eff_flat = np.ones_like(pt, dtype=float)
            eff_flat[flav == 5] = _lookup_th2_vals(pt[flav == 5],   abseta[flav == 5],   pt_edges, eta_edges, eff_b)
            eff_flat[flav == 4] = _lookup_th2_vals(pt[flav == 4],   abseta[flav == 4],   pt_edges, eta_edges, eff_c)
            eff_flat[flav == 0] = _lookup_th2_vals(pt[flav == 0],   abseta[flav == 0],   pt_edges, eta_edges, eff_udg)
        
            counts  = ak.num(jets_for_btag.pt, axis=1)
            effs_mc = _unflatten_like(eff_flat, counts)
        
            # --- compute full fixed-WP weight --- #
            w_btag_full = self.btag_event_weight_full(
                jets=jets_for_btag,
                effs_mc=effs_mc,
                working_point=wp_name,
                score_field=score_field,
                systematic="central"
            )
            
        output["n_lep_step0"].fill(n=n_leptons, weight=weights.weight())
        
        #------------------------------------------------------------------------------------------------------------------------------------------#
        # =========================
        # Step 1: SFOS + pT 
        # =========================
        # --- offline thresholds ( ---
        PT_E_LEAD  = 25.0
        PT_E_SUB   = 15.0
        PT_MU_LEAD = 20.0
        PT_MU_SUB  = 10.0

        # need ≥2 leptons 
        has_2lep = ak.num(leptons) >= 2
        if not np.any(ak.to_numpy(has_2lep)):
            return output

        # work only on events with ≥2 leptons 
        leps_ge2 = leptons[has_2lep]
        lead2    = leps_ge2[:, :2]

        sf_small = ak.fill_none(lead2[:, 0].lepton_type == lead2[:, 1].lepton_type, False)
        os_small = ak.fill_none(lead2[:, 0].charge * lead2[:, 1].charge == -1,       False)

        # channel tags for the pair
        ee_pair_small   = ak.fill_none((lead2[:, 0].lepton_type == "e")  & (lead2[:, 1].lepton_type == "e"),  False)
        mumu_pair_small = ak.fill_none((lead2[:, 0].lepton_type == "mu") & (lead2[:, 1].lepton_type == "mu"), False)

        # channel-specific pT thresholds on the same pair
        pt_ee_small   = ak.fill_none((lead2[:, 0].pt > PT_E_LEAD)  & (lead2[:, 1].pt > PT_E_SUB),   False)
        pt_mumu_small = ak.fill_none((lead2[:, 0].pt > PT_MU_LEAD) & (lead2[:, 1].pt > PT_MU_SUB),  False)

        step1_small = sf_small & os_small & ((ee_pair_small & pt_ee_small) | (mumu_pair_small & pt_mumu_small))

        # write back to full-length masks (old style)
        mask_step1 = np.zeros(len(events), dtype=bool)
        mask_ee    = np.zeros(len(events), dtype=bool)
        mask_mumu  = np.zeros(len(events), dtype=bool)

        idx_has2 = np.where(ak.to_numpy(has_2lep))[0]
        mask_step1[idx_has2] = ak.to_numpy(step1_small)

        # per-channel masks (also include SFOS+pT so they’re subsets of mask_step1)
        mask_ee[idx_has2]   = ak.to_numpy(ee_pair_small   & pt_ee_small   & sf_small & os_small)
        mask_mumu[idx_has2] = ak.to_numpy(mumu_pair_small & pt_mumu_small & sf_small & os_small)
        # cutflow: step1
        w_all = weights.weight()
        output["eventflow_boosted"].fill(cut="step1",  weight=np.sum(w_all[mask_step1]))
        output["eventflow_resolved"].fill(cut="step1", weight=np.sum(w_all[mask_step1]))
        if np.any(mask_ee):
            output["ee_eventflow_boosted"].fill(cut="step1",  weight=np.sum(w_all[mask_ee]))
            output["ee_eventflow_resolved"].fill(cut="step1", weight=np.sum(w_all[mask_ee]))
            output["ee_n_lep_step1"].fill(n=n_leptons[mask_ee], weight=w_all[mask_ee])
        if np.any(mask_mumu):
            output["mumu_eventflow_boosted"].fill(cut="step1",  weight=np.sum(w_all[mask_mumu]))
            output["mumu_eventflow_resolved"].fill(cut="step1", weight=np.sum(w_all[mask_mumu]))
            output["mumu_n_lep_step1"].fill(n=n_leptons[mask_mumu], weight=w_all[mask_mumu])

        output["n_lep_step1"].fill(n=n_leptons[mask_step1], weight=w_all[mask_step1])
        # =========================
        # Trigger 
        # =========================
        MU_BM = (1 << 0)  
        E_BM  = (1 << 3)

        trig_word = np.asarray(ak.to_numpy(events.trigger_type)) if "trigger_type" in events.fields else np.zeros(len(events), dtype=np.int64)
        hasTrig_mumu = (trig_word & MU_BM) != 0
        hasTrig_ee   = (trig_word & E_BM)  != 0

        mask_step_trig_mumu = mask_mumu & hasTrig_mumu
        mask_step_trig_ee   = mask_ee   & hasTrig_ee
        mask_step_trig      = mask_step_trig_mumu | mask_step_trig_ee

        # cutflow: trigger
        output["eventflow_boosted"].fill(cut="trigger",  weight=np.sum(w_all[mask_step_trig]))
        output["eventflow_resolved"].fill(cut="trigger", weight=np.sum(w_all[mask_step_trig]))
        if np.any(mask_step_trig_mumu):
            output["mumu_eventflow_boosted"].fill(cut="trigger",  weight=np.sum(w_all[mask_step_trig_mumu]))
            output["mumu_eventflow_resolved"].fill(cut="trigger", weight=np.sum(w_all[mask_step_trig_mumu]))
        if np.any(mask_step_trig_ee):
            output["ee_eventflow_boosted"].fill(cut="trigger",  weight=np.sum(w_all[mask_step_trig_ee]))
            output["ee_eventflow_resolved"].fill(cut="trigger", weight=np.sum(w_all[mask_step_trig_ee]))

        # =========================
        # Z window on the same pair
        # =========================
        leps_trig = leptons[mask_step_trig]                   
        mll = (make_vector(leps_trig[:, 0]) + make_vector(leps_trig[:, 1])).mass
        output["mass_Z_bef"].fill(m_ll=mll, weight=w_all[mask_step_trig])
        in_zwin = ak.to_numpy((mll > 80.0) & (mll < 100.0))

        mask_mll = np.zeros(len(events), dtype=bool)
        mask_mll[np.where(mask_step_trig)[0]] = in_zwin

        mask_step_trig_z      = mask_step_trig      & mask_mll
        mask_step_trig_z_ee   = mask_step_trig_ee   & mask_mll
        mask_step_trig_z_mumu = mask_step_trig_mumu & mask_mll

        # cutflow: zwindow
        output["eventflow_boosted"].fill(cut="zwindow",  weight=np.sum(w_all[mask_step_trig_z]))
        output["eventflow_resolved"].fill(cut="zwindow", weight=np.sum(w_all[mask_step_trig_z]))
        
        output["n_jets_bef"].fill(n=n_single_jets[mask_step_trig_z], weight=w_all[mask_step_trig_z])
        if np.any(mask_step_trig_z_ee):
            output["ee_eventflow_boosted"].fill(cut="zwindow",  weight=np.sum(w_all[mask_step_trig_z_ee]))
            output["ee_eventflow_resolved"].fill(cut="zwindow", weight=np.sum(w_all[mask_step_trig_z_ee]))
        if np.any(mask_step_trig_z_mumu):
            output["mumu_eventflow_boosted"].fill(cut="zwindow",  weight=np.sum(w_all[mask_step_trig_z_mumu]))
            output["mumu_eventflow_resolved"].fill(cut="zwindow", weight=np.sum(w_all[mask_step_trig_z_mumu]))

        # =========================
        # pT histos of leading/subleading leptons
        # (before vs after trigger), per channel
        # =========================
        def _lead_sub_pt(pair2):
        # pair2 is leptons[:, :2] for selected events
         return pair2[:, 0].pt, pair2[:, 1].pt

        # Make sure masks are NumPy arrays (full-length, event-level)
        ms1_ee = np.asarray(mask_ee)                # passed Step 1 (ee)
        mtr_ee = np.asarray(mask_step_trig_ee)      # passed Step 1 + trigger (ee)
        ms1_mm = np.asarray(mask_mumu)              # passed Step 1 (mumu)
        mtr_mm = np.asarray(mask_step_trig_mumu)    # passed Step 1 + trigger (mumu)

        # Weights for all events (slice per mask below)
        w_all = weights.weight()

        # ---------- ee channel ----------
        # BEFORE trigger (after Step 1)
        if np.any(ms1_ee):
            ee_pairs_pre = leptons[ms1_ee][:, :2]  
            pt0, pt1 = _lead_sub_pt(ee_pairs_pre)
            w = w_all[ms1_ee]
            output["ee_pt_lead_pretrig"].fill(pt_pretrig=pt0, weight=w)
            output["ee_pt_sub_pretrig"].fill( pt_pretrig=pt1, weight=w)

        # AFTER trigger
        if np.any(mtr_ee):
            ee_pairs_post = leptons[mtr_ee][:, :2]
            pt0, pt1 = _lead_sub_pt(ee_pairs_post)
            w = w_all[mtr_ee]
            output["ee_pt_lead_posttrig"].fill(pt_posttrig=pt0, weight=w)
            output["ee_pt_sub_posttrig"].fill( pt_posttrig=pt1, weight=w)

        # ---------- mumu channel ----------
        # BEFORE trigger (after Step 1)
        if np.any(ms1_mm):
            mm_pairs_pre = leptons[ms1_mm][:, :2]
            pt0, pt1 = _lead_sub_pt(mm_pairs_pre)
            w = w_all[ms1_mm]
            output["mumu_pt_lead_pretrig"].fill(pt_pretrig=pt0, weight=w)
            output["mumu_pt_sub_pretrig"].fill(pt_pretrig=pt1, weight=w)

        # AFTER trigger
        if np.any(mtr_mm):
            mm_pairs_post = leptons[mtr_mm][:, :2]
            pt0, pt1 = _lead_sub_pt(mm_pairs_post)
            w = w_all[mtr_mm]
            output["mumu_pt_lead_posttrig"].fill(pt_posttrig=pt0, weight=w)
            output["mumu_pt_sub_posttrig"].fill( pt_posttrig=pt1, weight=w)
        #============================================#
        #                                           #
        # Boosted analysis: ma: 12, 15, 20, 25, 30  #
        #                                           #
        #============================================#
        
       #============================================#

        #######################################
        # STEP 2a: At least 2 double AK4 jets #
        #######################################
        mask_step2a = mask_step_trig_z & (n_double_jets >= 2)
        w2a = weights.weight()[mask_step2a]

        
        output["eventflow_boosted"].fill(cut="step2", weight=np.sum(w2a))
        output["n_bjets_double_bef"].fill(n=n_double_bjets[mask_step2a], weight=w2a)

        # slice boosted collections at step2a
        double_jets_2a       = double_jets[mask_step2a]
        double_bjets_2a      = double_bjets[mask_step2a]
        double_untag_jets_2a = double_untag_jets[mask_step2a]

        # b-tag score distributions of leading/subleading double-AK4 jets
        has_ge1_db = ak.num(double_jets_2a) >= 1
        has_ge2_db = ak.num(double_jets_2a) >= 2

        if ak.any(has_ge1_db):
            lead_db_score = ak.to_numpy(ak.fill_none(double_jets_2a[has_ge1_db][:, 0].btagUParTAK4probbb, 0.0))
            w_lead_db     = ak.to_numpy(w2a[has_ge1_db])
            output["double_btag_score_lead"].fill(score=lead_db_score, weight=w_lead_db)

        if ak.any(has_ge2_db):
            sublead_db_score = ak.to_numpy(ak.fill_none(double_jets_2a[has_ge2_db][:, 1].btagUParTAK4probbb, 0.0))
            w_sublead_db     = ak.to_numpy(w2a[has_ge2_db])
            output["double_btag_score_sublead"].fill(score=sublead_db_score, weight=w_sublead_db)

        # per-channel cutflow at step2
        idx2a      = np.where(mask_step2a)[0]
        mu_mask_2a = mask_step_trig_z_mumu[idx2a] if 'mask_step_trig_z_mumu' in locals() else np.zeros(len(idx2a), dtype=bool)
        ee_mask_2a = mask_step_trig_z_ee[idx2a]   if 'mask_step_trig_z_ee'   in locals() else np.zeros(len(idx2a), dtype=bool)
        if np.any(mu_mask_2a):
            output["mumu_eventflow_boosted"].fill(cut="step2", weight=np.sum(w2a[mu_mask_2a]))
            output["mumu_n_bjets_double_bef"].fill(n=n_double_bjets[mask_step2a][mu_mask_2a], weight=w2a[mu_mask_2a])
        if np.any(ee_mask_2a):
            output["ee_eventflow_boosted"].fill(cut="step2", weight=np.sum(w2a[ee_mask_2a]))
            output["ee_n_bjets_double_bef"].fill(n=n_double_bjets[mask_step2a][ee_mask_2a], weight=w2a[ee_mask_2a])

        #################################################
        # STEP 3a: At least 2 double **b-tag** AK4 jets #
        #################################################
        full_mask_double     = mask_step_trig_z & (n_double_bjets >= 2)

        # slice step3 (pre-tight-btag) collections
        weights_boosted      = weights.weight()[full_mask_double]
        double_bjets_boosted = double_bjets[full_mask_double]
        double_jets_boosted  = double_jets[full_mask_double]
        double_untag_boosted = double_untag_jets[full_mask_double]

        # cutflow step3 (pre-tight-btag) + basic counters
        output["eventflow_boosted"].fill(cut="step3", weight=np.sum(weights_boosted))
        output["n_bjets_bef_boosted"].fill(n=n_double_bjets[full_mask_double], weight=weights_boosted)

        n_jets_boo     = ak.num(double_jets_boosted)
        n_untagged_boo = ak.num(double_untag_boosted)
        output["n_jets_bef_boosted"].fill(   n=n_jets_boo,     weight=weights_boosted)
        output["n_untag_bef_boosted"].fill(  n=n_untagged_boo, weight=weights_boosted)

        met_boosted = PuppiMETCorr[full_mask_double]
        ht_boosted  = ak.sum(double_jets_boosted.pt, axis=1)

        # channel masks aligned to step3 slice
        idx_boosted      = np.where(full_mask_double)[0]
        mumu_mask_double = np.asarray(mask_step_trig_z_mumu[idx_boosted]) if 'mask_step_trig_z_mumu' in locals() else np.zeros(len(idx_boosted), dtype=bool)
        ee_mask_double   = np.asarray(mask_step_trig_z_ee[idx_boosted])   if 'mask_step_trig_z_ee'   in locals() else np.zeros(len(idx_boosted), dtype=bool)

        # ---  b-tag in boosted: lead >= 0.38, sublead >= 0.14 ---
        lead_btag = ak.fill_none(double_bjets_boosted[:, 0].btagUParTAK4probbb, 0.0)
        sub_btag  = ak.fill_none(double_bjets_boosted[:, 1].btagUParTAK4probbb, 0.0)
        pass_btag_boosted_local = (lead_btag >= 0.38) & (sub_btag >= 0.14)  
        pass_np = ak.to_numpy(pass_btag_boosted_local)                     

       
        mask_btag_boosted = np.zeros(len(events), dtype=bool)
        mask_btag_boosted[idx_boosted] = pass_np

       
        output["eventflow_boosted"].fill(cut="btag", weight=np.sum(weights.weight()[mask_btag_boosted]))

        weights_boosted_sel      =        weights_boosted[pass_np]
        double_bjets_boosted_sel =   double_bjets_boosted[pass_np]
        double_jets_boosted_sel  =    double_jets_boosted[pass_np]
        double_untag_boosted_sel = double_untag_boosted[pass_np]
        met_boosted_sel          =          met_boosted[pass_np]
        ht_boosted_sel           =           ht_boosted[pass_np]

        mu_after_sel = mumu_mask_double[pass_np] 
        ee_after_sel =   ee_mask_double[pass_np]

        btag_max_boosted = double_bjets_boosted_sel[:, 0].btagUParTAK4probbb
        btag_min_boosted = double_bjets_boosted_sel[:, 1].btagUParTAK4probbb
        output["btag_max_boosted"].fill( btag=btag_max_boosted,                      weight=weights_boosted_sel)
        output["btag_min_boosted"].fill( btag=btag_min_boosted,                      weight=weights_boosted_sel)
        output["btag_prod_boosted"].fill(btag=btag_min_boosted * btag_max_boosted,   weight=weights_boosted_sel)

        lead_bb    = double_bjets_boosted_sel[:, 0]
        sublead_bb = double_bjets_boosted_sel[:, 1]
        output["pt_b1_boosted"].fill(pt_b1=lead_bb.pt,    weight=weights_boosted_sel)
        output["pt_b2_boosted"].fill(pt_b2=sublead_bb.pt, weight=weights_boosted_sel)

        # build 4-vectors
        lead_bb_vec    = make_vector(lead_bb)
        sublead_bb_vec = make_vector(sublead_bb)
        higgs_boost    = lead_bb_vec + sublead_bb_vec

        output["mass_H_boosted"].fill(m_H=higgs_boost.mass, weight=weights_boosted_sel)
        output["pt_H_boosted"].fill(  pt_H=higgs_boost.pt,   weight=weights_boosted_sel)
        output["eta_H_boosted"].fill( eta=higgs_boost.eta,  weight=weights_boosted_sel)

        dm_boosted = np.abs(lead_bb.mass - sublead_bb.mass)
        output["dm_bb_bb_min_boosted"].fill(dm=dm_boosted, weight=weights_boosted_sel)
        output["dR_bb_bb_ave_boosted"].fill(dr=lead_bb_vec.delta_r(sublead_bb_vec), weight=weights_boosted_sel)

        # leptons (selected)
        leptons_boosted_sel = leptons[full_mask_double][pass_np]
        lead_lep_boo = leptons_boosted_sel[:, 0]
        sub_lep_boo  = leptons_boosted_sel[:, 1]
        vec_lead_lep_boo = make_vector(lead_lep_boo)
        vec_sub_lep_boo  = make_vector(sub_lep_boo)
        dilepton_boosted  = vec_lead_lep_boo + vec_sub_lep_boo

        output["pt_ll_boosted"].fill(  pt_ll=dilepton_boosted.pt,    weight=weights_boosted_sel)
        output["mass_ll_boosted"].fill(m_ll=dilepton_boosted.mass,   weight=weights_boosted_sel)
        output["dr_ll_boosted"].fill(    dr=vec_lead_lep_boo.delta_r(vec_sub_lep_boo), weight=weights_boosted_sel)

        # ratios / angles
        eps = 1e-6
        Z_pt_safe_boo = ak.where(dilepton_boosted.pt == 0, eps, dilepton_boosted.pt)
        pt_ratio_boo  = higgs_boost.pt / Z_pt_safe_boo
        output["pt_ratio_boosted"].fill(ratio=pt_ratio_boo, weight=weights_boosted_sel)
        output["dphi_HZ_boosted"].fill(dphi=np.abs(higgs_boost.delta_phi(dilepton_boosted)), weight=weights_boosted_sel)
        output["deta_HZ_boosted"].fill(deta=delta_eta_vec(higgs_boost, dilepton_boosted),    weight=weights_boosted_sel)
        output["dr_HZ_boosted"].fill(    dr=higgs_boost.delta_r(dilepton_boosted),           weight=weights_boosted_sel)

        # proxy jet for Δφ and pt
        lead_untag_boo = ak.firsts(double_untag_boosted_sel)
        fallback_bjet  = ak.firsts(double_bjets_boosted_sel[
            ak.argmin(double_bjets_boosted_sel.btagUParTAK4probbb, axis=1, keepdims=True)
        ])
        has_untag_boo  = ak.num(double_untag_boosted_sel) > 0
        proxy_jet_boo  = ak.where(has_untag_boo, lead_untag_boo, fallback_bjet)

        dphi_proxy_Z_boo = delta_phi_raw(proxy_jet_boo.phi, dilepton_boosted.phi)
        pt_proxy_boo     = proxy_jet_boo.pt

        # ---------- per-channel histograms AFTER asymmetric b-tag ----------
        # (use *_sel arrays with *_after_sel masks and weights_boosted_sel)

        if np.any(mu_after_sel):
            w_mu = weights_boosted_sel[mu_after_sel]
            output["mumu_eventflow_boosted"].fill(cut="btag", weight=np.sum(w_mu))

            output["mumu_met_boosted"].fill(met=met_boosted_sel[mu_after_sel].pt, weight=w_mu)
            output["mumu_HT_boosted"].fill( ht=ht_boosted_sel[mu_after_sel],      weight=w_mu)

            output["mumu_btag_max_boosted"].fill(btag=btag_max_boosted[mu_after_sel], weight=w_mu)
            output["mumu_btag_min_boosted"].fill(btag=btag_min_boosted[mu_after_sel], weight=w_mu)
            output["mumu_btag_prod_boosted"].fill(btag=(btag_min_boosted * btag_max_boosted)[mu_after_sel], weight=w_mu)

            output["mumu_mass_H_boosted"].fill(m_H=higgs_boost[mu_after_sel].mass, weight=w_mu)
            output["mumu_pt_H_boosted"].fill(  pt_H=higgs_boost[mu_after_sel].pt,  weight=w_mu)
            output["mumu_eta_H_boosted"].fill( eta=higgs_boost[mu_after_sel].eta,  weight=w_mu)

            output["mumu_pt_b1_boosted"].fill(pt_b1=lead_bb[mu_after_sel].pt,    weight=w_mu)
            output["mumu_pt_b2_boosted"].fill(pt_b2=sublead_bb[mu_after_sel].pt, weight=w_mu)

            dm_boosted_mu = np.abs(lead_bb[mu_after_sel].mass - sublead_bb[mu_after_sel].mass)
            output["mumu_dm_bb_bb_min_boosted"].fill(dm=dm_boosted_mu, weight=w_mu)

            output["mumu_pt_Z_boosted"].fill(  pt_ll=dilepton_boosted[mu_after_sel].pt,   weight=w_mu)
            output["mumu_mass_Z_boosted"].fill(m_ll=dilepton_boosted[mu_after_sel].mass,  weight=w_mu)

            output["mumu_pt_ratio_boosted"].fill(ratio=pt_ratio_boo[mu_after_sel], weight=w_mu)
            output["mumu_dphi_HZ_boosted"].fill(dphi=np.abs(higgs_boost[mu_after_sel].delta_phi(dilepton_boosted[mu_after_sel])), weight=w_mu)
            output["mumu_deta_HZ_boosted"].fill(deta=delta_eta_vec(higgs_boost[mu_after_sel], dilepton_boosted[mu_after_sel]),     weight=w_mu)
            output["mumu_dr_HZ_boosted"].fill(  dr=higgs_boost[mu_after_sel].delta_r(dilepton_boosted[mu_after_sel]),              weight=w_mu)

            output["mumu_dr_ll_boosted"].fill(  dr=vec_lead_lep_boo[mu_after_sel].delta_r(vec_sub_lep_boo[mu_after_sel]),          weight=w_mu)
            output["mumu_dR_bb_bb_ave_boosted"].fill(dr=lead_bb_vec.delta_r(sublead_bb_vec)[mu_after_sel], weight=w_mu)

            output["mumu_n_untag_boosted"].fill(n=ak.to_numpy(ak.num(double_untag_boosted_sel[mu_after_sel])), weight=w_mu)
            output["mumu_n_jets_boosted"].fill( n=ak.to_numpy(ak.num(double_jets_boosted_sel[mu_after_sel], axis=1)), weight=w_mu)
            output["mumu_n_bjets_boosted"].fill(n=ak.to_numpy(ak.num(double_bjets_boosted_sel[mu_after_sel], axis=1)), weight=w_mu)

        if np.any(ee_after_sel):
            w_ee = weights_boosted_sel[ee_after_sel]
            output["ee_eventflow_boosted"].fill(cut="btag", weight=np.sum(w_ee))

            output["ee_met_boosted"].fill(met=met_boosted_sel[ee_after_sel].pt, weight=w_ee)
            output["ee_HT_boosted"].fill( ht=ht_boosted_sel[ee_after_sel],      weight=w_ee)

            output["ee_btag_max_boosted"].fill(btag=btag_max_boosted[ee_after_sel], weight=w_ee)
            output["ee_btag_min_boosted"].fill(btag=btag_min_boosted[ee_after_sel], weight=w_ee)
            output["ee_btag_prod_boosted"].fill(btag=(btag_min_boosted * btag_max_boosted)[ee_after_sel], weight=w_ee)

            output["ee_mass_H_boosted"].fill(m_H=higgs_boost[ee_after_sel].mass, weight=w_ee)
            output["ee_pt_H_boosted"].fill(  pt_H=higgs_boost[ee_after_sel].pt,  weight=w_ee)
            output["ee_eta_H_boosted"].fill( eta=higgs_boost[ee_after_sel].eta,  weight=w_ee)

            output["ee_pt_b1_boosted"].fill(pt_b1=lead_bb[ee_after_sel].pt,    weight=w_ee)
            output["ee_pt_b2_boosted"].fill(pt_b2=sublead_bb[ee_after_sel].pt, weight=w_ee)

            dm_boosted_ee = np.abs(lead_bb[ee_after_sel].mass - sublead_bb[ee_after_sel].mass)
            output["ee_dm_bb_bb_min_boosted"].fill(dm=dm_boosted_ee, weight=w_ee)

            output["ee_pt_Z_boosted"].fill(  pt_ll=dilepton_boosted[ee_after_sel].pt,   weight=w_ee)
            output["ee_mass_Z_boosted"].fill(m_ll=dilepton_boosted[ee_after_sel].mass,  weight=w_ee)

            output["ee_pt_ratio_boosted"].fill(ratio=pt_ratio_boo[ee_after_sel], weight=w_ee)
            output["ee_dphi_HZ_boosted"].fill(dphi=np.abs(higgs_boost[ee_after_sel].delta_phi(dilepton_boosted[ee_after_sel])), weight=w_ee)
            output["ee_deta_HZ_boosted"].fill(deta=delta_eta_vec(higgs_boost[ee_after_sel], dilepton_boosted[ee_after_sel]),     weight=w_ee)
            output["ee_dr_HZ_boosted"].fill(  dr=higgs_boost[ee_after_sel].delta_r(dilepton_boosted[ee_after_sel]),              weight=w_ee)

            output["ee_n_untag_boosted"].fill(n=ak.to_numpy(ak.num(double_untag_boosted_sel[ee_after_sel])), weight=w_ee)
            output["ee_n_jets_boosted"].fill( n=ak.to_numpy(ak.num(double_jets_boosted_sel[ee_after_sel], axis=1)), weight=w_ee)
            output["ee_n_bjets_boosted"].fill(n=ak.to_numpy(ak.num(double_bjets_boosted_sel[ee_after_sel], axis=1)), weight=w_ee)
            output["ee_dr_ll_boosted"].fill(  dr=vec_lead_lep_boo[ee_after_sel].delta_r(vec_sub_lep_boo[ee_after_sel]),          weight=w_ee)
            output["ee_dR_bb_bb_ave_boosted"].fill(dr=lead_bb_vec.delta_r(sublead_bb_vec)[ee_after_sel], weight=w_ee)

        # ---------- BDT inputs (selected events only) ----------
        bdt_boosted = {
            "H_mass":         ak.to_numpy(higgs_boost.mass),
            "H_pt":           ak.to_numpy(higgs_boost.pt),
            "H_eta":          ak.to_numpy(higgs_boost.eta),
            "Z_pt":           ak.to_numpy(dilepton_boosted.pt),
            "HT":             ak.to_numpy(ht_boosted_sel),
            "pt_ratio":       ak.to_numpy(pt_ratio_boo),
            "puppimet_pt":    ak.to_numpy(met_boosted_sel.pt),
            "btag_max":       ak.to_numpy(btag_max_boosted),
            "btag_min":       ak.to_numpy(btag_min_boosted),
            "btag_prod":      ak.to_numpy(btag_min_boosted * btag_max_boosted),
            "dr_bb_bb_ave":   ak.to_numpy(lead_bb_vec.delta_r(sublead_bb_vec)),
            "dm_bb_bb_min":   ak.to_numpy(dm_boosted),
            "dphi_HZ":        ak.to_numpy(np.abs(higgs_boost.delta_phi(dilepton_boosted))),
            "deta_HZ":        ak.to_numpy(delta_eta_vec(higgs_boost, dilepton_boosted)),
            "dr_HZ":          ak.to_numpy(higgs_boost.delta_r(dilepton_boosted)),
            "pt_untag_max":   ak.to_numpy(ak.fill_none(pt_proxy_boo, np.nan)),
            "dphi_untag_Z":   ak.to_numpy(np.abs(dphi_proxy_Z_boo)),
            "n_jets":         ak.to_numpy(n_jets_boo[pass_np]),
            "n_untag":        ak.to_numpy(n_untagged_boo[pass_np]),
            "dr_ll":          ak.to_numpy(vec_lead_lep_boo.delta_r(vec_sub_lep_boo)),
            "weight":         ak.to_numpy(weights_boosted_sel),
        }

        # ----- tree filling on selected boosted events -----
        if self.isMVA:
            self.compat_tree_variables(bdt_boosted)
            self.add_tree_entry("boosted", bdt_boosted)

        # ----- BDT evaluation on selected boosted events -----
        if self.run_eval and not self.isMVA:
            inputs_boosted = {k: bdt_boosted[k] for k in self.bdt_eval_boosted.var_list}
            bdt_score_boosted = np.ravel(self.bdt_eval_boosted.eval(inputs_boosted))

            output["bdt_score_boosted"].fill(bdt=bdt_score_boosted, weight=bdt_boosted["weight"])

            # Use the COMPRESSED masks for the selected slice
            e_mask_all  = ee_after_sel
            mu_mask_all = mu_after_sel
            w_all       = bdt_boosted["weight"]

            for i, cut in enumerate(self.optim_Cuts1_bdt):
                cut_mask = (bdt_score_boosted > cut)
                if not np.any(cut_mask):
                    continue

                ele_mask_cut = e_mask_all  & cut_mask
                mu_mask_cut  = mu_mask_all & cut_mask

                if np.any(ele_mask_cut):
                    w_e = w_all[ele_mask_cut]
                    s_e = bdt_score_boosted[ele_mask_cut]
                    output["ee_bdt_score_boosted"].fill(bdt=s_e, weight=w_e)
                    output["ee_bdt_shapes_boosted"].fill(cut_index=i, bdt=s_e, weight=w_e)
                    output["ee_higgsMass_shapes_boosted"].fill(cut_index=i, m_H=ak.to_numpy(higgs_boost.mass[ele_mask_cut]), weight=w_e)
                    output["ee_higgsPt_shapes_boosted"].fill(  cut_index=i, pt_H=ak.to_numpy(higgs_boost.pt[ele_mask_cut]),   weight=w_e)
                    output["ee_higgsEta_shapes_boosted"].fill( cut_index=i, eta=ak.to_numpy(higgs_boost.eta[ele_mask_cut]),    weight=w_e)
                    output["ee_ht_shapes_boosted"].fill(       cut_index=i, ht=ak.to_numpy(ht_boosted_sel[ele_mask_cut]),      weight=w_e)
                    output["ee_met_shapes_boosted"].fill(      cut_index=i, met=ak.to_numpy(met_boosted_sel.pt[ele_mask_cut]), weight=w_e)
                    output["ee_ptZ_shapes_boosted"].fill(      cut_index=i, pt=ak.to_numpy(dilepton_boosted.pt[ele_mask_cut]), weight=w_e)
                    output["ee_dphiZh_shapes_boosted"].fill(   cut_index=i, dphi=ak.to_numpy(np.abs(higgs_boost.delta_phi(dilepton_boosted)[ele_mask_cut])), weight=w_e)
                    output["ee_detaZh_shapes_boosted"].fill(   cut_index=i, deta=ak.to_numpy(delta_eta_vec(higgs_boost, dilepton_boosted)[ele_mask_cut]),     weight=w_e)
                    output["ee_dRbb_shapes_boosted"].fill(     cut_index=i, dr=ak.to_numpy(lead_bb_vec.delta_r(sublead_bb_vec)[ele_mask_cut]),                weight=w_e)
                    output["ee_dm_shapes_boosted"].fill(       cut_index=i, dm=ak.to_numpy(dm_boosted[ele_mask_cut]),                                         weight=w_e)
                    output["ee_dRZh_shapes_boosted"].fill(     cut_index=i, dr=ak.to_numpy(higgs_boost.delta_r(dilepton_boosted)[ele_mask_cut]),              weight=w_e)
                    output["ee_ptratio_shapes_boosted"].fill(  cut_index=i, ratio=ak.to_numpy(pt_ratio_boo[ele_mask_cut]),                                    weight=w_e)
                    output["ee_Njets_shapes_boosted"].fill(    cut_index=i, n_jets=ak.to_numpy(ak.num(double_jets_boosted_sel[ele_mask_cut], axis=1)),        weight=w_e)
                    output["ee_btag_min_shapes_boosted"].fill( cut_index=i, btag=ak.to_numpy(btag_min_boosted[ele_mask_cut]),                                 weight=w_e)

                if np.any(mu_mask_cut):
                    w_mu = w_all[mu_mask_cut]
                    s_mu = bdt_score_boosted[mu_mask_cut]
                    output["mumu_bdt_score_boosted"].fill(bdt=s_mu, weight=w_mu)
                    output["mumu_bdt_shapes_boosted"].fill(cut_index=i, bdt=s_mu, weight=w_mu)
                    output["mumu_higgsMass_shapes_boosted"].fill(cut_index=i, m_H=ak.to_numpy(higgs_boost.mass[mu_mask_cut]), weight=w_mu)
                    output["mumu_higgsPt_shapes_boosted"].fill(  cut_index=i, pt_H=ak.to_numpy(higgs_boost.pt[mu_mask_cut]),   weight=w_mu)
                    output["mumu_higgsEta_shapes_boosted"].fill( cut_index=i, eta=ak.to_numpy(higgs_boost.eta[mu_mask_cut]),    weight=w_mu)
                    output["mumu_ht_shapes_boosted"].fill(       cut_index=i, ht=ak.to_numpy(ht_boosted_sel[mu_mask_cut]),      weight=w_mu)
                    output["mumu_met_shapes_boosted"].fill(      cut_index=i, met=ak.to_numpy(met_boosted_sel.pt[mu_mask_cut]), weight=w_mu)
                    output["mumu_ptZ_shapes_boosted"].fill(      cut_index=i, pt_ll=ak.to_numpy(dilepton_boosted.pt[mu_mask_cut]), weight=w_mu)
                    output["mumu_dphiZh_shapes_boosted"].fill(   cut_index=i, dphi=ak.to_numpy(np.abs(higgs_boost.delta_phi(dilepton_boosted)[mu_mask_cut])),  weight=w_mu)
                    output["mumu_detaZh_shapes_boosted"].fill(   cut_index=i, deta=ak.to_numpy(delta_eta_vec(higgs_boost, dilepton_boosted)[mu_mask_cut]),      weight=w_mu)
                    output["mumu_dRbb_shapes_boosted"].fill(     cut_index=i, dr=ak.to_numpy(lead_bb_vec.delta_r(sublead_bb_vec)[mu_mask_cut]),                 weight=w_mu)
                    output["mumu_dm_shapes_boosted"].fill(       cut_index=i, dm=ak.to_numpy(dm_boosted[mu_mask_cut]),                                          weight=w_mu)
                    output["mumu_dRZh_shapes_boosted"].fill(     cut_index=i, dr=ak.to_numpy(higgs_boost.delta_r(dilepton_boosted)[mu_mask_cut]),               weight=w_mu)
                    output["mumu_ptratio_shapes_boosted"].fill(  cut_index=i, pt_ratio=ak.to_numpy(pt_ratio_boo[mu_mask_cut]),                                  weight=w_mu)
                    output["mumu_Njets_shapes_boosted"].fill(    cut_index=i, n_jets=ak.to_numpy(ak.num(double_jets_boosted_sel[mu_mask_cut], axis=1)),         weight=w_mu)
                    output["mumu_btag_min_shapes_boosted"].fill( cut_index=i, btag=ak.to_numpy(btag_min_boosted[mu_mask_cut]),                                   weight=w_mu)

                                
        #=====================================================#
        #                                                   #
        # Resolved analysis: ma: 30, 35, 40, 45, 50, 55, 60 #
        #                                                   #
        #=====================================================#
        
        #######################################
        # STEP 2b: At least 3 single AK4 jets #
        #######################################
        print("\nStarting STEP 2b: At least 3 single AK4 jets")
        mask_step2b = mask_step_trig_z & (n_single_jets >= 3)
        print(f"After STEP 2b: {np.sum(mask_step2b)} events remaining")
        
        w_btag_evt = np.ones(len(events), dtype="float64")
        
        # ---- b-tag SFs for RESOLVED only ---- #
        if self.isMC and (self._btag_sf_node is not None):
            w_btag_all = self.btag_event_weight_full(
            jets=single_jets,
            effs_mc=effs_mc,                 
            working_point="T",
            score_field="btagUParTAK4B",
            systematic="central",
        )
            w_btag_all = self.btag_event_weight_tagged_only(
            jets=single_jets,
            working_point="T",
            score_field="btagUParTAK4B",
            systematic="central",
        )
        
        weights.add("btag_UParTAK4B_T_resolved", w_btag_evt)

        w2b = weights.weight()[np.asarray(mask_step2b)]
        single_jets_2b = single_jets[mask_step2b]
        output["n_single_jets_bef"].fill(n=ak.num(single_jets_2b), weight=weights.weight()[mask_step2b])
        
        n_single_jets_2b = ak.num(single_jets_2b)
        single_untag_jets_2b = single_untag_jets[mask_step2b]
        n_single_untag_jets_2b = ak.num(single_untag_jets_2b)
        single_bjets_2b = single_bjets[mask_step2b]
        n_single_bjets_2b = ak.num(single_bjets_2b)
        output["n_bjets_single_bef"].fill(n=n_single_bjets_2b, weight=weights.weight()[mask_step2b])
        has_ge1_sj = ak.num(single_jets_2b) >= 1
        lead_sj_score = ak.to_numpy(single_jets_2b[has_ge1_sj][:, 0].btagUParTAK4B)
        w_lead_sj = ak.to_numpy(w2b[has_ge1_sj])
        
        has_ge2_sj = ak.num(single_jets_2b) >= 2
        sublead_sj_score = ak.to_numpy(single_jets_2b[has_ge2_sj][:, 1].btagUParTAK4B)
        w_sublead_sj = ak.to_numpy(w2b[has_ge2_sj])
        
        # Histogram plotting
        output["single_btag_score_lead"].fill(score=lead_sj_score, weight=w_lead_sj)
        output["single_btag_score_sublead"].fill(score=sublead_sj_score, weight=w_sublead_sj)
        
        output["eventflow_resolved"].fill(cut="step2", weight=np.sum(w2b))
        
       # local μ/e masks aligned to the STEP 2b slice
        idx2b     = np.where(mask_step2b)[0]
        mu_mask_2b = mask_step_trig_z_mumu[idx2b]
        ee_mask_2b = mask_step_trig_z_ee[idx2b]

        if np.any(mu_mask_2b):
            output["mumu_eventflow_resolved"].fill(cut="step2", weight=np.sum(w2b[mu_mask_2b]))
            output["mumu_n_bjets_single_bef"].fill(n=n_single_bjets_2b[mu_mask_2b], weight=w2b[mu_mask_2b])
        if np.any(ee_mask_2b):
            output["ee_eventflow_resolved"].fill(cut="step2", weight=np.sum(w2b[ee_mask_2b]))
            output["ee_n_bjets_single_bef"].fill(n=n_single_bjets_2b[ee_mask_2b], weight=w2b[ee_mask_2b])
        #############################################
        # STEP 3b: At least 3 single b-tag AK4 jets #
        #############################################
        full_mask_res = mask_step_trig_z & (n_single_bjets >= 3)
        output["eventflow_resolved"].fill(cut="step3", weight=np.sum(weights.weight()[full_mask_res]))
        
        weights_res = weights.weight()[full_mask_res]
        
        # Filtered objects
        single_bjets_resolved = single_bjets[full_mask_res]
        single_jets_resolved = single_jets[full_mask_res]
        single_untag_jets_resolved = single_untag_jets[full_mask_res]
        
        vec_single_bjets_resolved = make_vector(single_bjets_resolved)
        vec_single_jets_resolved = make_vector(single_jets_resolved)
        
        n_jets_res = ak.num(single_jets_resolved)
        n_bjets_res = ak.num(single_bjets_resolved)
        n_untagged_res = ak.num(single_untag_jets_resolved)
        
        output["n_untag_resolved"].fill(n=n_untagged_res, weight=weights_res)
        output["n_jets_resolved"].fill(n=n_jets_res, weight=weights_res)
        output["n_bjets_resolved"].fill(n=n_bjets_res, weight=weights_res)
        ##leptons
        leptons_res = leptons[full_mask_res]
        lead_lep_res = leptons_res[:, 0]
        sub_lep_res = leptons_res[:, 1]
        
        vec_lead_lep_res = make_vector(lead_lep_res)
        vec_sub_lep_res = make_vector(sub_lep_res)
        dilepton_res = vec_lead_lep_res + vec_sub_lep_res
        
        output["dr_ll_resolved"].fill(dr=vec_lead_lep_res.delta_r(vec_sub_lep_res), weight=weights_res)
        output["pt_Z_resolved"].fill(pt_ll=dilepton_res.pt, weight=weights_res)
        output["mass_Z_resolved"].fill(m_ll=dilepton_res.mass, weight=weights_res)
        
        ### higgs dependent
        mass_H, pt_H, phi_H, eta_H = higgs_kin(vec_single_bjets_resolved, vec_single_jets_resolved)
        vec_H_res = ak.zip({
            "pt": pt_H,
            "eta": eta_H,
            "phi": phi_H,
            "mass": mass_H,
        }, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
        
        output["mass_H_resolved"].fill(m_H=mass_H, weight=weights_res)
        output["pt_H_resolved"].fill(pt_H=pt_H, weight=weights_res)
        output["eta_H_resolved"].fill(eta=eta_H, weight=weights_res)
        
        output["dphi_HZ_resolved"].fill(dphi=np.abs(delta_phi_raw(phi_H, dilepton_res.phi)), weight=weights_res)
        
        Z_pt_safe_res = ak.where(dilepton_res.pt == 0, eps, dilepton_res.pt)
        pt_ratio_res = vec_H_res.pt / Z_pt_safe_res
        output["pt_ratio_resolved"].fill(ratio=pt_ratio_res, weight=weights_res)
        
        output["dr_HZ_resolved"].fill(dr=vec_H_res.delta_r(dilepton_res), weight=weights_res)
        output["deta_HZ_resolved"].fill(deta=delta_eta_vec(vec_H_res,dilepton_res), weight=weights_res)
        
        # --- Quantities independent of Higgs definition ---
        output["dm_bb_bb_min_resolved"].fill(
            dm=min_dm_bb_bb(vec_single_bjets_resolved, all_jets=vec_single_jets_resolved),
            weight=weights_res
        )
        
        output["dr_bb_bb_ave_resolved"].fill(
            dr=dr_bb_bb_avg(vec_single_bjets_resolved, all_jets=vec_single_jets_resolved),
            weight=weights_res
        )
        
        output["HT_resolved"].fill(
            ht=ak.sum(single_jets_resolved.pt, axis=1),
            weight=weights_res
        )
        
        met_res = PuppiMETCorr[full_mask_res]
        output["met_resolved"].fill(
            met=met_res.pt,
            weight=weights_res
        )
        
        output["btag_max_resolved"].fill(
            btag=ak.max(single_bjets_resolved.btagUParTAK4B, axis=1),
            weight=weights_res
        )
        
        output["dr_bb_ave_resolved"].fill(
            dr=dr_bb_avg(single_bjets_resolved),
            weight=weights_res
        )
        
        mbbj_resolved = m_bbj(vec_single_bjets_resolved, vec_single_jets_resolved)
        output["m_bbj_resolved"].fill(
            mbbj=mbbj_resolved,
            weight=weights_res
        )
        
        # Define a fallback: bjet with minimum b-tag
        fallback_bjet_res = ak.firsts(
            single_bjets_resolved[
                ak.argmin(single_bjets_resolved.btagUParTAK4B, axis=1, keepdims=True)
            ]
        )
        
        # Regular leading untagged jet
        leading_untagged_res = ak.firsts(single_untag_jets_resolved)
        
        # Condition: if there are untagged jets
        has_untagged_res = ak.num(single_untag_jets_resolved) > 0
        
        # Final jet: use untagged if it exists, else fallback
        proxy_jet_res = ak.where(has_untagged_res, leading_untagged_res, fallback_bjet_res)
        
        # Compute Δφ between selected proxy jet and MET
        dphi_proxy_Z_res = delta_phi_raw(proxy_jet_res.phi, dilepton_res.phi)
        output["dphi_untag_Z_resolved"].fill(
            dphi=np.abs(dphi_proxy_Z_res),
            weight=weights_res
        )
        
        # local μ/e masks aligned to resolved selection
        idx_res      = np.where(full_mask_res)[0]
        mumu_mask_res = mask_step_trig_z_mumu[idx_res]
        ee_mask_res   = mask_step_trig_z_ee[idx_res]

        if np.any(mumu_mask_res):
            w_res_mm = weights_res[mumu_mask_res]
            
            output["mumu_eventflow_resolved"].fill(cut="step3", weight=np.sum(w_res_mm))
            output["mumu_n_untag_resolved"].fill(n=n_untagged_res[mumu_mask_res], weight=w_res_mm)
            output["mumu_n_jets_resolved"].fill(n=n_jets_res[mumu_mask_res], weight=w_res_mm)
            output["mumu_n_bjets_resolved"].fill(n=n_bjets_res[mumu_mask_res], weight=w_res_mm)
            ##leptons
            output["mumu_dr_ll_resolved"].fill(dr=vec_lead_lep_res[mumu_mask_res].delta_r(vec_sub_lep_res[mumu_mask_res]), weight=w_res_mm)
            output["mumu_pt_Z_resolved"].fill(pt_ll=dilepton_res[mumu_mask_res].pt, weight=w_res_mm)
            output["mumu_mass_Z_resolved"].fill(m_ll=dilepton_res[mumu_mask_res].mass, weight=w_res_mm)
            
            ### higgs dependent
            mass_H_res_m, pt_H_res_m, phi_H_res_m, eta_H_res_m = higgs_kin(vec_single_bjets_resolved[mumu_mask_res], vec_single_jets_resolved[mumu_mask_res])
            vec_H_res_mu = ak.zip({
                "pt": pt_H_res_m,
                "eta": eta_H_res_m,
                "phi": phi_H_res_m,
                "mass": mass_H_res_m,
            }, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
            
            output["mumu_mass_H_resolved"].fill(m_H=mass_H_res_m, weight=w_res_mm)
            output["mumu_pt_H_resolved"].fill(pt_H=pt_H_res_m, weight=w_res_mm)
            output["mumu_eta_H_resolved"].fill(eta=eta_H_res_m, weight=w_res_mm)
            
            output["mumu_dphi_HZ_resolved"].fill(dphi=np.abs(delta_phi_raw(phi_H_res_m, dilepton_res[mumu_mask_res].phi)), weight=w_res_mm)
            
            pt_ratio_res_m = pt_ratio_res[mumu_mask_res]
            output["mumu_pt_ratio_resolved"].fill(ratio=pt_ratio_res_m, weight=w_res_mm)
            
            output["mumu_dr_HZ_resolved"].fill(dr=vec_H_res[mumu_mask_res].delta_r(dilepton_res[mumu_mask_res]), weight=w_res_mm)
            output["mumu_deta_HZ_resolved"].fill(deta=delta_eta_vec(vec_H_res[mumu_mask_res],dilepton_res[mumu_mask_res]), weight=w_res_mm)
            
            # --- Quantities independent of Higgs definition ---
            output["mumu_dm_bb_bb_min_resolved"].fill(
                dm=min_dm_bb_bb(vec_single_bjets_resolved[mumu_mask_res], all_jets=vec_single_jets_resolved[mumu_mask_res]),
                weight=w_res_mm
            )
            
            output["mumu_dr_bb_bb_ave_resolved"].fill(
                dr=dr_bb_bb_avg(vec_single_bjets_resolved[mumu_mask_res], all_jets=vec_single_jets_resolved[mumu_mask_res]),
                weight=w_res_mm
            )
            
            output["mumu_HT_resolved"].fill(
                ht=ak.sum(single_jets_resolved[mumu_mask_res].pt, axis=1),
                weight=w_res_mm
            )
            
            met_res_m = PuppiMETCorr[mumu_mask_res]
            output["mumu_met_resolved"].fill(
                met=met_res_m.pt,
                weight=w_res_mm
            )
            
            output["mumu_btag_max_resolved"].fill(
                btag=ak.max(single_bjets_resolved[mumu_mask_res].btagUParTAK4B, axis=1),
                weight=w_res_mm
            )
            
            output["mumu_btag_min_resolved"].fill(
                btag=ak.min(single_bjets_resolved[mumu_mask_res].btagUParTAK4B, axis=1),
                weight=w_res_mm
            )
            
            output["mumu_dr_bb_ave_resolved"].fill(
                dr=dr_bb_avg(single_bjets_resolved[mumu_mask_res]),
                weight=w_res_mm
            )
            
            output["mumu_m_bbj_resolved"].fill(
                mbbj=mbbj_resolved[mumu_mask_res],
                weight=w_res_mm
            )
            
            output["mumu_dphi_untag_Z_resolved"].fill(
                dphi=np.abs(dphi_proxy_Z_res[mumu_mask_res]),
                weight=w_res_mm
            )
            
        ##################################################
        if np.any(ee_mask_res):
           
            w_res_ee = weights_res[ee_mask_res]
            
            output["ee_eventflow_resolved"].fill(cut="step3", weight=np.sum(w_res_ee))
            output["ee_n_untag_resolved"].fill(n=n_untagged_res[ee_mask_res], weight=w_res_ee)
            output["ee_n_jets_resolved"].fill(n=n_jets_res[ee_mask_res], weight=w_res_ee)
            output["ee_n_bjets_resolved"].fill(n=n_bjets_res[ee_mask_res], weight=w_res_ee)
            ##leptons
            output["ee_dr_ll_resolved"].fill(dr=vec_lead_lep_res[ee_mask_res].delta_r(vec_sub_lep_res[ee_mask_res]), weight=w_res_ee)
            output["ee_pt_Z_resolved"].fill(pt_ll=dilepton_res[ee_mask_res].pt, weight=w_res_ee)
            output["ee_mass_Z_resolved"].fill(m_ll=dilepton_res[ee_mask_res].mass, weight=w_res_ee)
            
            ### higgs dependent
            mass_H_res_e, pt_H_res_e, phi_H_res_e, eta_H_res_e = higgs_kin(vec_single_bjets_resolved[ee_mask_res], vec_single_jets_resolved[ee_mask_res])
            vec_H_res_e = ak.zip({
                "pt": pt_H_res_e,
                "eta": eta_H_res_e,
                "phi": phi_H_res_e,
                "mass": mass_H_res_e,
            }, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
            
            output["ee_mass_H_resolved"].fill(m_H=mass_H_res_e, weight=w_res_ee)
            output["ee_pt_H_resolved"].fill(pt_H=pt_H_res_e, weight=w_res_ee)
            output["ee_eta_H_resolved"].fill(eta=eta_H_res_e, weight=w_res_ee)
            
            output["ee_dphi_HZ_resolved"].fill(dphi=np.abs(delta_phi_raw(phi_H_res_e, dilepton_res[ee_mask_res].phi)), weight=w_res_ee)
            
            pt_ratio_res_e = pt_ratio_res[ee_mask_res]
            output["ee_pt_ratio_resolved"].fill(ratio=pt_ratio_res_e, weight=w_res_ee)
            
            output["ee_dr_HZ_resolved"].fill(dr=vec_H_res[ee_mask_res].delta_r(dilepton_res[ee_mask_res]), weight=w_res_ee)
            output["ee_deta_HZ_resolved"].fill(deta=delta_eta_vec(vec_H_res[ee_mask_res],dilepton_res[ee_mask_res]), weight=w_res_ee)
            
            # --- Quantities independent of Higgs definition ---
            output["ee_dm_bb_bb_min_resolved"].fill(
                dm=min_dm_bb_bb(vec_single_bjets_resolved[ee_mask_res], all_jets=vec_single_jets_resolved[ee_mask_res]),
                weight=w_res_ee
            )
            
            output["ee_dr_bb_bb_ave_resolved"].fill(
                dr=dr_bb_bb_avg(vec_single_bjets_resolved[ee_mask_res], all_jets=vec_single_jets_resolved[ee_mask_res]),
                weight=w_res_ee
            )
            
            output["ee_HT_resolved"].fill(
                ht=ak.sum(single_jets_resolved[ee_mask_res].pt, axis=1),
                weight=w_res_ee
            )
            
            met_res_e = PuppiMETCorr[ee_mask_res]
            output["ee_met_resolved"].fill(
                met=met_res_e.pt,
                weight=w_res_ee
            )
            
            output["ee_btag_max_resolved"].fill(
                btag=ak.max(single_bjets_resolved[ee_mask_res].btagUParTAK4B, axis=1),
                weight=w_res_ee
            )
            
            output["ee_dr_bb_ave_resolved"].fill(
                dr=dr_bb_avg(single_bjets_resolved[ee_mask_res]),
                weight=w_res_ee
            )
            
            mbbj_resolved_e = m_bbj(vec_single_bjets_resolved[ee_mask_res], vec_single_jets_resolved[ee_mask_res])
            output["ee_m_bbj_resolved"].fill(
                mbbj=mbbj_resolved_e,
                weight=w_res_ee
            )
            
            output["ee_dphi_untag_Z_resolved"].fill(
                dphi=np.abs(dphi_proxy_Z_res[ee_mask_res]),
                weight=w_res_ee
            )
            
        bdt_resolved = {
            "H_mass": ak.to_numpy(vec_H_res.mass),
            "H_pt": ak.to_numpy(vec_H_res.pt),
            "H_eta": ak.to_numpy(vec_H_res.eta),
            "Z_pt": ak.to_numpy(dilepton_res.pt),
            "HT": ak.to_numpy(ak.sum(single_jets_resolved.pt, axis=1)),
            "pt_ratio": ak.to_numpy(pt_ratio_res),
            "puppimet_pt": ak.to_numpy(met_res.pt),
            "btag_max": ak.to_numpy(ak.max(single_bjets_resolved.btagUParTAK4B, axis=1)),
            "btag_min": ak.to_numpy(ak.min(single_bjets_resolved.btagUParTAK4B, axis=1)),
            "dr_bb_bb_ave": ak.to_numpy(dr_bb_bb_avg(vec_single_bjets_resolved, all_jets=vec_single_jets_resolved)),
            "dr_bb_ave": ak.to_numpy(dr_bb_avg(vec_single_bjets_resolved)),
            "dm_bb_bb_min": ak.to_numpy(min_dm_bb_bb(vec_single_bjets_resolved, all_jets=vec_single_jets_resolved)),
            "dphi_HZ": ak.to_numpy(np.abs(vec_H_res.delta_phi(dilepton_res))),
            "deta_HZ": ak.to_numpy(delta_eta_vec(dilepton_res,vec_H_res)),
            "dr_HZ": ak.to_numpy(vec_H_res.delta_r(dilepton_res)),
            "dphi_untag_Z": ak.to_numpy(np.abs(dphi_proxy_Z_res)),
            "n_jets": ak.to_numpy(n_jets_res),
            "n_untag": ak.to_numpy(n_untagged_res),
            "dr_ll": ak.to_numpy(vec_lead_lep_res.delta_r(vec_sub_lep_res)),
            "mbbj": ak.to_numpy(mbbj_resolved),
            "weight": ak.to_numpy(weights_res),
        }
        
        if self.isMVA:
            self.compat_tree_variables(bdt_resolved)
            self.add_tree_entry("resolved", bdt_resolved)
            
        if self.run_eval and not self.isMVA:
            inputs_resolved = {k: bdt_resolved[k] for k in self.bdt_eval_resolved.var_list}
            bdt_score_resolved = np.ravel(self.bdt_eval_resolved.eval(inputs_resolved))
            output["bdt_score_resolved"].fill(bdt=bdt_score_resolved, weight=bdt_resolved["weight"])
            
            e_mask_all4b = np.asarray(mask_step_trig_z_ee[full_mask_res])
            mu_mask_all4b = np.asarray(mask_step_trig_z_mumu[full_mask_res])
            
            for i, cut in enumerate(self.optim_Cuts1_bdt):
                cut_mask = (bdt_score_resolved > cut)
                if not np.any(cut_mask):
                    continue
                    
                ele_mask_cut = e_mask_all4b & cut_mask
                mu_mask_cut = mu_mask_all4b & cut_mask
                
                # ===== electrons =====
                if np.any(ele_mask_cut):
                    w = weights_res[ele_mask_cut]
                    s = bdt_score_resolved[ele_mask_cut]
                    
                    output["ee_bdt_score_resolved"].fill(bdt=s, weight=w)
                    output["ee_bdt_shapes_resolved"].fill(cut_index=i, bdt=s, weight=w)
                    output["ee_higgsMass_shapes_resolved"].fill(
                        cut_index=i, m=ak.to_numpy(vec_H_res.mass[ele_mask_cut]), weight=w)
                    output["ee_higgsPt_shapes_resolved"].fill(
                        cut_index=i, pt=ak.to_numpy(vec_H_res.pt[ele_mask_cut]), weight=w)
                    output["ee_higgsEta_shapes_resolved"].fill(
                        cut_index=i, eta=ak.to_numpy(vec_H_res.eta[ele_mask_cut]), weight=w)
                    output["ee_ht_shapes_resolved"].fill(
                        cut_index=i, ht=ak.sum(single_jets_resolved[ele_mask_cut].pt, axis=1), weight=w)
                    output["ee_met_shapes_resolved"].fill(
                        cut_index=i, met=ak.to_numpy(met_res.pt[ele_mask_cut]), weight=w)
                    output["ee_ptZ_shapes_resolved"].fill(
                        cut_index=i, pt=ak.to_numpy(dilepton_res.pt[ele_mask_cut]), weight=w)
                    output["ee_dphiZh_shapes_resolved"].fill(
                        cut_index=i, dphi=np.abs(vec_H_res.delta_phi(dilepton_res)[ele_mask_cut]), weight=w)
                    output["ee_detaZh_shapes_resolved"].fill(
                        cut_index=i, deta=delta_eta_vec(vec_H_res,dilepton_res)[ele_mask_cut], weight=w)
                    output["ee_dRbb_shapes_resolved"].fill(
                        cut_index=i, dr=dr_bb_avg(vec_single_bjets_resolved[ele_mask_cut]), weight=w)
                    output["ee_dRbbbb_shapes_resolved"].fill(
                        cut_index=i, dr=dr_bb_bb_avg(vec_single_bjets_resolved[ele_mask_cut], vec_single_jets_resolved[ele_mask_cut]), weight=w)
                    output["ee_dm_shapes_resolved"].fill(
                        cut_index=i, dm=min_dm_bb_bb(vec_single_bjets_resolved[ele_mask_cut], all_jets=vec_single_jets_resolved[ele_mask_cut]), weight=w)
                    output["ee_dRZh_shapes_resolved"].fill(
                        cut_index=i, dr=vec_H_res.delta_r(dilepton_res)[ele_mask_cut], weight=w)
                    output["ee_ptratio_shapes_resolved"].fill(
                        cut_index=i, pt_ratio=pt_ratio_res[ele_mask_cut], weight=w)
                    output["ee_Njets_shapes_resolved"].fill(
                        cut_index=i, n_jets=ak.num(single_jets_resolved[ele_mask_cut], axis=1), weight=w)
                    output["ee_btag_min_shapes_resolved"].fill(
                        cut_index=i, btag=ak.min(single_bjets_resolved[ele_mask_cut].btagUParTAK4B, axis=1), weight=w)
                    output["ee_mbbj_shapes_resolved"].fill(cut_index=i, m=mbbj_resolved[ele_mask_cut], weight=w)
                    
                # ===== muons =====
                if np.any(mu_mask_cut):
                    w = weights_res[mu_mask_cut]
                    s = bdt_score_resolved[mu_mask_cut]
                    
                    output["mumu_bdt_score_resolved"].fill(bdt=s, weight=w)
                    output["mumu_bdt_shapes_resolved"].fill(cut_index=i, bdt=s, weight=w)
                    output["mumu_higgsMass_shapes_resolved"].fill(
                        cut_index=i, m=ak.to_numpy(vec_H_res.mass[mu_mask_cut]), weight=w)
                    output["mumu_higgsPt_shapes_resolved"].fill(
                        cut_index=i, pt=ak.to_numpy(vec_H_res.pt[mu_mask_cut]), weight=w)
                    output["mumu_higgsEta_shapes_resolved"].fill(
                        cut_index=i, eta=ak.to_numpy(vec_H_res.eta[mu_mask_cut]), weight=w)
                    output["mumu_ht_shapes_resolved"].fill(
                        cut_index=i, ht=ak.sum(single_jets_resolved[mu_mask_cut].pt, axis=1), weight=w)
                    output["mumu_met_shapes_resolved"].fill(
                        cut_index=i, met=ak.to_numpy(met_res.pt[mu_mask_cut]), weight=w)
                    output["mumu_ptZ_shapes_resolved"].fill(
                        cut_index=i, pt=ak.to_numpy(dilepton_res.pt[mu_mask_cut]), weight=w)
                    output["mumu_dphiZh_shapes_resolved"].fill(
                        cut_index=i, dphi=np.abs(vec_H_res.delta_phi(dilepton_res)[mu_mask_cut]), weight=w)
                    output["mumu_detaZh_shapes_resolved"].fill(
                        cut_index=i, deta=delta_eta_vec(vec_H_res,dilepton_res)[mu_mask_cut], weight=w)
                    output["mumu_dRbb_shapes_resolved"].fill(
                        cut_index=i, dr=dr_bb_avg(vec_single_bjets_resolved[mu_mask_cut]), weight=w)
                    output["mumu_dRbbbb_shapes_resolved"].fill(
                        cut_index=i, dr=dr_bb_bb_avg(vec_single_bjets_resolved[mu_mask_cut], vec_single_jets_resolved[mu_mask_cut]), weight=w)
                    output["mumu_dm_shapes_resolved"].fill(
                        cut_index=i, dm=min_dm_bb_bb(vec_single_bjets_resolved[mu_mask_cut], all_jets=vec_single_jets_resolved[mu_mask_cut]), weight=w)
                    output["mumu_dRZh_shapes_resolved"].fill(
                        cut_index=i, dr=vec_H_res.delta_r(dilepton_res)[mu_mask_cut], weight=w)
                    output["mumu_ptratio_shapes_resolved"].fill(
                        cut_index=i, pt_ratio=pt_ratio_res[mu_mask_cut], weight=w)
                    output["mumu_Njets_shapes_resolved"].fill(
                        cut_index=i, n_jets=ak.num(single_jets_resolved[mu_mask_cut], axis=1), weight=w)
                    output["mumu_btag_min_shapes_resolved"].fill(
                        cut_index=i, btag=ak.min(single_bjets_resolved[mu_mask_cut].btagUParTAK4B, axis=1), weight=w)
                    output["mumu_mbbj_shapes_resolved"].fill(cut_index=i, m=mbbj_resolved[mu_mask_cut], weight=w)
     
        if self.isMVA:
            output["trees"] = self._trees
            for regime, trees in self._trees.items():
                print(f"\n[DEBUG] Regime '{regime}' has {len(trees)} entries")
                
        return output
        
    def postprocess(self, accumulator):
        return accumulator
