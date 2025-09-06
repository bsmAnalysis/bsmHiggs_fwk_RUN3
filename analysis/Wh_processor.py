import awkward as ak
import numpy as np
import uproot
import os
import json
import argparse
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import vector as vector_old
from coffea import processor
from coffea.analysis_tools import Weights
import coffea.util
import hist
from hist import Hist, axis
import itertools
from boost_histogram import storage
from utils.jet_tight_id import compute_jet_id
from collections import defaultdict
from collections import Counter
from utils.xgb_tools import XGBHelper
import correctionlib
import gzip

from utils.deltas_array import (
    delta_r,
    clean_by_dr,
    delta_phi,
    delta_eta,
    min_dphi_jets_lepton)

from utils.variables_def import (
    min_dm_bb_bb,
    dr_bb_bb_avg,
    m_bbj,
    trans_massW,
    higgs_kin
    )

from utils.functions import (
    make_vector,
    make_vector_met,
    make_regressed_vector,
    build_sum_vector,
    build_sum_reg_vector)


from utils.matching import (
    extract_gen_bb_pairs,
    make_vector_old)

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
    Works even when an event has 0 leptons or 0 jets.
    """
    if leptons is None:
        return ak.ones_like(jets.pt, dtype=bool)

    # Build all jet–lepton pairs per event: shape [evt, njet, nlep]
    pairs = ak.cartesian({"j": jets, "l": leptons}, axis=1, nested=True)

    # ΔR^2
    # dphi = ((Δφ + π) % (2π)) - π
    dphi = np.arctan2(np.sin(pairs.j.phi - pairs.l.phi), np.cos(pairs.j.phi - pairs.l.phi))
    deta = pairs.j.eta - pairs.l.eta
    dr2  = dphi * dphi + deta * deta

    # Keep jets that are farther than dr from ALL leptons.
    return ak.all(dr2 > (dr * dr), axis=2)

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
    eps  = np.finfo(np.float64).tiny
    onem = np.nextafter(1.0, 0.0)
    u1 = np.clip(u1, eps, onem)
    u2 = np.clip(u2, eps, onem)

    # Box–Muller
    r     = np.sqrt(-2.0 * np.log(u1))
    theta = np.float64(2.0 * np.pi) * u2
    z     = r * np.cos(theta)

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

    if   vals.shape == (ny, nx): out = vals[iy, ix]
    elif vals.shape == (nx, ny): out = vals[ix, iy]
    else: raise RuntimeError(f"Bad TH2 shape {vals.shape}")

    return out  

#----------------------------------------------------------------------------------------------------------------------------------------------

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


class Wh_Processor(processor.ProcessorABC):
    def __init__(self, xsec=1.0, nevts=1.0, isMC=True, dataset_name=None, isMVA=True, isQCD=False, runEval=False, verbose=False):
        self.xsec    = xsec
        self.nevts   = nevts
        self.isMC    = isMC
        self.isMVA   = isMVA
        self.isQCD   = isQCD
        self.runEval = runEval
        self.verbose = verbose
        self.dataset_name=dataset_name
        self._trees = {regime: defaultdict(list) for regime in ["boosted", "resolved"]} if isMVA else None
        self._histograms = {}
        
        self.bdt_eval_boosted  = XGBHelper(os.path.join("xgb_model", "bdt_model_boosted.json"), ["H_mass", "H_pt", "MTW", "W_pt", "HT", "MET_pt", "dr_bb", "dm_bb" ,
                                                                                                 "dphi_WH", "dphi_jet_lepton_min", "pt_lepton", "btag_prod", "deta_WH", "Njets"])        
        self.bdt_eval_resolved = XGBHelper(os.path.join("xgb_model", "bdt_model_resolved.json"),["H_mass", "MTW", "W_pt", "HT" , "btag_min",  "dr_bb_ave",  "dm_4b_min", "mbbj",
                                                                                                 "dphi_WH", "dphi_jet_lepton_min",  "Njets",  "pt_lepton",  "pt_b1",  "WH_pt_assymetry" ])   
        
        self.bdt_edges = np.linspace(0.0, 1.0, 51 + 1)  
        self.optim_Cuts1_bdt = self.bdt_edges[:-1].tolist()
        
        self.systematics_labels = [""]  
        # self.systematics_labels = [
        #     "", # nominal
        #     "_umetup","_umetdown",
        #     "_jerup","_jerdown",
        #     "_jesup","_jesdown",
        #     "_scale_mup","_scale_mdown",
        #     "_stat_eup","_stat_edown",
        #     "_sys_eup","_sys_edown",
        #     "_GS_eup","_GS_edown",
        #     "_resRho_eup","_resRho_edown",
        #     "_puup","_pudown",
        #     "_pdfup","_pdfdown",
        #     "_btagup","_btagdown",
        # ]

        nvarsToInclude = len(self.systematics_labels)
        nCuts = len(self.optim_Cuts1_bdt)
        
        HERE = os.path.dirname(__file__)
        CORR_DIR = os.path.join(HERE, "corrections")
        
        #--- Electron energy and smearing corrections (2024) ---#
        # https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#2022_2023_and_2024_Scale_and_Sme
        self.egm_json_path = os.path.join(CORR_DIR, "electronSS_EtDependent.json.gz")
        #self.egm_json_path = "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2024/SS/electronSS_EtDependent.json.gz"
        self._egm_scale = None
        self._egm_smear = None
        
        if os.path.exists(self.egm_json_path):
            try:
                egm_cset = correctionlib.CorrectionSet.from_file(self.egm_json_path)
                
                def _take(cset, name):
                    try: return cset[name]
                    except Exception: return None
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
        # https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3
        self.ele_id_root  = os.path.join(CORR_DIR, "merged_EGamma_SF2D_Tight.root")
        #self.ele_id_root = "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2024/EleID/passingCutBasedTight122XV1/merged_EGamma_SF2D_Tight.root"
        
        self._ele_id_edges_x = self._ele_id_edges_y = self._ele_id_vals = None
        try:
            with uproot.open(self.ele_id_root) as f:
                h = f["EGamma_SF2D"]
                self._ele_id_edges_x = h.axes[0].edges()
                self._ele_id_edges_y = h.axes[1].edges()
                self._ele_id_vals    = h.values()
        except Exception as e:
            print(f"[ANA:ElectronID] Failed to load TH2: {e}")
            
        #--- HLT Ele30 TightID (2023D proxy) / Need to change later (expected at the beginning of September) ---#
        self.ele_hlt_root = os.path.join(CORR_DIR, "egammaEffi.txt_EGM2D.root")
        #self.ele_hlt_root = "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2023/ForPrompt23D/tnpEleHLT/HLT_SF_Ele30_TightID/egammaEffi.txt_EGM2D.root"
        
        self._ele_hlt_edges_x = self._ele_hlt_edges_y = None
        self._ele_hlt_effD = self._ele_hlt_effM = None
        
        try:
            with uproot.open(self.ele_hlt_root) as f:
                keys = {k.split(";")[0] for k in f.keys()}
                if "EGamma_EffData2D" in keys and "EGamma_EffMC2D" in keys:
                    hD = f["EGamma_EffData2D"]; hM = f["EGamma_EffMC2D"]
                    self._ele_hlt_edges_x = hD.axes[0].edges()
                    self._ele_hlt_edges_y = hD.axes[1].edges()
                    self._ele_hlt_effD    = hD.values()
                    self._ele_hlt_effM    = hM.values()
                else:
                    print("[ANA:EleHLT] No eff histos found; Ele HLT SFs disabled.")
        except Exception as e:
            print(f"[ANA:EleHLT] Failed to load HLT eff: {e}")
            
            
        #--- Muon ID Tight and Iso SFs (2023D) ---#
        # https://muon-wiki.docs.cern.ch/guidelines/corrections/#medium-pt-reco-efficiencies
        self.mu_idiso_json = os.path.join(CORR_DIR, "ScaleFactors_Muon_Z_ID_ISO_2023_BPix_schemaV2.json") # https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/raw/master/Run3/2023_BPix/2023_Z/ScaleFactors_Muon_Z_ID_ISO_2023_BPix_schemaV2.json
        self._mu_id = self._mu_iso = None
        
        try:
            mu_cset = correctionlib.CorrectionSet.from_file(self.mu_idiso_json)
            self._mu_id  = mu_cset["NUM_TightID_DEN_TrackerMuons"]
            self._mu_iso = mu_cset["NUM_TightPFIso_DEN_TightID"]
            print("[ANA:Muon] Loaded ID+ISO corrections.")
        except Exception as e:
            print(f"[ANA:Muon] Failed to load ID/ISO JSON: {e}")
            
        #--- HLT IsoMu24 TightID (2023D proxy) ---#
        self.mu_hlt_json = os.path.join(CORR_DIR, "muon_Z.json") # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/POG/MUO/2023_Summer23BPix/muon_Z.json.gz
        self._mu_hlt = None
        
        try:
            hlt_cset = correctionlib.CorrectionSet.from_file(self.mu_hlt_json)
            self._mu_hlt = hlt_cset["NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight"]
            print("[ANA:MuonHLT] Loaded HLT IsoMu24 corrections.")
        except Exception as e:
            print(f"[ANA:MuonHLT] Failed to load HLT JSON: {e}")

        # ====================== #
        #    JERC (JEC + JER)    #
        # ====================== #
        # https://cms-jerc.web.cern.ch/Recommendations/
        self.jerc_json_path = os.path.join(CORR_DIR, "jet_jerc.json.gz")
        self._jec_L1 = self._jec_L2 = self._jec_L3 = self._jec_residual = None
        self._jer_sf = self._jer_res = None 
        self._jes_unc_total = None
        
        if os.path.exists(self.jerc_json_path):
            try:
                jerc = correctionlib.CorrectionSet.from_file(self.jerc_json_path)
                def _pick(*parts):
                    for k in jerc:
                        if all(p in k for p in parts):
                            return jerc[k], k
                    return None, None
                    
                # AK4 PFPuppi chain (Summer24Prompt24*)
                if self.isMC:
                    self._jec_L1, key_L1 = _pick("L1FastJet",  "AK4PFPuppi", "MC")
                    self._jec_L2, key_L2 = _pick("L2Relative", "AK4PFPuppi", "MC")
                    self._jec_L3, key_L3 = _pick("L3Absolute", "AK4PFPuppi", "MC")
                    key_resid = None
                else:
                    self._jec_L1, key_L1 = _pick("L1FastJet",  "AK4PFPuppi", "DATA")
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
                self._jer_sf,  jer_sf_key  = _get("Summer23BPixPrompt23_RunD_JRV1_MC ScaleFactor AK4PFPuppi")
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

                                
        Wh_Processor._btag_wp_threshold             = _btag_wp_threshold
        Wh_Processor.btag_sf_perjet                 = btag_sf_perjet
        Wh_Processor.btag_event_weight_tagged_only  = btag_event_weight_tagged_only
        Wh_Processor.btag_event_weight_full         = btag_event_weight_full
    

        ##############
        # HISTOGRAMS #
        ##############
        
        #GENERATOR LEVEL ANALYSIS
        for particle in ["gen:H", "gen:A", "gen:W", "gen:b1", "gen:b2", "gen:b3", "gen:b4", 
                         "gen:bbbb", "gen:bb1", "gen:bb2", "gen:bbbb", "gen:AA", "gen:A1", "gen:A2", 
                         "gen:q1", "gen:q2", "gen:q3", "gen:q4", "gen:lepton", "gen:neutrino"]:
            
            self._histograms[f"mass_{particle}"]         = Hist.new.Reg(200, 0, 1000,       name="m",    label=f"{particle} mass").Double()
            self._histograms[f"pt_{particle}"]           = Hist.new.Reg(100, 0, 1000,       name="pt",   label=f"{particle} pT").Double()
            self._histograms[f"eta_{particle}"]          = Hist.new.Reg(100, -6, 6,         name="eta",  label=f"{particle} eta").Double()
            self._histograms[f"phi_{particle}"]          = Hist.new.Reg(100, -np.pi, np.pi, name="phi",  label=f"{particle} phi").Double()
            self._histograms[f"E_{particle}"]            = Hist.new.Reg(1000, 0, 2000,      name="E",    label=f"{particle} E").Double()
            self._histograms[f"deta_{particle}"]         = Hist.new.Reg(200, 0, 10,         name="deta", label=f"{particle} deltaEta").Double()
            self._histograms[f"dphi_{particle}"]         = Hist.new.Reg(200, 0, 10,         name="dphi", label=f"{particle} deltaPhi").Double()
            self._histograms[f"dr_{particle}"]           = Hist.new.Reg(200, 0, 6,          name="dr",   label=f"{particle} deltaR").Double()
      
        
      
        #DETECTOR LEVEL ANALYSIS
        self._histograms["lepton_multi_bef"]          = Hist.new.Reg(8,   0, 8, name="n",     label="Lepton multiplicity (bef)").Weight()
        self._histograms["lepton_multi_aft"]          = Hist.new.Reg(8,   0, 8, name="n",     label="Lepton multiplicity (aft)").Weight()
        self._histograms["double_btag_score_lead"]    = Hist.new.Reg(100, 0, 1, name="score", label="Double tag UParT (lead)").Weight()
        self._histograms["double_btag_score_sublead"] = Hist.new.Reg(100, 0, 1, name="score", label="Double tag UParT (sublead)").Weight()    
        self._histograms["single_btag_score_lead"]    = Hist.new.Reg(100, 0, 1, name="score", label="Single tag UParT (lead)").Weight()
        self._histograms["single_btag_score_sublead"] = Hist.new.Reg(100, 0, 1, name="score", label="Single tag UParT (sublead)").Weight()

        
        self._histograms["all_optim_systs"] = Hist.new.StrCat(self.systematics_labels, name="syst").Weight()
        self._histograms["all_optim_cut"]   = Hist.new.IntCategory(range(nCuts), name="cut_index", label="cut index").Reg(1, 0, 1, name="var", label="BDT>").Weight()
        
        bdt_features = {"boosted":  ["H_mass", "H_pt", "MTW", "W_pt", "HT", "MET_pt", "dr_bb", "dm_bb" ,
                                     "dphi_WH", "dphi_jet_lepton_min", "pt_lepton", "btag_prod", "deta_WH", "Njets"],
                        "resolved": ["H_mass", "MTW", "W_pt", "HT" , "btag_min",  "dr_bb_ave",  "dm_4b_min", "mbbj",
                                     "dphi_WH", "dphi_jet_lepton_min",  "Njets",  "pt_lepton",  "pt_b1",  "WH_pt_assymetry"]}
                      
        for suffix in ["boosted", "resolved"]:
            for prefix in ["e", "mu"]:
                
                # BDT model helpers
                attr_name = f"{prefix}_bdt_score_{suffix}"
                model_path = os.path.join("xgb_model", f"bdt_model_{suffix}.json")
                features = bdt_features[suffix]
                setattr(self, attr_name, XGBHelper(model_path, features))
                
                for region in ["A", "B", "C", "D"]:
                    
                    # 2-D Shape Histograms     
                    #for syst in self.systematics_labels:
                    self._histograms[f"{prefix}_{region}_SR_3b_bdt_shapes_{suffix}"]        = Hist.new.IntCategory(range(nCuts), name="cut_index").Variable(self.bdt_edges, name="bdt").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_higgsMass_shapes_{suffix}"]  = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 1000.0,     name="H_mass").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_higgsPt_shapes_{suffix}"]    = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 500.0,      name="H_pt").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_b1Pt_shapes_{suffix}"]       = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 500.0,      name="pt_b1").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_ht_shapes_{suffix}"]         = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 800.0,      name="HT").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_pfmet_shapes_{suffix}"]      = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 400.0,      name="MET_pt").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_mtw_shapes_{suffix}"]        = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(40, 0.0, 400.0,      name="MTW").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_ptw_shapes_{suffix}"]        = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 500.0,      name="W_pt").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_dphiWh_shapes_{suffix}"]     = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, np.pi,      name="dphi_WH").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_dphijetlep_shapes_{suffix}"] = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, np.pi,      name="dphi_lep_met").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_dRave_shapes_{suffix}"]      = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 5.0,        name="dr_bb_ave").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_dRbb_shapes_{suffix}"]       = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 5.0,        name="dr_bb").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_dmmin_shapes_{suffix}"]      = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 250.0,      name="dm_4b_min").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_dm_shapes_{suffix}"]         = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 250.0,      name="dm_bb").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_dphijmet_shapes_{suffix}"]   = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, np.pi,      name="dphi_jet_lepton_min").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_lep_pt_raw_shapes_{suffix}"] = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 200.0,      name="pt_lepton").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_dRwh_shapes_{suffix}"]       = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 6.0,        name="dr_WH").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_ptratio_shapes_{suffix}"]    = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 50.0,       name="pt_ratio").Weight() 
                    self._histograms[f"{prefix}_{region}_SR_3b_wh_pt_asym_shapes_{suffix}"] = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 1.0,        name="WH_pt_assymetry").Weight()  
                    self._histograms[f"{prefix}_{region}_SR_3b_jets_shapes_{suffix}"]       = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 12.0,       name="n_jets").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_btag_prod_shapes_{suffix}"]  = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 1.0,        name="btag_prod").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_btag_min_shapes_{suffix}"]   = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 1.0,        name="btag_min").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_btag_max_shapes_{suffix}"]   = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 1.0,        name="btag_max").Weight()
                    self._histograms[f"{prefix}_{region}_SR_3b_mbbj_shapes_{suffix}"]       = Hist.new.IntCategory(range(nCuts), name="cut_index").Reg(50, 0.0, 1000,       name="mbbj").Weight()
                    
                    for objt in ["H", "A", "W", "lepton", "MET", "MT", "bjet", "jet", "double-b jet", "bbj", "WH", "MET-lepton", "W-jet",
                                 "jet-lepton_min", "bb1", "bb2", "4b", "bb_ave", "b1", "b2", "b3", "b4", "bb", "bdt", "wh_asym", "bb1", "bb2"
                                 "single_untag_jets", "single_jets", "single_bjets", "double_jets", "double_bjets", "double_untag_jets"]:
                             
                        self._histograms[f"eventflow_{suffix}"] = hist.Hist(hist.axis.StrCategory(
                            ["raw", "step1", "trigger", "step2", "step3", "step4"], name="cut"), storage=storage.Double())
                        
                        self._histograms[f"{prefix}_eventflow_{suffix}"] = hist.Hist(hist.axis.StrCategory(
                            ["raw", "step1", "trigger", "step2", "step3", "step4"], name="cut"), storage=storage.Double())
                        
                       
                        self._histograms[f"dm_bbbb_min_{suffix}"]                       = Hist.new.Reg(100, 0, 200,  name="dm",    label=f"|ΔM(bb,bb)| minimum {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_dm_bbbb_min_{suffix}"]     = Hist.new.Reg(100, 0, 200,  name="dm",    label=f"|ΔM(bb,bb)| minimum {suffix}").Weight()
                        self._histograms[f"mass_{objt}_{suffix}"]                       = Hist.new.Reg(100, 0, 1000, name="m",     label=f"{objt} Mass {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_mass_{objt}_{suffix}"]     = Hist.new.Reg(100, 0, 1000, name="m",     label=f"{objt} Mass {suffix}").Weight()
                        self._histograms[f"MTW_bef_{suffix}"]                           = Hist.new.Reg(100, 0, 800,  name="m",     label=f"MTW (bef) {suffix}").Weight()
                        self._histograms[f"MTW_{suffix}"]                               = Hist.new.Reg(100, 0, 800,  name="m",     label=f"MTW (aft) {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_MTW_{suffix}"]             = Hist.new.Reg(100, 0, 800,  name="m",     label=f"MTW (aft) {suffix}").Weight()
                        self._histograms[f"MET_bef_{suffix}"]                           = Hist.new.Reg(100, 0, 800,  name="pt",    label=f"MET (bef) {suffix}").Weight()
                        self._histograms[f"MET_{suffix}"]                               = Hist.new.Reg(100, 0, 800,  name="pt",    label=f"MET (aft) {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_MET_{suffix}"]             = Hist.new.Reg(100, 0, 800,  name="pt",    label=f"MET (aft) {suffix}").Weight()
                        self._histograms[f"pt_{objt}_{suffix}"]                         = Hist.new.Reg(100, 0, 1000, name="pt",    label=f"{objt} pT {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_pt_{objt}_{suffix}"]       = Hist.new.Reg(100, 0, 1000, name="pt",    label=f"{objt} pT {suffix}").Weight()
                        self._histograms[f"wh_pt_asym_{suffix}"]                        = Hist.new.Reg(50,  0, 1,    name="pt",    label=f"pt assymetry wh {suffix}").Weight()           
                        self._histograms[f"{prefix}_{region}_wh_pt_asym_{suffix}"]      = Hist.new.Reg(50,  0, 1,    name="pt",    label=f"pt assymetry wh {suffix}").Weight()   
                        self._histograms[f"phi_{objt}_{suffix}"]                        = Hist.new.Reg(50,  -np.pi, np.pi,name="phi",   label=f"{objt} phi {suffix}").Weight() 
                        self._histograms[f"{prefix}_{region}_phi_{objt}_{suffix}"]      = Hist.new.Reg(50,  -np.pi, np.pi,name="phi",   label=f"{objt} phi {suffix}").Weight() 
                        self._histograms[f"eta_{objt}_{suffix}"]                        = Hist.new.Reg(50,  -3, 3,   name="eta",   label=f"{objt} eta {suffix}").Weight() 
                        self._histograms[f"{prefix}_{region}_eta_{objt}_{suffix}"]      = Hist.new.Reg(50,  -3, 3,   name="eta",   label=f"{objt} eta {suffix}").Weight() 
                        self._histograms[f"HT_{suffix}"]                                = Hist.new.Reg(100, 0, 1500, name="ht",    label=f"HT {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_HT_{suffix}"]              = Hist.new.Reg(100, 0, 1500, name="ht",    label=f"HT {suffix}").Weight()
                        self._histograms[f"pt_ratio_{suffix}"]                          = Hist.new.Reg(50,  0, 5,    name="ratio", label=f"pT(H)/pT(W) {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_pt_ratio_{suffix}"]        = Hist.new.Reg(50,  0, 5,    name="ratio", label=f"pT(H)/pT(W) {suffix}").Weight()
                        self._histograms[f"dphi_{objt}_{suffix}"]                       = Hist.new.Reg(60,  0, np.pi,name="dphi",  label=f"Δφ({objt}) {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_dphi_{objt}_{suffix}"]     = Hist.new.Reg(60,  0, np.pi,name="dphi",  label=f"Δφ({objt}) {suffix}").Weight()    
                        self._histograms[f"deta_{objt}_{suffix}"]                       = Hist.new.Reg(64,  0, 6,    name="deta",  label=f"Δη({objt}) {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_deta_{objt}_{suffix}"]     = Hist.new.Reg(64,  0, 6,    name="deta",  label=f"Δη({objt}) {suffix}").Weight()                        
                        self._histograms[f"dr_{objt}_{suffix}"]                         = Hist.new.Reg(60,  0, 6,    name="dr",    label=f"ΔR({objt}) average {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_dr_{objt}_{suffix}"]       = Hist.new.Reg(60,  0, 6,    name="dr",    label=f"ΔR({objt}) average {suffix}").Weight()
                        self._histograms[f"{objt}_multi_bef_{suffix}"]                  = Hist.new.Reg(12,  0, 12,   name="n",     label=f"{objt} multiplicity (bef) {suffix}").Weight()
                        self._histograms[f"{objt}_multi_aft_{suffix}"]                  = Hist.new.Reg(12,  0, 12,   name="n",     label=f"{objt} multiplicity (aft) {suffix}").Weight()                         
                        self._histograms[f"btag_min_{objt}_{suffix}"]                   = Hist.new.Reg(50,  0, 1,    name="btag",  label=f"btag min {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_btag_min_{objt}_{suffix}"] = Hist.new.Reg(50,  0, 1,    name="btag",  label=f"btag min {suffix}").Weight()
                        self._histograms[f"btag_max_{objt}_{suffix}"]                   = Hist.new.Reg(50,  0, 1,    name="btag",  label=f"btag max {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_btag_max_{objt}_{suffix}"] = Hist.new.Reg(50,  0, 1,    name="btag",  label=f"btag max {suffix}").Weight()                      
                        self._histograms[f"bdt_score_{suffix}"]                         = Hist.new.Reg(100, 0, 1,    name="bdt",   label=f"bdt score {suffix}").Weight()              
                        self._histograms[f"{prefix}_bdt_score_{suffix}"]                = Hist.new.Reg(100, 0, 1,    name="bdt",   label=f"bdt score {suffix}").Weight()            
                        self._histograms[f"btag_prod_{suffix}"]                         = Hist.new.Reg(50, 0, 1,     name="btag_prod",   label=f"btag product {suffix}").Weight()            
                        self._histograms[f"{prefix}_{region}_btag_prod_{suffix}"]       = Hist.new.Reg(50, 0, 1,     name="btag_prod",   label=f"btag_prod {suffix}").Weight()            
                        self._histograms[f"{prefix}_double_btag_score"]                 = Hist.new.Reg(100, 0, 1,    name="score", label="Double tag UParT score").Weight()
                        self._histograms[f"{prefix}_single_btag_score"]                 = Hist.new.Reg(100, 0, 1,    name="score", label="Single tag UParT score").Weight()

                        
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
            
        isSignal = (
            (self.dataset_name.startswith("ZH_ZToAll_HToAATo4B") or
             self.dataset_name.startswith("WH_WToAll_HToAATo4B") and self.isMC)
        )
        
        output = {key: hist if not hasattr(hist, "copy") else hist.copy() for key, hist in self._histograms.items()}
        
        
        ####################################
        #      GENERATOR LEVEL ANALYSIS    #
        ####################################
        genLevel = False       
        if genLevel:
            if isSignal:
                
                genparts = events.GenPart
                flags = genparts.statusFlags
                isHard  = (((flags >> 7) & 1) == 1) | (((flags >> 8) & 1) == 1)
                mask_HL = isHard 
                
                # load Higgs
                genHiggs  = genparts[(genparts.pdgId == 25) & (genparts.status == 22)]
                # load W
                maskW     = (abs(genparts.pdgId == 24))
                genW      = genparts[maskW & (genparts.status == 22)]   
                motherIsW = maskW[genparts.genPartIdxMother]  
                # load A's
                maskA     = (genparts.pdgId == 36) 
                genA      = genparts[maskA]          
                motherIsA = maskA[genparts.genPartIdxMother] 
                # load the b quarks
                maskB     = (abs(genparts.pdgId) == 5) & (genparts.status == 23) & motherIsA & mask_HL
                genB      = genparts[maskB]
                # catch b's with the same mother 
                bb_pairs = extract_gen_bb_pairs(genparts)
                # load the leptons
                islepton  = ((abs(genparts.pdgId) == 11) | (abs(genparts.pdgId) == 13) | (abs(genparts.pdgId) == 15)) & motherIsW 
                genLepton = genparts[islepton] 
                genLepton = genLepton[ak.argsort(genLepton.pt, ascending=False)] #sort on pT
                # load the neutrinos 
                isneutrino  = ((abs(genparts.pdgId) == 12) | (abs(genparts.pdgId) == 14) | (abs(genparts.pdgId) == 16)) & motherIsW 
                genNeutrino = genparts[isneutrino] 
                genNeutrino = genNeutrino[ak.argsort(genNeutrino.pt, ascending=False)] #sort on pT
                 
                genHiggs = genHiggs[:, 0]; 
                p4_Higgs = make_vector_old(genHiggs)
                
                output["pt_gen:H"].fill(pt=genHiggs.pt)
                output["eta_gen:H"].fill(eta=genHiggs.eta)
                output["mass_gen:H"].fill(m=genHiggs.mass)
                output["phi_gen:H"].fill(phi=genHiggs.phi)
                
                genW = ak.flatten(genW)
                p4_W = make_vector_old(genW)
                
                output["pt_gen:W"].fill(pt=genW.pt)
                output["eta_gen:W"].fill(eta=genW.eta)
                output["mass_gen:W"].fill(m=genW.mass)
                output["phi_gen:W"].fill(phi=genW.phi)
                
                mask_has_two_A = ak.num(genA) >= 2
                genA_pairs = genA[mask_has_two_A]
                genA1 = genA[:, 0]; p4_A1 = make_vector_old(genA1)
                genA2 = genA[:, 1]; p4_A2 = make_vector_old(genA2)
                genA  = ak.flatten(ak.Array([genA1, genA2]), axis=1)
                
                output["pt_gen:A"].fill(pt=genA.pt)
                output["eta_gen:A"].fill(eta=genA.eta)
                output["mass_gen:A"].fill(m=genA.mass)
                output["phi_gen:A"].fill(phi=genA.phi)
                
                genB = genB[ak.argsort(genB.pt, ascending=False)] #sort on pT
                        
                output["pt_gen:b1"].fill(pt=genB[:, 0].pt)
                output["eta_gen:b1"].fill(eta=genB[:, 0].eta)
                output["phi_gen:b1"].fill(phi=genB[:, 0].phi)
                output["mass_gen:b1"].fill(m=genB[:, 0].mass)
    
                output["pt_gen:b2"].fill(pt=genB[:, 1].pt)
                output["eta_gen:b2"].fill(eta=genB[:, 1].eta)
                output["phi_gen:b2"].fill(phi=genB[:, 1].phi)
                output["mass_gen:b2"].fill(m=genB[:, 1].mass)
    
                output["pt_gen:b3"].fill(pt=genB[:, 2].pt)
                output["eta_gen:b3"].fill(eta=genB[:, 2].eta)
                output["phi_gen:b3"].fill(phi=genB[:, 2].phi)
                output["mass_gen:b3"].fill(m=genB[:, 2].mass)
    
                output["pt_gen:b4"].fill(pt=genB[:, 3].pt)
                output["eta_gen:b4"].fill(eta=genB[:, 3].eta)
                output["phi_gen:b4"].fill(phi=genB[:, 3].phi)
                output["mass_gen:b4"].fill(m=genB[:, 3].mass)
                
                genLepton = ak.flatten(genLepton)
                p4_Lepton = make_vector_old(genLepton)
                
                output["pt_gen:lepton"].fill(pt=genLepton.pt)
                output["eta_gen:lepton"].fill(eta=genLepton.eta)
                output["mass_gen:lepton"].fill(m=genLepton.mass)
                output["phi_gen:lepton"].fill(phi=genLepton.phi)
                
                genNeutrino = ak.flatten(genNeutrino)
                output["pt_gen:neutrino"].fill(pt=genNeutrino.pt)
                
                output["dr_gen:AA"].fill(dr=p4_A1.deltaR(p4_A2))
                
                bb1, bb2 = extract_gen_bb_pairs(genparts)
                
                mask_valid_bb1 = ak.num(bb1) == 2
                mask_valid_bb2 = ak.num(bb2) == 2
                
                vec_b1 = make_vector_old(bb1[mask_valid_bb1][:, 0])
                vec_b2 = make_vector_old(bb1[mask_valid_bb1][:, 1])
                vec_b3 = make_vector_old(bb2[mask_valid_bb2][:, 0])
                vec_b4 = make_vector_old(bb2[mask_valid_bb2][:, 1])
                
                dr_bb1 = vec_b1.deltaR(vec_b2)
                dr_bb2 = vec_b3.deltaR(vec_b4)
                
                output["dr_gen:bb1"].fill(dr=dr_bb1)
                output["dr_gen:bb2"].fill(dr=dr_bb2)
                
                mass_bb1 = (vec_b1 + vec_b2).mass
                mass_bb2 = (vec_b3 + vec_b4).mass
                
                vec_sum = vec_b1 + vec_b2 + vec_b3 + vec_b4
                mass_bbbb = vec_sum.mass
                pt_bbbb = vec_sum.pt
                
                output["mass_gen:bb1"].fill(m=mass_bb1)
                output["mass_gen:bb2"].fill(m=mass_bb2)
                
                output["mass_gen:bbbb"].fill(m=mass_bbbb)
                output["pt_gen:bbbb"].fill(pt=pt_bbbb)
                           
            
        ###################################
        #      DETECTOR LEVEL ANALYSIS    #
        ###################################
        
        # Better add this in the skimmer
        if ("PV" in events.fields) and ("npvsGood" in ak.fields(events.PV)):
            pv_mask = events.PV.npvsGood >= 1
        else:
            pv_mask = ak.ones_like(events.event, dtype=bool)  
        
        # ====================== #
        # STEP 1 : Build weights #
        # ====================== #   
        events  = events[pv_mask]
        n_ev    = len(events)
        weights = Weights(n_ev)
        
        if self.isMC:
            weights.add("norm", np.full(n_ev, self.xsec / self.nevts, dtype="float64"))
        else:
            weights.add("ones", np.ones(n_ev, dtype="float64"))
            
        print(f"[ENV] host={os.uname().nodename}  isMC={self.isMC}  dataset={self.dataset_name}")
        print(f"[ENV] xsec/nevts={self.xsec}/{self.nevts} = {self.xsec/self.nevts:.3e}")

        w_now = weights.weight()
        print(f"[WEIGHTS] n_ev={len(w_now)}  sum={np.sum(w_now):.6g}  mean={np.mean(w_now):.6g}  "
                  f"min={np.min(w_now):.3g}  max={np.max(w_now):.3g}")

                                      
        # =============================== #
        # STEP 2 : EGM energy corrections #
        # =============================== #
        # https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#2022_2023_and_2024_Scale_and_Sme
        # Align electron energy response/resolution between data and simulation.
        ele_all = events.Electron
        
        if "event" not in ak.fields(ele_all):
            ele_all = ak.with_field(ele_all, ak.broadcast_arrays(events.event, ele_all.pt)[0], "event")
            
        scEta    = ele_all.superclusterEta
        absScEta = np.abs(scEta)
        
        if (not self.isMC) and (self._egm_scale is not None):
            # DATA: Multiply electron pT by a data scale factor from EGM JSON (depends on run, scEta, |scEta|, r9, pt, seedGain).
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
            # MC: Smear electron pT using a Gaussian width from EGM JSON (depends on pt, r9, |scEta|), with a deterministic RNG per electron.
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
            # Electrons with corrected pt (all other fields unchanged).
            ElectronCorr = ak.with_field(ele_all, ele_corr_pt, "pt")
        
        else:
            ElectronCorr = ele_all  
                
        # ================================== #
        # STEP 3 : JEC + JER for AK4 Puppi   #
        # ================================== # 
        # https://cms-jerc.web.cern.ch/Recommendations/#2024
        # JEC brings jets onto the correct scale; JER makes MC jet resolution match data.
        jets_in = events.Jet
   
        if "event" not in ak.fields(jets_in):
            jets_in = ak.with_field(jets_in, ak.broadcast_arrays(events.event, jets_in.pt)[0], "event")
            
        rawFactor = getattr(jets_in, "rawFactor", ak.zeros_like(jets_in.pt))
        pt_raw    = jets_in.pt   * (1.0 - rawFactor)
        mass_raw  = jets_in.mass * (1.0 - rawFactor)  
        
        rho_evt  = events.fixedGridRhoFastjetAll              
        counts   = ak.num(pt_raw, axis=1)
        rho      = ak.broadcast_arrays(rho_evt, pt_raw)[0]
        
        pt_step = pt_raw
        
        # --- Apply L1 → L2 → L3 (and Data residual for data) using correctionlib on (area, eta, phi, pt_step, ρ). --- #
        
        # JEC chain (all jets) #
        # L1
        if self._jec_L1 is not None:
            c = self._jec_L1
            corr_flat = _eval_corr_vectorized(
                c,
                JetA=jets_in.area, JetEta=jets_in.eta, JetPt=pt_step, Rho=rho, JetPhi=jets_in.phi
            )
            if corr_flat is not None:
                cfac = _unflatten_like(corr_flat, counts)
                pt_step = pt_step * cfac
                
        # L2
        if self._jec_L2 is not None:
            c = self._jec_L2
            corr_flat = _eval_corr_vectorized(
                c,
                JetA=jets_in.area, JetEta=jets_in.eta, JetPt=pt_step, Rho=rho, JetPhi=jets_in.phi
            )
            if corr_flat is not None:
                cfac = _unflatten_like(corr_flat, counts)
                pt_step = pt_step * cfac
                
        # L3
        if self._jec_L3 is not None:
            c = self._jec_L3
            corr_flat = _eval_corr_vectorized(
                c,
                JetA=jets_in.area, JetEta=jets_in.eta, JetPt=pt_step, Rho=rho, JetPhi=jets_in.phi
            )
            if corr_flat is not None:
                cfac = _unflatten_like(corr_flat, counts)
                pt_step = pt_step * cfac
                
        # Data-only residual
        if (not self.isMC) and (self._jec_residual is not None):
            c = self._jec_residual

            run_b = ak.broadcast_arrays(events.run, pt_raw)[0]
        
            kwargs = {
                "JetA": jets_in.area,
                "JetEta": jets_in.eta,
                "JetPhi": jets_in.phi,
                "Rho": rho,
                "JetPt": pt_raw,
                "Run": run_b,
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
        
        if self.isMC and (self._jer_sf is not None):
            # JEC-corrected kinematics (inputs for JER)
            pt  = pt_jec
            eta = jets_in.eta
        
            if hasattr(jets_in, "pt_genMatched"):
                pt_gen  = jets_in.pt_genMatched      # NaN where unmatched / on data
                has_gen = np.isfinite(pt_gen)        # True only where a match exists
                                
            # SF nominal
            sf_nom_flat = _eval_corr_vectorized(self._jer_sf, JetEta=eta, JetPt=pt, systematic="nom")
            sf_nom      = _unflatten_like(sf_nom_flat, counts) if sf_nom_flat is not None else ak.ones_like(pt)

            # Resolution
            if self._jer_res is not None:
                res_flat = _eval_corr_vectorized(self._jer_res, JetEta=eta, Rho=rho, JetPt=pt)
                res = _unflatten_like(res_flat, counts)
                
            # ----- tight gen matching per twiki -----#
            # https://cms-jerc.web.cern.ch/JER/
                       
            # |pT - pT_gen| < 3 * σ_JER * pT  (evaluate only where matched)
            ptdiff_ok = ak.fill_none(
                np.abs(ak.mask(pt, has_gen) - ak.mask(pt_gen, has_gen)) <
                (3.0 * ak.mask(res, has_gen) * ak.mask(pt, has_gen)), False
            )
        
            match_tight = has_gen & ptdiff_ok

            # Start from un-smeared
            pt_corr = pt
        
            # matched & tight: scaling
            pt_matched = ak.where((pt_gen + sf_nom * (pt - pt_gen)) > 0.0,
                                  pt_gen + sf_nom * (pt - pt_gen), 0.0)
            pt_corr = ak.where(match_tight, pt_matched, pt_corr)
        
            # others: stochastic
            n            = _unflatten_like(_rng_normal_like(jets_in), counts)
            sigma        = res * np.sqrt(np.maximum(sf_nom**2 - 1.0, 0.0))
            smear_factor = 1.0 + sigma * n
            pt_corr      = ak.where(~match_tight, pt * smear_factor, pt_corr)
        
            # Propagate mass
            jer_factor = ak.where(pt > 0, pt_corr / pt, 1.0)
            mass_corr  = mass_jec * jer_factor
        else:
            pt_corr  = pt_jec
            mass_corr = mass_jec
        
        jets = ak.with_field(jets_in, ak.values_astype(pt_corr,  "float32"), "pt")
        jets = ak.with_field(jets,    ak.values_astype(mass_corr,"float32"), "mass")
        jets = ak.with_field(jets,    jec_factor, "jecFactor")
        jerFactor = ak.values_astype(ak.where(pt_jec > 0, pt_corr / pt_jec, 1.0) if self.isMC else ak.ones_like(pt_jec), "float32")
        jets = ak.with_field(jets, jerFactor, "jerFactor")        

        # ================== #
        # STEP 4: Type-1 MET #
        # ================== #        
        met_in = events.PuppiMET
        met_px, met_py = _ptphi_to_pxpy(met_in.pt, met_in.phi)
        
        jets_nom  = events.Jet
        pt_old    = jets_nom.pt 
        rawFactor = ak.fill_none(getattr(jets_nom, "rawFactor", ak.zeros_like(jets_nom.pt)), 0.0)
        pt_raw    = jets_nom.pt * (1.0 - rawFactor)
        
        # --- Build L2×L3-only pT for PUPPI Type-1 "new" JEC --- #
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections#Type_I_Correction_Propagation_of
        counts_j = ak.num(jets_nom.pt, axis=1)
        rho_forL = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, pt_raw)[0]
        
        pt_L2L3 = pt_raw
        if self._jec_L2 is not None:
            c2_flat = _eval_corr_vectorized(self._jec_L2,
                                            JetA=jets_nom.area, JetEta=jets_nom.eta,
                                            JetPt=pt_L2L3, Rho=rho_forL, JetPhi=jets_nom.phi)
            if c2_flat is not None:
                pt_L2L3 = pt_L2L3 * _unflatten_like(c2_flat, counts_j)
        
        if self._jec_L3 is not None:
            c3_flat = _eval_corr_vectorized(self._jec_L3,
                                            JetA=jets_nom.area, JetEta=jets_nom.eta,
                                            JetPt=pt_L2L3, Rho=rho_forL, JetPhi=jets_nom.phi)
            if c3_flat is not None:
                pt_L2L3 = pt_L2L3 * _unflatten_like(c3_flat, counts_j)
                          
        # --- Jet mask for PUPPI Type-1 --- #
        overlap_mu = _mask_lepton_overlap(jets_nom, events.Muon,  dr=0.4)
        overlap_el = _mask_lepton_overlap(jets_nom, ElectronCorr, dr=0.4)
        jet_for_met = (
            (pt_L2L3 > 15.0)
            & (np.abs(jets_nom.eta) < 4.8)
            & ak.values_astype(jets_nom.passJetIdTightLepVeto, bool)
            & overlap_mu & overlap_el
        )
        
        dpt_jec = ak.where(jet_for_met, (pt_L2L3 - pt_old), 0.0)
        
        # --- JER propagation (MC): apply nominal jer_ratio on top of the new JEC --- #
        
        if self.isMC:
            jer_ratio = ak.where(pt_jec > 0, pt_corr / pt_jec, 1.0)   
            pt_final_met = pt_L2L3 * jer_ratio
            dpt_jer = ak.where(jet_for_met, (pt_final_met - pt_L2L3), 0.0)
        else:
            dpt_jer = ak.zeros_like(dpt_jec)
            
        # --- Update MET --- #
        dpx = ak.sum((dpt_jec + dpt_jer) * np.cos(jets_nom.phi), axis=1)
        dpy = ak.sum((dpt_jec + dpt_jer) * np.sin(jets_nom.phi), axis=1)
        met_px_corr = met_px - dpx
        met_py_corr = met_py - dpy
        
        # MC: propagate electron energy correction to MET
        if self.isMC:
            dpx_el = ak.sum((ElectronCorr.pt - ele_all.pt) * np.cos(ele_all.phi), axis=1)
            dpy_el = ak.sum((ElectronCorr.pt - ele_all.pt) * np.sin(ele_all.phi), axis=1)
            met_px_corr = met_px_corr - ak.values_astype(dpx_el, "float64")
            met_py_corr = met_py_corr - ak.values_astype(dpy_el, "float64")
        
        met_pt_corr, met_phi_corr = _pxpy_to_ptphi(met_px_corr, met_py_corr)
        PuppiMETCorr = ak.zip(
            {"pt": ak.values_astype(met_pt_corr, "float32"),
             "phi": ak.values_astype(met_phi_corr, "float32")},
            with_name="MET",
        )
        
        print("[JEC/JER] pt_jec stats:", 
                  f"min={ak.min(jets.pt):.2f}  max={ak.max(jets.pt):.2f}  mean={ak.mean(jets.pt):.2f}")
        print("[MET] PuppiMETCorr pt:",
                  f"min={ak.min(PuppiMETCorr.pt):.2f}  max={ak.max(PuppiMETCorr.pt):.2f}  mean={ak.mean(PuppiMETCorr.pt):.2f}")


        
        # Stash the systematics
        systs = {"jets": {}, "met": {}, "weights": {}}
                
        # # ============================== #
        # # STEP 5 : JER up/down (MC only) #
        # # ============================== #
        # if self.isMC and (self._jer_sf is not None):
        #     jets_in   = events.Jet
        #     rawFactor = getattr(jets_in, "rawFactor", ak.zeros_like(jets_in.pt))
        #     pt_raw    = jets_in.pt * (1.0 - rawFactor)
        
        #     # inputs at JEC level
        #     eta    = jets_in.eta
        #     pt     = pt_jec
        #     rho    = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, pt)[0]
        #     counts = ak.num(pt, axis=1)
        
        #     # gen matching
        #     genIdx  = jets_in.genJetIdx
        #     has_gen = ak.values_astype(genIdx >= 0, bool)
        
        #     if hasattr(events, "GenJet"):
        #         safe_idx    = ak.mask(genIdx, has_gen)
        #         matched_gen = events.GenJet[safe_idx]
        #         pt_gen      = ak.fill_none(matched_gen.pt, 0.0)
        #     else:
        #         has_gen = ak.zeros_like(has_gen, dtype=bool)
        #         pt_gen  = ak.zeros_like(pt)
        
        #     # SF up/down
        #     sf_up   = _unflatten_like(_eval_corr_vectorized(self._jer_sf, JetEta=eta, JetPt=pt, systematic="up"),   counts)
        #     sf_down = _unflatten_like(_eval_corr_vectorized(self._jer_sf, JetEta=eta, JetPt=pt, systematic="down"), counts)
        
        #     # resolution at JEC kinematics
        #     if self._jer_res is not None:
        #         res = _unflatten_like(_eval_corr_vectorized(self._jer_res, JetEta=eta, Rho=rho, JetPt=pt), counts)
        #     else:
        #         res = ak.zeros_like(pt)
        
        #     # ----- same tight matching as in STEP 3 -----
        #     R_CONE = 0.4
        #     dr_cut = 0.5 * R_CONE
        
        #     j_phi_m = ak.mask(jets_in.phi, has_gen)
        #     j_eta_m = ak.mask(jets_in.eta, has_gen)
        #     g_phi   = ak.mask(matched_gen.phi if hasattr(events, "GenJet") else None, has_gen)
        #     g_eta   = ak.mask(matched_gen.eta if hasattr(events, "GenJet") else None)
        
        #     dphi = np.arctan2(np.sin(j_phi_m - g_phi), np.cos(j_phi_m - g_phi))
        #     deta = j_eta_m - g_eta
        #     dr2  = dphi * dphi + deta * deta
        #     dr_ok = ak.fill_none(dr2 < (dr_cut * dr_cut), False)
        
        #     ptdiff_ok = ak.fill_none(
        #         np.abs(ak.mask(pt, has_gen) - ak.mask(pt_gen, has_gen)) <
        #         (3.0 * ak.mask(res, has_gen) * ak.mask(pt, has_gen)), False
        #     )
        
        #     match_tight = has_gen & dr_ok & ptdiff_ok
        #     # -------------------------------------------
        
        #     # deterministic RNG per (event, jet)
        #     if "event" not in ak.fields(jets_in):
        #         jets_in = ak.with_field(jets_in, ak.broadcast_arrays(events.event, jets_in.pt)[0], "event")
        #     n = _unflatten_like(_rng_normal_like(jets_in), counts)
        
        #     def _smear(pt_in, pt_gen_in, match_in, sf_in, res_in, n_in):
        #         pt_m  = ak.where((pt_gen_in + sf_in * (pt_in - pt_gen_in)) > 0.0,
        #                          pt_gen_in + sf_in * (pt_in - pt_gen_in), 0.0)
        #         sigma = res_in * np.sqrt(np.maximum(sf_in**2 - 1.0, 0.0))
        #         pt_u  = pt_in * (1.0 + sigma * n_in)
        #         return ak.where(match_in, pt_m, pt_u)
        
        #     pt_jerUp   = _smear(pt, pt_gen, match_tight, sf_up,   res, n)
        #     pt_jerDown = _smear(pt, pt_gen, match_tight, sf_down, res, n)
        
        #     # propagate to mass from JEC mass
        #     mass_jerUp   = mass_jec * ak.where(pt_jec > 0, pt_jerUp   / pt_jec, 1.0)
        #     mass_jerDown = mass_jec * ak.where(pt_jec > 0, pt_jerDown / pt_jec, 1.0)
        
        #     jets_jerUp   = ak.with_field(ak.with_field(jets_in, ak.values_astype(pt_jerUp,   "float32"), "pt"),
        #                                  ak.values_astype(mass_jerUp,   "float32"), "mass")
        #     jets_jerDown = ak.with_field(ak.with_field(jets_in, ak.values_astype(pt_jerDown, "float32"), "pt"),
        #                                  ak.values_astype(mass_jerDown, "float32"), "mass")
        
        #     # --- MET variants (use local jet_for_met; rely on met_px/met_py from step 4) ---
        #     def _met_variant(pt_step_var, pt_final_var):
        #         jet_for_met = (
        #             (pt_raw > 15.0)
        #             & (np.abs(jets_in.eta) < 4.8)
        #             & ak.values_astype(jets_in.passJetIdTightLepVeto, bool)
        #         )
        #         dpt_jec = ak.where(jet_for_met, pt_step_var  - pt_raw, 0.0)
        #         dpt_jer = ak.where(jet_for_met, pt_final_var - pt_step_var, 0.0)
        #         dpx = ak.sum((dpt_jec + dpt_jer) * np.cos(jets_in.phi), axis=1)
        #         dpy = ak.sum((dpt_jec + dpt_jer) * np.sin(jets_in.phi), axis=1)
        
        #         px = met_px - dpx
        #         py = met_py - dpy
        
        #         # electron shift (MC)
        #         dpx_el = ak.sum((ElectronCorr.pt - ele_all.pt) * np.cos(ele_all.phi), axis=1)
        #         dpy_el = ak.sum((ElectronCorr.pt - ele_all.pt) * np.sin(ele_all.phi), axis=1)
        #         px = px - ak.values_astype(dpx_el, "float64")
        #         py = py - ak.values_astype(dpy_el, "float64")
        
        #         ptv, phiv = _pxpy_to_ptphi(px, py)
        #         return ak.zip({"pt": ak.values_astype(ptv, "float32"),
        #                        "phi": ak.values_astype(phiv, "float32")}, with_name="MET")
        
        #     PuppiMETCorr_jerUp   = _met_variant(pt_jec, pt_jerUp)
        #     PuppiMETCorr_jerDown = _met_variant(pt_jec, pt_jerDown)
        
        #     systs["jets"]["_jerup"]   = jets_jerUp
        #     systs["jets"]["_jerdown"] = jets_jerDown
        #     systs["met"]["_jerup"]    = PuppiMETCorr_jerUp
        #     systs["met"]["_jerdown"]  = PuppiMETCorr_jerDown

                
        # # ============================ #
        # # STEP 6 : JES "Total" up/down # 
        # # ============================ #
        # if self._jes_unc_total is not None:
        #     jets_in   = events.Jet
        #     rawFactor = getattr(jets_in, "rawFactor", ak.zeros_like(jets_in.pt))
        #     pt_raw    = jets_in.pt * (1.0 - rawFactor)
        #     mass_raw  = jets_in.mass * (1.0 - rawFactor)  
        #     counts    = ak.num(pt_jec, axis=1)            
        
        #     # fractional uncertainty u(eta, pt_jec)
        #     unc_flat = _eval_corr_vectorized(self._jes_unc_total, JetEta=jets_in.eta, JetPt=pt_jec)
        #     unc = _unflatten_like(unc_flat, counts)
        
        #     # reuse nominal JER ratio (so JES varies only the JEC stage)
        #     jer_ratio = ak.where(pt_jec > 0, pt_corr / pt_jec, 1.0)
        
        #     # vary at JEC stage, then apply same JER ratio
        #     pt_step_jesUp   = pt_jec * (1.0 + unc)
        #     pt_step_jesDown = pt_jec * (1.0 - unc)
        
        #     pt_jesUp   = pt_step_jesUp   * jer_ratio
        #     pt_jesDown = pt_step_jesDown * jer_ratio
        
        #     # propagate mass with same pT ratio w.r.t. *raw*
        #     mass_jesUp   = ak.values_astype(mass_raw * ak.where(pt_raw > 0, pt_jesUp   / pt_raw, 1.0), "float32")
        #     mass_jesDown = ak.values_astype(mass_raw * ak.where(pt_raw > 0, pt_jesDown / pt_raw, 1.0), "float32")
        
        #     jets_jesUp   = ak.with_field(jets_in, ak.values_astype(pt_jesUp,   "float32"), "pt")
        #     jets_jesUp   = ak.with_field(jets_jesUp,   mass_jesUp,   "mass")
        #     jets_jesDown = ak.with_field(jets_in, ak.values_astype(pt_jesDown, "float32"), "pt")
        #     jets_jesDown = ak.with_field(jets_jesDown, mass_jesDown, "mass")
        
        #     # MET variant helper (recompute jet_for_met locally)
        #     def _met_variant(pt_step_var, pt_final_var):
        #         dpt_jec_var = ak.where(jet_for_met, pt_step_var  - pt_old, 0.0)    
        #         dpt_jer_var = ak.where(jet_for_met, pt_final_var - pt_step_var, 0.0) 
                
        #         dpx = ak.sum((dpt_jec_var + dpt_jer_var) * np.cos(jets_nom.phi), axis=1)
        #         dpy = ak.sum((dpt_jec_var + dpt_jer_var) * np.sin(jets_nom.phi), axis=1)
            
        #         px = met_px - dpx
        #         py = met_py - dpy
            
        #         if self.isMC:
        #             dpx_el = ak.sum((ElectronCorr.pt - ele_all.pt) * np.cos(ele_all.phi), axis=1)
        #             dpy_el = ak.sum((ElectronCorr.pt - ele_all.pt) * np.sin(ele_all.phi), axis=1)
        #             px = px - ak.values_astype(dpx_el, "float64")
        #             py = py - ak.values_astype(dpy_el, "float64")
            
        #         ptv, phiv = _pxpy_to_ptphi(px, py)
        #         return ak.zip({"pt": ak.values_astype(ptv, "float32"),
        #                        "phi": ak.values_astype(phiv, "float32")}, with_name="MET")
        
        #     PuppiMETCorr_jesUp   = _met_variant(pt_step_jesUp,   pt_jesUp)
        #     PuppiMETCorr_jesDown = _met_variant(pt_step_jesDown, pt_jesDown)
            
        #     systs["jets"]["_jesup"]   = jets_jesUp
        #     systs["jets"]["_jesdown"] = jets_jesDown
        #     systs["met"]["_jesup"]    = PuppiMETCorr_jesUp
        #     systs["met"]["_jesdown"]  = PuppiMETCorr_jesDown
            
                
        # # ============================================ #
        # # STEP 7 : Unclustered MET (_umetup/_umetdown) #
        # # ============================================ #
        # pm_fields = set(ak.fields(events.PuppiMET))
        # need = {"ptUnclusteredUp", "phiUnclusteredUp", "ptUnclusteredDown", "phiUnclusteredDown"}
        
        # if need.issubset(pm_fields):
        #     # baseline: your corrected MET
        #     px_corr, py_corr = _ptphi_to_pxpy(PuppiMETCorr.pt, PuppiMETCorr.phi)
        
        #     # deltas from Nano (raw PuppiMET → UnclusteredUp/Down)
        #     px0,  py0  = _ptphi_to_pxpy(events.PuppiMET.pt,                events.PuppiMET.phi)
        #     pxUp, pyUp = _ptphi_to_pxpy(events.PuppiMET.ptUnclusteredUp,   events.PuppiMET.phiUnclusteredUp)
        #     pxDn, pyDn = _ptphi_to_pxpy(events.PuppiMET.ptUnclusteredDown, events.PuppiMET.phiUnclusteredDown)
        
        #     dpx_up = pxUp - px0
        #     dpy_up = pyUp - py0
        #     dpx_dn = pxDn - px0
        #     dpy_dn = pyDn - py0
        
        #     # apply those deltas on top of your corrected MET
        #     pt_up, phi_up = _pxpy_to_ptphi(px_corr + dpx_up, py_corr + dpy_up)
        #     pt_dn, phi_dn = _pxpy_to_ptphi(px_corr + dpx_dn, py_corr + dpy_dn)
        
        #     PuppiMETCorr_umetUp = ak.with_field(PuppiMETCorr, ak.values_astype(pt_up,  "float32"), "pt")
        #     PuppiMETCorr_umetUp = ak.with_field(PuppiMETCorr_umetUp, ak.values_astype(phi_up, "float32"), "phi")
        
        #     PuppiMETCorr_umetDown = ak.with_field(PuppiMETCorr, ak.values_astype(pt_dn,  "float32"), "pt")
        #     PuppiMETCorr_umetDown = ak.with_field(PuppiMETCorr_umetDown, ak.values_astype(phi_dn, "float32"), "phi")
        
        #     systs["met"]["_umetup"]   = PuppiMETCorr_umetUp
        #     systs["met"]["_umetdown"] = PuppiMETCorr_umetDown
                            
        # # ================================================= #
        # # STEP 8 : Theory weight systematics (PDF / scales) #
        # # ================================================= #
        # w_one = np.ones(n_ev, dtype="float64")
        
        # #--- PDF up/down ---#
        # if hasattr(events, "LHEPdfWeight"):
        #     pdfw = events.LHEPdfWeight  
        #     has_any = (ak.num(pdfw, axis=1) > 0)
        #     # mean and std per event
        #     pdf_mean = ak.where(has_any, ak.mean(pdfw, axis=1), 1.0)
        #     pdf_var  = ak.where(has_any, ak.mean((pdfw - pdf_mean)**2, axis=1), 0.0)
        #     pdf_std  = ak.to_numpy(ak.fill_none(np.sqrt(ak.to_numpy(pdf_var)), 0.0))
        #     w_pdf_up   = 1.0 + pdf_std
        #     w_pdf_down = 1.0 - pdf_std
        # else:
        #     w_pdf_up = w_one
        #     w_pdf_down = w_one
        
        # systs["weights"]["_pdfup"]   = w_pdf_up
        # systs["weights"]["_pdfdown"] = w_pdf_down


            
###################################################### S T A R T   T H E   A N A L Y S I S ##################################################### 
            
        # STEP0: Raw events
        output["eventflow_boosted"].fill(cut="raw", weight=np.sum(weights.weight()))
        output["eventflow_resolved"].fill(cut="raw", weight=np.sum(weights.weight()))
        
        output["e_eventflow_boosted"].fill(cut="raw", weight=np.sum(weights.weight()))
        output["e_eventflow_resolved"].fill(cut="raw", weight=np.sum(weights.weight()))
        
        output["mu_eventflow_boosted"].fill(cut="raw", weight=np.sum(weights.weight()))
        output["mu_eventflow_resolved"].fill(cut="raw", weight=np.sum(weights.weight()))
                      
        # ========== Object Configuration ========== #
        
        ## LEPTONS
        if self.isQCD:
            # QCD-enriched region selection (Loose ID for QCD & Non-Isolated)
            muons = events.Muon[(events.Muon.pt > 20) & (np.abs(events.Muon.eta) < 2.4) 
                                & events.Muon.tightId & (events.Muon.pfRelIso04_all >= 0.15)]    
            electrons = ElectronCorr[(ElectronCorr.pt > 20) & (np.abs(ElectronCorr.eta) < 2.5) 
                                        & (ElectronCorr.cutBased >= 2) & (ElectronCorr.pfRelIso03_all >= 0.15)]
        else:
            # Standard signal region selection
            muons = events.Muon[(events.Muon.pt > 20) & (np.abs(events.Muon.eta) < 2.4) 
                                & events.Muon.tightId & (events.Muon.pfRelIso04_all < 0.15)]
            electrons = ElectronCorr[(ElectronCorr.pt > 20) & (np.abs(ElectronCorr.eta) < 2.5) 
                                        & (ElectronCorr.cutBased >= 4) & (ElectronCorr.pfRelIso03_all < 0.15)]

        muons = ak.with_field(muons, "mu", "lepton_type")
        electrons = ak.with_field(electrons, "e", "lepton_type")
        
        leptons    = ak.concatenate([muons, electrons], axis=1)
        leptons    = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
        n_leptons  = ak.num(leptons)
        leading_pt = ak.firsts(leptons.pt)

                                    
        ## JETS       
        goodJet = (
            (jets.pt > 20)
            & (np.abs(jets.eta) < 2.5)
            & jets.passJetIdTightLepVeto
        )

        # WP constants 
        BTAG_WP_TIGHT   = float(self._btag_wp_threshold("T"))     # AK4 single b-tag (UParT AK4B)
        DBTAG_WP_MEDIUM = 0.38                                    # AK4 double-b (UParT AK4probbb)
                
        single_jets = jets[goodJet]
        double_jets = jets[goodJet]
             
        # ========== Jet Cleaning ========== #
        single_jets = clean_by_dr(single_jets, leptons, 0.4)
        double_jets = clean_by_dr(double_jets, leptons, 0.4)
        
        # Sort by btag score
        single_jets = single_jets[ak.argsort(single_jets.btagUParTAK4B, ascending=False)]
        double_jets = double_jets[ak.argsort(double_jets.btagUParTAK4probbb, ascending=False)]
        
        n_single_jets = ak.num(single_jets)
        n_double_jets = ak.num(double_jets)
        
        # Single AK4 jets
        single_bjets       = single_jets[single_jets.btagUParTAK4B >= BTAG_WP_TIGHT]
        single_bjets       = single_bjets[ak.argsort(single_bjets.btagUParTAK4B, ascending=False)]
        single_untag_jets  = single_jets[single_jets.btagUParTAK4B <  BTAG_WP_TIGHT]
        ssingle_untag_jets = single_untag_jets[ak.argsort(single_untag_jets.btagUParTAK4B, ascending=True)]
        
        n_single_bjets = ak.num(single_bjets)
        
        # Double AK4 jets       
        double_jets    = double_jets[ak.argsort(double_jets.btagUParTAK4probbb, axis=-1, ascending=False)] 
        n_double_jets  = ak.num(double_jets)
        
        double_bjets      = double_jets[double_jets.btagUParTAK4probbb >= DBTAG_WP_MEDIUM]
        double_bjets      = double_bjets[ak.argsort(double_bjets.btagUParTAK4probbb, ascending=False)]
        double_untag_jets = double_jets[double_jets.btagUParTAK4probbb <  DBTAG_WP_MEDIUM]
        double_untag_jets = double_untag_jets[ak.argsort(double_untag_jets.pt, ascending=False)]
        
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
        
            # --- define ε binning (use SF JSON pT edges for consistency; simple |eta| binning) --- #
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
            
        
#=============================================================== EVENT SELECTION ===============================================================#

        # CUTFLOW:
        # mask_step1  : at least 1 lepton
        # mask_step2a : at least 2 AK4 double jets
        # mask_step2b : at least 3 AK4 single jets
        # mask_step3a : at least 2 AK4 double b-tag jets
        # mask_step3b : at least 3 AK4 single b-tag jets
        # mask_step4  : MET>25, MTW>50
        
        
        for i, cut in enumerate(self.optim_Cuts1_bdt):
            output["all_optim_cut"].fill(cut_index=i, var=0.0, weight=cut)

        for label in self.systematics_labels:
            output["all_optim_systs"].fill(syst=label, weight=1)
        

        ###############################
        # STEP 1: Exactly one lepton #
        ###############################
        print("\nStarting STEP 1: Exactly one lepton")
        
        has_1lep = n_leptons == 1
        print(f"Events with exactly one lepton: {ak.sum(has_1lep)}/{len(has_1lep)}")
    
        lep  = leptons[has_1lep]    
        lead = ak.firsts(lep)

        if len(lead) == 0:
            return output
        
        pt_sel = (
            ((lead.lepton_type == "e")  & (lead.pt > 35)) |
            ((lead.lepton_type == "mu") & (lead.pt > 30))
        )
        print(f"\nLeptons passing pt cuts: {ak.sum(pt_sel)} / {len(pt_sel)}")
        
        mask_step1 = np.zeros(len(events), dtype=bool)
        mask_step1[np.where(ak.to_numpy(has_1lep))[0]] = ak.to_numpy(pt_sel)
        print(f"Events passing step1: {np.sum(mask_step1)} / {len(mask_step1)}")
        
        leptons_1   = leptons[mask_step1]
        n_leptons_1 = ak.num(leptons_1)
        
        # Event categorization by lepton type
        pass_step1   = np.where(mask_step1)[0]
        tag_cat      = np.full(len(events), "", dtype="U2")
        
        if pass_step1.size > 0:
            lead_lep_step1  = leptons_1[:, 0]
            lead_type_step1 = ak.to_numpy(lead_lep_step1.lepton_type)
            tag_cat[pass_step1] = lead_type_step1
            
            # print("\n--- Leading leptons (step1) ---")
            # for i in range(min(len(lead_type_step1), 10)):
            #     print(f"[{i}] Type: {lead_type_step1[i]}")
            
        output["eventflow_boosted"].fill(cut="step1", weight=np.sum(weights.weight()[mask_step1]))
        output["eventflow_resolved"].fill(cut="step1", weight=np.sum(weights.weight()[mask_step1]))
        
        mask_mu = (tag_cat == "mu")
        mask_e  = (tag_cat == "e")
        
        if np.any(mask_mu & mask_step1):
            output["mu_eventflow_boosted"].fill(cut="step1", weight=np.sum(weights.weight()[mask_step1 & mask_mu]))
            output["mu_eventflow_resolved"].fill(cut="step1", weight=np.sum(weights.weight()[mask_step1 & mask_mu]))                                                  
        if np.any(mask_e & mask_step1):
            output["e_eventflow_boosted"].fill(cut="step1", weight=np.sum(weights.weight()[mask_step1 & mask_e]))
            output["e_eventflow_resolved"].fill(cut="step1", weight=np.sum(weights.weight()[mask_step1 & mask_e]))
            
        print("[CHK] step1:",
          f"tot={np.sum(mask_step1)}  mu={np.sum(mask_step1 & (tag_cat=='mu'))}  e={np.sum(mask_step1 & (tag_cat=='e'))}")

            
        # Triggers + Noise filters
        #---------------------------------------------------------------------
        hasTrigger = events.has_trigger
        trigger_2  = (events.trigger_type & (1 << 2)) != 0  # IsoMu24
        trigger_4  = (events.trigger_type & (1 << 4)) != 0  # Ele30_WPTight_Gsf
        
        hasTriggerMu = ((hasTrigger) & (trigger_2))
        hasTriggerE  = ((hasTrigger) & (trigger_4))
                
        final_trigger_mask          = np.zeros(len(events), dtype=bool)
        final_trigger_mask[mask_mu] = ak.to_numpy(hasTriggerMu[mask_mu])
        final_trigger_mask[mask_e]  = ak.to_numpy(hasTriggerE[mask_e])
        
        mask_step1 = mask_step1 & final_trigger_mask
        
        #------------------------------------------------------------------------------------------------------------------------------------------#
        # Lepton weights
        #------------------------------------------------------------------------------------------------------------------------------------------#
        w_lep = np.ones(len(events), dtype=float)
        
        # Masks for channel at step1
        mask_mu_evt = np.zeros(len(events), dtype=bool)
        mask_e_evt  = np.zeros(len(events), dtype=bool)
        pass_idx    = np.where(mask_step1)[0]
        
        if pass_idx.size:
            lead_step1 = leptons[mask_step1][:, 0]
            ch = ak.to_numpy(lead_step1.lepton_type)
            mask_mu_evt[pass_idx] = (ch == "mu")
            mask_e_evt[pass_idx]  = (ch == "e")
        
        
        # ---------- MUON CHANNEL ----------
        sel_mu_evt = mask_step1 & mask_mu_evt
        
        if np.any(sel_mu_evt):
            lead_mu   = leptons[sel_mu_evt][:, 0]
            mu_eta    = ak.to_numpy(lead_mu.eta).astype(float)
            mu_abseta = np.abs(mu_eta)
            mu_pt     = ak.to_numpy(lead_mu.pt).astype(float)
        
            mu_eta    = np.clip(mu_eta,    np.nextafter(-2.4, -1.0), np.nextafter( 2.4, -1.0))
            mu_abseta = np.clip(mu_abseta, np.nextafter( 0.0,  1.0), np.nextafter( 2.4, -1.0))
            mu_pt     = np.maximum(mu_pt,  np.nextafter(15.0, 1.0))
            mu_pt_hlt = np.maximum(mu_pt,  26.000001)
        
            w_mu = np.ones_like(mu_pt, dtype=float)
            
            try:
                if self._mu_id  is not None: w_mu *= self._mu_id.evaluate(mu_eta,     mu_pt,     "nominal")
                if self._mu_iso is not None: w_mu *= self._mu_iso.evaluate(mu_eta,    mu_pt,     "nominal")
                if self._mu_hlt is not None: w_mu *= self._mu_hlt.evaluate(mu_abseta, mu_pt_hlt, "nominal")
            except Exception as e:
                print(f"[ANA:Muon] SF eval failed: {e}")
            w_lep[sel_mu_evt] = w_mu
        
        # ---------- ELECTRON CHANNEL ----------
        sel_e_evt = mask_step1 & mask_e_evt
        if np.any(ak.to_numpy(sel_e_evt)):
            # take the highest-pt electron per selected event
            ele_evt = ElectronCorr[sel_e_evt]  # use your corrected electrons
            lead_ele = ele_evt[ak.argmax(ele_evt.pt, axis=1, keepdims=True)][:, 0]
        
            nsel = int(np.count_nonzero(ak.to_numpy(sel_e_evt)))
        
            # Electron ID Tight SF (TH2: x=eta, y=pt)
            w_ele_id = np.ones(nsel, dtype=float)
            if (self._ele_id_vals is not None):
                sf2d = _lookup_th2_vals(
                    lead_ele.eta, lead_ele.pt,
                    self._ele_id_edges_x, self._ele_id_edges_y, self._ele_id_vals
                )
                w_ele_id = np.asarray(sf2d, dtype=float)
        
            # Electron HLT Ele30 TightID using eff(Data/MC)
            w_ele_hlt = np.ones_like(w_ele_id)
            if (self._ele_hlt_effD is not None) and (self._ele_hlt_effM is not None):
                effD = _lookup_th2_vals(
                    lead_ele.superclusterEta, lead_ele.pt,
                    self._ele_hlt_edges_x, self._ele_hlt_edges_y, self._ele_hlt_effD
                )
                effM = _lookup_th2_vals(
                    lead_ele.superclusterEta, lead_ele.pt,
                    self._ele_hlt_edges_x, self._ele_hlt_edges_y, self._ele_hlt_effM
                )
                epsD = np.clip(np.asarray(effD, dtype=float), 0.0, 1.0)
                epsM = np.clip(np.asarray(effM, dtype=float), 0.0, 1.0)
                np.divide(epsD, epsM, out=w_ele_hlt, where=(epsM > 0))
        
            w_ele = np.nan_to_num(w_ele_id * w_ele_hlt, nan=1.0, posinf=1.0, neginf=1.0)
            w_lep[ak.to_numpy(sel_e_evt)] = w_ele
                    
        if self.isMC:
            weights.add("lepton_SFs", w_lep)
            
 #------------------------------------------------------------------------------------------------------------------------------------------#
            
        
        leptons_1   = leptons[mask_step1]
        n_leptons_1 = ak.num(leptons_1)
                
        single_jets_1            = single_jets[mask_step1]
        n_single_jets_1          = ak.num(single_jets_1)
        
        single_untag_jets_1      = single_untag_jets[mask_step1]
        n_single_untag_jets_1    = ak.num(single_untag_jets_1)
        
        double_jets_1            = double_jets[mask_step1]
        n_double_jets_1          = ak.num(double_jets_1)
        
        double_untag_jets_1      = double_untag_jets[mask_step1]
        n_double_untag_jets_1    = ak.num(double_untag_jets_1)
        
        # Histogram plotting     
        output["eventflow_boosted"].fill(cut="trigger",  weight=np.sum(weights.weight()[mask_step1]))
        output["eventflow_resolved"].fill(cut="trigger", weight=np.sum(weights.weight()[mask_step1]))
        
        output["lepton_multi_bef"].fill(n=n_leptons,   weight=weights.weight())
        output["lepton_multi_aft"].fill(n=n_leptons_1, weight=weights.weight()[mask_step1])
        
        output["single_jets_multi_bef_resolved"].fill(n=n_single_jets_1 ,             weight=weights.weight()[mask_step1])       
        output["double_jets_multi_bef_boosted"].fill(n=n_double_jets_1 ,            weight=weights.weight()[mask_step1])
        
        if ak.any(mask_mu[mask_step1]):
            output["mu_eventflow_boosted"].fill(cut="trigger",  weight=np.sum(weights.weight()[mask_step1 & mask_mu]))
            output["mu_eventflow_resolved"].fill(cut="trigger", weight=np.sum(weights.weight()[mask_step1 & mask_mu]))                                                  
        if ak.any(mask_e[mask_step1]):
            output["e_eventflow_boosted"].fill(cut="trigger",  weight=np.sum(weights.weight()[mask_step1 & mask_e]))
            output["e_eventflow_resolved"].fill(cut="trigger", weight=np.sum(weights.weight()[mask_step1 & mask_e]))
        
        #============================================#
        #                                            #
        #  Boosted analysis: ma: 12, 15, 20, 25, 30  #
        #                                            #
        #============================================#

        #######################################
        # STEP 2a: At least 2 double AK4 jets #
        #######################################
                    
        print("\nStarting STEP 2a: At least 2 double AK4 jets")                                                                                                                                
        mask_step2a = mask_step1 & (n_double_jets >= 2)
        print(f"After STEP 2a: {np.sum(mask_step2a)} events remaining")

        output["eventflow_boosted"].fill(cut="step2", weight=np.sum(weights.weight()[mask_step2a]))
        
        double_jets_2a    = double_jets[mask_step2a]
        n_double_jets_2a  = ak.num(double_jets_2a)
        
        double_bjets_2a   = double_bjets[mask_step2a]
        n_double_bjets_2a = ak.num(double_bjets_2a)
       
        double_untag_jets_2a   = double_untag_jets[mask_step2a]
        n_double_untag_jets_2a = ak.num(double_untag_jets_2a)
        
        w2a = weights.weight()[mask_step2a]
        
        has_ge1_db = (n_double_jets_2a >= 1)
        has_ge2_db = (n_double_jets_2a >= 2)
        
        lead_db       = double_jets_2a[has_ge1_db][:, 0]
        lead_db_score = ak.to_numpy(lead_db.btagUParTAK4probbb)
        w_lead_db     = ak.to_numpy(w2a[has_ge1_db])
        
        sublead_db      = double_jets_2a[has_ge2_db][:, 1]
        sublead_db_score = ak.to_numpy(sublead_db.btagUParTAK4probbb)
        w_sublead_db     = ak.to_numpy(w2a[has_ge2_db])
               
        # Histogram plotting
        output["double_jets_multi_aft_boosted"].fill(n=n_double_jets_2a ,            weight=w2a)
        output["double_bjets_multi_bef_boosted"].fill(n=n_double_bjets_2a,           weight=w2a)
        output["double_btag_score_lead"].fill(score=lead_db_score,                   weight=w_lead_db)
        output["double_btag_score_sublead"].fill(score=sublead_db_score,             weight=w_sublead_db)
      
        if ak.any(mask_mu[mask_step2a]):
            mu_mask_2a = ak.to_numpy(mask_mu[mask_step2a])
            w2a_mu = w2a[mu_mask_2a] 
            
            output["mu_eventflow_boosted"].fill(cut="step2", weight=np.sum(w2a_mu))
                                            
        if ak.any(mask_e[mask_step2a]):
            e_mask_2a = ak.to_numpy(mask_e[mask_step2a])
            w2a_e = w2a[e_mask_2a]
            
            output["e_eventflow_boosted"].fill(cut="step2", weight=np.sum(w2a_e))
            


        #############################################
        # STEP 3a: At least 2 double b-tag AK4 jets #
        #############################################
        
        print("\nStarting STEP 3a: At least 2 double b-tag AK4 jets")
        mask_step3a = mask_step2a & (n_double_bjets >= 2)
        print(f"After STEP 3a: {np.sum(mask_step3a)} events remaining")
        
        double_bjets_3a      = double_bjets[mask_step3a]
        n_double_bjets_3a    = ak.num(double_bjets_3a)
        
        lead_l_3a            = leptons[mask_step3a][:, 0]     
        vec_lead_l_3a        = make_vector(lead_l_3a)
        
        met_3a               = PuppiMETCorr[mask_step3a]
        vec_met_3a           = make_vector_met(met_3a)       
        mTW_3a               = trans_massW(vec_lead_l_3a, vec_met_3a)
        
        # Histogram plotting
        w3a = weights.weight()[mask_step3a]
        
        output["eventflow_boosted"].fill(cut="step3", weight=np.sum(w3a))
        
        output["double_bjets_multi_aft_boosted"].fill(n=n_double_bjets_3a, weight=w3a)
        
        output["MTW_bef_boosted"].fill(m=mTW_3a,     weight=w3a)
        output["MET_bef_boosted"].fill(pt=met_3a.pt, weight=w3a)
        
        if ak.any(mask_mu[mask_step3a]):
            mu3a = np.asarray(mask_mu[mask_step3a])
            w3a_mu = w3a[mu3a]
            
            output["mu_eventflow_boosted"].fill(cut="step3", weight=np.sum(w3a_mu))     
                                  
        if ak.any(mask_e[mask_step3a]):
            e3a = np.asarray(mask_e[mask_step3a])
            w3a_e = w3a[e3a]
            
            output["e_eventflow_boosted"].fill(cut="step3", weight=np.sum(w3a_e))
                   
        #################################
        # STEP 4a: At MET>25 and MTW>50 #
        #################################
        
        print("\nStarting STEP 4a: MET>25 and MTW>50")
        
        mask_met_3a_full = np.zeros(len(events), dtype=bool)
        mask_mtw_3a_full = np.zeros(len(events), dtype=bool)
        
        mask_met_3a_full[mask_step3a] = (met_3a.pt > 25)
        mask_mtw_3a_full[mask_step3a] = (mTW_3a > 50)           
               
        mask_step4a = mask_step3a & mask_met_3a_full & mask_mtw_3a_full
        print(f"After STEP 4a: {np.sum(mask_step4a)} events remaining")
        print(f"Events passing MET cut only: {np.sum(mask_step3a & mask_met_3a_full)}")
        print(f"Events passing MTW cut only: {np.sum(mask_step3a & mask_mtw_3a_full)}")
        
        double_jets_4a       = double_jets[mask_step4a]
        double_bjets_4a      = double_bjets[mask_step4a]
        double_untag_jets_4a = double_untag_jets[mask_step4a]
                  
        HT_4a                = ak.sum(double_jets_4a.pt, axis=1)
        
        lead_bb_4a           = double_bjets_4a[:, 0]
        sublead_bb_4a        = double_bjets_4a[:, 1]
        
        lead_l_4a            = leptons[mask_step4a][:, 0]  
        met_4a               = PuppiMETCorr[mask_step4a]       
        mTW_4a               = trans_massW(lead_l_4a, met_4a)
        
        vec_lead_l_4a        = make_vector(lead_l_4a)
        vec_met_4a           = make_vector_met(met_4a)
        vec_W_4a             = vec_lead_l_4a + vec_met_4a
        
        vec_lead_bb_4a       = make_vector(lead_bb_4a)
        vec_sublead_bb_4a    = make_vector(sublead_bb_4a) 
        vec_H_4a             = vec_lead_bb_4a + vec_sublead_bb_4a  
        
        btag_max_4a          = double_bjets_4a[:, 0].btagUParTAK4probbb
        btag_min_4a          = double_bjets_4a[:, 1].btagUParTAK4probbb
        btag_prod_4a         = lead_bb_4a.btagUParTAK4probbb * sublead_bb_4a.btagUParTAK4probbb
        
        dphi_wh_4a           = np.abs(vec_H_4a.delta_phi(vec_W_4a))      
        dphi_metlep_4a       = np.abs(vec_met_4a.delta_phi(vec_lead_l_4a))   
        deta_wh_4a           = np.abs(vec_W_4a.eta - vec_H_4a.eta)
        dr_wh_4a             = vec_H_4a.delta_r(vec_W_4a)              
        dmbb_4a              = np.abs(vec_lead_bb_4a.mass - vec_sublead_bb_4a.mass)              
        min_dphi_lepjet_4a   = min_dphi_jets_lepton(jets=double_jets_4a, leptons=lead_l_4a)              
        dr_bb_4a             = vec_lead_bb_4a.delta_r(vec_sublead_bb_4a)       
        pt_ratio_4a          = np.where(vec_W_4a.pt > 0, vec_H_4a.pt / vec_W_4a.pt, -1)
        wh_pt_asymmetry_4a   = np.abs(vec_H_4a.pt - vec_W_4a.pt) / (vec_H_4a.pt + vec_W_4a.pt)        
        
      
        # Histogram plotting
        w4a = weights.weight()[np.asarray(mask_step4a)]
        
        output["eventflow_boosted"].fill(cut="step4", weight=np.sum(w4a))
        
        output["HT_boosted"].fill(ht=HT_4a,                                 weight=w4a)
        output["pt_bb1_boosted"].fill(pt=lead_bb_4a.pt,                     weight=w4a)
        output["pt_bb2_boosted"].fill(pt=sublead_bb_4a.pt,                  weight=w4a)
        output["pt_lepton_boosted"].fill(pt=lead_l_4a.pt,                   weight=w4a)
        output["MET_boosted"].fill(pt=met_4a.pt,                            weight=w4a)
        output["MTW_boosted"].fill(m=mTW_4a,                                weight=w4a)
        output["pt_W_boosted"].fill(pt=vec_W_4a.pt,                         weight=w4a)
        output["mass_H_boosted"].fill(m=vec_H_4a.mass,                      weight=w4a)
        output["pt_H_boosted"].fill(pt=vec_H_4a.pt,                         weight=w4a)
        output["btag_max_double_bjets_boosted"].fill(btag=btag_max_4a,      weight=w4a)
        output["btag_min_double_bjets_boosted"].fill(btag=btag_min_4a,      weight=w4a)
        output["dphi_WH_boosted"].fill(dphi=dphi_wh_4a,                     weight=w4a)
        output["dr_WH_boosted"].fill(dr=dr_wh_4a,                           weight=w4a)
        output["dphi_jet-lepton_min_boosted"].fill(dphi=min_dphi_lepjet_4a, weight=w4a)
        output["dphi_MET-lepton_boosted"].fill(dphi=dphi_metlep_4a,         weight=w4a)
        output["dr_bb_boosted"].fill(dr=dr_bb_4a,                           weight=w4a)
        output["pt_ratio_boosted"].fill(ratio=pt_ratio_4a,                  weight=w4a)
        output["btag_prod_boosted"].fill(btag_prod=btag_prod_4a,            weight=w4a)
        output["deta_WH_boosted"].fill(deta=deta_wh_4a,                     weight=w4a)
        output["eta_bb1_boosted"].fill(eta=lead_bb_4a.eta,                  weight=w4a)
        output["eta_bb2_boosted"].fill(eta=sublead_bb_4a.eta,               weight=w4a)
        output["phi_bb1_boosted"].fill(phi=lead_bb_4a.phi,                  weight=w4a)
        output["phi_bb2_boosted"].fill(phi=sublead_bb_4a.phi,               weight=w4a)
        #output["eta_MET_boosted"].fill(eta=met_4a.eta,                      weight=w4a)
        output["phi_MET_boosted"].fill(phi=met_4a.phi,                      weight=w4a)
        
        mu_m4a = np.asarray(mask_mu[mask_step4a])
        e_m4a  = np.asarray(mask_e[mask_step4a])
                
        if np.any(mu_m4a):
            w_mu4a = w4a[mu_m4a]
            
            output["mu_eventflow_boosted"].fill(cut="step4", weight=np.sum(w_mu4a))
            
            output["mu_A_HT_boosted"].fill(ht=HT_4a[mu_m4a],                                 weight=w_mu4a)
            output["mu_A_pt_bb1_boosted"].fill(pt=lead_bb_4a[mu_m4a].pt,                     weight=w_mu4a)
            output["mu_A_pt_bb2_boosted"].fill(pt=sublead_bb_4a[mu_m4a].pt,                  weight=w_mu4a)
            output["mu_A_pt_lepton_boosted"].fill(pt=lead_l_4a[mu_m4a].pt,                   weight=w_mu4a)
            output["mu_A_MET_boosted"].fill(pt=met_4a[mu_m4a].pt,                            weight=w_mu4a)
            output["mu_A_MTW_boosted"].fill(m=mTW_4a[mu_m4a],                                weight=w_mu4a)
            output["mu_A_pt_W_boosted"].fill(pt=vec_W_4a[mu_m4a].pt,                         weight=w_mu4a)
            output["mu_A_mass_H_boosted"].fill(m=vec_H_4a[mu_m4a].mass,                      weight=w_mu4a)
            output["mu_A_pt_H_boosted"].fill(pt=vec_H_4a[mu_m4a].pt,                         weight=w_mu4a)
            output["mu_A_btag_max_double_bjets_boosted"].fill(btag=btag_max_4a[mu_m4a],      weight=w_mu4a)
            output["mu_A_btag_min_double_bjets_boosted"].fill(btag=btag_min_4a[mu_m4a],      weight=w_mu4a)
            output["mu_A_dphi_WH_boosted"].fill(dphi=dphi_wh_4a[mu_m4a],                     weight=w_mu4a)
            output["mu_A_dr_WH_boosted"].fill(dr=dr_wh_4a[mu_m4a],                           weight=w_mu4a)
            output["mu_A_dphi_jet-lepton_min_boosted"].fill(dphi=min_dphi_lepjet_4a[mu_m4a], weight=w_mu4a)
            output["mu_A_dphi_MET-lepton_boosted"].fill(dphi=dphi_metlep_4a[mu_m4a],         weight=w_mu4a)
            output["mu_A_dr_bb_boosted"].fill(dr=dr_bb_4a[mu_m4a],                           weight=w_mu4a)
            output["mu_A_btag_prod_boosted"].fill(btag_prod=btag_prod_4a[mu_m4a],            weight=w_mu4a)
            output["mu_A_deta_WH_boosted"].fill(deta=deta_wh_4a[mu_m4a],                     weight=w_mu4a)
            output["mu_A_eta_bb1_boosted"].fill(eta=lead_bb_4a[mu_m4a].eta,                  weight=w_mu4a)
            output["mu_A_eta_bb2_boosted"].fill(eta=sublead_bb_4a[mu_m4a].eta,               weight=w_mu4a)
            output["mu_A_phi_bb1_boosted"].fill(phi=lead_bb_4a[mu_m4a].phi,                  weight=w_mu4a)
            output["mu_A_phi_bb2_boosted"].fill(phi=sublead_bb_4a[mu_m4a].phi,               weight=w_mu4a)
            #output["mu_A_eta_MET_boosted"].fill(eta=met_4a[mu_m4a].eta,                      weight=w_mu4a)
            output["mu_A_phi_MET_boosted"].fill(phi=met_4a[mu_m4a].phi,                      weight=w_mu4a)
            
                                                   
        if np.any(e_m4a):
            w_e4a = w4a[e_m4a]
            output["e_eventflow_boosted"].fill(cut="step4", weight=np.sum(w_e4a))
            
            output["e_A_HT_boosted"].fill(ht=HT_4a[e_m4a],                                 weight=w_e4a)
            output["e_A_pt_bb1_boosted"].fill(pt=lead_bb_4a[e_m4a].pt,                     weight=w_e4a)
            output["e_A_pt_bb2_boosted"].fill(pt=sublead_bb_4a[e_m4a].pt,                  weight=w_e4a)
            output["e_A_pt_lepton_boosted"].fill(pt=lead_l_4a[e_m4a].pt,                   weight=w_e4a)
            output["e_A_MET_boosted"].fill(pt=met_4a[e_m4a].pt,                            weight=w_e4a)
            output["e_A_MTW_boosted"].fill(m=mTW_4a[e_m4a],                                weight=w_e4a)
            output["e_A_pt_W_boosted"].fill(pt=vec_W_4a[e_m4a].pt,                         weight=w_e4a)
            output["e_A_mass_H_boosted"].fill(m=vec_H_4a[e_m4a].mass,                      weight=w_e4a)
            output["e_A_pt_H_boosted"].fill(pt=vec_H_4a[e_m4a].pt,                         weight=w_e4a)
            output["e_A_btag_max_double_bjets_boosted"].fill(btag=btag_max_4a[e_m4a],      weight=w_e4a)
            output["e_A_btag_min_double_bjets_boosted"].fill(btag=btag_min_4a[e_m4a],      weight=w_e4a)
            output["e_A_dphi_WH_boosted"].fill(dphi=dphi_wh_4a[e_m4a],                     weight=w_e4a)
            output["e_A_dr_WH_boosted"].fill(dr=dr_wh_4a[e_m4a],                           weight=w_e4a)
            output["e_A_dphi_jet-lepton_min_boosted"].fill(dphi=min_dphi_lepjet_4a[e_m4a], weight=w_e4a)
            output["e_A_dphi_MET-lepton_boosted"].fill(dphi=dphi_metlep_4a[e_m4a],         weight=w_e4a)
            output["e_A_dr_bb_boosted"].fill(dr=dr_bb_4a[e_m4a],                           weight=w_e4a)
            output["e_A_pt_ratio_boosted"].fill(ratio=pt_ratio_4a[e_m4a],                  weight=w_e4a)
            output["e_A_btag_prod_boosted"].fill(btag_prod=btag_prod_4a[e_m4a],            weight=w_e4a)
            output["e_A_deta_WH_boosted"].fill(deta=deta_wh_4a[e_m4a],                     weight=w_e4a)
            output["e_A_eta_bb1_boosted"].fill(eta=lead_bb_4a[e_m4a].eta,                  weight=w_e4a)
            output["e_A_eta_bb2_boosted"].fill(eta=sublead_bb_4a[e_m4a].eta,               weight=w_e4a)
            output["e_A_phi_bb1_boosted"].fill(phi=lead_bb_4a[e_m4a].phi,                  weight=w_e4a)
            output["e_A_phi_bb2_boosted"].fill(phi=sublead_bb_4a[e_m4a].phi,               weight=w_e4a)
            #output["e_A_eta_MET_boosted"].fill(eta=met_4a[e_m4a].eta,                      weight=w_e4a)
            output["e_A_phi_MET_boosted"].fill(phi=met_4a[e_m4a].phi,                      weight=w_e4a)
        
        weights_boosted = w4a
        n_boosted = len(weights_boosted)
        print(f"\nNumber of events after selection: {n_boosted}")
        bdt_boosted = {
            "H_mass"             : ak.to_numpy(vec_H_4a.mass),
            "H_pt"               : ak.to_numpy(vec_H_4a.pt),
            "MTW"                : ak.to_numpy(mTW_4a),
            "W_pt"               : ak.to_numpy(vec_W_4a.pt),
            "HT"                 : ak.to_numpy(HT_4a),
            "MET_pt"             : ak.to_numpy(vec_met_4a.pt),
            "btag_max"           : ak.to_numpy(btag_max_4a),
            "btag_min"           : ak.to_numpy(btag_min_4a),
            "dr_bb"              : ak.to_numpy(dr_bb_4a),
            "dm_bb"              : ak.to_numpy(dmbb_4a),
            "dphi_WH"            : ak.to_numpy(np.abs(dphi_wh_4a)),
            "dr_WH"              : ak.to_numpy(dr_wh_4a),
            "dphi_jet_lepton_min": ak.to_numpy(np.abs(min_dphi_lepjet_4a)),
            "pt_ratio"           : ak.to_numpy(pt_ratio_4a),
            "pt_lepton"          : ak.to_numpy(vec_lead_l_4a.pt),
            "pt_b1"              : ak.to_numpy(vec_lead_bb_4a.pt),
            "WH_pt_assymetry"    : ak.to_numpy(wh_pt_asymmetry_4a),
            "btag_prod"          : ak.to_numpy(btag_prod_4a),
            "deta_WH"            : ak.to_numpy(deta_wh_4a),
            "Njets"              : ak.to_numpy(ak.num(double_jets_4a)),
            "weight"             : ak.to_numpy(weights_boosted),
        }


        if self.isMVA:
            self.compat_tree_variables(bdt_boosted)
            self.add_tree_entry("boosted", bdt_boosted)


        if self.runEval and not self.isMVA :
            inputs_boosted = {k: bdt_boosted[k] for k in self.bdt_eval_boosted.var_list}
            bdt_score_boosted = np.ravel(self.bdt_eval_boosted.eval(inputs_boosted))
            output["bdt_score_boosted"].fill(bdt=bdt_score_boosted, weight=bdt_boosted["weight"])
            
            mu_mask_all4a = np.asarray(mask_mu[mask_step4a])
            e_mask_all4a  = np.asarray(mask_e[mask_step4a])
            w_all4a       = weights_boosted
            
            for i, cut in enumerate(self.optim_Cuts1_bdt):
                cut_mask = (bdt_score_boosted > cut)
                if not np.any(cut_mask):
                    continue
                        
                ele_mask_cut = e_mask_all4a & cut_mask
                mu_mask_cut  = mu_mask_all4a & cut_mask
               
                # ===== electrons =====
                if np.any(ele_mask_cut):
                    w_e = w_all4a[ele_mask_cut]
                    s_e = bdt_score_boosted[ele_mask_cut]
                    
                    output["e_bdt_score_boosted"].fill(bdt=s_e, weight=w_e)
                    output["e_A_SR_3b_bdt_shapes_boosted"].fill(cut_index=i, bdt=s_e, weight=w_e)
        
                    output["e_A_SR_3b_higgsMass_shapes_boosted"].fill(
                        cut_index=i, H_mass=ak.to_numpy(vec_H_4a.mass[ele_mask_cut]), weight=w_e)
                    output["e_A_SR_3b_higgsPt_shapes_boosted"].fill(
                        cut_index=i, H_pt=ak.to_numpy(vec_H_4a.pt[ele_mask_cut]), weight=w_e)
                    output["e_A_SR_3b_b1Pt_shapes_boosted"].fill(
                        cut_index=i, pt_b1=ak.to_numpy(vec_lead_bb_4a.pt[ele_mask_cut]), weight=w_e)
                    output["e_A_SR_3b_ht_shapes_boosted"].fill(
                        cut_index=i, HT=HT_4a[ele_mask_cut], weight=w_e)
                    output["e_A_SR_3b_pfmet_shapes_boosted"].fill(
                        cut_index=i, MET_pt=ak.to_numpy(vec_met_4a.pt[ele_mask_cut]), weight=w_e)
                    output["e_A_SR_3b_mtw_shapes_boosted"].fill(
                        cut_index=i, MTW=mTW_4a[ele_mask_cut], weight=w_e)
                    output["e_A_SR_3b_ptw_shapes_boosted"].fill(
                        cut_index=i, W_pt=ak.to_numpy(vec_W_4a.pt[ele_mask_cut]), weight=w_e)
                    output["e_A_SR_3b_dphiWh_shapes_boosted"].fill(
                        cut_index=i, dphi_WH=np.abs(dphi_wh_4a[ele_mask_cut]), weight=w_e)
                    output["e_A_SR_3b_dphijetlep_shapes_boosted"].fill(
                        cut_index=i, dphi_lep_met=np.abs(min_dphi_lepjet_4a[ele_mask_cut]), weight=w_e)
                    output["e_A_SR_3b_dRbb_shapes_boosted"].fill(
                        cut_index=i, dr_bb=dr_bb_4a[ele_mask_cut], weight=w_e)
                    output["e_A_SR_3b_dm_shapes_boosted"].fill(
                        cut_index=i, dm_bb=dmbb_4a[ele_mask_cut], weight=w_e)
                    output["e_A_SR_3b_lep_pt_raw_shapes_boosted"].fill(
                        cut_index=i, pt_lepton=ak.to_numpy(vec_lead_l_4a.pt[ele_mask_cut]), weight=w_e)
                    output["e_A_SR_3b_dRwh_shapes_boosted"].fill(
                        cut_index=i, dr_WH=dr_wh_4a[ele_mask_cut], weight=w_e)
                    output["e_A_SR_3b_ptratio_shapes_boosted"].fill(
                        cut_index=i, pt_ratio=pt_ratio_4a[ele_mask_cut], weight=w_e)
                    output["e_A_SR_3b_jets_shapes_boosted"].fill(
                        cut_index=i, n_jets=ak.num(double_jets_4a[ele_mask_cut]), weight=w_e)
                    output["e_A_SR_3b_btag_prod_shapes_boosted"].fill(
                        cut_index=i, btag_prod=btag_prod_4a[ele_mask_cut], weight=w_e)
            
                # ===== muons =====
                if np.any(mu_mask_cut):
                    w_mu = w_all4a[mu_mask_cut]
                    s_mu = bdt_score_boosted[mu_mask_cut]
        
                    output["mu_bdt_score_boosted"].fill(bdt=s_mu, weight=w_mu)
                    output["mu_A_SR_3b_bdt_shapes_boosted"].fill(cut_index=i, bdt=s_mu, weight=w_mu)
        
                    output["mu_A_SR_3b_higgsMass_shapes_boosted"].fill(
                        cut_index=i, H_mass=ak.to_numpy(vec_H_4a.mass[mu_mask_cut]), weight=w_mu)
                    output["mu_A_SR_3b_higgsPt_shapes_boosted"].fill(
                        cut_index=i, H_pt=ak.to_numpy(vec_H_4a.pt[mu_mask_cut]), weight=w_mu)
                    output["mu_A_SR_3b_b1Pt_shapes_boosted"].fill(
                        cut_index=i, pt_b1=ak.to_numpy(vec_lead_bb_4a.pt[mu_mask_cut]), weight=w_mu)
                    output["mu_A_SR_3b_ht_shapes_boosted"].fill(
                        cut_index=i, HT=HT_4a[mu_mask_cut], weight=w_mu)
                    output["mu_A_SR_3b_pfmet_shapes_boosted"].fill(
                        cut_index=i, MET_pt=ak.to_numpy(vec_met_4a.pt[mu_mask_cut]), weight=w_mu)
                    output["mu_A_SR_3b_mtw_shapes_boosted"].fill(
                        cut_index=i, MTW=mTW_4a[mu_mask_cut], weight=w_mu)
                    output["mu_A_SR_3b_ptw_shapes_boosted"].fill(
                        cut_index=i, W_pt=ak.to_numpy(vec_W_4a.pt[mu_mask_cut]), weight=w_mu)
                    output["mu_A_SR_3b_dphiWh_shapes_boosted"].fill(
                        cut_index=i, dphi_WH=np.abs(dphi_wh_4a[mu_mask_cut]), weight=w_mu)
                    output["mu_A_SR_3b_dphijetlep_shapes_boosted"].fill(
                        cut_index=i, dphi_lep_met=np.abs(min_dphi_lepjet_4a[mu_mask_cut]), weight=w_mu)
                    output["mu_A_SR_3b_dRbb_shapes_boosted"].fill(
                        cut_index=i, dr_bb=dr_bb_4a[mu_mask_cut], weight=w_mu)
                    output["mu_A_SR_3b_dm_shapes_boosted"].fill(
                        cut_index=i, dm_bb=dmbb_4a[mu_mask_cut], weight=w_mu)
                    output["mu_A_SR_3b_lep_pt_raw_shapes_boosted"].fill(
                        cut_index=i, pt_lepton=ak.to_numpy(vec_lead_l_4a.pt[mu_mask_cut]), weight=w_mu)
                    output["mu_A_SR_3b_dRwh_shapes_boosted"].fill(
                        cut_index=i, dr_WH=dr_wh_4a[mu_mask_cut], weight=w_mu)
                    output["mu_A_SR_3b_ptratio_shapes_boosted"].fill(
                        cut_index=i, pt_ratio=pt_ratio_4a[mu_mask_cut], weight=w_mu)
                    output["mu_A_SR_3b_jets_shapes_boosted"].fill(
                        cut_index=i, n_jets=ak.num(double_jets_4a[mu_mask_cut]), weight=w_mu)
                    output["mu_A_SR_3b_btag_prod_shapes_boosted"].fill(
                        cut_index=i, btag_prod=btag_prod_4a[mu_mask_cut], weight=w_mu)

        
        #=====================================================#                                                                                                                                                    
        #                                                     #
        #  Resolved analysis: ma: 30, 35, 40, 45, 50, 55, 60  #
        #                                                     #                                                                                                       
        #=====================================================#  
        
        #######################################
        # STEP 2b: At least 3 single AK4 jets #
        #######################################
                    
        print("\nStarting STEP 2b: At least 3 single AK4 jets")                                                                                                                                
        mask_step2b = mask_step1 & (n_single_jets >= 3) 
        print(f"After STEP 2b: {np.sum(mask_step2b)} events remaining")
        
        base_w_snapshot = weights.weight().copy()
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
        
        single_jets_2b         = single_jets[mask_step2b]
        n_single_jets_2b       = ak.num(single_jets_2b)
        
        single_untag_jets_2b   = single_untag_jets[mask_step2b]
        n_single_untag_jets_2b = ak.num(single_untag_jets_2b)
        
        single_bjets_2b        = single_bjets[mask_step2b]
        n_single_bjets_2b      = ak.num(single_bjets_2b)
        
        has_ge1_sj    = ak.num(single_jets_2b) >= 1
        lead_sj_score = ak.to_numpy(single_jets_2b[has_ge1_sj][:, 0].btagUParTAK4B)
        w_lead_sj     = ak.to_numpy(w2b[has_ge1_sj])
                
        has_ge2_sj       = ak.num(single_jets_2b) >= 2
        sublead_sj_score = ak.to_numpy(single_jets_2b[has_ge2_sj][:, 1].btagUParTAK4B)
        w_sublead_sj     = ak.to_numpy(w2b[has_ge2_sj])
                     
        # Histogram plotting        
        output["single_btag_score_lead"].fill(score=lead_sj_score, weight=w_lead_sj)
        output["single_btag_score_sublead"].fill(score=sublead_sj_score, weight=w_sublead_sj)
        
        output["eventflow_resolved"].fill(cut="step2", weight=np.sum(w2b))
        
        output["single_jets_multi_aft_resolved"].fill(n=n_single_jets_2b,             weight=w2b)
        output["single_bjets_multi_bef_resolved"].fill(n=n_single_bjets_2b,           weight=w2b)    
        
        ele_mask_2b = np.asarray(mask_e[mask_step2b])
        mu_mask_2b  = np.asarray(mask_mu[mask_step2b])
        
        if ak.any(mu_mask_2b):
            output["mu_eventflow_resolved"].fill(cut="step2", weight=np.sum(w2b[mu_mask_2b]))

        if ak.any(ele_mask_2b):
            output["e_eventflow_resolved"].fill(cut="step2", weight=np.sum(w2b[ele_mask_2b]))
       
        #############################################
        # STEP 3b: At least 3 single b-tag AK4 jets #
        #############################################
        
        print("\nStarting STEP 3b: At least 3 single b-tag AK4 jets")
        
        mask_step3b = mask_step2b & (n_single_bjets >= 3)
        print(f"After STEP 3b: {np.sum(mask_step3b)} events remaining")
                 
        single_bjets_3b   = single_bjets[mask_step3b]
        n_single_bjets_3b = ak.num(single_bjets_3b) 
        
        lead_l_3b = leptons[mask_step3b][:, 0]  
        vec_lead_l_3b = make_vector(lead_l_3b)
        
        met_3b = PuppiMETCorr[mask_step3b]
        vec_met_3b = make_vector_met(met_3b)
        
        mTW_3b = trans_massW(vec_lead_l_3b, vec_met_3b)
        
        # Histogram plotting
        w3b = weights.weight()[np.asarray(mask_step3b)]
        
        output["eventflow_resolved"].fill(cut="step3", weight=np.sum(w3b))
        
        output["single_bjets_multi_aft_resolved"].fill(n=n_single_bjets_3b, weight=w3b)
        output["MTW_bef_resolved"].fill(m=mTW_3b,                           weight=w3b)
        output["MET_bef_resolved"].fill(pt=met_3b.pt,                       weight=w3b)
        
        ele_mask_3b = np.asarray(mask_e[mask_step3b])
        mu_mask_3b  = np.asarray(mask_mu[mask_step3b])
        
        if ak.any(ele_mask_3b):
            output["e_eventflow_resolved"].fill(cut="step3",  weight=np.sum(w3b[ele_mask_3b]))

        if ak.any(mu_mask_3b):
            output["mu_eventflow_resolved"].fill(cut="step3", weight=np.sum(w3b[mu_mask_3b]))
        
        #################################
        # STEP 4b: At MET>25 and MTW>50 #
        #################################
        
        print("\nStarting STEP 4b: MET>25 and MTW>50")
        
        mask_met_3b_full = np.zeros(len(events), dtype=bool)
        mask_mtw_3b_full = np.zeros(len(events), dtype=bool)
        
        mask_met_3b_full[np.asarray(mask_step3b)] = (met_3b.pt > 25)
        mask_mtw_3b_full[np.asarray(mask_step3b)] = (mTW_3b   > 50)
        
        mask_step4b = mask_step3b & mask_met_3b_full & mask_mtw_3b_full
        print(f"After STEP 4b: {np.sum(mask_step4b)} events remaining")
        print(f"Events passing MET cut only: {np.sum(mask_step3b & mask_met_3b_full)}")
        print(f"Events passing MTW cut only: {np.sum(mask_step3b & mask_mtw_3b_full)}")
        
        single_jets_4b       = single_jets[mask_step4b]
        single_bjets_4b      = single_bjets[mask_step4b]
        
        vec_single_jets_4b   = make_vector(single_jets_4b)
        vec_single_bjets_4b  = make_vector(single_bjets_4b)
        
        n_sjs  = ak.num(single_jets_4b)
        
        mass_H, pt_H, phi_H, eta_H = higgs_kin(vec_single_bjets_4b, vec_single_jets_4b)
        
        met_4b             = PuppiMETCorr[mask_step4b]
        vec_met_4b         = make_vector_met(met_4b)
        
        lead_l_4b          = leptons[mask_step4b][:, 0]  
        vec_lead_l_4b      = make_vector(lead_l_4b)
        
        mTW_4b             = trans_massW(vec_lead_l_4b, vec_met_4b)
        vec_W_4b           = vec_met_4b + vec_lead_l_4b
        HT_4b              = ak.sum(single_jets_4b.pt, axis=1)
        
        dphi_metlep_4b     = np.abs(vec_met_4b.delta_phi(vec_lead_l_4b))
        dphi_wh_4b         = dphi_wh_4b = np.abs(((phi_H - vec_W_4b.phi + np.pi) % (2*np.pi)) - np.pi)
        deta_wh_4b         = np.abs(vec_W_4b.eta - eta_H)
        dr_wh_4b           = np.sqrt(deta_wh_4b**2 + dphi_wh_4b**2)
              
        btag_max_4b        = ak.max(single_bjets_4b.btagUParTAK4B, axis=1)
        btag_min_4b        = ak.min(single_bjets_4b.btagUParTAK4B, axis=1)
        btag_prod_4b       = single_bjets_4b[:, 0].btagUParTAK4B * single_bjets_4b[:, 1].btagUParTAK4B
             
        dr_bb_avg_4b       = dr_bb_bb_avg(single_bjets_4b)             
        pt_ratio_4b        = np.where(vec_W_4b.pt > 0, pt_H / vec_W_4b.pt, -1)
        min_dphi_lepjet_4b = min_dphi_jets_lepton(jets=single_jets_4b, leptons=lead_l_4b)         
        dm4b_4b            = min_dm_bb_bb(make_vector(single_bjets_4b), all_jets=make_vector(single_jets_4b))  
        
        mbbj_4b            = m_bbj(vec_single_bjets_4b, vec_single_jets_4b)        
        lead_b_4b          = single_bjets_4b[:, 0]
        sublead_b_4b       = single_bjets_4b[:, 1]
        vec_lead_b_4b      = make_vector(lead_b_4b)
        vec_sublead_b_4b   = make_vector(sublead_b_4b)
        wh_pt_asymmetry_4b = np.abs(pt_H - vec_W_4b.pt) / (pt_H + vec_W_4b.pt)       
                   
        ele_mask_4b  = np.asarray(mask_e[mask_step4b])
        mu_mask_4b   = np.asarray(mask_mu[mask_step4b])
        
        # Histogram plotting
        w4b = weights.weight()[np.asarray(mask_step4b)]
        
        output["eventflow_resolved"].fill(cut="step4", weight=np.sum(w4b))
        
        output["mass_H_resolved"].fill(m=mass_H,                             weight=w4b)
        output["pt_H_resolved"].fill(pt=pt_H,                                weight=w4b)
        output["MET_resolved"].fill(pt=met_4b.pt,                            weight=w4b)
        output["pt_lepton_resolved"].fill(pt=lead_l_4b.pt,                   weight=w4b)
        output["MTW_resolved"].fill(m=mTW_4b,                                weight=w4b)
        output["pt_W_resolved"].fill(pt=vec_W_4b.pt,                         weight=w4b)
        output["HT_resolved"].fill(ht=HT_4b,                                 weight=w4b)
        output["btag_max_single_bjets_resolved"].fill(btag=btag_max_4b,      weight=w4b)
        output["btag_min_single_bjets_resolved"].fill(btag=btag_min_4b,      weight=w4b)
        output["dr_bb_resolved"].fill(dr=dr_bb_avg_4b,                       weight=w4b)
        output["pt_ratio_resolved"].fill(ratio=pt_ratio_4b,                  weight=w4b)
        output["dphi_jet-lepton_min_resolved"].fill(dphi=min_dphi_lepjet_4b, weight=w4b)
        output["dm_bbbb_min_resolved"].fill(dm=dm4b_4b,                      weight=w4b)
        output["pt_b1_resolved"].fill(pt=lead_b_4b.pt,                       weight=w4b)
        output["pt_b2_resolved"].fill(pt=sublead_b_4b.pt,                    weight=w4b)
        output["mass_bbj_resolved"].fill(m=mbbj_4b,                          weight=w4b)
        output["btag_prod_resolved"].fill(btag_prod=btag_prod_4b,            weight=w4b)
        output["wh_pt_asym_resolved"].fill(pt=wh_pt_asymmetry_4b,            weight=w4b)
        output["deta_WH_resolved"].fill(deta=deta_wh_4b,                     weight=w4b)
        output["dphi_WH_resolved"].fill(dphi=dphi_wh_4b,                     weight=w4b)
        output["dphi_MET-lepton_resolved"].fill(dphi=dphi_metlep_4b,         weight=w4b)
        output["dr_WH_resolved"].fill(dr=dr_wh_4b,                           weight=w4b) 
        output["eta_b1_resolved"].fill(eta=lead_b_4b.eta,                    weight=w4b) 
        output["eta_b2_resolved"].fill(eta=sublead_b_4b.eta,                 weight=w4b) 
        output["eta_b3_resolved"].fill(eta=single_bjets_4b[:, 2].eta,        weight=w4b) 
        output["phi_b1_resolved"].fill(phi=lead_b_4b.phi,                    weight=w4b) 
        output["phi_b2_resolved"].fill(phi=sublead_b_4b.phi,                 weight=w4b) 
        output["phi_b3_resolved"].fill(phi=single_bjets_4b[:, 2].phi,        weight=w4b) 
        #output["eta_MET_resolved"].fill(eta=met_4b.eta,                      weight=w4b) 
        output["phi_MET_resolved"].fill(phi=met_4b.phi,                      weight=w4b) 
        
        has4 = ak.num(single_bjets_4b) >= 4            
        j4   = single_bjets_4b[has4][:, 3]             
        w4   = w4b[has4]
        
        output["eta_b4_resolved"].fill(eta=j4.eta, weight=w4)
        output["phi_b4_resolved"].fill(phi=j4.phi, weight=w4)
                
        if np.any(mu_mask_4b):
            wmu = w4b[mu_mask_4b]
            output["mu_eventflow_resolved"].fill(cut="step4", weight=np.sum(wmu))
        
            output["mu_A_HT_resolved"].fill(ht=HT_4b[mu_mask_4b],                                 weight=wmu)
            output["mu_A_pt_b1_resolved"].fill(pt=lead_b_4b[mu_mask_4b].pt,                       weight=wmu)
            output["mu_A_pt_b2_resolved"].fill(pt=sublead_b_4b[mu_mask_4b].pt,                    weight=wmu)
            output["mu_A_pt_lepton_resolved"].fill(pt=lead_l_4b[mu_mask_4b].pt,                   weight=wmu)
            output["mu_A_MET_resolved"].fill(pt=met_4b[mu_mask_4b].pt,                            weight=wmu)
            output["mu_A_MTW_resolved"].fill(m=mTW_4b[mu_mask_4b],                                weight=wmu)
            output["mu_A_pt_W_resolved"].fill(pt=vec_W_4b[mu_mask_4b].pt,                         weight=wmu)
            output["mu_A_mass_H_resolved"].fill(m=mass_H[mu_mask_4b],                             weight=wmu)
            output["mu_A_pt_H_resolved"].fill(pt=pt_H[mu_mask_4b],                                weight=wmu)
            output["mu_A_btag_min_single_bjets_resolved"].fill(btag=btag_min_4b[mu_mask_4b],      weight=wmu)
            output["mu_A_btag_max_single_bjets_resolved"].fill(btag=btag_max_4b[mu_mask_4b],      weight=wmu)
            output["mu_A_dphi_WH_resolved"].fill(dphi=dphi_wh_4b[mu_mask_4b],                     weight=wmu)
            output["mu_A_dr_WH_resolved"].fill(dr=dr_wh_4b[mu_mask_4b],                           weight=wmu)
            output["mu_A_dphi_jet-lepton_min_resolved"].fill(dphi=min_dphi_lepjet_4b[mu_mask_4b], weight=wmu)
            output["mu_A_dphi_MET-lepton_resolved"].fill(dphi=dphi_metlep_4b[mu_mask_4b],         weight=wmu)
            output["mu_A_dr_bb_ave_resolved"].fill(dr=dr_bb_avg_4b[mu_mask_4b],                   weight=wmu)
            output["mu_A_pt_ratio_resolved"].fill(ratio=pt_ratio_4b[mu_mask_4b],                  weight=wmu)
            output["mu_A_btag_prod_resolved"].fill(btag_prod=btag_prod_4b[mu_mask_4b],            weight=wmu)
            output["mu_A_deta_WH_resolved"].fill(deta=deta_wh_4b[mu_mask_4b],                     weight=wmu)
            output["mu_A_dm_bbbb_min_resolved"].fill(dm=dm4b_4b[mu_mask_4b],                      weight=wmu)
            output["mu_A_mass_bbj_resolved"].fill(m=mbbj_4b[mu_mask_4b],                          weight=wmu)
            output["mu_A_eta_b1_resolved"].fill(eta=lead_b_4b[mu_mask_4b].eta,                    weight=wmu)
            output["mu_A_eta_b2_resolved"].fill(eta=sublead_b_4b[mu_mask_4b].eta,                 weight=wmu)
            output["mu_A_eta_b3_resolved"].fill(eta=single_bjets_4b[:, 2][mu_mask_4b].eta,        weight=wmu)
            output["mu_A_phi_b1_resolved"].fill(phi=lead_b_4b[mu_mask_4b].phi,                    weight=wmu)
            output["mu_A_phi_b2_resolved"].fill(phi=sublead_b_4b[mu_mask_4b].phi,                 weight=wmu)
            output["mu_A_phi_b3_resolved"].fill(phi=single_bjets_4b[:, 2][mu_mask_4b].phi,        weight=wmu)
            #output["mu_A_eta_MET_resolved"].fill(eta=met_4b[mu_mask_4b].eta,                      weight=wmu)
            output["mu_A_phi_MET_resolved"].fill(phi=met_4b[mu_mask_4b].phi,                      weight=wmu)
        
            has4_mu = (ak.num(single_bjets_4b) >= 4) & mu_mask_4b
            if ak.any(has4_mu):
                j4_mu = single_bjets_4b[has4_mu][:, 3]
                wmu4  = w4b[has4_mu]
                output["mu_A_eta_b4_resolved"].fill(eta=j4_mu.eta, weight=wmu4)
                output["mu_A_phi_b4_resolved"].fill(phi=j4_mu.phi, weight=wmu4)
                                                   
        if np.any(ele_mask_4b):
            wel = w4b[ele_mask_4b]
            output["e_eventflow_resolved"].fill(cut="step4", weight=np.sum(wel))
        
            output["e_A_HT_resolved"].fill(ht=HT_4b[ele_mask_4b],                                 weight=wel)
            output["e_A_pt_b1_resolved"].fill(pt=lead_b_4b[ele_mask_4b].pt,                       weight=wel)
            output["e_A_pt_b2_resolved"].fill(pt=sublead_b_4b[ele_mask_4b].pt,                    weight=wel)
            output["e_A_pt_lepton_resolved"].fill(pt=lead_l_4b[ele_mask_4b].pt,                   weight=wel)
            output["e_A_MET_resolved"].fill(pt=met_4b[ele_mask_4b].pt,                            weight=wel)
            output["e_A_MTW_resolved"].fill(m=mTW_4b[ele_mask_4b],                                weight=wel)
            output["e_A_pt_W_resolved"].fill(pt=vec_W_4b[ele_mask_4b].pt,                         weight=wel)
            output["e_A_mass_H_resolved"].fill(m=mass_H[ele_mask_4b],                             weight=wel)
            output["e_A_pt_H_resolved"].fill(pt=pt_H[ele_mask_4b],                                weight=wel)
            output["e_A_btag_min_single_bjets_resolved"].fill(btag=btag_min_4b[ele_mask_4b],      weight=wel)
            output["e_A_btag_max_single_bjets_resolved"].fill(btag=btag_max_4b[ele_mask_4b],      weight=wel)
            output["e_A_dphi_WH_resolved"].fill(dphi=dphi_wh_4b[ele_mask_4b],                     weight=wel)
            output["e_A_dr_WH_resolved"].fill(dr=dr_wh_4b[ele_mask_4b],                           weight=wel)
            output["e_A_dphi_jet-lepton_min_resolved"].fill(dphi=min_dphi_lepjet_4b[ele_mask_4b], weight=wel)
            output["e_A_dphi_MET-lepton_resolved"].fill(dphi=dphi_metlep_4b[ele_mask_4b],         weight=wel)
            output["e_A_dr_bb_ave_resolved"].fill(dr=dr_bb_avg_4b[ele_mask_4b],                   weight=wel)
            output["e_A_pt_ratio_resolved"].fill(ratio=pt_ratio_4b[ele_mask_4b],                  weight=wel)
            output["e_A_btag_prod_resolved"].fill(btag_prod=btag_prod_4b[ele_mask_4b],            weight=wel)
            output["e_A_deta_WH_resolved"].fill(deta=deta_wh_4b[ele_mask_4b],                     weight=wel)
            output["e_A_dm_bbbb_min_resolved"].fill(dm=dm4b_4b[ele_mask_4b],                      weight=wel)
            output["e_A_mass_bbj_resolved"].fill(m=mbbj_4b[ele_mask_4b],                          weight=wel)
            output["e_A_eta_b1_resolved"].fill(eta=lead_b_4b[ele_mask_4b].eta,                    weight=wel)
            output["e_A_eta_b2_resolved"].fill(eta=sublead_b_4b[ele_mask_4b].eta,                 weight=wel)
            output["e_A_eta_b3_resolved"].fill(eta=single_bjets_4b[:, 2][ele_mask_4b].eta,        weight=wel)
            output["e_A_phi_b1_resolved"].fill(phi=lead_b_4b[ele_mask_4b].phi,                    weight=wel)
            output["e_A_phi_b2_resolved"].fill(phi=sublead_b_4b[ele_mask_4b].phi,                 weight=wel)
            output["e_A_phi_b3_resolved"].fill(phi=single_bjets_4b[:, 2][ele_mask_4b].phi,        weight=wel)
            #output["e_A_eta_MET_resolved"].fill(eta=met_4b[ele_mask_4b].eta,                      weight=wel)
            output["e_A_phi_MET_resolved"].fill(phi=met_4b[ele_mask_4b].phi,                      weight=wel)
            
            has4_e = (ak.num(single_bjets_4b) >= 4) & ele_mask_4b
            if ak.any(has4_e):
                j4_e = single_bjets_4b[has4_e][:, 3]
                we4  = w4b[has4_e]
                output["e_A_eta_b4_resolved"].fill(eta=j4_e.eta, weight=we4)
                output["e_A_phi_b4_resolved"].fill(phi=j4_e.phi, weight=we4)
        
        weights_resolved = w4b
        n_resolved= len(weights_resolved)
        print(f"Number of events after selection: {n_resolved}")
        bdt_resolved = {
            "H_mass"                  : ak.to_numpy(mass_H),
            "H_pt"                    : ak.to_numpy(pt_H),
            "MTW"                     : ak.to_numpy(mTW_4b),
            "W_pt"                    : ak.to_numpy(vec_W_4b.pt),
            "HT"                      : ak.to_numpy(HT_4b),
            "MET_pt"                  : ak.to_numpy(met_4b.pt),
            "btag_max"                : ak.to_numpy(btag_max_4b),
            "btag_min"                : ak.to_numpy(btag_min_4b),
            "dr_bb_ave"               : ak.to_numpy(dr_bb_avg_4b),
            "dm_4b_min"               : ak.to_numpy(dm4b_4b),
            "mbbj"                    : ak.to_numpy(mbbj_4b),
            "dphi_WH"                 : ak.to_numpy(np.abs(dphi_wh_4b)),
            "dphi_jet_lepton_min"     : ak.to_numpy(np.abs(min_dphi_lepjet_4b)),                    
            "Njets"                   : ak.to_numpy(n_sjs),
            "pt_ratio"                : ak.to_numpy(pt_ratio_4b),
            "pt_lepton"               : ak.to_numpy(lead_l_4b.pt),
            "pt_b1"                   : ak.to_numpy(lead_b_4b.pt),
            "WH_pt_assymetry"         : ak.to_numpy(wh_pt_asymmetry_4b),
            "btag_prod"               : ak.to_numpy(btag_prod_4b),
            "weight"                  : ak.to_numpy(weights_resolved),
            }

        if self.isMVA:
            self.compat_tree_variables(bdt_resolved)
            self.add_tree_entry("resolved", bdt_resolved)
            
        if self.runEval and not self.isMVA :
            inputs_resolved    = {k: bdt_resolved[k] for k in self.bdt_eval_resolved.var_list}
            bdt_score_resolved = np.ravel(self.bdt_eval_resolved.eval(inputs_resolved))
            output["bdt_score_resolved"].fill(bdt=bdt_score_resolved, weight=bdt_resolved["weight"])
            
            e_mask_all4b  = np.asarray(mask_e[mask_step4b])
            mu_mask_all4b = np.asarray(mask_mu[mask_step4b])
                      
            for i, cut in enumerate(self.optim_Cuts1_bdt):
                cut_mask = (bdt_score_resolved > cut)
                if not np.any(cut_mask):
                    continue
                        
                ele_mask_cut = e_mask_all4b & cut_mask
                mu_mask_cut  = mu_mask_all4b & cut_mask
                
                # ===== electrons =====
                if np.any(ele_mask_cut):
                    w = weights_resolved[ele_mask_cut]
                    s = bdt_score_resolved[ele_mask_cut]
        
                    output["e_bdt_score_resolved"].fill(bdt=s, weight=w)
                    output["e_A_SR_3b_bdt_shapes_resolved"].fill(cut_index=i, bdt=s, weight=w)
          
                    output["e_A_SR_3b_higgsMass_shapes_resolved"].fill(cut_index=i,  H_mass=ak.to_numpy(mass_H)[ele_mask_cut],              weight=w)
                    output["e_A_SR_3b_higgsPt_shapes_resolved"].fill  (cut_index=i,  H_pt=ak.to_numpy(pt_H  )[ele_mask_cut],                weight=w)
                    output["e_A_SR_3b_b1Pt_shapes_resolved"].fill     (cut_index=i,  pt_b1=ak.to_numpy(lead_b_4b.pt)[ele_mask_cut],         weight=w)
                    output["e_A_SR_3b_ht_shapes_resolved"].fill       (cut_index=i,  HT=HT_4b[ele_mask_cut],                                weight=w)
                    output["e_A_SR_3b_pfmet_shapes_resolved"].fill    (cut_index=i,  MET_pt=ak.to_numpy(met_4b.pt)[ele_mask_cut],           weight=w)
                    output["e_A_SR_3b_mtw_shapes_resolved"].fill      (cut_index=i,  MTW=mTW_4b[ele_mask_cut],                              weight=w)
                    output["e_A_SR_3b_ptw_shapes_resolved"].fill      (cut_index=i,  W_pt=ak.to_numpy(vec_W_4b.pt)[ele_mask_cut],           weight=w)       
                    output["e_A_SR_3b_dRwh_shapes_resolved"].fill     (cut_index=i,  dr_WH=dr_wh_4b[ele_mask_cut],                          weight=w)
                    output["e_A_SR_3b_dphiWh_shapes_resolved"].fill   (cut_index=i,  dphi_WH=np.abs(dphi_wh_4b[ele_mask_cut]),              weight=w)
                    output["e_A_SR_3b_dphijetlep_shapes_resolved"].fill(cut_index=i, dphi_lep_met=np.abs(min_dphi_lepjet_4b[ele_mask_cut]), weight=w)        
                    output["e_A_SR_3b_dRave_shapes_resolved"].fill    (cut_index=i,  dr_bb_ave=dr_bb_avg_4b[ele_mask_cut],                  weight=w)
                    output["e_A_SR_3b_dmmin_shapes_resolved"].fill    (cut_index=i,  dm_4b_min=dm4b_4b[ele_mask_cut],                       weight=w)
                    output["e_A_SR_3b_lep_pt_raw_shapes_resolved"].fill(cut_index=i, pt_lepton=ak.to_numpy(lead_l_4b.pt)[ele_mask_cut],     weight=w)
                    output["e_A_SR_3b_wh_pt_asym_shapes_resolved"].fill(cut_index=i, WH_pt_assymetry=wh_pt_asymmetry_4b[ele_mask_cut],      weight=w)       
                    output["e_A_SR_3b_jets_shapes_resolved"].fill(cut_index=i,       n_jets=ak.num(single_jets_4b[ele_mask_cut]),           weight=w)       
                    output["e_A_SR_3b_btag_prod_shapes_resolved"].fill(cut_index=i,  btag_prod=btag_prod_4b[ele_mask_cut],                  weight=w)
                    output["e_A_SR_3b_btag_min_shapes_resolved"].fill (cut_index=i,  btag_min=btag_min_4b [ele_mask_cut],                   weight=w)
                    output["e_A_SR_3b_btag_max_shapes_resolved"].fill (cut_index=i,  btag_max=btag_max_4b [ele_mask_cut],                   weight=w)
                    output["e_A_SR_3b_mbbj_shapes_resolved"].fill     (cut_index=i,  mbbj=mbbj_4b[ele_mask_cut],                            weight=w)
        
                # ===== muons =====
                if np.any(mu_mask_cut):
                    w = weights_resolved[mu_mask_cut]
                    s = bdt_score_resolved[mu_mask_cut]
        
                    output["mu_bdt_score_resolved"].fill(bdt=s, weight=w)
                    output["mu_A_SR_3b_bdt_shapes_resolved"].fill(cut_index=i, bdt=s, weight=w)
        
                    output["mu_A_SR_3b_higgsMass_shapes_resolved"].fill(cut_index=i, H_mass=ak.to_numpy(mass_H)[mu_mask_cut],               weight=w)
                    output["mu_A_SR_3b_higgsPt_shapes_resolved"].fill  (cut_index=i, H_pt=ak.to_numpy(pt_H  )[mu_mask_cut],                 weight=w)
                    output["mu_A_SR_3b_b1Pt_shapes_resolved"].fill     (cut_index=i, pt_b1=ak.to_numpy(lead_b_4b.pt)[mu_mask_cut],          weight=w)
                    output["mu_A_SR_3b_ht_shapes_resolved"].fill       (cut_index=i, HT=HT_4b[mu_mask_cut],                                 weight=w)
                    output["mu_A_SR_3b_pfmet_shapes_resolved"].fill    (cut_index=i, MET_pt=ak.to_numpy(met_4b.pt)[mu_mask_cut],            weight=w)
                    output["mu_A_SR_3b_mtw_shapes_resolved"].fill      (cut_index=i, MTW=mTW_4b[mu_mask_cut],                               weight=w)
                    output["mu_A_SR_3b_ptw_shapes_resolved"].fill      (cut_index=i, W_pt=ak.to_numpy(vec_W_4b.pt)[mu_mask_cut],            weight=w)      
                    output["mu_A_SR_3b_dRwh_shapes_resolved"].fill     (cut_index=i, dr_WH=dr_wh_4b[mu_mask_cut],                           weight=w)
                    output["mu_A_SR_3b_dphiWh_shapes_resolved"].fill   (cut_index=i, dphi_WH=np.abs(dphi_wh_4b[mu_mask_cut]),               weight=w)
                    output["mu_A_SR_3b_dphijetlep_shapes_resolved"].fill(cut_index=i, dphi_lep_met=np.abs(min_dphi_lepjet_4b[mu_mask_cut]), weight=w)      
                    output["mu_A_SR_3b_dRave_shapes_resolved"].fill    (cut_index=i, dr_bb_ave=dr_bb_avg_4b[mu_mask_cut],                   weight=w)
                    output["mu_A_SR_3b_dmmin_shapes_resolved"].fill    (cut_index=i, dm_4b_min=dm4b_4b[mu_mask_cut],                        weight=w)
                    output["mu_A_SR_3b_lep_pt_raw_shapes_resolved"].fill(cut_index=i, pt_lepton=ak.to_numpy(lead_l_4b.pt)[mu_mask_cut],     weight=w)
                    output["mu_A_SR_3b_wh_pt_asym_shapes_resolved"].fill(cut_index=i, WH_pt_assymetry=wh_pt_asymmetry_4b[mu_mask_cut],      weight=w)      
                    output["mu_A_SR_3b_jets_shapes_resolved"].fill(cut_index=i,n_jets=ak.num(single_jets_4b[mu_mask_cut]),                  weight=w)       
                    output["mu_A_SR_3b_btag_prod_shapes_resolved"].fill(cut_index=i, btag_prod=btag_prod_4b[mu_mask_cut],                   weight=w)
                    output["mu_A_SR_3b_btag_min_shapes_resolved"].fill (cut_index=i, btag_min=btag_min_4b [mu_mask_cut],                    weight=w)
                    output["mu_A_SR_3b_btag_max_shapes_resolved"].fill (cut_index=i, btag_max=btag_max_4b [mu_mask_cut],                    weight=w)
                    output["mu_A_SR_3b_mbbj_shapes_resolved"].fill     (cut_index=i, mbbj=mbbj_4b[mu_mask_cut],                             weight=w)
                    
        if self.isMVA:
            output["trees"] = self._trees
            for regime, trees in self._trees.items():
                print(f"\n[DEBUG] Regime '{regime}' has {len(trees)} entries")    
                
                
        # VERBOSES USED FOR DEBUGGING
        verbose = False
        if verbose:
            # Gen level 
            if genLevel:
                print("(Events, Multiplicity) from: ... ")
                print("Higgs:           (" + str(len(genHiggs)) + ", " + str(ak.max(ak.num(genHiggs))) + ")")
                print("W:               (" + str(len(genW)) + ", " + str(ak.max(ak.num(genW))) + ")")
                print("A:               (" + str(len(genA)) + ", " + str(ak.max(ak.num(genA))) + ")")
                print("b quarks:        (" + str(len(genB)) + ", " + str(ak.max(ak.num(genB))) + ")")
                print("leptons:         (" + str(len(genLepton)) + ", " + str(ak.max(ak.num(genLepton))) + ")")
                print("neutrinos:       (" + str(len(genNeutrino)) + ", " + str(ak.max(ak.num(genNeutrino))) + ")")
            
            # Lepton configuration
            print("\nNumber of muons:", ak.num(muons))
            print("Number of electrons:", ak.num(electrons))
            print("Number of leptons (pT sorted):", n_leptons)        
            print("Leading lepton pT:", leading_pt)
            
            # Jet configuration
            print("\nInitial single jets count:", ak.num(single_jets))
            print("Initial single b-jets count:", ak.num(single_bjets))
            print("Initial single untagged jets count:", ak.num(single_untag_jets))
            print("\nInitial double jets count:", ak.num(double_jets))
            print("Initial double b-jets count:", ak.num(double_bjets))
            print("Initial double untagged jets count:", ak.num(double_untag_jets))
            
            # Trigger check
            print("\nTrigger-matched mu events:", np.sum(mask_step1 & mask_mu))
            print("Trigger-matched e events:", np.sum(mask_step1 & mask_e))
            
            # Step 1: at least one lepton checks
            print("\nLepton type counts (after pT cuts):", np.unique(tag_cat[mask_step1], return_counts=True))
            print("PDG IDs of leading leptons (step1):", lead_lep_step1.lepton_type[:10])
            print("Assigned tags:", tag_cat[pass_step1][:10])
            #more muons than electrons: better reconstruction efficiency
            
            # Boosted analysis checks
            print("\nDouble jet multiplicities (step2a):", ak.to_list(n_double_jets_2a[:10]))
            print("Double b-jet multiplicities (step2a):", ak.to_list(n_double_bjets_2a[:10]))
            print("Untagged double jets (step2a):", ak.to_list(n_double_untag_jets_2a[:10]))
            print("Double b-jet counts (after cut):", ak.to_list(n_double_bjets[mask_step3a][:10]))
            
            print("\nEvent counts for each variable in STEP 4a (after MET>25 & MTW>50):")
            print(f"Lead lepton pT:              {ak.count_nonzero(~ak.is_none(lead_l_4a.pt))}")
            print(f"MET pT:                      {ak.count_nonzero(~ak.is_none(met_4a.pt))}")
            print(f"MTW:                         {ak.count_nonzero(~ak.is_none(mTW_4a))}")
            print(f"W boson pT:                  {ak.count_nonzero(~ak.is_none(vec_W_4a.pt))}")
            print(f"H mass:                      {ak.count_nonzero(~ak.is_none(vec_H_4a.mass))}")
            print(f"BTag max:                    {ak.count_nonzero(~ak.is_none(btag_max_4a))}")
            print(f"BTag min:                    {ak.count_nonzero(~ak.is_none(btag_min_4a))}")
            print(f"Δφ(H,W):                     {ak.count_nonzero(~ak.is_none(dphi_wh_4a))}")
            print(f"ΔR(H,W):                     {ak.count_nonzero(~ak.is_none(dr_wh_4a))}")
            print(f"|Δm(bb)|:                    {ak.count_nonzero(~ak.is_none(dmbb_4a))}")
            print(f"Min Δφ(jet, lepton):         {ak.count_nonzero(~ak.is_none(min_dphi_lepjet_4a))}")
            print(f"ΔR(bb):                      {ak.count_nonzero(~ak.is_none(dr_bb_4a))}")
            print(f"pT(H)/pT(W):                 {ak.count_nonzero(~ak.is_none(pt_ratio_4a))}")
            print(f"WH pT asymmetry:             {ak.count_nonzero(~ak.is_none(wh_pt_asymmetry_4a))}")
            print(f"b-tag product:               {ak.count_nonzero(~ak.is_none(btag_prod_4a))}")
            
            
            print("\nEvent counts for each variable in STEP 4b (after MET>25 & MTW>50):")
            print(f"Lead lepton pT:              {ak.count_nonzero(~ak.is_none(lead_l_4b.pt))}")
            print(f"MET pT:                      {ak.count_nonzero(~ak.is_none(met_4b.pt))}")
            print(f"MTW:                         {ak.count_nonzero(~ak.is_none(mTW_4b))}")
            print(f"W boson pT:                  {ak.count_nonzero(~ak.is_none(vec_W_4b.pt))}")
            print(f"H mass:                      {ak.count_nonzero(~ak.is_none(mass_H))}")
            print(f"BTag max:                    {ak.count_nonzero(~ak.is_none(btag_max_4b))}")
            print(f"BTag min:                    {ak.count_nonzero(~ak.is_none(btag_min_4b))}")
            print(f"Δφ(H,W):                     {ak.count_nonzero(~ak.is_none(dphi_wh_4b))}")
            print(f"Mass bbj:                    {ak.count_nonzero(~ak.is_none(mbbj_4b))}")
            print(f"Min Δφ(jet, lepton):         {ak.count_nonzero(~ak.is_none(min_dphi_lepjet_4b))}")
            print(f"ΔR(bb) average:              {ak.count_nonzero(~ak.is_none(dr_bb_avg_4b))}")
            print(f"pT(H)/pT(W):                 {ak.count_nonzero(~ak.is_none(pt_ratio_4b))}")
            print(f"WH pT asymmetry:             {ak.count_nonzero(~ak.is_none(wh_pt_asymmetry_4b))}")
            print(f"b-tag product:               {ak.count_nonzero(~ak.is_none(btag_prod_4b))}")
            
            # Btagging SF weights ---- #
            if self.isMC:
                with_sf  = weights.weight()
                base_res = base_w_snapshot[np.asarray(mask_step2b, dtype=bool)]
                with_res = with_sf[np.asarray(mask_step2b, dtype=bool)]
                print("\n[DEBUG] Resolved yield comparison:")
                print(f"  events (mask_step2b) = {np.sum(mask_step2b)}")
                print(f"  sum of weights (no btag SF)  = {np.sum(base_res):.6f}")
                print(f"  sum of weights (with btag SF)= {np.sum(with_res):.6f}")
                if np.sum(base_res) > 0:
                    print(f"  ratio (with / without)       = {np.sum(with_res)/np.sum(base_res):.6f}")
        

        return output

    def postprocess(self, accumulator):
        return accumulator
