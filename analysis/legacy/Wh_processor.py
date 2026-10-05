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

def _stats(x, title="", *, flatten=True):
    """
    Print basic stats for numpy/awkward/list-like inputs.
    - Works for jagged Awkward arrays (flattens by default).
    - Ignores non-finite values.
    """
    import numpy as np
    try:
        import awkward as ak  
    except Exception:
        ak = None

    # Flatten if it's awkward
    if ak is not None and flatten:
        try:
            x = ak.flatten(x, axis=None)
        except Exception:
            pass

    # Convert to numpy, then to float array
    try:
        xv = ak.to_numpy(x) if ak is not None else np.asarray(x)
    except Exception:
        xv = np.asarray(x)

    xv = np.asarray(xv, dtype=float).ravel()

    if xv.size == 0:
        print(f"{title} empty" if title else "empty")
        return

    m = np.isfinite(xv)
    if not np.any(m):
        print(f"{title} all values non-finite" if title else "all values non-finite")
        return

    xv = xv[m]
    q50 = np.median(xv)
    lab = f"[{title}]" if title else ""
    print(f"{lab} n={xv.size}  min={xv.min():.4g}  q50={q50:.4g}  mean={xv.mean():.4g}  max={xv.max():.4g}")


#----------------------------------------------------------------------------------------------------------------------------------------------

def _ptphi_to_pxpy(pt, phi):
    return pt * np.cos(phi), pt * np.sin(phi)

#----------------------------------------------------------------------------------------------------------------------------------------------

def _pxpy_to_ptphi(px, py):
    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)
    return pt, phi

#----------------------------------------------------------------------------------------------------------------------------------------------

def _mask_lepton_overlap(jets, leptons, dr=0.4):
    """
    Per jet: True if ΔR(jet, ANY lepton) > dr.
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
        
        HERE = os.path.abspath(os.path.dirname(__file__))
        CORR_DIR = os.environ.get("CORR_DIR", os.path.join(HERE, "corrections"))
        
        #===================================================================================================================================================
        # --- Electron energy and smearing corrections (2024) --- #
        #===================================================================================================================================================
         
        # https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#2022_2023_and_2024_Scale_and_Sme
        # PATH: /eos/cms/store/group/phys_egamma/ScaleFactors/Data2024/SS/
        self.egm_json_path = os.path.join(CORR_DIR, "electronSS_EtDependent_v1.json.gz")
        
        self._egm_scale = None
        self._egm_smear = None
        
        if os.path.exists(self.egm_json_path):
            try:
                egm_cset = correctionlib.CorrectionSet.from_file(self.egm_json_path)
                
                print(f"\n[ANA:EGM] Available keys in {self.egm_json_path}: {list(egm_cset.keys())}")
                
                self._egm_scale = egm_cset["EGMScale_ElePTsplit_2024"]
                self._egm_smear = egm_cset["EGMSmearAndSyst_ElePTsplit_2024"]
                             
                if (self._egm_scale is None) and (self._egm_smear is None):
                    print(f"[ANA:EGM] No expected keys in {self.egm_json_path}. Available: {list(egm_cset.keys())}")
                else:
                    print("[ANA:EGM] Loaded electron scale/smear JSON.")
                
            except Exception as e:
                print(f"[ANA:EGM] Failed to load: {e}")
        else:
            print("[ANA:EGM] electronSS_EtDependent.json.gz not found; skipping EGM energy corrections.")


        #===================================================================================================================================================
        # --- Electron Reco and ID SFs (2024) --- #
        #===================================================================================================================================================
        
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammSFandSSRun3#Electron_and_Electron_ID_JSON_fo
        # PATH: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2024_Summer24/
        self.ele_reco_json_path = os.path.join(CORR_DIR, "electron_v1.json.gz")
        self.ele_id_json_path   = os.path.join(CORR_DIR, "electronID_v1.json.gz")
        
        ELE_MAP_NAME = "Electron-ID-SF"
         
        self._ele_reco_corr = None
        self._ele_id_corr   = None
        
        # --- Reco SF JSON --- #
        if os.path.exists(self.ele_reco_json_path):
            try:
                import gzip
                raw = gzip.open(self.ele_reco_json_path, "rt").read() 
                
                # replace edges (-inf/inf) with very large numbers
                raw = raw.replace('"-inf"', '-1e6').replace('"inf"', '1e6')
                cset_reco = correctionlib.CorrectionSet.from_string(raw)
                self._ele_reco_corr = cset_reco[ELE_MAP_NAME]
                print(f"\n[ANA:ElectronRECOSF] Available keys in {self.ele_reco_json_path}: {list(cset_reco.keys())}")
                print(f"[ANA:ElectronRECOSF] Loaded (sanitized) from {self.ele_reco_json_path}")
            except Exception as e:
                print(f"[ANA:ElectronRECOSF] Failed to load {self.ele_reco_json_path}: {e}")
        else:
            print(f"[ANA:ElectronRECOSF] {self.ele_reco_json_path} not found; skipping Reco SFs.")

        
        # --- ID SF JSON --- #
        if os.path.exists(self.ele_id_json_path):
            try:
                cset_id = correctionlib.CorrectionSet.from_file(self.ele_id_json_path)
                print(f"[ANA:ElectronIDSF] Available keys in {self.ele_id_json_path}: {list(cset_id.keys())}")
                if ELE_MAP_NAME in cset_id.keys():
                    self._ele_id_corr = cset_id[ELE_MAP_NAME]
                    print("[ANA:ElectronIDSF] Loaded ID SF JSON.")
                else:
                    print(f"[ANA:ElectronIDSF] '{ELE_MAP_NAME}' not found in {self.ele_id_json_path}.")
            except Exception as e:
                print(f"[ANA:ElectronIDSF] Failed to load {self.ele_id_json_path}: {e}")
        else:
            print(f"[ANA:ElectronIDSF] {self.ele_id_json_path} not found; skipping ID SFs.")
                
        
        
        #===================================================================================================================================================            
        # --- HLT Ele30 TightID (2023D proxy) / Need to change later (expected at the beginning of September) --- #
        # --- It will not be used until the update --- #
        #===================================================================================================================================================
        # PATH: /eos/cms/store/group/phys_egamma/ScaleFactors/Data2024/
        
        
        #===================================================================================================================================================    
        # --- Medium pT: RECO efficiencies (2024) --- #
        #===================================================================================================================================================
        # No correction is recommended for Run 3 data. 
        # Data/MC scale factors are expected to be 1 and therefore no correction is needed/provided.
        
            
        #===================================================================================================================================================    
        # --- Medium pT: ID efficiencies (2024) --- #
        #===================================================================================================================================================
        # https://muon-wiki.docs.cern.ch/guidelines/corrections/#medium-pt-id-efficiencies
        # PATH : https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/blob/master/Run3/2024/2024_Z/ScaleFactors_Muon_ID_ISO_2024_schemaV2.json
        
        #===================================================================================================================================================    
        # --- Medium pT: ISO efficiencies (2024) --- #
        #===================================================================================================================================================
        # https://muon-wiki.docs.cern.ch/guidelines/corrections/#medium-pt-iso-efficiencies
        # PATH : https://gitlab.cern.ch/cms-muonPOG/muonefficiencies/-/blob/master/Run3/2024/2024_Z/ScaleFactors_Muon_ID_ISO_2024_schemaV2.json
        
        #===================================================================================================================================================    
        # --- Medium pT: Trigger efficiencies (2024) --- #
        #===================================================================================================================================================
        # https://muon-wiki.docs.cern.ch/guidelines/corrections/#medium-pt-trigger-efficiencies
        # TBD
        
        #===================================================================================================================================================    
        # --- Medium pT: Scale and Resolution (2024) --- #
        #===================================================================================================================================================
        # https://muon-wiki.docs.cern.ch/guidelines/corrections/#medium-pt-scale-and-resolution
        # TBD
        

        # =================================================================================================================================================== #
        # --- JERC (JEC + JER) — Run-3 PUPPI setup + JES (27 NP preferred) + JER (6 NP scheme ready) --- #
        # #=================================================================================================================================================== #
        # https://cms-jerc.web.cern.ch/Recommendations/
        # https://indico.cern.ch/event/1546228/contributions/6567938/attachments/3095763/5484272/JetMET_01July2025_JhLee%20.pdf
        # PATH: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME
          
        self.jerc_json_path = os.path.join(CORR_DIR, "jet_jerc.json.gz")
        self._jec_L1 = self._jec_L2 = self._jec_L3 = self._jec_residual = None
        self._jer_sf = self._jer_res = None
        self._jes_unc_sources = []    # list of NPs 
        
        if os.path.exists(self.jerc_json_path):
            jerc = correctionlib.CorrectionSet.from_file(self.jerc_json_path)
            all_keys = list(jerc.keys())
            
            print("\n[ANA:JERC] Available keys in jet_jerc.json.gz:")
            for k in all_keys:
                print("   -", k)
            
            tag = "MC" if self.isMC else "DATA"
            
            # --- Base JEC steps (skip L1 for PUPPI in Run-3) --- #
            # https://cms-jerc.web.cern.ch/JEC/#l1-pileup
            # Note: Starting from Run 3, the use of PUPPI jets eliminates the need for L1 corrections. 
            # To maintain compatibility with Run 2 scripts, a dummy file is provided.
            kL2  = [k for k in all_keys if ("L2Relative"  in k and "AK4PFPuppi" in k and tag in k)]
            kL3  = [k for k in all_keys if ("L3Absolute"  in k and "AK4PFPuppi" in k and tag in k)]
            kRes = [] if self.isMC else [k for k in all_keys if ("L2L3Residual" in k and "AK4PFPuppi" in k and "DATA" in k)]
        
            if len(kL2) == 1:
                self._jec_L2 = jerc[kL2[0]]
            else:
                print(f"[ANA:JEC] WARNING: L2 not unique: {kL2}")
        
            if len(kL3) == 1:
                self._jec_L3 = jerc[kL3[0]]
            else:
                print(f"[ANA:JEC] WARNING: L3 not unique: {kL3}")
        
            if (not self.isMC) and len(kRes) == 1:
                self._jec_residual = jerc[kRes[0]]
            print(f"[ANA:JEC] L2={kL2[0] if kL2 else None}  L3={kL3[0] if kL3 else None}  Residual={kRes[0] if kRes else None}")
        
            # --- JER (resolution + scale factor) --- #
            kJERres = [k for k in all_keys if ("PtResolution" in k and "AK4PFPuppi" in k and "MC" in k)]
            kJERsf  = [k for k in all_keys if ("ScaleFactor"  in k and "AK4PFPuppi" in k and "MC" in k)]
            
            if len(kJERres) == 1:
                self._jer_res = jerc[kJERres[0]]
            else:
                print(f"[ANA:JER] WARNING: RES not unique: {kJERres}")
            if len(kJERsf) == 1:
                self._jer_sf  = jerc[kJERsf[0]]
            else:
                print(f"[ANA:JER] WARNING: SF not unique: {kJERsf}")
            print(f"[ANA:JER] res: {kJERres[0] if kJERres else None}   sf: {kJERsf[0] if kJERsf else None}")
        
            # --- JES uncertainties --- #
            #JES_SCHEME = "regrouped11"   
            JES_SCHEME = "full27"   
        
            self._jes_unc_sources = []
            
            if JES_SCHEME == "regrouped11":
                jes_keys = [
                    "Summer24Prompt24_V1_MC_Regrouped_Absolute_2024_AK4PFPuppi",
                    "Summer24Prompt24_V1_MC_Regrouped_Absolute_AK4PFPuppi",
                    "Summer24Prompt24_V1_MC_Regrouped_BBEC1_2024_AK4PFPuppi",
                    "Summer24Prompt24_V1_MC_Regrouped_BBEC1_AK4PFPuppi",
                    "Summer24Prompt24_V1_MC_Regrouped_EC2_2024_AK4PFPuppi",
                    "Summer24Prompt24_V1_MC_Regrouped_EC2_AK4PFPuppi",
                    "Summer24Prompt24_V1_MC_Regrouped_FlavorQCD_AK4PFPuppi",
                    "Summer24Prompt24_V1_MC_Regrouped_HF_2024_AK4PFPuppi",
                    "Summer24Prompt24_V1_MC_Regrouped_HF_AK4PFPuppi",
                    "Summer24Prompt24_V1_MC_Regrouped_RelativeBal_AK4PFPuppi",
                    "Summer24Prompt24_V1_MC_Regrouped_RelativeSample_2024_AK4PFPuppi",
                ]
                print(f"[ANA:JES] Regrouped scheme: using {len(jes_keys)} sources.")
                for k in jes_keys:
                    assert k in all_keys, f"[ANA:JES] Missing regrouped key: {k}"
                    self._jes_unc_sources.append(jerc[k])
                    print("  [JES NP]", k)
            
            elif JES_SCHEME == "full27":
                raw_names = [
                    "AbsoluteStat","AbsoluteScale","AbsoluteMPFBias","AbsoluteSample",
                    "Fragmentation","SinglePionECAL","SinglePionHCAL","TimePtEta",
                    "RelativeJEREC1","RelativeJEREC2","RelativeJERHF",
                    "RelativePtBB","RelativePtEC1","RelativePtEC2","RelativePtHF",
                    "RelativeFSR","RelativeStatEC","RelativeStatFSR","RelativeStatHF",
                    "RelativeBal","RelativeSample",
                    "PileUpDataMC","PileUpPtRef","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF",
                    "FlavorQCD",
                ]
                jes_keys = [f"Summer24Prompt24_V1_MC_{n}_AK4PFPuppi" for n in raw_names]
                print(f"[ANA:JES] Full-sources scheme: using {len(jes_keys)} sources.")
                for k in jes_keys:
                    assert k in all_keys, f"[ANA:JES] Missing raw source key: {k}"
                    self._jes_unc_sources.append(jerc[k])
                    print("  [JES NP]", k)
            else:
                raise RuntimeError(f"[ANA:JES] Unknown JES_SCHEME={JES_SCHEME}")
                    
        # =======================================================================================================================================
        # --- b-tagging (UParTAK4) --- #
        # =======================================================================================================================================
        # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer24/
        # PATH: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2024_Summer24
        # PATH: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/BTV/2024_Summer24
        
        self.btag_json_path = os.path.join(CORR_DIR, "btagging_preliminary.json")
        self._btag_wp_vals  = None      
        self._btag_sf_node  = None     
        
        if os.path.exists(self.btag_json_path):
            try:
                btag_cset = correctionlib.CorrectionSet.from_file(self.btag_json_path)
                all_keys = list(btag_cset.keys())
                print(f"\n[ANA:BTAG] Available keys in {self.btag_json_path}: {all_keys}")
        
                # exact names we expect in the Run-3 UParTAK4 preliminary JSON
                need_wp   = "UParTAK4_wp_values"
                need_kinf = "UParTAK4_kinfit"
        
                assert need_wp   in btag_cset.keys(),  f"[ANA:BTAG] Missing '{need_wp}' in {self.btag_json_path}"
                assert need_kinf in btag_cset.keys(),  f"[ANA:BTAG] Missing '{need_kinf}' in {self.btag_json_path}"
        
                self._btag_wp_vals = btag_cset[need_wp]
                self._btag_sf_node = btag_cset[need_kinf]
        
                print("[ANA:BTAG] Loaded UParTAK4 WP + kinfit SFs.")
                try:
                    print("[ANA:BTAG] WP inputs: ",   [v.name for v in self._btag_wp_vals.inputs])
                    print("[ANA:BTAG] SF inputs: ",   [v.name for v in self._btag_sf_node.inputs])
                except Exception:
                    pass
        
            except Exception as e:
                print(f"[ANA:BTAG] Failed to load {self.btag_json_path}: {e}")
        else:
            print("[ANA:BTAG] btagging_preliminary.json not found; skipping b-tag SFs.")

    

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
        
        self._histograms["mu_trg_eff2d"] = Hist.new.Reg(40, 0, 400, name="pt",  label="leading muon pT [GeV]").Reg(100, 0, 1,  name="eff", label="trigger efficiency").Weight()
        self._histograms["e_trg_eff2d"]  = Hist.new.Reg(40, 0, 400, name="pt",  label="leading electron pT [GeV]").Reg(100, 0, 1,  name="eff", label="trigger efficiency").Weight()
        
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
                
                SR_REGION    = "B" if self.isQCD else "A"
                CTRL_REGION  = "D" if self.isQCD else "C"
                REGIONS_RUN  = [SR_REGION, CTRL_REGION]  
                SIDE_REGIONS = [CTRL_REGION] 
                for region in REGIONS_RUN:
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
                                 "jet-lepton_min", "bb1", "bb2", "4b", "bb_ave", "b1", "b2", "b3", "b4", "bb", "bdt", "wh_asym",
                                 "single_untag_jets", "single_jets", "single_bjets", "double_jets", "double_bjets", "double_untag_jets",
                                 "j1", "j2", "j3", "j4"]:
                             
                        self._histograms[f"eventflow_{suffix}"] = hist.Hist(hist.axis.StrCategory(
                            ["raw", "step1", "trigger", "step2", "step3", "step4"], name="cut"), storage=storage.Double())
                        
                        self._histograms[f"{prefix}_eventflow_{suffix}"] = hist.Hist(hist.axis.StrCategory(
                            ["raw", "step1", "trigger", "step2", "step3", "step4"], name="cut"), storage=storage.Double())
                        
                        self._histograms[f"dm_bbbb_min_{suffix}"]                       = Hist.new.Reg(100, 0, 200,  name="dm",    label=f"|ΔM(bb,bb)| minimum {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_dm_bbbb_min_{suffix}"]     = Hist.new.Reg(100, 0, 200,  name="dm",    label=f"|ΔM(bb,bb)| minimum {suffix}").Weight()
                        self._histograms[f"mass_{objt}_{suffix}"]                       = Hist.new.Reg(100, 0, 1000, name="m",     label=f"{objt} Mass {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_mass_{objt}_{suffix}"]     = Hist.new.Reg(100, 0, 1000, name="m",     label=f"{objt} Mass {suffix}").Weight()
                        self._histograms[f"MTW_bef_{suffix}"]                           = Hist.new.Reg(100, 0, 800,  name="m",     label=f"MTW (bef) {suffix}").Weight()
                        self._histograms[f"{prefix}_MTW_bef_{suffix}"]                  = Hist.new.Reg(100, 0, 800,  name="m",     label=f"MTW (bef) {suffix}").Weight()
                        self._histograms[f"MTW_{suffix}"]                               = Hist.new.Reg(100, 0, 800,  name="m",     label=f"MTW (aft) {suffix}").Weight()
                        self._histograms[f"{prefix}_{region}_MTW_{suffix}"]             = Hist.new.Reg(100, 0, 800,  name="m",     label=f"MTW (aft) {suffix}").Weight()
                        self._histograms[f"MET_bef_{suffix}"]                           = Hist.new.Reg(100, 0, 800,  name="pt",    label=f"MET (bef) {suffix}").Weight()
                        self._histograms[f"{prefix}_MET_bef_{suffix}"]                  = Hist.new.Reg(100, 0, 800,  name="pt",    label=f"MET (bef) {suffix}").Weight()
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
        
        # ====================== #
        # STEP 1 : Build weights #
        # ====================== #   
        print("\nSTEP 1: Build Weights")
        
        n_ev    = len(events)
        weights = Weights(n_ev)
        
        if self.isMC:
            weights.add("norm", np.full(n_ev, self.xsec / self.nevts, dtype="float64"))
        else:
            weights.add("ones", np.ones(n_ev, dtype="float64"))

        w_now = weights.weight()
        print(f"[WEIGHTS] n_ev={len(w_now)}  sum={np.sum(w_now):.6g}  mean={np.mean(w_now):.6g}  "
                  f"min={np.min(w_now):.3g}  max={np.max(w_now):.3g}")

                                      
        # =============================== #
        # STEP 2 : EGM energy corrections #
        # =============================== #
        # https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#2022_2023_and_2024_Scale_and_Sme
        # Align electron energy response/resolution between data and simulation.
        
        print("\nSTEP 2: EGM scale and smearing corrections")
        
        ele_all = events.Electron
        ele_all = ak.with_field(ele_all, ak.broadcast_arrays(events.event, ele_all.pt)[0], "event")
                        
        scEta    = ele_all.superclusterEta
        absScEta = np.abs(scEta)
        
        print("[DEBUG] EGM scale axis order:", [v.name for v in self._egm_scale.inputs])
        # EGM scale axis order: ['syst', 'pt', 'r9', 'AbsScEta'] 
        print("[DEBUG] EGM smear axis order:", [v.name for v in self._egm_smear.inputs])
        # EGM smear axis order: ['syst', 'pt', 'r9', 'AbsScEta']
        
        # DATA: Multiply electron pT by a data scale factor from EGM JSON (depends on pt, r9, |scEta|).
        if (not self.isMC) and (self._egm_scale is not None):
            # run_e = ak.broadcast_arrays(events.run, ele_all.pt)[0]
            counts = ak.num(ele_all.pt, axis=1)
        
            scale_flat = self._egm_scale.evaluate(
                "scale",
                #ak.to_numpy(ak.flatten(run_e)),
                #ak.to_numpy(ak.flatten(scEta)),
                ak.to_numpy(ak.flatten(ele_all.pt)),
                ak.to_numpy(ak.flatten(ele_all.r9)),
                ak.to_numpy(ak.flatten(absScEta)), 
                #ak.to_numpy(ak.flatten(ele_all.seedGain)),
            )
            scale = _unflatten_like(scale_flat, counts)
            
            _stats(scale, "EGM scale (DATA)")
            _stats(ele_all.pt, "Electron pt (pre EGM)")
            _stats(ele_all.pt * scale, "Electron pt (post EGM)")

            _val = ele_all.pt * scale
            ele_corr_pt = ak.values_astype(ak.where(_val > 0.0, _val, 0.0), "float32")
            ElectronCorr = ak.with_field(ele_all, ele_corr_pt, "pt")
        
        # MC: Smear electron pT using a Gaussian width from EGM JSON (depends on pt, r9, |scEta|), with a deterministic RNG per electron.
        elif self.isMC and (self._egm_smear is not None):
            counts = ak.num(ele_all.pt, axis=1)
        
            smear_width_flat = self._egm_smear.evaluate(
                "smear",
                ak.to_numpy(ak.flatten(ele_all.pt)),
                ak.to_numpy(ak.flatten(ele_all.r9)),
                ak.to_numpy(ak.flatten(absScEta)),
            )
            smear_width = _unflatten_like(smear_width_flat, counts)
        
            n = _unflatten_like(_rng_normal_like(ele_all), counts)  # deterministic per electron
            
            # per-electron resolution
            _stats(smear_width, "EGM smear width (MC)")  
            # Gaussian random numbers drawn for each electron             
            _stats(n, "EGM smear RNG n (MC)")         
            
            val = ele_all.pt * (1.0 + smear_width * n)
            ele_corr_pt = ak.values_astype(ak.where(val > 0.0, val, 0.0), "float32")
            
            # Electrons with corrected pt (all other fields unchanged).
            ElectronCorr = ak.with_field(ele_all, ele_corr_pt, "pt")      
        else:
            ElectronCorr = ele_all  
            
        if self.isMC:
            _stats((ElectronCorr.pt/ele_all.pt) - 1.0, "EGM relative pt shift (MC)")
        else:
            _stats((ElectronCorr.pt/ele_all.pt) - 1.0, "EGM relative pt scale shift (DATA)")

                
        # ================================== #
        # STEP 3 : JEC + JER for AK4 Puppi   #
        # ================================== # 
        # https://cms-jerc.web.cern.ch/Recommendations/#2024
        # JEC brings jets onto the correct scale; JER makes MC jet resolution match data.
        
        print("\nSTEP 3: JEC + JER for AK4 Puppi")
        
        jets_in = events.Jet
   
        rawFactor = getattr(jets_in, "rawFactor", ak.zeros_like(jets_in.pt))
        pt_raw    = jets_in.pt   * (1.0 - rawFactor)
        mass_raw  = jets_in.mass * (1.0 - rawFactor)      
        rho_evt   = events.fixedGridRhoFastjetAll      
        rho       = ak.broadcast_arrays(rho_evt, pt_raw)[0]        
        counts    = ak.num(pt_raw, axis=1)
        n_tot     = int(ak.sum(counts))
        
        pt_step = pt_raw
        
        
        # --- Apply L2 (and Data residual for data) using correctionlib on (area, eta, phi, pt_step, ρ). --- #
        # https://cms-jerc.web.cern.ch/JEC/
        # Note: Starting from Run 3, the use of PUPPI jets eliminates the need for L1corrections. 
        # To maintain compatibility with Run 2 scripts, a dummy file is provided.
        # L3 is unity. Mandatory L2 (MC), L2 + L2L3Residuals (DATA)
                           
        # L2Relative
        if self._jec_L2 is not None:
            order = [v.name for v in self._jec_L2.inputs]
            print("[JEC:L2 inputs]", order)
            # [JEC:L2 inputs] ['JetEta', 'JetPhi', 'JetPt']
            
            args_L2 = []
            for name in order:
                if   name == "JetA":   args_L2.append(ak.to_numpy(ak.flatten(jets_in.area)))
                elif name == "JetEta": args_L2.append(ak.to_numpy(ak.flatten(jets_in.eta)))
                elif name == "JetPhi": args_L2.append(ak.to_numpy(ak.flatten(jets_in.phi)))
                elif name == "Rho":    args_L2.append(ak.to_numpy(ak.flatten(rho)))
                elif name == "JetPt":  args_L2.append(ak.to_numpy(ak.flatten(pt_step)))
                elif name == "run":    args_L2.append(ak.to_numpy(ak.flatten(ak.broadcast_arrays(events.run, pt_step)[0])))
                else: raise RuntimeError(f"Unexpected input '{name}' in JEC L2")
                
            cfac = ak.unflatten(self._jec_L2.evaluate(*args_L2), counts)
            pt_step = pt_step * cfac
            
                
        # Data-only residual
        if (not self.isMC) and (self._jec_residual is not None):
            order = [v.name for v in self._jec_residual.inputs]
            print("[JEC:Residual inputs]", order)
            
            run_b = ak.broadcast_arrays(events.run, pt_step)[0]
            args_RES  = []
            for name in order:
                if   name == "JetA":   args_RES.append(ak.to_numpy(ak.flatten(jets_in.area)))
                elif name == "JetEta": args_RES.append(ak.to_numpy(ak.flatten(jets_in.eta)))
                elif name == "JetPhi": args_RES.append(ak.to_numpy(ak.flatten(jets_in.phi)))
                elif name == "Rho":    args_RES.append(ak.to_numpy(ak.flatten(rho)))
                elif name == "JetPt":  args_RES.append(ak.to_numpy(ak.flatten(pt_step)))
                elif name == "run":    args_RES.append(ak.to_numpy(ak.flatten(run_b)))
                else: raise RuntimeError(f"Unexpected input '{name}' in JEC residual")
                
            cfac = ak.unflatten(self._jec_residual.evaluate(*args_RES), counts)
            pt_step = pt_step * cfac
                        
        # --- JEC factor and mass --- #
        jec_factor = ak.where(pt_raw > 0, pt_step / pt_raw, 1.0)
        jec_factor = ak.where(np.isfinite(jec_factor), jec_factor, 1.0)  # guard NaN/Inf
        nf = int(ak.sum(~np.isfinite(jec_factor)))
        if nf:
            print(f"[JEC] WARNING: found {nf} non-finite JEC factors; replacing with 1.0")
        jec_factor = ak.values_astype(_clip_nextafter(jec_factor, 0.0, np.inf), "float32")
        
        pt_jec      = pt_step
        mass_jec    = mass_raw * jec_factor
        
        # --- JER (MC) --- #
        if self.isMC and (self._jer_sf is not None):
            pt  = pt_jec
            eta = jets_in.eta
            
            # SF (nominal)
            order = [v.name for v in self._jer_sf.inputs]
            print("[JER:SF inputs]", order)
            # [JER:SF inputs] ['JetEta', 'JetPt', 'systematic']
            
            arrs = {
                "JetEta":     ak.to_numpy(ak.flatten(jets_in.eta)),
                "JetPt":      ak.to_numpy(ak.flatten(pt_jec)),
                "systematic": "nom",  
                }
            args = [arrs[name] for name in order]
            
            sf_nom_flat = self._jer_sf.evaluate(*args)
            sf_nom      = ak.unflatten(sf_nom_flat, counts)
        
            # Resolution
            if self._jer_res is not None:
                order = [v.name for v in self._jer_res.inputs]
                print("[JER:Res inputs]", order)
                args = []
                for name in order:
                    if name   == "JetEta": args.append(ak.to_numpy(ak.flatten(eta)))
                    elif name == "Rho":    args.append(ak.to_numpy(ak.flatten(rho)))
                    elif name == "JetPt":  args.append(ak.to_numpy(ak.flatten(pt)))
                    else: raise RuntimeError(f"Unexpected input '{name}' in JER Res")
                    
                res = ak.unflatten(self._jer_res.evaluate(*args), counts)
            else:
                res = ak.zeros_like(pt)
                
            # gen-match if available (NaN for unmatched)
            if hasattr(jets_in, "pt_genMatched"):
                pt_gen  = jets_in.pt_genMatched
                has_gen = ak.fill_none(np.isfinite(pt_gen), False)
            else:
                pt_gen  = ak.full_like(pt, np.nan)
                has_gen = ak.zeros_like(pt, dtype=bool)
                
            # tight match: |pT - pT_gen| < 3 * σ * pT
            # https://cms-jerc.web.cern.ch/JER/
            ptdiff_ok   = ak.fill_none(np.abs(ak.mask(pt, has_gen) - ak.mask(pt_gen, has_gen)) < (3.0 * ak.mask(res, has_gen) * ak.mask(pt, has_gen)), False)
            match_tight = has_gen & ptdiff_ok
            
            # start from un-smeared
            pt_corr = pt
            
            # matched & tight: scale
            pt_matched = ak.where((pt_gen + sf_nom * (pt - pt_gen)) > 0.0, pt_gen + sf_nom * (pt - pt_gen), 0.0)
            pt_corr = ak.where(match_tight, pt_matched, pt_corr)
            
            # unmatched or not-tight: stochastic smear (deterministic RNG)
            rng   = np.random.default_rng(12345)
            nsm   = ak.unflatten(rng.standard_normal(n_tot), counts)
            sigma = res * np.sqrt(np.maximum(sf_nom**2 - 1.0, 0.0))
            smear = 1.0 + sigma * nsm
            pt_corr = ak.where(~match_tight, pt * smear, pt_corr)
            pt_corr = ak.where(pt_corr > 1e-6, pt_corr, 1e-6)
        
            jer_factor = ak.values_astype(ak.where(pt > 0, pt_corr / pt, 1.0), "float32")
            mass_corr  = mass_jec * jer_factor
        else:
            pt_corr    = pt_jec
            mass_corr  = mass_jec
            jer_factor = ak.values_astype(ak.ones_like(pt_jec), "float32")
              
        # write back
        jets = ak.with_field(jets_in, ak.values_astype(pt_corr,   "float32"), "pt")
        jets = ak.with_field(jets,    ak.values_astype(mass_corr, "float32"), "mass")
        jets = ak.with_field(jets,    jec_factor, "jecFactor")
        jets = ak.with_field(jets,    jer_factor, "jerFactor")
        
        # raw inputs
        _stats(rawFactor, "Jet rawFactor")
        _stats(pt_raw,    "Jet pt_raw")
        # per-step JEC factors 
        # L2
        fac_L2 = ak.unflatten(self._jec_L2.evaluate(*args_L2), counts)
        _stats(fac_L2, "JEC L2 factor")
        if (not self.isMC):
            fac_RES = ak.unflatten(self._jec_residual.evaluate(*args_RES), counts)
            _stats(fac_RES, "JEC Residual factor")
        # outlier fraction
        frac_hi = np.mean(ak.to_numpy(ak.flatten(jec_factor > 2.0)))
        print(f"[JEC] frac(jec_factor>2) = {frac_hi:.3%}")
        

        # ================== #
        # STEP 4: Type-1 MET #
        # ================== #        
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections#Type_I_Correction_Propagation_of
        # https://indico.cern.ch/event/1546228/contributions/6567938/attachments/3095763/5484272/JetMET_01July2025_JhLee%20.pdf
        # We do not include x-y corrections: impacts MET phi, recommended for PF MET only
        
        print("\nSTEP 4: Type-1 MET")
        
        met_in = events.PuppiMET
        met_px, met_py = _ptphi_to_pxpy(met_in.pt, met_in.phi)
        
        jets_nom  = events.Jet 
        rawFactor = ak.fill_none(getattr(jets_nom, "rawFactor", ak.zeros_like(jets_nom.pt)), 0.0)
        pt_raw    = jets_nom.pt * (1.0 - rawFactor)
        
        # --- Build L2×L3-only pT for PUPPI Type-1 "new" JEC --- #
        
        counts_j  = ak.num(jets_nom.pt, axis=1)
        rho_forL  = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, pt_raw)[0]
        pt_L2L3   = pt_raw
        
        # L2Relative
        if self._jec_L2 is not None:            
            order_L2 = [v.name for v in self._jec_L2.inputs]
            print("[Type-1] JEC L2 inputs:", order_L2)
            
            args_L2_met = []
            for name in order_L2:
                if   name == "JetA":   args_L2_met.append(ak.to_numpy(ak.flatten(jets_nom.area)))
                elif name == "JetEta": args_L2_met.append(ak.to_numpy(ak.flatten(jets_nom.eta)))
                elif name == "JetPhi": args_L2_met.append(ak.to_numpy(ak.flatten(jets_nom.phi)))
                elif name == "Rho":    args_L2_met.append(ak.to_numpy(ak.flatten(rho_forL)))
                elif name == "JetPt":  args_L2_met.append(ak.to_numpy(ak.flatten(pt_L2L3)))
                elif name == "run":    args_L2_met.append(ak.to_numpy(ak.flatten(ak.broadcast_arrays(events.run, pt_L2L3)[0])))
                else: raise RuntimeError(f"[Type-1 MET] Unexpected input '{name}' in JEC L2.")
                
            c2_flat = self._jec_L2.evaluate(*args_L2_met)
            
            if not np.all(np.isfinite(c2_flat)):
                raise RuntimeError("[Type-1 MET] Non-finite values from JEC L2.")
                
            c2 = ak.unflatten(c2_flat, counts_j)
            pt_L2L3 = pt_L2L3 * c2
                          
        # --- Jet mask for PUPPI Type-1 --- #
        overlap_mu = _mask_lepton_overlap(jets_nom, events.Muon,  dr=0.4)
        overlap_el = _mask_lepton_overlap(jets_nom, ElectronCorr, dr=0.4)
        jet_for_met = (
            (pt_L2L3 > 15.0)
            & (np.abs(jets_nom.eta) < 4.8)
            & ak.values_astype(jets_nom.passJetIdTight, bool)
            & overlap_mu & overlap_el
        )
        
        if ak.any(ak.is_none(jet_for_met)):
            raise RuntimeError("[Type-1 MET] jet_for_met has None values; investigate overlap masks or jetId fields.")

        
        pt_old  = jets_nom.pt
        dpt_jec = ak.where(jet_for_met, (pt_L2L3 - pt_old), 0.0)
        
        # --- JER propagation (MC): scale L2L3 by (pt_corr / pt_jec) from Step 3 --- #
        
        if self.isMC:
            jer_ratio    = ak.where(pt_jec > 0, pt_corr / pt_jec, 1.0)
            pt_final_met = pt_L2L3 * jer_ratio
            dpt_jer      = ak.where(jet_for_met, (pt_final_met - pt_L2L3), 0.0)
        else:
            dpt_jer = ak.zeros_like(dpt_jec)
            
        # --- Update MET --- #
        dpx = ak.sum((dpt_jec + dpt_jer) * np.cos(jets_nom.phi), axis=1)
        dpy = ak.sum((dpt_jec + dpt_jer) * np.sin(jets_nom.phi), axis=1)
        met_px_corr = met_px - dpx
        met_py_corr = met_py - dpy
        
        # MC: propagate electron energy correction to MET       
        dpx_el = ak.sum((ElectronCorr.pt - ele_all.pt) * np.cos(ele_all.phi), axis=1)
        dpy_el = ak.sum((ElectronCorr.pt - ele_all.pt) * np.sin(ele_all.phi), axis=1)
            
        met_px_corr = met_px_corr - ak.values_astype(dpx_el, "float64")
        met_py_corr = met_py_corr - ak.values_astype(dpy_el, "float64")
        met_pt_corr, met_phi_corr = _pxpy_to_ptphi(met_px_corr, met_py_corr)
        
        PuppiMETCorr = ak.zip({"pt":  ak.values_astype(met_pt_corr,  "float32"), "phi": ak.values_astype(met_phi_corr, "float32")}, with_name="MET")
        
        # sanity checks
        n_jets_all = ak.sum(ak.num(jets_nom.pt, axis=1))
        n_jets_used = ak.sum(ak.num(jets_nom.pt[jet_for_met], axis=1))
        frac_used = float(n_jets_used) / float(n_jets_all) if n_jets_all>0 else 0.0
        print(f"[Type-1] jets used: {int(n_jets_used)}/{int(n_jets_all)}  ({frac_used:.1%})")
     
        _stats(dpt_jec,          "MET Δpt from JEC (per jet)")
        _stats(dpt_jer,          "MET Δpt from JER (per jet)")
        _stats(dpx,              "MET Δpx sum (jets)")
        _stats(dpy,              "MET Δpy sum (jets)")
        _stats(dpx_el,           "MET Δpx from electrons")
        _stats(dpy_el,           "MET Δpy from electrons")
        _stats(PuppiMETCorr.pt,  "PuppiMETCorr pt (final)")
        _stats(PuppiMETCorr.phi, "PuppiMETCorr phi (final)")
        
        
        # Stash the systematics
        systs = {"jets": {}, "met": {}, "weights": {}}
         
        
        # ============================== #
        # STEP 5 : JER up/down (MC only) #
        # ============================== #
                
        # ============================ #
        # STEP 6 : JES "Total" up/down # 
        # ============================ #
        
        # ============================================ #
        # STEP 7 : Unclustered MET (_umetup/_umetdown) #
        # ============================================ #
                            
        # ================================================= #
        # STEP 8 : Theory weight systematics (PDF / scales) #
        # ================================================= #
        
        # TBD

            
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
            muons     = events.Muon[(events.Muon.pt > 20)   & (np.abs(events.Muon.eta) < 2.4)  & events.Muon.tightId          & (events.Muon.pfRelIso04_all >= 0.15)]    
            electrons = ElectronCorr[(ElectronCorr.pt > 20) & (np.abs(ElectronCorr.eta) < 2.5) & (ElectronCorr.cutBased >= 2) & (ElectronCorr.pfRelIso03_all >= 0.15)]
        else:
            # Standard signal region selection
            muons     = events.Muon[(events.Muon.pt > 20)   & (np.abs(events.Muon.eta) < 2.4)  & events.Muon.tightId          & (events.Muon.pfRelIso04_all < 0.15)]
            electrons = ElectronCorr[(ElectronCorr.pt > 20) & (np.abs(ElectronCorr.eta) < 2.5) & (ElectronCorr.cutBased >= 4) & (ElectronCorr.pfRelIso03_all < 0.15)]

        muons      = ak.with_field(muons, "mu", "lepton_type")
        electrons  = ak.with_field(electrons, "e", "lepton_type")
        
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
        BTAG_WP_TIGHT = None
        if self._btag_wp_vals is not None:
            try:
                BTAG_WP_TIGHT = float(self._btag_wp_vals.evaluate("T"))  # AK4 single b-tag (UParT AK4B)
                print(f"\n[BTAG] WP(T) from JSON: {BTAG_WP_TIGHT:.5f}")
            except Exception as e:
                print(f"[BTAG] WARNING: failed to read WP(T) from JSON: {e}")   
        DBTAG_WP_MEDIUM = 0.38    # AK4 double-b (UParT AK4probbb)
                
        single_jets = jets[goodJet]
        double_jets = jets[goodJet]
             
        # ========== Jet Cleaning ========== #
        single_jets = clean_by_dr(single_jets, leptons, 0.4)
        double_jets = clean_by_dr(double_jets, leptons, 0.4)
        
        # Sort by btag score
        single_jets   = single_jets[ak.argsort(single_jets.btagUParTAK4B, ascending=False)]
        double_jets   = double_jets[ak.argsort(double_jets.btagUParTAK4probbb, ascending=False)]
        
        n_single_jets = ak.num(single_jets)
        n_double_jets = ak.num(double_jets)
        
        # Single AK4 jets
        single_bjets       = single_jets[single_jets.btagUParTAK4B >= BTAG_WP_TIGHT]
        single_bjets       = single_bjets[ak.argsort(single_bjets.btagUParTAK4B, ascending=False)]
        single_untag_jets  = single_jets[single_jets.btagUParTAK4B <  BTAG_WP_TIGHT]
        single_untag_jets  = single_untag_jets[ak.argsort(single_untag_jets.btagUParTAK4B, ascending=True)]
        
        n_single_bjets     = ak.num(single_bjets)
        
        # Double AK4 jets           
        double_bjets      = double_jets[double_jets.btagUParTAK4probbb >= DBTAG_WP_MEDIUM]
        double_bjets      = double_bjets[ak.argsort(double_bjets.btagUParTAK4probbb, ascending=False)]
        double_untag_jets = double_jets[double_jets.btagUParTAK4probbb <  DBTAG_WP_MEDIUM]
        double_untag_jets = double_untag_jets[ak.argsort(double_untag_jets.pt, ascending=False)]
        
        n_double_bjets    = ak.num(double_bjets)
        
        # ===================== #
        # b-tag efficiencies ε  #
        # ===================== #
        # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/#scale-factor-recommendations-for-event-reweighting   
        
        if self.isMC and (self._btag_sf_node is not None):
            print("\nb-tag efficiencies ε and SFs")
            
            jets_for_btag = single_jets      
            score_field   = "btagUParTAK4B"        
            wp_name       = "T"    

            if not hasattr(jets_for_btag, score_field):
                raise RuntimeError(f"[BTAG] jets missing '{score_field}' needed for {wp_name} selection.")
            if not hasattr(jets_for_btag, "hadronFlavour"):
                raise RuntimeError("[BTAG] jets missing 'hadronFlavour'.")
                
            # flatten inputs
            counts      = ak.num(jets_for_btag.pt, axis=1)
            pt_flat     = ak.to_numpy(ak.flatten(jets_for_btag.pt))
            eta_flat    = ak.to_numpy(ak.flatten(jets_for_btag.eta))
            abseta_flat = np.abs(eta_flat)
            flav_flat   = ak.to_numpy(ak.flatten(jets_for_btag.hadronFlavour))
            score_flat  = ak.to_numpy(ak.flatten(getattr(jets_for_btag, score_field)))
            passed_flat = (score_flat >= BTAG_WP_TIGHT)
        
            if np.any(~np.isfinite(pt_flat)) or np.any(~np.isfinite(abseta_flat)):
                raise RuntimeError("[BTAG] Non-finite pt/eta encountered in jets_for_btag.")
            if pt_flat.size != abseta_flat.size or pt_flat.size != flav_flat.size:
                raise RuntimeError("[BTAG] Inconsistent jet array lengths for btag inputs.")
                
            n_nonb = int(np.sum(flav_flat != 5))
            if n_nonb > 0:
                print(f"[BTAG] INFO: Found {n_nonb} non-b jets in jets_for_btag; treating them as SF=1, ε=0 (neutral).")
       
            # --- define ε binning --- #
            pt_edges  = np.array([20., 30., 50., 70., 100., 140., 200., 300., 600.], dtype=float)
            eta_edges = np.array([0.0, 2.5], dtype=float)  

            nx, ny = len(pt_edges) - 1, len(eta_edges) - 1
            
            ix_all = np.clip(np.digitize(pt_flat,     pt_edges,  right=False) - 1, 0, nx - 1)
            iy_all = np.clip(np.digitize(abseta_flat, eta_edges, right=False) - 1, 0, ny - 1)
        
            # build ε table for b only
            sel_b = (flav_flat == 5)
            den_b = np.zeros((ny, nx), dtype=np.int64)
            num_b = np.zeros((ny, nx), dtype=np.int64)
            if np.any(sel_b):
                np.add.at(den_b, (iy_all[sel_b], ix_all[sel_b]), 1)
                np.add.at(num_b, (iy_all[sel_b], ix_all[sel_b]), passed_flat[sel_b].astype(np.int64))
                glob_b = float(np.mean(passed_flat[sel_b]))
            else:
                glob_b = 0.0
                
            with np.errstate(divide="ignore", invalid="ignore"):
                eff_b = num_b / np.maximum(den_b, 1)
            
            n_empty_b = int(np.sum(den_b == 0))
            if n_empty_b:
                print(f"[BTAG] ε_b: {n_empty_b} empty bins -> filled with global rate={glob_b:.4f}")
                eff_b[den_b == 0] = glob_b

            eff_b = np.clip(eff_b, 1e-6, 1 - 1e-6)
        
            # per-jet ε: b gets table lookup; non-b gets 0 
            eff_flat = np.zeros_like(pt_flat, dtype=float)
            if np.any(sel_b):
                eff_flat[sel_b] = eff_b[iy_all[sel_b], ix_all[sel_b]]
                
            # --- per-jet SF from JSON: b only; non-b stays 1 --- #
            order = [v.name for v in self._btag_sf_node.inputs]
            if order != ['systematic','working_point','flavor','abseta','pt']:
                print(f"[BTAG] WARNING: unexpected SF node inputs {order} (continuing)")
            
            sf_flat = np.ones_like(pt_flat, dtype=float)
            if np.any(sel_b):
                try:
                    idx_b = np.nonzero(sel_b)[0]
                    out   = np.empty(idx_b.size, dtype=float)
                    for k, i in enumerate(idx_b):
                        out[k] = self._btag_sf_node.evaluate(
                            "central",          
                            wp_name,            
                            int(flav_flat[i]), 
                            float(abseta_flat[i]), 
                            float(pt_flat[i])     
                        )
                    sf_flat[idx_b] = out
                except Exception as e:
                    print(f"[BTAG] WARNING: SF evaluate failed for b jets: {e}. Using SF=1 for those jets.")

            if np.any(~np.isfinite(sf_flat)):
                raise RuntimeError("[BTAG] Non-finite SF values encountered.")

            # --- per-jet factor and event weight (fixed-WP) --- #
            denom = (1.0 - eff_flat)
            if np.any(denom <= 0):
                bad = np.where(denom <= 0)[0][:5]
                raise RuntimeError(f"[BTAG] ε>=1 in some bins (indices {bad}); check ε_b table.")
            untag_factor = (1.0 - eff_flat * sf_flat) / denom
            jet_factor   = np.where(passed_flat, sf_flat, untag_factor)

            jet_factor_awk = ak.unflatten(jet_factor, counts)
            n_j_in_evt     = ak.num(jet_factor_awk, axis=1)
            prod_nonempty  = ak.prod(jet_factor_awk, axis=1)
            w_btag_full    = ak.where(n_j_in_evt > 0, prod_nonempty, 1.0)
            
            self._w_btag_evt_fullT = ak.to_numpy(w_btag_full) 
            if self._w_btag_evt_fullT.shape[0] != len(events):
                raise RuntimeError("[BTAG] cached event-weight length mismatch")

            # quick diags
            _stats(ak.flatten(jet_factor_awk), "BTAG per-jet factor (b-only)")
            _stats(w_btag_full,                "BTAG event weight (b-only)")
            
        
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
            lead_type_step1 = ak.to_numpy(leptons_1[:, 0].lepton_type)
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
        
        output["lepton_multi_bef"].fill(n=n_leptons,   weight=weights.weight())
        output["lepton_multi_aft"].fill(n=n_leptons_1, weight=weights.weight()[mask_step1])

        # Triggers 
        #---------------------------------------------------------------------
        # selections before/after trigger 
        sel_mu_bef = mask_step1 & (tag_cat == "mu")
        sel_e_bef  = mask_step1 & (tag_cat == "e")

        trigger_mu  = (events.trigger_type & (1 << 2)) != 0  # IsoMu24
        trigger_el  = (events.trigger_type & (1 << 4)) != 0  # Ele30_WPTight_Gsf
        
        sel_mu_aft = sel_mu_bef & ak.to_numpy(trigger_mu)
        sel_e_aft  = sel_e_bef  & ak.to_numpy(trigger_el)
        
        mu_pt_bef = ak.to_numpy(leptons[sel_mu_bef][:, 0].pt) if ak.any(sel_mu_bef) else np.array([], float)
        mu_pt_aft = ak.to_numpy(leptons[sel_mu_aft][:, 0].pt) if ak.any(sel_mu_aft) else np.array([], float)

        e_pt_bef  = ak.to_numpy(leptons[sel_e_bef][:, 0].pt)  if ak.any(sel_e_bef)  else np.array([], float)
        e_pt_aft  = ak.to_numpy(leptons[sel_e_aft][:, 0].pt)  if ak.any(sel_e_aft)  else np.array([], float)
        
        # ---- MU: fill 2D (pt, eff) once per pt-bin ----
        if mu_pt_bef.size:
            bins_mu = output["mu_trg_eff2d"].axes["pt"].edges
            den_mu, _ = np.histogram(mu_pt_bef, bins=bins_mu)                # <- unweighted (data-style)
            num_mu, _ = np.histogram(mu_pt_aft, bins=bins_mu)
            # # If you prefer MC-weighted efficiency, use:
            # w_evt = ak.to_numpy(weights.weight())
            # den_mu, _ = np.histogram(mu_pt_bef, bins=bins_mu, weights=w_evt[sel_mu_bef])
            # num_mu, _ = np.histogram(mu_pt_aft, bins=bins_mu, weights=w_evt[sel_mu_aft])
        
            with np.errstate(divide="ignore", invalid="ignore"):
                eff_mu = np.where(den_mu > 0, num_mu / den_mu, 0.0)
            centers_mu = 0.5 * (bins_mu[:-1] + bins_mu[1:])
            m = den_mu > 0
            if np.any(m):
                output["mu_trg_eff2d"].fill(pt=centers_mu[m], eff=eff_mu[m], weight=den_mu[m])
        
        # ---- E: fill 2D (pt, eff) once per pt-bin ----
        if e_pt_bef.size:
            bins_e = output["e_trg_eff2d"].axes["pt"].edges
            den_e, _ = np.histogram(e_pt_bef, bins=bins_e)
            num_e, _ = np.histogram(e_pt_aft, bins=bins_e)
            # weighted version (optional):
            # w_evt = ak.to_numpy(weights.weight())
            # den_e, _ = np.histogram(e_pt_bef, bins=bins_e, weights=w_evt[sel_e_bef])
            # num_e, _ = np.histogram(e_pt_aft, bins=bins_e, weights=w_evt[sel_e_aft])
        
            with np.errstate(divide="ignore", invalid="ignore"):
                eff_e = np.where(den_e > 0, num_e / den_e, 0.0)
            centers_e = 0.5 * (bins_e[:-1] + bins_e[1:])
            m = den_e > 0
            if np.any(m):
                output["e_trg_eff2d"].fill(pt=centers_e[m], eff=eff_e[m], weight=den_e[m])

        final_trigger_mask          = np.zeros(len(events), dtype=bool)
        final_trigger_mask[mask_mu] = ak.to_numpy(trigger_mu[mask_mu])
        final_trigger_mask[mask_e]  = ak.to_numpy(trigger_el[mask_e])
        
        mask_step1 = mask_step1 & final_trigger_mask
        
        #------------------------------------------------------------------------------------------------------------------------------------------#
        # Lepton weights
        #------------------------------------------------------------------------------------------------------------------------------------------#
        w_lep = np.ones(len(events), dtype=float)
        # TBD later
        
        # ---------- ELECTRON CHANNEL ---------- #
        
        # ---------    MUON CHANNEL   ---------- #
            

         
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
        
        output["single_jets_multi_bef_resolved"].fill(n=n_single_jets_1 ,           weight=weights.weight()[mask_step1])       
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
        
        m2a = np.asarray(mask_step2a)
        w2a = weights.weight()[m2a]
        
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
      
        mu_mask_2a = np.asarray(mask_mu[mask_step2a])
        if np.any(mu_mask_2a):
            w2a_mu = w2a[mu_mask_2a]
            
            output["mu_eventflow_boosted"].fill(cut="step2", weight=np.sum(w2a_mu))
                                            
        e_mask_2a = np.asarray(mask_e[mask_step2a])
        if np.any(e_mask_2a):
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
        w3a  = weights.weight()[np.asarray(mask_step3a)]
        mu3a = np.asarray(mask_mu[mask_step3a])
        e3a  = np.asarray(mask_e[mask_step3a])
        
        output["eventflow_boosted"].fill(cut="step3", weight=np.sum(w3a))
        
        output["double_bjets_multi_aft_boosted"].fill(n=n_double_bjets_3a, weight=w3a)
        output["MTW_bef_boosted"].fill(m=mTW_3a,     weight=w3a)
        output["MET_bef_boosted"].fill(pt=met_3a.pt, weight=w3a)
        
        if np.any(mu3a):
            w3a_mu = w3a[mu3a]
            
            output["mu_eventflow_boosted"].fill(cut="step3",      weight=np.sum(w3a_mu))  
            output["mu_MTW_bef_boosted"].fill(m=mTW_3a[mu3a],     weight=w3a_mu)
            output["mu_MET_bef_boosted"].fill(pt=met_3a[mu3a].pt, weight=w3a_mu)
                                  
        if np.any(e3a):
            w3a_e = w3a[e3a]
            
            output["e_eventflow_boosted"].fill(cut="step3",     weight=np.sum(w3a_e))
            output["e_MTW_bef_boosted"].fill(m=mTW_3a[e3a],     weight=w3a_e)  
            output["e_MET_bef_boosted"].fill(pt=met_3a[e3a].pt, weight=w3a_e)  
                   
        #################################
        # STEP 4a: At MET>25 and MTW>50 #
        #################################
        
        print("\nStarting STEP 4a: MET>25 and MTW>50")
        
        mask_met_3a_full = np.zeros(len(events), dtype=bool)
        mask_mtw_3a_full = np.zeros(len(events), dtype=bool)
        
        idx3a = np.asarray(mask_step3a)
        mask_met_3a_full[idx3a] = ak.to_numpy(met_3a.pt > 25)
        mask_mtw_3a_full[idx3a] = ak.to_numpy(mTW_3a   > 50)    
               
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
        
        vec_lead_l_4a        = make_vector(lead_l_4a)
        vec_met_4a           = make_vector_met(met_4a)
        vec_W_4a             = vec_lead_l_4a + vec_met_4a
        mTW_4a               = trans_massW(vec_lead_l_4a, vec_met_4a)
        
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

        if self.isMC:
            w_btag_evt = getattr(self, "_w_btag_evt_fullT", None)
            if w_btag_evt is None:
                raise RuntimeError("[BTAG] per-event weights not cached; ensure the ε/SF block ran before STEP 2b.")
            if w_btag_evt.shape[0] != len(events):
                raise RuntimeError("[BTAG] cached per-event weight length != number of events")
        else:
            w_btag_evt = np.ones(len(events), dtype="float64")
        
        if self.isMC:
            m2b      = np.asarray(mask_step2b, dtype=bool)
            base_res = base_w_snapshot[m2b]
            with_res = base_res * w_btag_evt[m2b]
            print("\n[DEBUG] Resolved yield comparison:")
            print(f"  events (mask_step2b) = {np.sum(m2b)}")
            print(f"  sum of weights (no btag SF)  = {np.sum(base_res):.6f}")
            print(f"  sum of weights (with btag SF)= {np.sum(with_res):.6f}")
            if np.sum(base_res) > 0:
                print(f"  ratio (with / without)       = {np.sum(with_res)/np.sum(base_res):.6f}")
        
        # final per-event weights for STEP 2b histos (resolved-only = base * btag)
        w2b = base_w_snapshot[np.asarray(mask_step2b)] * w_btag_evt[np.asarray(mask_step2b)]

        
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
        
        if np.any(mu_mask_2b):
            output["mu_eventflow_resolved"].fill(cut="step2", weight=np.sum(w2b[mu_mask_2b]))
            

        if np.any(ele_mask_2b):
            output["e_eventflow_resolved"].fill(cut="step2", weight=np.sum(w2b[ele_mask_2b]))
            
       
        #############################################
        # STEP 3b: At least 3 single b-tag AK4 jets #
        #############################################
        
        print("\nStarting STEP 3b: At least 3 single b-tag AK4 jets")
        
        mask_step3b = mask_step2b & (n_single_bjets >= 3)
        
        print(f"After STEP 3b: {np.sum(mask_step3b)} events remaining")
        
        single_bjets_3b   = single_bjets[mask_step3b]
        single_jets_3b    = single_jets[mask_step3b]
        n_single_bjets_3b = ak.num(single_bjets_3b) 
        
        lead_l_3b = leptons[mask_step3b][:, 0]  
        vec_lead_l_3b = make_vector(lead_l_3b)
        
        met_3b = PuppiMETCorr[mask_step3b]
        vec_met_3b = make_vector_met(met_3b)
        
        mTW_3b = trans_massW(vec_lead_l_3b, vec_met_3b)
        
        # Histogram plotting
        w3b = weights.weight()[np.asarray(mask_step3b)] * w_btag_evt[np.asarray(mask_step3b)]
        
        output["eventflow_resolved"].fill(cut="step3", weight=np.sum(w3b))
        
        output["single_bjets_multi_aft_resolved"].fill(n=n_single_bjets_3b, weight=w3b)
        output["MTW_bef_resolved"].fill(m=mTW_3b,                           weight=w3b)
        output["MET_bef_resolved"].fill(pt=met_3b.pt,                       weight=w3b)
        
        ele_mask_3b = np.asarray(mask_e[mask_step3b])
        mu_mask_3b  = np.asarray(mask_mu[mask_step3b])
        
        if np.any(ele_mask_3b):
            output["e_eventflow_resolved"].fill(cut="step3",             weight=np.sum(w3b[ele_mask_3b]))
            output["e_MTW_bef_resolved"].fill(m=mTW_3b[ele_mask_3b],     weight=w3b[ele_mask_3b])
            output["e_MET_bef_resolved"].fill(pt=met_3b[ele_mask_3b].pt, weight=w3b[ele_mask_3b])
            
        if np.any(mu_mask_3b):
            output["mu_eventflow_resolved"].fill(cut="step3",            weight=np.sum(w3b[mu_mask_3b]))
            output["mu_MTW_bef_resolved"].fill(m=mTW_3b[mu_mask_3b],     weight=w3b[mu_mask_3b])
            output["mu_MET_bef_resolved"].fill(pt=met_3b[mu_mask_3b].pt, weight=w3b[mu_mask_3b])
        
        #################################
        # STEP 4b: MET>25 and MTW>50    #
        # + ABCD classification         #
        #################################
        
        print("\nStarting STEP 4b: MET>25 and MTW>50")
        
        SR_REGION    = "B" if self.isQCD else "A"
        CTRL_REGION  = "D" if self.isQCD else "C"
        REGIONS_RUN  = [SR_REGION, CTRL_REGION]
        SIDE_REGIONS = [CTRL_REGION]
        
        # --- ABCD CLASSIFICATION --- #
        sel3b = np.asarray(mask_step3b)
        if np.any(sel3b):
            t3b = ak.to_numpy((met_3b.pt > 25) & (mTW_3b > 50))
            l3b = ak.to_numpy((met_3b.pt < 25) & (mTW_3b < 50))
            keep_abcd_3b = t3b | l3b      
            
            # Region labeling: A/C for SR (iso), B/D for QCD (anti-iso)
            region_3b = np.full(np.count_nonzero(sel3b), "", dtype=object)
            if self.isQCD:
                region_3b[t3b] = "B"
                region_3b[l3b] = "D"
            else:
                region_3b[t3b] = "A"
                region_3b[l3b] = "C"
                
            # Channel on step3b slice (leading lepton)
            lead_l_3b_all = leptons[mask_step3b][:, 0]
            ch_mu_3b      = (ak.to_numpy(lead_l_3b_all.lepton_type) == "mu")
            ch_e_3b       = (ak.to_numpy(lead_l_3b_all.lepton_type) == "e")
            
            def _regmask3b(lbl):
                return keep_abcd_3b & (region_3b == lbl)

            def _chmask3b(lbl, is_mu):
                return (ch_mu_3b if is_mu else ch_e_3b) & _regmask3b(lbl)
             
            w3b_sel = (weights.weight() * w_btag_evt)[sel3b]
            
            # Build step3b vectors/kinematics
            v_sjs_3b = make_vector(single_jets_3b)
            v_sbs_3b = make_vector(single_bjets_3b)
            vW_3b    = vec_lead_l_3b  + vec_met_3b
            
            lead_j_3b    = single_jets_3b[:, 0]
            sublead_j_3b = single_jets_3b[:, 1]
            j3_3b        = single_jets_3b[:, 2]
        
            mH_3b, ptH_3b, phiH_3b, etaH_3b = higgs_kin(v_sbs_3b, v_sjs_3b)
            HT_3b = ak.sum(single_jets_3b.pt, axis=1)
        
            dphi_metlep_3b = np.abs(vec_met_3b.delta_phi(vec_lead_l_3b))
            dphi_wh_3b     = np.abs(((phiH_3b - vW_3b.phi + np.pi) % (2*np.pi)) - np.pi)
            deta_wh_3b     = np.abs(vW_3b.eta - etaH_3b)
            dr_wh_3b       = np.sqrt(deta_wh_3b**2 + dphi_wh_3b**2)
        
            btag_max_3b    = ak.max(single_bjets_3b.btagUParTAK4B, axis=1)
            btag_min_3b    = ak.min(single_bjets_3b.btagUParTAK4B, axis=1)
            btag_prod_3b   = single_bjets_3b[:, 0].btagUParTAK4B * single_bjets_3b[:, 1].btagUParTAK4B
            dr_bb_ave_3b   = dr_bb_bb_avg(single_bjets_3b)
            pt_ratio_3b    = ak.where(vW_3b.pt > 0, ptH_3b / vW_3b.pt, -1)            
            min_dphi_lj_3b = min_dphi_jets_lepton(jets=single_jets_3b, leptons=lead_l_3b)
            dm4b_3b        = min_dm_bb_bb(make_vector(single_bjets_3b), all_jets=make_vector(single_jets_3b))
            mbbj_3b        = m_bbj(v_sbs_3b, v_sjs_3b)
            lead_b_3b      = single_bjets_3b[:, 0]

            # Fill per-region C/D shapes
            side_regions = SIDE_REGIONS
            for ch_lbl, is_mu in [("mu", True), ("e", False)]:
                for reg_lbl in side_regions:
                    m_evt = _chmask3b(reg_lbl, is_mu)
                    if not np.any(m_evt):
                        continue
                    ww = w3b_sel[m_evt]
                    
                    def H1(name):  
                        return output[f"{ch_lbl}_{reg_lbl}_{name}_resolved"]
                    
                    H1("HT").fill(ht=HT_3b[m_evt],                                      weight=ww)
                    H1("pt_lepton").fill(pt=ak.to_numpy(lead_l_3b.pt)[m_evt],           weight=ww)
                    H1("MET").fill(pt=ak.to_numpy(met_3b.pt)[m_evt],                    weight=ww)
                    H1("MTW").fill(m=mTW_3b[m_evt],                                     weight=ww)
                    H1("pt_W").fill(pt=ak.to_numpy(vW_3b.pt)[m_evt],                    weight=ww)
                    H1("mass_H").fill(m=ak.to_numpy(mH_3b)[m_evt],                      weight=ww)
                    H1("pt_H").fill(pt=ak.to_numpy(ptH_3b)[m_evt],                      weight=ww)
                    H1("pt_j1").fill(pt=lead_j_3b[m_evt].pt,                            weight=ww)
                    H1("pt_j2").fill(pt=sublead_j_3b[m_evt].pt,                         weight=ww)
                    H1("pt_j3").fill(pt=j3_3b[m_evt].pt,                                weight=ww)
                    H1("eta_j1").fill(eta=lead_j_3b[m_evt].eta,                         weight=ww)
                    H1("eta_j2").fill(eta=sublead_j_3b[m_evt].eta,                      weight=ww)
                    H1("eta_j3").fill(eta=j3_3b[m_evt].eta,                             weight=ww)
                    H1("phi_j1").fill(phi=lead_j_3b[m_evt].phi,                         weight=ww)
                    H1("phi_j2").fill(phi=sublead_j_3b[m_evt].phi,                      weight=ww)
                    H1("phi_j3").fill(phi=j3_3b[m_evt].phi,                             weight=ww)
                    H1("pt_b1").fill(pt=ak.to_numpy(single_bjets_3b[:,0].pt)[m_evt],    weight=ww)
                    H1("pt_b2").fill(pt=ak.to_numpy(single_bjets_3b[:,1].pt)[m_evt],    weight=ww)
                    H1("pt_b3").fill(pt=ak.to_numpy(single_bjets_3b[:,2].pt)[m_evt],    weight=ww)
                    H1("eta_b1").fill(eta=lead_b_3b[m_evt].eta,                         weight=ww)
                    H1("eta_b2").fill(eta=single_bjets_3b[:,1][m_evt].eta,              weight=ww)
                    H1("eta_b3").fill(eta=ak.to_numpy(single_bjets_3b[:,2].eta)[m_evt], weight=ww)
                    H1("phi_b1").fill(phi=lead_b_3b[m_evt].phi,                         weight=ww)
                    H1("phi_b2").fill(phi=single_bjets_3b[:,1][m_evt].phi,              weight=ww)
                    H1("phi_b3").fill(phi=single_bjets_3b[:,2][m_evt].phi,              weight=ww)
            
                    has4_cd  = np.asarray(ak.num(single_jets_3b)  >= 4) & m_evt
                    if ak.any(has4_cd):
                        j4_cd  = single_jets_3b[has4_cd][:, 3]
                        w4_cd  = w3b_sel[has4_cd]
                        H1("pt_j4").fill(pt=j4_cd.pt,    weight=w4_cd)
                        H1("eta_j4").fill(eta=j4_cd.eta, weight=w4_cd)
                        H1("phi_j4").fill(phi=j4_cd.phi, weight=w4_cd)
                        
                    has4b_cd = np.asarray(ak.num(single_bjets_3b) >= 4) & m_evt
                    if ak.any(has4b_cd):
                        b4_cd = single_bjets_3b[has4b_cd][:, 3]
                        w4_cd = w3b_sel[has4b_cd]
                        H1("pt_b4").fill(pt=b4_cd.pt,    weight=w4_cd)
                        H1("eta_b4").fill(eta=b4_cd.eta, weight=w4_cd)
                        H1("phi_b4").fill(phi=b4_cd.phi, weight=w4_cd)
                                
                    H1("dphi_WH").fill(dphi=dphi_wh_3b[m_evt],                      weight=ww)
                    H1("deta_WH").fill(deta=deta_wh_3b[m_evt],                      weight=ww)
                    H1("dr_WH").fill(dr=dr_wh_3b[m_evt],                            weight=ww)
                    H1("dphi_MET-lepton").fill(dphi=dphi_metlep_3b[m_evt],          weight=ww)
                    H1("dphi_jet-lepton_min").fill(dphi=min_dphi_lj_3b[m_evt],      weight=ww)         
                    H1("btag_min_single_bjets").fill(btag=btag_min_3b[m_evt],       weight=ww)
                    H1("btag_max_single_bjets").fill(btag=btag_max_3b[m_evt],       weight=ww)
                    H1("btag_prod").fill(btag_prod=btag_prod_3b[m_evt],             weight=ww)           
                    H1("dr_bb_ave").fill(dr=dr_bb_ave_3b[m_evt],                    weight=ww)
                    H1("dm_bbbb_min").fill(dm=dm4b_3b[m_evt],                       weight=ww)
                    H1("mass_bbj").fill(m=mbbj_3b[m_evt],                           weight=ww)           
                    H1("pt_ratio").fill(ratio=pt_ratio_3b[m_evt],                   weight=ww)
                    H1("phi_MET").fill(phi=ak.to_numpy(met_3b.phi)[m_evt],          weight=ww)
                    
                    H = self._histograms
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_higgsMass_shapes_resolved"].fill ( cut_index=0, H_mass=ak.to_numpy(mH_3b)[m_evt],           weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_higgsPt_shapes_resolved"].fill   ( cut_index=0, H_pt=ak.to_numpy(ptH_3b)[m_evt],            weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_b1Pt_shapes_resolved"].fill      ( cut_index=0, pt_b1=ak.to_numpy(lead_b_3b.pt)[m_evt],     weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_ht_shapes_resolved"].fill        ( cut_index=0, HT=HT_3b[m_evt],                            weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_pfmet_shapes_resolved"].fill     ( cut_index=0, MET_pt=ak.to_numpy(met_3b.pt)[m_evt],       weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_mtw_shapes_resolved"].fill       ( cut_index=0, MTW=mTW_3b[m_evt],                          weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_ptw_shapes_resolved"].fill       ( cut_index=0, W_pt=ak.to_numpy(vW_3b.pt)[m_evt],          weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_dphiWh_shapes_resolved"].fill    ( cut_index=0, dphi_WH=dphi_wh_3b[m_evt],                  weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_dphijetlep_shapes_resolved"].fill( cut_index=0, dphi_lep_met=min_dphi_lj_3b[m_evt],         weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_dRave_shapes_resolved"].fill     ( cut_index=0, dr_bb_ave=dr_bb_ave_3b[m_evt],              weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_dmmin_shapes_resolved"].fill     ( cut_index=0, dm_4b_min=dm4b_3b[m_evt],                   weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_lep_pt_raw_shapes_resolved"].fill( cut_index=0, pt_lepton=ak.to_numpy(lead_l_3b.pt)[m_evt], weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_jets_shapes_resolved"].fill      ( cut_index=0, n_jets=ak.num(single_jets_3b[m_evt]),       weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_btag_prod_shapes_resolved"].fill ( cut_index=0, btag_prod=btag_prod_3b[m_evt],              weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_btag_min_shapes_resolved"].fill  ( cut_index=0, btag_min=btag_min_3b[m_evt],                weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_btag_max_shapes_resolved"].fill  ( cut_index=0, btag_max=btag_max_3b[m_evt],                weight=ww)
                    H[f"{ch_lbl}_{reg_lbl}_SR_3b_mbbj_shapes_resolved"].fill      ( cut_index=0, mbbj=mbbj_3b[m_evt],                        weight=ww)

        
        mask_met_3b_full = np.zeros(len(events), dtype=bool)
        mask_mtw_3b_full = np.zeros(len(events), dtype=bool)
        
        mask_met_3b_full[np.asarray(sel3b)] = np.asarray(met_3b.pt > 25)
        mask_mtw_3b_full[np.asarray(sel3b)] = np.asarray(mTW_3b   > 50)
        
        mask_step4b = mask_step3b & mask_met_3b_full & mask_mtw_3b_full
        print(f"After STEP 4b: {np.sum(mask_step4b)} events remaining")
        print(f"Events passing MET cut only: {np.sum(mask_step3b & mask_met_3b_full)}")
        print(f"Events passing MTW cut only: {np.sum(mask_step3b & mask_mtw_3b_full)}")
        
        single_jets_4b       = single_jets[mask_step4b]
        single_bjets_4b      = single_bjets[mask_step4b]
        
        vec_single_jets_4b   = make_vector(single_jets_4b)
        vec_single_bjets_4b  = make_vector(single_bjets_4b)
        
        n_sjs  = ak.num(single_jets_4b)
        
        lead_l_4b          = leptons[mask_step4b][:, 0]  
        vec_lead_l_4b      = make_vector(lead_l_4b)
        
        met_4b             = PuppiMETCorr[mask_step4b]
        vec_met_4b         = make_vector_met(met_4b)
        
        mTW_4b             = trans_massW(vec_lead_l_4b, vec_met_4b)
        
        mass_H, pt_H, phi_H, eta_H = higgs_kin(vec_single_bjets_4b, vec_single_jets_4b)
        
        vec_W_4b           = vec_met_4b + vec_lead_l_4b
        HT_4b              = ak.sum(single_jets_4b.pt, axis=1)
        
        dphi_metlep_4b     = np.abs(vec_met_4b.delta_phi(vec_lead_l_4b))
        dphi_wh_4b         = np.abs(((phi_H - vec_W_4b.phi + np.pi) % (2*np.pi)) - np.pi)
        deta_wh_4b         = np.abs(vec_W_4b.eta - eta_H)
        dr_wh_4b           = np.sqrt(deta_wh_4b**2 + dphi_wh_4b**2)
              
        btag_max_4b        = ak.max(single_bjets_4b.btagUParTAK4B, axis=1)
        btag_min_4b        = ak.min(single_bjets_4b.btagUParTAK4B, axis=1)
        btag_prod_4b       = single_bjets_4b[:, 0].btagUParTAK4B * single_bjets_4b[:, 1].btagUParTAK4B
             
        dr_bb_avg_4b       = dr_bb_bb_avg(single_bjets_4b)             
        pt_ratio_4b        = ak.where(vec_W_4b.pt > 0, pt_H   / vec_W_4b.pt, -1)
        min_dphi_lepjet_4b = min_dphi_jets_lepton(jets=single_jets_4b, leptons=lead_l_4b)         
        dm4b_4b            = min_dm_bb_bb(make_vector(single_bjets_4b), all_jets=make_vector(single_jets_4b))  
        
        mbbj_4b            = m_bbj(vec_single_bjets_4b, vec_single_jets_4b)    
        lead_j_4b          = single_jets_4b[:, 0]
        sublead_j_4b       = single_jets_4b[:, 1]
        lead_b_4b          = single_bjets_4b[:, 0]
        sublead_b_4b       = single_bjets_4b[:, 1]
        vec_lead_b_4b      = make_vector(lead_b_4b)
        vec_sublead_b_4b   = make_vector(sublead_b_4b)
        wh_pt_asymmetry_4b = np.abs(pt_H - vec_W_4b.pt) / (pt_H + vec_W_4b.pt)       
                           
        ele_mask_4b  = np.asarray(mask_e[mask_step4b])
        mu_mask_4b   = np.asarray(mask_mu[mask_step4b])
        w4b          = weights.weight()[np.asarray(mask_step4b)] * w_btag_evt[np.asarray(mask_step4b)]
        
        # Histogram plotting
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
        output["pt_j1_resolved"].fill(pt=lead_j_4b.pt,                       weight=w4b)
        output["pt_j2_resolved"].fill(pt=sublead_j_4b.pt,                    weight=w4b)
        output["pt_j3_resolved"].fill(pt=single_jets_4b[:, 2].pt,            weight=w4b) 
        output["mass_bbj_resolved"].fill(m=mbbj_4b,                          weight=w4b)
        output["btag_prod_resolved"].fill(btag_prod=btag_prod_4b,            weight=w4b)
        output["wh_pt_asym_resolved"].fill(pt=wh_pt_asymmetry_4b,            weight=w4b)
        output["deta_WH_resolved"].fill(deta=deta_wh_4b,                     weight=w4b)
        output["dphi_WH_resolved"].fill(dphi=dphi_wh_4b,                     weight=w4b)
        output["dphi_MET-lepton_resolved"].fill(dphi=dphi_metlep_4b,         weight=w4b)
        output["dr_WH_resolved"].fill(dr=dr_wh_4b,                           weight=w4b) 
        output["eta_j1_resolved"].fill(eta=lead_j_4b.eta,                    weight=w4b) 
        output["eta_j2_resolved"].fill(eta=sublead_j_4b.eta,                 weight=w4b) 
        output["eta_j3_resolved"].fill(eta=single_jets_4b[:, 2].eta,         weight=w4b) 
        output["phi_j1_resolved"].fill(phi=lead_j_4b.phi,                    weight=w4b) 
        output["phi_j2_resolved"].fill(phi=sublead_j_4b.phi,                 weight=w4b) 
        output["phi_j3_resolved"].fill(phi=single_jets_4b[:, 2].phi,         weight=w4b) 
        output["phi_MET_resolved"].fill(phi=met_4b.phi,                      weight=w4b) 
        output["pt_b1_resolved"].fill(pt=lead_b_4b.pt,                       weight=w4b)
        output["pt_b2_resolved"].fill(pt=sublead_b_4b.pt,                    weight=w4b)
        output["eta_b1_resolved"].fill(eta=lead_b_4b.eta,                    weight=w4b)
        output["eta_b2_resolved"].fill(eta=sublead_b_4b.eta,                 weight=w4b)
        output["phi_b1_resolved"].fill(phi=lead_b_4b.phi,                    weight=w4b)
        output["phi_b2_resolved"].fill(phi=sublead_b_4b.phi,                 weight=w4b)
        output["pt_b3_resolved"].fill(pt=single_bjets_4b[:, 2].pt,           weight=w4b)
        output["eta_b3_resolved"].fill(eta=single_bjets_4b[:, 2].eta,        weight=w4b)
        output["phi_b3_resolved"].fill(phi=single_bjets_4b[:, 2].phi,        weight=w4b)
        
        
        has4 = ak.num(single_jets_4b) >= 4            
        j4   = single_jets_4b[has4][:, 3]             
        w4   = w4b[has4]
        output["pt_j4_resolved"].fill(pt=j4.pt,    weight=w4)
        output["eta_j4_resolved"].fill(eta=j4.eta, weight=w4)
        output["phi_j4_resolved"].fill(phi=j4.phi, weight=w4)
        
        has4b_4b = ak.num(single_bjets_4b) >= 4
        if ak.any(has4b_4b):
            b4_4b  = single_bjets_4b[has4b_4b][:, 3]
            w4b_4  = w4b[has4b_4b]
            output["pt_b4_resolved"].fill(pt=b4_4b.pt,     weight=w4b_4)
            output["eta_b4_resolved"].fill(eta=b4_4b.eta,  weight=w4b_4)
            output["phi_b4_resolved"].fill(phi=b4_4b.phi,  weight=w4b_4)
                
        for ch_lbl, ch_mask in [("mu", mu_mask_4b), ("e", ele_mask_4b)]:
            if not np.any(ch_mask):
                continue
            ww = w4b[ch_mask]
        
            # eventflow per channel
            output[f"{ch_lbl}_eventflow_resolved"].fill(cut="step4", weight=np.sum(ww))
        
            def H(suffix):
                return output[f"{ch_lbl}_{SR_REGION}_{suffix}_resolved"]
        
            # === per-channel, per-region SR shapes === #
            H("HT").fill(ht=HT_4b[ch_mask],                                    weight=ww)
            H("pt_j1").fill(pt=lead_j_4b[ch_mask].pt,                          weight=ww)
            H("pt_j2").fill(pt=sublead_j_4b[ch_mask].pt,                       weight=ww)
            H("pt_j3").fill(pt=single_jets_4b[:, 2][ch_mask].pt,               weight=ww)
            H("pt_lepton").fill(pt=lead_l_4b[ch_mask].pt,                      weight=ww)
            H("MET").fill(pt=met_4b[ch_mask].pt,                               weight=ww)
            H("MTW").fill(m=mTW_4b[ch_mask],                                   weight=ww)
            H("pt_W").fill(pt=vec_W_4b[ch_mask].pt,                            weight=ww)
            H("mass_H").fill(m=mass_H[ch_mask],                                weight=ww)
            H("pt_H").fill(pt=pt_H[ch_mask],                                   weight=ww)
            H("btag_min_single_bjets").fill(btag=btag_min_4b[ch_mask],         weight=ww)
            H("btag_max_single_bjets").fill(btag=btag_max_4b[ch_mask],         weight=ww)
            H("dphi_WH").fill(dphi=dphi_wh_4b[ch_mask],                        weight=ww)
            H("dr_WH").fill(dr=dr_wh_4b[ch_mask],                              weight=ww)
            H("dphi_jet-lepton_min").fill(dphi=min_dphi_lepjet_4b[ch_mask],    weight=ww)
            H("dphi_MET-lepton").fill(dphi=dphi_metlep_4b[ch_mask],            weight=ww)
            H("dr_bb_ave").fill(dr=dr_bb_avg_4b[ch_mask],                      weight=ww)
            H("pt_ratio").fill(ratio=pt_ratio_4b[ch_mask],                     weight=ww)
            H("btag_prod").fill(btag_prod=btag_prod_4b[ch_mask],               weight=ww)
            H("deta_WH").fill(deta=deta_wh_4b[ch_mask],                        weight=ww)
            H("dm_bbbb_min").fill(dm=dm4b_4b[ch_mask],                         weight=ww)
            H("mass_bbj").fill(m=mbbj_4b[ch_mask],                             weight=ww)
            H("eta_j1").fill(eta=lead_j_4b[ch_mask].eta,                       weight=ww)
            H("eta_j2").fill(eta=sublead_j_4b[ch_mask].eta,                    weight=ww)
            H("eta_j3").fill(eta=single_jets_4b[:, 2][ch_mask].eta,            weight=ww)
            H("phi_j1").fill(phi=lead_j_4b[ch_mask].phi,                       weight=ww)
            H("phi_j2").fill(phi=sublead_j_4b[ch_mask].phi,                    weight=ww)
            H("phi_j3").fill(phi=single_jets_4b[:, 2][ch_mask].phi,            weight=ww)
            H("phi_MET").fill(phi=met_4b[ch_mask].phi,                         weight=ww)
            H("pt_b1").fill(pt=lead_b_4b[ch_mask].pt,                          weight=ww)
            H("pt_b2").fill(pt=sublead_b_4b[ch_mask].pt,                       weight=ww)
            H("eta_b1").fill(eta=lead_b_4b[ch_mask].eta,                       weight=ww)
            H("eta_b2").fill(eta=sublead_b_4b[ch_mask].eta,                    weight=ww)
            H("phi_b1").fill(phi=lead_b_4b[ch_mask].phi,                       weight=ww)
            H("phi_b2").fill(phi=sublead_b_4b[ch_mask].phi,                    weight=ww)
            H("pt_b3").fill(pt=single_bjets_4b[:, 2][ch_mask].pt,              weight=ww)
            H("eta_b3").fill(eta=single_bjets_4b[:, 2][ch_mask].eta,           weight=ww)
            H("phi_b3").fill(phi=single_bjets_4b[:, 2][ch_mask].phi,           weight=ww)
        
            has4_ch  = np.asarray(ak.num(single_jets_4b)  >= 4) & ch_mask
            if ak.any(has4_ch):
                j4_ch = single_jets_4b[has4_ch][:, 3]
                w4_ch = w4b[has4_ch]
                H("pt_j4").fill(pt=j4_ch.pt,    weight=w4_ch)
                H("eta_j4").fill(eta=j4_ch.eta, weight=w4_ch)
                H("phi_j4").fill(phi=j4_ch.phi, weight=w4_ch)
                
            has4b_ch = np.asarray(ak.num(single_bjets_4b) >= 4) & ch_mask
            if ak.any(has4b_ch):
                b4_ch = single_bjets_4b[has4b_ch][:, 3]
                w4_ch = w4b[has4b_ch]
                H("pt_b4").fill(pt=b4_ch.pt,     weight=w4_ch)
                H("eta_b4").fill(eta=b4_ch.eta,  weight=w4_ch)
                H("phi_b4").fill(phi=b4_ch.phi,  weight=w4_ch)
        
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
        
            # dynamic SR region (A for SR run, B for QCD run)
            sr_region = "B" if self.isQCD else "A"
        
            for i, cut in enumerate(self.optim_Cuts1_bdt):
                cut_mask = (bdt_score_resolved > cut)
                if not np.any(cut_mask):
                    continue
        
                # loop over channels in one go
                for ch_lbl, ch_mask_all in [("e", e_mask_all4b), ("mu", mu_mask_all4b)]:
                    ch_mask_cut = ch_mask_all & cut_mask
                    if not np.any(ch_mask_cut):
                        continue
        
                    w = weights_resolved[ch_mask_cut]
                    s = bdt_score_resolved[ch_mask_cut]
        
                    # 1D (non-regioned) bdt score per channel
                    output[f"{ch_lbl}_bdt_score_resolved"].fill(bdt=s, weight=w)
        
                    # helper for the 2D shapes with dynamic region A/B
                    def H2D(name):
                        return output[f"{ch_lbl}_{SR_REGION}_SR_3b_{name}_shapes_resolved"]
        
                    H2D("bdt").fill                (cut_index=i, bdt=s,                                                weight=w)
                    H2D("higgsMass").fill          (cut_index=i, H_mass=ak.to_numpy(mass_H)[ch_mask_cut],              weight=w)
                    H2D("higgsPt").fill            (cut_index=i, H_pt=ak.to_numpy(pt_H)[ch_mask_cut],                  weight=w)
                    H2D("b1Pt").fill               (cut_index=i, pt_b1=ak.to_numpy(lead_b_4b.pt)[ch_mask_cut],         weight=w)
                    H2D("ht").fill                 (cut_index=i, HT=HT_4b[ch_mask_cut],                                weight=w)
                    H2D("pfmet").fill              (cut_index=i, MET_pt=ak.to_numpy(met_4b.pt)[ch_mask_cut],           weight=w)
                    H2D("mtw").fill                (cut_index=i, MTW=mTW_4b[ch_mask_cut],                              weight=w)
                    H2D("ptw").fill                (cut_index=i, W_pt=ak.to_numpy(vec_W_4b.pt)[ch_mask_cut],           weight=w)
                    H2D("dRwh").fill               (cut_index=i, dr_WH=dr_wh_4b[ch_mask_cut],                          weight=w)
                    H2D("dphiWh").fill             (cut_index=i, dphi_WH=np.abs(dphi_wh_4b[ch_mask_cut]),              weight=w)
                    H2D("dphijetlep").fill         (cut_index=i, dphi_lep_met=np.abs(min_dphi_lepjet_4b[ch_mask_cut]), weight=w)
                    H2D("dRave").fill              (cut_index=i, dr_bb_ave=dr_bb_avg_4b[ch_mask_cut],                  weight=w)
                    H2D("dmmin").fill              (cut_index=i, dm_4b_min=dm4b_4b[ch_mask_cut],                       weight=w)
                    H2D("lep_pt_raw").fill         (cut_index=i, pt_lepton=ak.to_numpy(lead_l_4b.pt)[ch_mask_cut],     weight=w)
                    H2D("wh_pt_asym").fill         (cut_index=i, WH_pt_assymetry=wh_pt_asymmetry_4b[ch_mask_cut],      weight=w)
                    H2D("jets").fill               (cut_index=i, n_jets=ak.num(single_jets_4b[ch_mask_cut]),           weight=w)
                    H2D("btag_prod").fill          (cut_index=i, btag_prod=btag_prod_4b[ch_mask_cut],                  weight=w)
                    H2D("btag_min").fill           (cut_index=i, btag_min=btag_min_4b[ch_mask_cut],                    weight=w)
                    H2D("btag_max").fill           (cut_index=i, btag_max=btag_max_4b[ch_mask_cut],                    weight=w)
                    H2D("mbbj").fill               (cut_index=i, mbbj=mbbj_4b[ch_mask_cut],                            weight=w)
                    
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
            
            

        return output

    def postprocess(self, accumulator):
        return accumulator
