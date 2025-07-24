import numpy as np
import awkward as ak
import os
import json
import argparse
from coffea import processor
from coffea.analysis_tools import Weights
import hist
from collections import defaultdict
from coffea.nanoevents.methods import vector
from hist import Hist
import coffea.util
import itertools
from boost_histogram import storage
from utils.deltas_array import (
    delta_r,
    clean_by_dr,
    delta_phi,
    delta_eta
)
from utils.variables_def import (
    min_dm_bb_bb,
    dr_bb_bb_avg,
    min_dm_doubleb_bb,
    dr_doubleb_bb,
    m_bbj,
    dr_bb_avg
)
from utils.jet_tight_id import compute_jet_id
from utils.xgb_tools import XGBHelper
import itertools

def safe_array(array, target_len, fill_value=np.nan):
    array = ak.fill_none(array, fill_value)
    np_array = ak.to_numpy(array)

    if np.isscalar(np_array):
        np_array = np.full(target_len, fill_value)

    if len(np_array) == target_len:
        return np_array
    elif len(np_array) < target_len:
        pad_width = target_len - len(np_array)
        return np.concatenate([np_array, np.full(pad_width, fill_value)])
    else:
        return np_array[:target_len]

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
    },with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)

def build_sum_vector(jets, mask, n):
    '''
    Constructs a summed Lorentz vector from the first `n` jets per event **after applying a mask**.
    
    '''
    jets_masked = jets[mask]
    
    vec_sum = None
    for i in range(n):
        component = make_vector(jets_masked[:, i])
        vec_sum = component if vec_sum is None else vec_sum + component

    return vec_sum

def build_sum_reg_vector(jets, mask, n):
    '''                                                                                                                              
    Constructs a summed Lorentz vector from the first `n` jets per event 
    '''
    jets_masked = jets[mask]

    vec_sum = None
    for i in range(n):
        component = make_regressed_vector(jets_masked[:, i])
        vec_sum = component if vec_sum is None else vec_sum + component

    return vec_sum

class TOTAL_Processor(processor.ProcessorABC):
    def __init__(self, xsec=0.89, nevts=3000, isMC=False, dataset_name=None, is_MVA=False, run_eval=True):
        self.xsec = xsec
        self.nevts = nevts
        self.dataset_name = dataset_name
        self.isMC = isMC
        self.is_MVA= is_MVA
        self._trees = {regime: defaultdict(list) for regime in ["boosted", "resolved", "merged"]} if is_MVA else None
        self.run_eval= run_eval
        self._histograms = {}
        self.bdt_eval_boost=XGBHelper(os.path.join("xgb_model", "bdt_model_boosted.json"), ["H_mass", "H_pt", "HT","puppimet_pt","btag_max","btag_min","dr_bb_ave","dm_bb_bb_min","dphi_H_MET","pt_untag_max","dphi_untag_MET","pt_tag_max","n_untag"])
        self.bdt_eval_mer=XGBHelper(os.path.join("xgb_model", "bdt_model_merged.json"), ["H_mass", "H_pt", "HT","puppimet_pt","n_untag","pt_untag_max", "dphi_untag_MET", "btag_min","dr_bb_ave","dm_bb_bb_min", "dphi_H_MET"])
        self.bdt_eval_res=XGBHelper(os.path.join("xgb_model", "bdt_model_resolved.json"), ["H_mass", "H_pt", "HT","dphi_H_MET","dm_bb_bb_min","mbbj","puppimet_pt","n_untag","btag_min", "pt_untag_max","dr_bb_ave","dphi_untag_MET"])
        self._histograms["cutflow_2l"] = hist.Hist(
            hist.axis.StrCategory(["raw", "2lep", "Mll", "boosted", "merged", "resolved"], name="cut"),
            storage=storage.Double()
        )
        self._histograms["n_doubleb_merged"] = hist.Hist(hist.axis.Integer(0, 6, name="n", label="N double-b jets (merged)"), storage=storage.Double())
        self._histograms["n_singleb_merged"] = hist.Hist(hist.axis.Integer(0, 6, name="n", label="N single-b jets (merged)"), storage=storage.Double())
        
        for suffix in ["_boosted", "_merged", "_resolved"]:
            self._histograms[f"bdt_score{suffix}"] = (
                Hist.new.Reg(100, 0, 1, name="bdt", label=f"bdt score {suffix}").Weight()
            )
            self._histograms[f"mass_H{suffix}"] = (
                Hist.new.Reg(80, 0, 400, name="m", label=f"Higgs mass {suffix}").Weight()
            )
            self._histograms[f"pt_H{suffix}"] = (
                Hist.new.Reg(100, 0, 1000, name="pt", label=f"Higgs pt {suffix}").Weight()
            )
            
            self._histograms[f"dr_bb_ave{suffix}"] = (
                Hist.new.Reg(50, 0, 5, name="dr", label=f"A pt {suffix}").Weight()
            )
           
            self._histograms[f"pt_b_max{suffix}"] = (
                Hist.new.Reg(100, 0, 1000, name="pt", label=f"max b pt {suffix}").Weight()
            )
            self._histograms[f"pt_untag_max{suffix}"] = (
                Hist.new.Reg(100, 0, 1000, name="pt", label=f"max untag pt {suffix}").Weight()
            )
            self._histograms[f"n_untag{suffix}"] = hist.Hist(hist.axis.Integer(0, 6, name="n", label="N untag"), storage=storage.Double())
            self._histograms[f"HT{suffix}"] = (
                Hist.new.Reg(100, 0, 1000, name="ht", label=f"HT (scalar sum of pT of double jets) {suffix}").Weight()
            )
            self._histograms[f"puppimet_pt{suffix}"] = (
                Hist.new.Reg(100, 0, 1000, name="met", label=f"met pt {suffix}").Weight()
            )
            self._histograms[f"btag_min{suffix}"] = (
                Hist.new.Reg(50, 0, 1, name="btag", label=f"btag min {suffix}").Weight()
            )
            self._histograms[f"btag_max{suffix}"] = (
                Hist.new.Reg(50, 0, 1, name="btag", label=f"btag max {suffix}").Weight()
            )
            self._histograms[f"dphi_H_MET{suffix}"] = Hist.new.Reg(50, 0, 5, name="dphi", label=f"dphi(H, Z) {suffix}").Weight()
            self._histograms[f"dphi_untag_MET{suffix}"] = Hist.new.Reg(50, 0, 5, name="dphi", label=f"dphi(j, met) {suffix}").Weight()

            self._histograms[f"dm_bb_bb_min{suffix}"] = (
               Hist.new.Reg(100, 0, 200, name="dm", label=f"|ΔM(bb, bb)|_min {suffix}").Weight()
            )
            
            self._histograms[f"dr_bb_bb_ave{suffix}"] = (
               Hist.new.Reg(100, 0, 10, name="dr", label=f"|ΔR(bb, bb)|_ave {suffix}").Weight()
            )
            self._histograms["m_bbj"] = (
                Hist.new.Reg(100, 0, 600, name="m", label="m(bbj) resolved").Weight()
            )

    @property
    def histograms(self):
        return self._histograms

    def add_tree_entry(self, regime, data_dict):
        if not self._trees or regime not in self._trees:
            return
        for key, val in data_dict.items():
            self._trees[regime][key].extend(np.atleast_1d(val))

    
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

        is_signal = (
            (self.dataset_name.startswith("ZH_ZToAll_HToAATo4B") or
             self.dataset_name.startswith("WH_WToAll_HToAATo4B") and self.is_MC)
        )
        #for bdt train label
        #label_value = 1 if is_signal else 0

        weight_array = np.ones(n) * ( self.xsec / self.nevts)
        weights = Weights(n)
        weights.add("norm", weight_array)
        output = {key: hist if not hasattr(hist, "copy") else hist.copy() for key, hist in self._histograms.items()}
        
        output["cutflow_2l"].fill(cut="raw", weight=np.sum(weights.weight()))
        
        #object configuration
        muons = events.Muon[(events.Muon.pt > 10) & (np.abs(events.Muon.eta) < 2.5) & events.Muon.tightId & (events.Muon.pfRelIso03_all < 0.15)]
        electrons = events.Electron[(events.Electron.pt > 15) & (np.abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased >= 4) & (events.Electron.pfRelIso03_all < 0.15)]

        leptons = ak.concatenate([muons, electrons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
        n_leptons = ak.num(leptons)
       
        # Decide whether to compute Jet ID (XROOTD) or use existing branches (from skimmed file on eos)
        if is_signal:
            # Compute on the fly
            tight_id, tight_lep_veto = compute_jet_id(events.Jet)
            
        else:
            # Use already-stored tight ID from skimmed data
            tight_id = events.Jet.passJetIdTight
            tight_lep_veto = events.Jet.passJetIdTightLepVeto
        #single jets§
        single_jets =events.Jet[(events.Jet.pt_regressed > 20) & (np.abs(events.Jet.eta) < 2.5) &  tight_id & tight_lep_veto ]
        #cc single jets with leptons.
        single_jets = clean_by_dr(single_jets, leptons, 0.4)
        single_jets = single_jets[ak.argsort(single_jets.btagUParTAK4B , axis=-1, ascending=False)] 
        #single b jets
        single_bjets= single_jets[single_jets.btagUParTAK4B > 0.4648]
        single_untag_jets=single_jets[single_jets.btagUParTAK4B < 0.4648]
        single_untag_jets=single_untag_jets[ak.argsort(single_untag_jets.pt , axis=-1, ascending=False)] 
        #sort single bjets by b tag score
        single_bjets = single_bjets[ak.argsort(single_bjets.btagUParTAK4B, ascending=False)]
        n_single_bjets=ak.num(single_bjets)
        #double jets
        double_jets =events.Jet[(events.Jet.pt_regressed > 20) & (np.abs(events.Jet.eta) < 2.5) &  tight_id & tight_lep_veto &
                                (events.Jet.svIdx1 != -1) & (events.Jet.svIdx2 != -1) &
                                (events.Jet.svIdx1 != events.Jet.svIdx2)        
                            ]
        #cc double jets with leptons
        double_jets = clean_by_dr(double_jets, leptons, 0.4)
        double_jets = double_jets[ak.argsort(double_jets.btagUParTAK4probbb, axis=-1, ascending=False)] 
        double_bjets= double_jets[double_jets.btagUParTAK4probbb > 0.38]
        double_untag_jets= double_jets[double_jets.btagUParTAK4probbb < 0.38]

        #double bjets
        double_bjets = double_bjets[ak.argsort(double_bjets.btagUParTAK4probbb, ascending=False)]
        double_untag_jets = double_untag_jets[ak.argsort(double_untag_jets.pt_regressed, ascending=False)]
        #cc doubleb jets with single b jets->keep the cross cleaned single b jets
        single_jets_cc = clean_by_dr(single_jets,double_jets, 0.4)
        single_jets_cc = single_jets_cc[ak.argsort(single_jets_cc.btagUParTAK4B , axis=-1, ascending=False)] 
        single_bjets_cc= single_jets_cc[single_jets_cc.btagUParTAK4B > 0.4648]
        single_untag_jets_cc= single_jets_cc[single_jets_cc.btagUParTAK4B < 0.4648]
        single_untag_jets_cc = single_untag_jets_cc[ak.argsort(single_untag_jets_cc.pt, axis=-1, ascending=False)] 
        n_single_bjets_cc=ak.num(single_bjets_cc)
        n_double_bjets=ak.num(double_bjets)
        #0lep
        mask_0lep = n_leptons == 0
        mask_met = events.PuppiMET.pt > 170
        mask0 = mask_0lep & mask_met
        
        #boosted 
        full_mask_double = mask0 & (n_double_bjets >= 2)
        output["cutflow_2l"].fill(cut="boosted", weight=np.sum(weights.weight()[full_mask_double]))
        weights_boosted= weights.weight()[full_mask_double]
        double_bjets_boosted = double_bjets[full_mask_double]
        double_jets_boosted = double_jets[full_mask_double]
        double_untag_jets_boosted = double_untag_jets[full_mask_double]
        
        n_untagged_boo = ak.num(double_untag_jets[full_mask_double])
        output["n_untag_boosted"].fill(n=n_untagged_boo, weight=weights_boosted)
        met_boosted = events.PuppiMET[full_mask_double]

        ht_boosted = ak.sum(double_jets_boosted.pt_regressed, axis=1)
        output["HT_boosted"].fill(ht=ht_boosted, weight=weights_boosted)
        puppimet_boo= events.PuppiMET[full_mask_double]
        pt_max_boosted=double_bjets_boosted[:, 0].pt_regressed
        btag_max_boosted=double_bjets_boosted[:, 0].btagUParTAK4probbb
        btag_min_boosted=double_bjets_boosted[:, 1].btagUParTAK4probbb
        output["btag_max_boosted"].fill(btag=btag_max_boosted, weight=weights_boosted)
        output["btag_min_boosted"].fill(btag=btag_min_boosted, weight=weights_boosted)

        lead_bb = double_bjets_boosted[:, 0]
        sublead_bb = double_bjets_boosted[:, 1]  
        # Build Lorentz vectors from regressed pt manually
        lead_bb_vec = make_regressed_vector(double_bjets_boosted[:, 0])
        sublead_bb_vec = make_regressed_vector(double_bjets_boosted[:, 1])
        
        # Higgs candidate = vector sum of lead and sublead
        higgs_boost = lead_bb_vec + sublead_bb_vec

        output["mass_H_boosted"].fill(m=higgs_boost.mass, weight=weights_boosted)
        output["pt_H_boosted"].fill(pt=higgs_boost.pt, weight=weights_boosted)
        output["dphi_H_MET_boosted"].fill(dphi=higgs_boost.delta_phi(met_boosted), weight=weights_boosted)
        dm_boosted = np.abs(lead_bb.mass - sublead_bb.mass)
        output["dm_bb_bb_min_boosted"].fill(dm=dm_boosted, weight=weights_boosted)
        lead_untag_boosted = ak.firsts(double_untag_jets_boosted)
        valid_mask_boo = ~ak.is_none(lead_untag_boosted)
       
        #double_bjets_sorted_by_pt = ak.sort(double_bjets_boosted, axis=-1, ascending=False, key=lambda jet: jet.pt_regressed)
        pt_tag_max_boo = ak.max(double_bjets_boosted.pt_regressed, axis=1)

        
        output["pt_untag_max_boosted"].fill(
            pt=lead_untag_boosted[valid_mask_boo].pt_regressed,
            weight=weights_boosted[valid_mask_boo]
        )

        output["dphi_untag_MET_boosted"].fill(
            dphi=lead_untag_boosted[valid_mask_boo].delta_phi(met_boosted[valid_mask_boo]),
            weight=weights_boosted[valid_mask_boo]
        )
      
        n_boost = len(weights_boosted)
        bdt_boosted = {
            "H_mass": safe_array(higgs_boost.mass, n_boost),
            "H_pt": safe_array(higgs_boost.pt, n_boost),
            "HT": safe_array(ht_boosted, n_boost),
            "puppimet_pt": safe_array(puppimet_boo.pt, n_boost),
            "btag_max": safe_array(btag_max_boosted, n_boost),
            "btag_min": safe_array(btag_min_boosted, n_boost),
            "dr_bb_ave": safe_array(dr_bb_avg(double_bjets_boosted), n_boost),
            "dm_bb_bb_min": safe_array(dm_boosted, n_boost),
            "dphi_H_MET": safe_array(np.abs(higgs_boost.delta_phi(met_boosted)), n_boost),
            "pt_untag_max": safe_array(lead_untag_boosted[valid_mask_boo].pt_regressed, n_boost),
            "dphi_untag_MET": safe_array(
                np.abs(lead_untag_boosted[valid_mask_boo].delta_phi(met_boosted[valid_mask_boo])), n_boost
            ),
            "pt_tag_max": safe_array(pt_tag_max_boo, n_boost),
            "n_untag": safe_array(n_untagged_boo, n_boost),
            "weight": ak.to_numpy(weights_boosted),
        }

        if self.is_MVA:
            self.compat_tree_variables(bdt_boosted)
            self.add_tree_entry("boosted", bdt_boosted)

        ### MODEL EVAL BOOST###
        if self.run_eval and not self.is_MVA :
            inputs_boost = {key: bdt_boosted[key] for key in self.bdt_eval_boost.var_list}
            bdt_score_boost = self.bdt_eval_boost.eval(inputs_boost)
            output["bdt_score_boosted"].fill(bdt=bdt_score_boost, weight=weights_boosted)

            
        #resolved
        # For resolved regime
        full_mask_res = mask0 & (n_single_bjets >= 3)
        output["cutflow_2l"].fill(cut="resolved", weight=np.sum(weights.weight()[full_mask_res]))
        weights_res = weights.weight()[full_mask_res]
        # Filtered objects
        single_bjets_resolved = single_bjets[full_mask_res]
        single_jets_resolved = single_jets[full_mask_res]
        single_untag_jets_resolved = single_untag_jets[full_mask_res]

        #single_bjets_sorted_by_pt = ak.sort(single_bjets_resolved, axis=-1, ascending=False, key=lambda jet: jet.pt_regressed)
        pt_tag_max_res = ak.max(double_bjets_boosted.pt_regressed, axis=1)


        vec_single_bjets_resolved = make_regressed_vector(single_bjets_resolved)
        vec_single_jets_resolved = make_regressed_vector(single_jets_resolved)
        # Select top 3 or 4 b-tagged jets
        # Boolean masks per condition
        mask_3b_3j = (ak.num(single_bjets_resolved) == 3) & (ak.num(single_jets_resolved) == 3)
        mask_4b = ak.num(single_bjets_resolved) == 4
        mask_3b_4j = (ak.num(single_bjets_resolved) == 3) & (ak.num(single_jets_resolved) == 4)

        # Combine first two into same behavior (3b+3j or 4b)
        vec_H1_res = build_sum_reg_vector(single_bjets_resolved, mask_3b_3j , n=3)
        vec_H2_res = build_sum_reg_vector(single_bjets_resolved, mask_4b , n=4)
        vec_H3_res = build_sum_reg_vector(single_jets_resolved,mask_3b_4j, n=4)
        
        vec_H_res = ak.concatenate([vec_H1_res, vec_H2_res, vec_H3_res])
        weights_res_all = ak.concatenate([
            weights_res[mask_3b_3j],
            weights_res[mask_4b],
            weights_res[mask_3b_4j]
        ])

        output["mass_H_resolved"].fill(m=vec_H_res.mass, weight=weights_res_all)
        output["pt_H_resolved"].fill(pt=vec_H_res.pt, weight=weights_res_all)
       
        # MET
        puppimet_res = events.PuppiMET[full_mask_res]
        puppimet_res1 = puppimet_res[mask_3b_3j]
        puppimet_res2 = puppimet_res[mask_4b]
        puppimet_res3 = puppimet_res[mask_3b_4j]

        puppimet_all_res = ak.concatenate([puppimet_res1, puppimet_res2, puppimet_res3])

        
        vec_H_res= make_vector(vec_H_res)
        output["dphi_H_MET_merged"].fill(dphi=vec_H_res.delta_phi(make_vector_met(puppimet_all_res)),weight=weights_res_all)

        # --- Quantities independent of Higgs definition ---
        output["dm_bb_bb_min_resolved"].fill(
            dm=min_dm_bb_bb(vec_single_bjets_resolved, all_jets=vec_single_jets_resolved),
            weight=weights_res
        )

        output["dr_bb_bb_ave_resolved"].fill(
            dr=dr_bb_bb_avg(vec_single_bjets_resolved, all_jets=vec_single_jets_resolved),
            weight=weights_res
        )

        mbbj_resolved = m_bbj(vec_single_bjets_resolved, vec_single_jets_resolved)
        valid_mask_mbbj = ~ak.is_none(mbbj_resolved)
        output["m_bbj"].fill(
            m=mbbj_resolved[valid_mask_mbbj],
            weight=weights_res[valid_mask_mbbj]
        )

        output["HT_resolved"].fill(
            ht=ak.sum(single_jets_resolved.pt_regressed, axis=1),
            weight=weights_res
        )

        output["puppimet_pt_resolved"].fill(
            met=puppimet_res.pt,
            weight=weights_res
        )

        output["n_untag_resolved"].fill(
            n=ak.num(single_untag_jets_resolved),
            weight=weights_res
        )

        output["btag_max_resolved"].fill(
            btag=ak.max(single_bjets_resolved.btagUParTAK4B, axis=1),
            weight=weights_res
        )

        pt_max_untagged_res = ak.firsts(single_untag_jets_resolved).pt_regressed
        valid = ~ak.is_none(pt_max_untagged_res)

        output["pt_untag_max_resolved"].fill(
            pt=ak.to_numpy(pt_max_untagged_res[valid]),
            weight=ak.to_numpy(weights_res[valid])
        )
        
        output["dr_bb_ave_resolved"].fill(
            dr=dr_bb_avg(single_bjets_resolved),
            weight=weights_res
        )
        
        first_untagged = ak.firsts(single_untag_jets_resolved)
        valid_mask = ~ak.is_none(first_untagged)
        
        output["dphi_untag_MET_resolved"].fill(
            dphi=first_untagged[valid_mask].delta_phi(puppimet_res[valid_mask]),
            weight=weights_res[valid_mask]
        )
        n_res = len(weights_res)
        bdt_resolved = {
            "H_mass": safe_array(vec_H_res.mass, n_res),
            "H_pt": safe_array(vec_H_res.pt, n_res),
            "HT": safe_array(ak.sum(single_jets_resolved.pt_regressed, axis=1), n_res),
            "dphi_H_MET": safe_array(
                np.abs(vec_H_res.delta_phi(make_vector_met(puppimet_all_res))), n_res
            ),
            "dm_bb_bb_min": safe_array(
                min_dm_bb_bb(vec_single_bjets_resolved, all_jets=vec_single_jets_resolved),
                n_res
            ),
            "m_bbj": safe_array(mbbj_resolved[valid_mask_mbbj], n_res),
            "puppimet_pt": safe_array(puppimet_res.pt, n_res),
            "n_untag": safe_array(ak.num(single_untag_jets_resolved), n_res),
            "btag_min": safe_array(ak.min(single_bjets_resolved.btagUParTAK4B, axis=1), n_res),
            "pt_untag_max": safe_array(pt_max_untagged_res[valid], n_res),
            "dr_bb_ave": safe_array(dr_bb_avg(single_bjets_resolved), n_res),
            "dphi_untag_MET": safe_array(
                np.abs(first_untagged[valid_mask].delta_phi(puppimet_res[valid_mask])),
                n_res
            ),
            
            "weight": ak.to_numpy(weights_res),
        }

        if self.is_MVA:
            self.compat_tree_variables(bdt_resolved)
            self.add_tree_entry("resolved", bdt_resolved)
         ### MODEL EVAL resolved###
         
        if self.run_eval and not self.is_MVA :
            #bdt_score_res= self.bdt_eval_res.eval(inputs_res)
            #output["bdt_score_resolved"].fill(bdt=bdt_score_res, weight=weights_res)
            inputs_res = {key: bdt_resolved[key] for key in self.bdt_eval_res.var_list}
            bdt_score_res = self.bdt_eval_res.eval(inputs_res)
            output["bdt_score_resolved"].fill(bdt=bdt_score_res, weight=weights_res)

        #merged
        full_mask_merged = mask0 & ((n_double_bjets >= 1) & (n_single_bjets_cc >= 1))
        output["cutflow_2l"].fill(cut="merged", weight=np.sum(weights.weight()[full_mask_merged]))
        double_bjets_merged = double_bjets[full_mask_merged]
        single_bjets_merged = single_bjets_cc[full_mask_merged]
        pt_tag_max_mer = double_bjets_merged[:, 0].pt_regressed
        double_jets_merged = double_jets[full_mask_merged]
        single_jets_merged = single_jets_cc[full_mask_merged]
        single_untag_jets_merged = single_untag_jets_cc[full_mask_merged]
        weights_merged = weights.weight()[full_mask_merged]
        output["n_doubleb_merged"].fill(n=ak.num(double_bjets_merged),weight=weights_merged)
        output["n_singleb_merged"].fill(n=ak.num(single_bjets_merged), weight=weights_merged)
        n_dbb = ak.num(double_bjets_merged)
        n_sbb = ak.num(single_bjets_merged)
        n_sj  = ak.num(single_jets_merged)

        # Case 1: 1 double-b + 1 single-b + 1 jet
        mask_case1 = (n_dbb == 1) & (n_sbb == 1) & (n_sj == 1)

        # Case 2: 1 double-b + 2 single-b
        mask_case2 = (n_dbb == 1) & (n_sbb == 2)

        # Case 3: 1 double-b + 1 single-b + ≥2 jets
        mask_case3 = (n_dbb == 1) & (n_sbb == 1) & (n_sj > 1)
        vec_H1 = build_sum_reg_vector(double_bjets_merged,mask_case1,1)+ build_sum_vector(single_bjets_merged,mask_case1,1)
      
        vec_H2 = build_sum_reg_vector(double_bjets_merged,mask_case2,1)+ build_sum_vector(single_bjets_merged,mask_case2,2)
       
        vec_H3 = build_sum_reg_vector(double_bjets_merged,mask_case3,1)+ build_sum_vector(single_jets_merged,mask_case3,2)
        
        vec_H_merged = ak.concatenate([vec_H1, vec_H2, vec_H3])
        weights_merged_all = ak.concatenate([
            weights_merged[mask_case1],
            weights_merged[mask_case2],
            weights_merged[mask_case3]
        ])



        output["mass_H_merged"].fill(m=vec_H_merged.mass, weight=weights_merged_all)
        output["pt_H_merged"].fill(pt=vec_H_merged.pt, weight=weights_merged_all)
        # MET
        puppimet_merged = events.PuppiMET[full_mask_merged]
        puppimet_case1 = puppimet_merged[mask_case1]
        puppimet_case2 = puppimet_merged[mask_case2]
        puppimet_case3 = puppimet_merged[mask_case3]

        puppimet_all = ak.concatenate([puppimet_case1, puppimet_case2, puppimet_case3])

        
        vec_H_merged= make_vector(vec_H_merged)
        output["dphi_H_MET_merged"].fill(dphi=np.abs(ak.to_numpy(vec_H_merged.delta_phi(make_vector_met(puppimet_all)))),weight=weights_merged_all)

        # Other variables that don’t depend on Higgs candidate
        
        ht_single = ak.sum(single_jets_merged.pt_regressed, axis=1)
        ht_double = ak.sum(double_jets_merged.pt_regressed, axis=1)
        ht_total = ht_single + ht_double
        output["HT_merged"].fill(
            ht=ak.to_numpy(ht_total),
            weight=ak.to_numpy(weights_merged)
        )

        output["puppimet_pt_merged"].fill(
            met=puppimet_merged.pt,
            weight=weights_merged
        )
        
        # Count is safe as-is
        output["n_untag_merged"].fill(
            n=ak.num(single_untag_jets_merged),
            weight=weights_merged
        )
        
        # First jet selections
        first_untag_merged = ak.firsts(single_untag_jets_merged)
        valid_mask_merged = ~ak.is_none(first_untag_merged)
        
        # pt histogram
        output["pt_untag_max_merged"].fill(
            pt=ak.to_numpy(first_untag_merged[valid_mask_merged].pt_regressed),
            weight=ak.to_numpy(weights_merged[valid_mask_merged])
        )
        
        # dphi histogram
        output["dphi_untag_MET_merged"].fill(
            dphi=np.abs(ak.to_numpy(first_untag_merged[valid_mask_merged].delta_phi(puppimet_merged[valid_mask_merged]))),
            weight=ak.to_numpy(weights_merged[valid_mask_merged])
        )

    
        output["btag_min_merged"].fill(
            btag=ak.min(single_bjets_merged.btagUParTAK4B, axis=1),
            weight=weights_merged
        )

        output["dr_bb_ave_merged"].fill(
            dr=dr_bb_avg(single_bjets_merged),
            weight=weights_merged
        )

        output["dm_bb_bb_min_merged"].fill(
            dm=min_dm_bb_bb(make_regressed_vector(single_bjets_merged), all_jets=make_regressed_vector(single_jets_merged)),
            weight=weights_merged
        )
        n_merged = len(weights_merged)
        bdt_merged = {
            "H_mass": safe_array(vec_H_merged.mass, n_merged),
            "H_pt": safe_array(vec_H_merged.pt, n_merged),
            "HT": safe_array(ht_total, n_merged),
            "puppimet_pt": safe_array(puppimet_merged.pt, n_merged),
            "n_untag": safe_array(ak.num(single_untag_jets_merged), n_merged),
            "pt_untag_max": safe_array(first_untag_merged[valid_mask_merged].pt_regressed, n_merged),
            "dphi_untag_MET": safe_array(
                np.abs(first_untag_merged[valid_mask_merged].delta_phi(puppimet_merged[valid_mask_merged])),n_merged),
            "btag_min": safe_array(ak.min(single_bjets_merged.btagUParTAK4B, axis=1), n_merged),
            "dr_bb_ave": safe_array(dr_bb_avg(single_bjets_merged), n_merged),
            "dm_bb_bb_min": safe_array(
                min_dm_bb_bb(make_regressed_vector(single_bjets_merged), all_jets=make_regressed_vector(single_jets_merged)),
                n_merged
            ),
            
            "dphi_H_MET": safe_array(
                np.abs(vec_H_merged.delta_phi(make_vector_met(puppimet_all))), n_merged
            ),
            
            "weight": ak.to_numpy(weights_merged),
        }
        if self.is_MVA:
            self.compat_tree_variables(bdt_merged)
            self.add_tree_entry("merged", bdt_merged)

         ### MODEL EVAL Merged###                                                                       
        if self.run_eval and not self.is_MVA :
            inputs_mer = {key: bdt_merged[key] for key in self.bdt_eval_mer.var_list}
            bdt_score_mer = self.bdt_eval_mer.eval(inputs_mer)
            output["bdt_score_merged"].fill(bdt=bdt_score_mer, weight=weights_merged)

        if self.is_MVA:
            output["trees"] = self._trees
            for regime, trees in self._trees.items():
                print(f"[DEBUG] Regime '{regime}' has {len(trees)} entries")
        return output 
    def postprocess(self, accumulator):
      
        return accumulator
