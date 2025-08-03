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
    delta_eta,
    delta_phi_raw
)
from utils.variables_def import (
    min_dm_bb_bb,
    dr_bb_bb_avg,
    min_dm_doubleb_bb,
    dr_doubleb_bb,
    m_bbj,
    dr_bb_avg,
    higgs_kin
)

from utils.xgb_tools import XGBHelper


import itertools

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



class TOTAL_Processor(processor.ProcessorABC):
    def __init__(self, xsec=0.89, nevts=3000, isMC=True, dataset_name=None, is_MVA=False, run_eval=True):
        self.xsec = xsec
        self.nevts = nevts
        self.dataset_name = dataset_name
        self.isMC = isMC
        self.is_MVA= is_MVA
        self._trees = {regime: defaultdict(list) for regime in ["boosted", "resolved"]} if is_MVA else None
        self.run_eval= run_eval
        self._histograms = {}
        
        self.bdt_eval_boost=XGBHelper(os.path.join("xgb_model", "bdt_model_boosted.json"), ["H_mass", "H_pt", "HT","pt_ratio","puppimet_pt","btag_max","btag_min","dr_bb_ave","dm_bb_bb_min","dphi_H_Z","pt_untag_max","pt_tag_max","n_untag"])
        self.bdt_eval_res=XGBHelper(os.path.join("xgb_model", "bdt_model_resolved.json"), ["H_mass", "H_pt", "HT","dphi_H_MET","dm_bb_bb_min","m_bbj","puppimet_pt","n_untag","btag_min", "pt_untag_max","dr_bb_ave","dphi_untag_MET"])
        
        self._histograms["cutflow_2l"] = hist.Hist(
            hist.axis.StrCategory(["raw", "2lep", "Mll", "boosted", "resolved"], name="cut"),
            storage=storage.Double()
        )
       
        for suffix in ["_boosted",  "_resolved"]:
            self._histograms[f"bdt_score{suffix}"] = (
                Hist.new.Reg(300, 0, 1, name="bdt", label=f"bdt score {suffix}").Weight()
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
            self._histograms[f"btag_prod{suffix}"] = (
                Hist.new.Reg(50, 0, 1, name="btag", label=f"btag prod {suffix}").Weight()
            )
            #self._histograms[f"dphi_H_Z{suffix}"] = Hist.new.Reg(50, 0, 5, name="dphi", label=f"dphi(H, Z) {suffix}").Weight()
            self._histograms[f"dphi_untag_Z{suffix}"] = Hist.new.Reg(50, 0, 5, name="dphi", label=f"dphi(j, Z) {suffix}").Weight()

            self._histograms[f"dm_bb_bb_min{suffix}"] = (
               Hist.new.Reg(100, 0, 200, name="dm", label=f"|ΔM(bb, bb)|_min {suffix}").Weight()
            )
            
            self._histograms[f"dr_bb_bb_ave{suffix}"] = (
               Hist.new.Reg(100, 0, 10, name="dr", label=f"|ΔR(bb, bb)|_ave {suffix}").Weight()
            )
            self._histograms["m_bbj"] = (
                Hist.new.Reg(100, 0, 600, name="m", label="m(bbj) resolved").Weight()
            )
            self._histograms[f"pt_ll{suffix}"] = Hist.new.Reg(100, 0, 1000, name="ptll", label=f"Dilepton pt {suffix}").Weight()
            self._histograms[f"mass_ll{suffix}"] = Hist.new.Reg(80, 0, 200, name="mll", label=f"Dilepton mass {suffix}").Weight()
            self._histograms[f"pt_ratio{suffix}"] = Hist.new.Reg(50, 0, 5, name="ratio", label=f"pt(H)/pt(Z) {suffix}").Weight()
            self._histograms[f"dr_HZ{suffix}"] = Hist.new.Reg(60, 0, 6, name="dr", label=f"ΔR(H, Z) {suffix}").Weight()
            self._histograms[f"dphi_HZ{suffix}"] = Hist.new.Reg(64, 0, np.pi, name="dphi", label=f"Δφ(H, Z) {suffix}").Weight()
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
            (self.dataset_name.startswith("ZH-ZToAll-HToAATo4B") or
             self.dataset_name.startswith("WH-WToAll-HToAATo4B") and self.is_MC)
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
        #2lep
        has_2lep = n_leptons >= 2
        if ak.any(has_2lep):
            lep = leptons[has_2lep]
            lead = lep[:, 0]
            sublead = lep[:, 1]

            pt_sel = (((np.abs(lead.pdgId) == 11) & (lead.pt > 25) & (sublead.pt > 15)) |
                      ((np.abs(lead.pdgId) == 13) & (lead.pt > 15) & (sublead.pt > 10)))

            same_flavor = (np.abs(lead.pdgId) == np.abs(sublead.pdgId))
            opposite_sign = lead.charge != sublead.charge
            dilep_mass = (lead + sublead).mass
            in_z_window = (dilep_mass > 80) & (dilep_mass < 100)

            local_mask = pt_sel & same_flavor & opposite_sign
            selected = ak.to_numpy(has_2lep)
            full_mask_2l = np.zeros(len(events), dtype=bool)
            full_mask_2l[selected] = ak.to_numpy(local_mask)

            output["cutflow_2l"].fill(cut="2lep", weight=np.sum(weights.weight()[full_mask_2l]))
            full_mask_mll = full_mask_2l.copy()
            full_mask_mll[selected] &= ak.to_numpy(in_z_window)
            output["cutflow_2l"].fill(cut="Mll", weight=np.sum(weights.weight()[full_mask_mll]))
        
            #boosted 
            full_mask_double = full_mask_mll & (n_double_bjets >= 2)
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
            output["btag_prod_boosted"].fill(btag=btag_min_boosted*btag_max_boosted, weight=weights_boosted)
            lead_bb = double_bjets_boosted[:, 0]
            sublead_bb = double_bjets_boosted[:, 1]  
            # Build Lorentz vectors from regressed pt manually
            lead_bb_vec = make_regressed_vector(double_bjets_boosted[:, 0])
            sublead_bb_vec = make_regressed_vector(double_bjets_boosted[:, 1])
            
            # Higgs candidate = vector sum of lead and sublead
            higgs_boost = lead_bb_vec + sublead_bb_vec

            output["mass_H_boosted"].fill(m=higgs_boost.mass, weight=weights_boosted)
            output["pt_H_boosted"].fill(pt=higgs_boost.pt, weight=weights_boosted)
            dm_boosted = np.abs(lead_bb.mass - sublead_bb.mass)
            output["dm_bb_bb_min_boosted"].fill(dm=dm_boosted, weight=weights_boosted)
            
            ##leptons
            leptons_boosted = leptons[full_mask_double]
            lead_lep_boosted = leptons_boosted[:,0]
            sub_lep_boosted = leptons_boosted[:,1]
            vec_lead_lep_boo= make_vector(lead_lep_boosted)
            vec_sub_lep_boo= make_vector(sub_lep_boosted)
            dilepton_boosted= vec_lead_lep_boo+ vec_sub_lep_boo
            output["pt_ll_boosted"].fill(ptll=dilepton_boosted.pt, weight=weights_boosted)
            output["mass_ll_boosted"].fill(mll=dilepton_boosted.mass, weight=weights_boosted)
            pt_ratio_boo = higgs_boost.pt / dilepton_boosted.pt
            output["pt_ratio_boosted"].fill(ratio=pt_ratio_boo, weight=weights_boosted)
            output["dphi_HZ_boosted"].fill(dphi=np.abs(higgs_boost.delta_phi(dilepton_boosted)), weight=weights_boosted)
            dr_boo = higgs_boost.delta_r(dilepton_boosted)
            output["dr_HZ_boosted"].fill(dr=dr_boo, weight=weights_boosted)
            lead_untag_boosted = ak.firsts(double_untag_jets_boosted)
            # Fallback: bjet with minimum b-tag
            fallback_bjet_boo = ak.firsts(
                double_bjets_boosted[
                    ak.argmin(double_bjets_boosted.btagUParTAK4probbb, axis=1, keepdims=True)
                ]
            )

            # Use leading untagged jet if it exists
            has_untagged_boo = ak.num(double_untag_jets_boosted) > 0
            proxy_jet_boo = ak.where(has_untagged_boo, lead_untag_boosted, fallback_bjet_boo)

            # Use proxy for Δφ and pt
            dphi_proxy_Z_boo = proxy_jet_boo.delta_phi(dilepton_boosted)
            pt_proxy_boo = proxy_jet_boo.pt_regressed

            
           
        
            n_boost = len(weights_boosted)
            bdt_boosted = {
                "H_mass": ak.to_numpy(higgs_boost.mass),
                "H_pt": ak.to_numpy(higgs_boost.pt),
                "HT": ak.to_numpy(ht_boosted),
                "pt_ratio": ak.to_numpy(pt_ratio_boo),
                "puppimet_pt": ak.to_numpy(puppimet_boo.pt),
                "btag_max": ak.to_numpy(btag_max_boosted),
                "btag_min": ak.to_numpy(btag_min_boosted),
                "btag_prod": ak.to_numpy(btag_min_boosted * btag_max_boosted),
                "dr_bb_ave": ak.to_numpy(dr_bb_avg(double_bjets_boosted)),
                "dm_bb_bb_min": ak.to_numpy(dm_boosted),
                "dphi_HZ": ak.to_numpy(np.abs(higgs_boost.delta_phi(dilepton_boosted))),
                "dr_HZ": ak.to_numpy(higgs_boost.delta_r(dilepton_boosted)),
                "pt_untag_max": ak.to_numpy(ak.fill_none(pt_proxy_boo, np.nan)),
                "dphi_untag_Z": ak.to_numpy(np.abs(dphi_proxy_Z_boo)),
                "n_untag": ak.to_numpy(n_untagged_boo),
                "pt_ll": ak.to_numpy(dilepton_boosted.pt),
                "mass_ll": ak.to_numpy(dilepton_boosted.mass),
                "weight": ak.to_numpy(weights_boosted),
            }

            if self.is_MVA:
                self.compat_tree_variables(bdt_boosted)
                self.add_tree_entry("boosted", bdt_boosted)

            ### MODEL EVAL BOOST###
            if self.run_eval and not self.is_MVA:
                inputs_boost = {
                    key: np.asarray(bdt_boosted[key], dtype=np.float64)
                    for key in self.bdt_eval_boost.var_list
                }
                bdt_score_boost = self.bdt_eval_boost.eval(inputs_boost)
                output["bdt_score_boosted"].fill(
                    bdt=np.ravel(bdt_score_boost),
                    weight=np.ravel(np.asarray(weights_boosted, dtype=np.float64))
                )
            #resolved
            # For resolved regime
            full_mask_res =  full_mask_mll & (n_single_bjets >= 3)
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
        
            ##leptons
            leptons_res = leptons[full_mask_res]
            lead_lep_res = leptons_res[:,0]
            sub_lep_res = leptons_res[:,1]
            vec_lead_lep_res= make_vector(lead_lep_res)
            vec_sub_lep_res= make_vector(sub_lep_res)
            dilepton_res= vec_lead_lep_res+ vec_sub_lep_res
            output["pt_ll_resolved"].fill(ptll=dilepton_res.pt, weight=weights_res)
            output["mass_ll_resolved"].fill(mll=dilepton_res.mass, weight=weights_res)
            ### higgs depentent
            mass_H, pt_H, phi_H, eta_H = higgs_kin(single_bjets_resolved, single_jets_resolved)
            vec_H_res= ak.zip({
                "pt": pt_H,
                "eta": eta_H,
                "phi": phi_H,
                "mass": mass_H,
            }, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior) 
            output["mass_H_resolved"].fill(m=mass_H, weight=weights_res)
            output["pt_H_resolved"].fill(pt=pt_H, weight=weights_res)
            output["dphi_HZ_resolved"].fill(
            dphi=np.abs(delta_phi_raw(phi_H, dilepton_res.phi)),
            weight=weights_res
        )

            pt_ratio_res = vec_H_res.pt / dilepton_res.pt
            output["pt_ratio_resolved"].fill(ratio=pt_ratio_res, weight=weights_res)

            output["dr_HZ_resolved"].fill(dr=vec_H_res.delta_r(dilepton_res), weight=weights_res)
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
                ht=ak.sum(single_jets_resolved.pt_regressed, axis=1),
                weight=weights_res
            )
            puppimet_res= events.PuppiMET[full_mask_res]
            output["puppimet_pt_resolved"].fill(
                met=puppimet_res.pt,
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
            output["m_bbj"].fill( m=mbbj_resolved,weight=weights_res)
            
            
            # Define a fallback: bjet with minimum b-tag
            fallback_bjet = ak.firsts(
                single_bjets_resolved[
                    ak.argmin(single_bjets_resolved.btagUParTAK4B, axis=1, keepdims=True)
                ]
            )

            leading_untagged = ak.firsts(single_untag_jets_resolved)
            has_untagged = ak.num(single_untag_jets_resolved) > 0
            proxy_jet = ak.where(has_untagged, leading_untagged, fallback_bjet)
            dphi_proxy_Z = proxy_jet.delta_phi(dilepton_res)
            output["dphi_untag_Z_resolved"].fill(
                dphi=np.abs(dphi_proxy_Z),
                weight=weights_res
            )
            n_res = len(weights_res)
            bdt_resolved = {
                "H_mass": ak.to_numpy(vec_H_res.mass),
                "H_pt": ak.to_numpy(vec_H_res.pt),
                "HT": ak.to_numpy(ak.sum(single_jets_resolved.pt_regressed, axis=1)),
                "dphi_HZ": ak.to_numpy(np.abs(vec_H_res.delta_phi(dilepton_res))),
                "dr_HZ": ak.to_numpy(vec_H_res.delta_r(dilepton_res)),
                "dm_bb_bb_min": ak.to_numpy(min_dm_bb_bb(vec_single_bjets_resolved, all_jets=vec_single_jets_resolved)),
                
                "puppimet_pt": ak.to_numpy(puppimet_res.pt),
                "n_untag": ak.to_numpy(ak.num(single_untag_jets_resolved)),
                "btag_min": ak.to_numpy(ak.min(single_bjets_resolved.btagUParTAK4B, axis=1)),
                "dphi_untag_Z": ak.to_numpy(np.abs(dphi_proxy_Z)),
                "dr_bb_ave": ak.to_numpy(dr_bb_avg(single_bjets_resolved)),
                "pt_ll": ak.to_numpy(dilepton_res.pt),
                "mass_ll": ak.to_numpy(dilepton_res.mass),
                "weight": ak.to_numpy(weights_res),
            }

            if self.is_MVA:
                self.compat_tree_variables(bdt_resolved)
                self.add_tree_entry("resolved", bdt_resolved)

            ### MODEL EVAL resolved###
        
            if self.run_eval and not self.is_MVA:
                inputs_res = {
                    key: np.asarray(bdt_resolved[key], dtype=np.float64)
                    for key in self.bdt_eval_res.var_list
                }
                bdt_score_res = self.bdt_eval_res.eval(inputs_res)
                output["bdt_score_resolved"].fill(
                    bdt=np.ravel(bdt_score_res),
                    weight=np.ravel(np.asarray(weights_res, dtype=np.float64))
                )

            if self.is_MVA:
                output["trees"] = self._trees
                for regime, trees in self._trees.items():
                    print(f"[DEBUG] Regime '{regime}' has {len(trees)} entries")
        return output 
    def postprocess(self, accumulator):
      
        return accumulator
