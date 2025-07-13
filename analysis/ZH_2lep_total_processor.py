import numpy as np
import awkward as ak
import os
import json
import argparse
import vector
vector.register_awkward()
from coffea import processor
from coffea.analysis_tools import Weights
import hist
from hist import Hist
import coffea.util
from boost_histogram import storage
from utils.matching import (
    delta_r,
    clean_by_dr,
    extract_gen_bb_pairs,
    is_jet_matched_to_bquark_pair,
    match_jets_to_single_qg
)
from utils.jet_tight_id import compute_jet_id

def min_dm_bb_bb(bjets):
    
    bb_pairs = ak.combinations(bjets, 2, axis=1)
    bb_masses = (bb_pairs[:, :, 0] + bb_pairs[:, :, 1]).mass
    bb_combos = ak.combinations(bb_masses, 2, axis=1)
    dm = np.abs(bb_combos[:, :, 0] - bb_combos[:, :, 1])
    return np.min(dm, axis=1)

class TOTAL_Processor(processor.ProcessorABC):
    def __init__(self, xsec=0.89, lumi=41.5e3, nevts=3000, dataset_name=None):
        self.xsec = xsec
        self.lumi = lumi
        self.nevts = nevts
        self.dataset_name = dataset_name
        #self.trees = {r: [] for r in ["boost", "merged", "resolved"]}
        #self.trees_weights = {r: [] for r in ["boost", "merged", "resolved"]}
        #self.trees_labels = {r: [] for r in ["boost", "merged", "resolved"]}

        self._histograms = {}

        self._histograms["cutflow_2l"] = hist.Hist(
            hist.axis.StrCategory(["raw", "2lep", "Mll", "boosted", "merged", "resolved"], name="cut"),
            storage=storage.Double()
        )
        self._histograms["n_doubleb_merged"] = hist.Hist(hist.axis.Integer(0, 6, name="n", label="N double-b jets (merged)"), storage=storage.Double())
        self._histograms["n_singleb_merged"] = hist.Hist(hist.axis.Integer(0, 6, name="n", label="N single-b jets (merged)"), storage=storage.Double())

        for suffix in ["_boosted", "_merged", "_resolved"]:
            self._histograms[f"mass_H{suffix}"] = (
                Hist.new.Reg(80, 0, 400, name="m", label=f"Higgs mass {suffix}").Weight()
            )
            self._histograms[f"pt_H{suffix}"] = (
                Hist.new.Reg(100, 0, 1000, name="pt", label=f"Higgs pt {suffix}").Weight()
            )
            self._histograms[f"pt_bb{suffix}"] = (
                Hist.new.Reg(100, 0, 1000, name="pt", label=f"A pt {suffix}").Weight()
            )
            self._histograms[f"HT{suffix}"] = (
                Hist.new.Reg(100, 0, 1000, name="ht", label=f"HT (scalar sum of pT of double jets) {suffix}").Weight()
            )
            self._histograms[f"puppimet_pt{suffix}"] = (
                Hist.new.Reg(100, 0, 1000, name="met", label=f"met pt {suffix}").Weight()
            )
            self._histograms[f"pt_ll{suffix}"] = Hist.new.Reg(100, 0, 1000, name="ptll", label=f"Dilepton pt {suffix}").Weight()
            self._histograms[f"mass_ll{suffix}"] = Hist.new.Reg(80, 0, 200, name="mll", label=f"Dilepton mass {suffix}").Weight()
            self._histograms[f"pt_ratio{suffix}"] = Hist.new.Reg(50, 0, 5, name="ratio", label=f"pt(H)/pt(Z) {suffix}").Weight()
            self._histograms[f"dr_HZ{suffix}"] = Hist.new.Reg(60, 0, 6, name="dr", label=f"ΔR(H, Z) {suffix}").Weight()
            #self._histograms[f"dphi_HZ{suffix}"] = Hist.new.Reg(64, 0, np.pi, name="dphi", label=f"Δφ(H, Z) {suffix}").Weight()
            self._histograms[f"dm_bb_bb_min{suffix}"] = (
               Hist.new.Reg(100, 0, 200, name="dm", label=f"|ΔM(bb, bb)|_min {suffix}").Weight()
            )
            

    @property
    def histograms(self):
        return self._histograms

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

        weight_array = np.ones(n) * (self.lumi * self.xsec / self.nevts)
        weights = Weights(n)
        weights.add("norm", weight_array)
        output = {key: hist if not hasattr(hist, "copy") else hist.copy() for key, hist in self._histograms.items()}
        
        output["cutflow_2l"].fill(cut="raw", weight=np.sum(weights.weight()))
        
        #object configuration
        muons = events.Muon[(events.Muon.pt > 10) & (np.abs(events.Muon.eta) < 2.5) & events.Muon.tightId & (events.Muon.pfRelIso03_all < 0.15)]
        electrons = events.Electron[(events.Electron.pt > 15) & (np.abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased >= 4)]

        leptons = ak.concatenate([muons, electrons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
        n_leptons = ak.num(leptons)
       
        # Decide whether to compute Jet ID (XROOTD) or use existing branches (from skimmed file on eos)
        if self.dataset_name and self.dataset_name.startswith("eos"):
            # Use already-stored tight ID from skimmed data
            tight_id = events.Jet.passJetIdTight
            tight_lep_veto = events.Jet.passJetIdTightLepVeto
        else:
            # Compute on the fly
            tight_id, tight_lep_veto = compute_jet_id(events.Jet)
        #single jets
        single_jets =events.Jet[(events.Jet.pt > 20) & (np.abs(events.Jet.eta) < 2.5) &  tight_id & tight_lep_veto ]
        #cc single jets with leptons.
        single_jets = clean_by_dr(single_jets, leptons, 0.4)
        single_jets = single_jets[ak.argsort(single_jets.pt, axis=-1, ascending=False)] 
        #single b jets
        single_bjets= single_jets[single_jets.btagDeepFlavB > 0.3]
        #sort single bjets by b tag score
        single_bjets = single_bjets[ak.argsort(single_bjets.btagDeepFlavB, ascending=False)]
        n_single_bjets=ak.num(single_bjets)
        #double jets
        double_jets =events.Jet[(events.Jet.pt > 20) & (np.abs(events.Jet.eta) < 2.5) &  tight_id & tight_lep_veto &
                                (events.Jet.svIdx1 != -1) & (events.Jet.svIdx2 != -1) &
                                (events.Jet.svIdx1 != events.Jet.svIdx2)        
                            ]
        #cc double jets with leptons
        double_jets = clean_by_dr(double_jets, leptons, 0.4)
        double_jets = double_jets[ak.argsort(double_jets.pt, axis=-1, ascending=False)] 
        double_bjets= double_jets[double_jets.btagUParTAK4probbb > 0.38]
        #double bjets
        double_bjets = double_bjets[ak.argsort(double_bjets.btagUParTAK4probbb, ascending=False)]
        #cc doubleb jets with single b jets->keep the cross cleaned single b jets
        single_bjets_cc = clean_by_dr(single_bjets,double_bjets, 0.4)
        single_bjets_cc = single_bjets_cc[ak.argsort(single_bjets_cc.btagDeepFlavB, ascending=False)]
        n_single_bjets_cc=ak.num(single_bjets_cc)
        n_double_bjets=ak.num(double_bjets)
        
        

        # 2 lepton selection
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
            double_bjets_boosted = double_bjets[full_mask_double]
            double_jets_boosted = double_jets[full_mask_double]
            n_double_bjets_boosted = n_double_bjets[full_mask_double]
            weights_boosted= weights.weight()[full_mask_double]
            ht_boosted = ak.sum(double_jets_boosted.pt, axis=1)
            output["HT_boosted"].fill(ht=ht_boosted, weight=weights_boosted)
            lead_bb = double_bjets_boosted[:, 0]
            sublead_bb = double_bjets_boosted[:, 1]  
            higgs_boost = lead_bb + sublead_bb  
            output["mass_H_boosted"].fill(m=higgs_boost.mass, weight=weights_boosted)
            output["pt_H_boosted"].fill(pt=higgs_boost.pt, weight=weights_boosted)
            lep_boosted = leptons[full_mask_double]
            dilepton_boosted = lep_boosted[:, 0] + lep_boosted[:, 1]

            output["pt_ll_boosted"].fill(ptll=dilepton_boosted.pt, weight=weights_boosted)
            output["mass_ll_boosted"].fill(mll=dilepton_boosted.mass, weight=weights_boosted)

            pt_ratio = higgs_boost.pt / dilepton_boosted.pt
            output["pt_ratio_boosted"].fill(ratio=pt_ratio, weight=weights_boosted)

            dr = higgs_boost.delta_r(dilepton_boosted)
            output["dr_HZ_boosted"].fill(dr=dr, weight=weights_boosted)

            dm_boosted = np.abs(lead_bb.mass - sublead_bb.mass)
            output["dm_bb_bb_min_boosted"].fill(dm=dm_boosted, weight=weights_boosted)
            #resolved
            # For resolved regime
            full_mask_res = full_mask_mll & (n_single_bjets >= 3)
            output["cutflow_2l"].fill(cut="resolved", weight=np.sum(weights.weight()[full_mask_res]))
            
            # Filtered objects
            single_bjets_resolved = single_bjets[full_mask_res]
            single_jets_resolved = single_jets[full_mask_res]
            #dilepton_resolved = lep[full_mask_res][:, 0] + lep[full_mask_res][:, 1]
            puppimet_resolved = events.PuppiMET.pt[full_mask_res]
            weights_res = weights.weight()[full_mask_res]
            # Use 3 or 4 leading b-tagged jets per event
            jets_for_H = ak.where(
                ak.num(single_bjets_resolved, axis=1) >= 4,
                single_bjets_resolved[:, :4],
                single_bjets_resolved
            )
            
            # Build proper Lorentz vectors per jet
            vecs = vector.awkward.zip({
                "pt": jets_for_H.pt,
                "eta": jets_for_H.eta,
                "phi": jets_for_H.phi,
                "mass": jets_for_H.mass,
            }, with_name="Momentum4D")

            # Sum per event (axis=1) to get Higgs candidate
            vec_H = ak.sum(vecs, axis=1)
            
            # Fill histograms
            output["mass_H_resolved"].fill(m=ak.to_numpy(vec_H.mass), weight=weights_res)
            output["pt_H_resolved"].fill(pt=ak.to_numpy(vec_H.pt), weight=weights_res)
            
            output["HT_resolved"].fill(ht=ak.sum(single_jets_resolved.pt, axis=1), weight=weights_res)
            output["puppimet_pt_resolved"].fill(met=puppimet_resolved, weight=weights_res)
           
            #merged
            full_mask_merged = full_mask_mll & ((n_double_bjets >= 1) & (n_single_bjets_cc >= 1))

            #full_mask_merged = full_mask_mll & (n_double_bjets>=1 & n_single_bjets_cc >= 1)
            output["cutflow_2l"].fill(cut="merged", weight=np.sum(weights.weight()[full_mask_merged]))
            double_bjets_merged = double_bjets[full_mask_merged]
            single_bjets_merged = single_bjets_cc[full_mask_merged]

            double_jets_merged = double_jets[full_mask_merged]
            single_jets_mer = single_jets[full_mask_merged]
            single_jets_merged = clean_by_dr(single_jets_mer,double_jets_merged, 0.4)
            weights_merged = weights.weight()[full_mask_merged]
            output["n_doubleb_merged"].fill(n=ak.num(double_bjets_merged), weight=weights_merged)
            output["n_singleb_merged"].fill(n=ak.num(single_bjets_merged), weight=weights_merged)
            higgs_candidates = []

            for db, sb in zip(double_bjets_merged, single_bjets_merged):
                if len(db) >= 1 and len(sb) >= 1:
                    jets_to_combine = ak.concatenate([db[:1], sb[:2]], axis=0)
                    
                    vecs = ak.zip({
                        "pt": jets_to_combine.pt,
                        "eta": jets_to_combine.eta,
                        "phi": jets_to_combine.phi,
                        "mass": jets_to_combine.mass,
                    }, with_name="Momentum4D")
                    
                    higgs_candidates.append(ak.sum(vecs, axis=0))
                else:
                    higgs_candidates.append(None)

            higgs_candidates = ak.Array(higgs_candidates)
            
            valid_mask = ~ak.is_none(higgs_candidates)
           

            full_mask_valid_merged = np.zeros(len(events), dtype=bool)
            full_mask_valid_merged[full_mask_merged] = ak.to_numpy(valid_mask)
            
            weights_valid = weights.weight()[full_mask_valid_merged]
            higgs_merged = higgs_candidates[valid_mask]
            
            puppimet_merged = events.PuppiMET.pt[full_mask_valid_merged]
            lep_merged = leptons[full_mask_valid_merged]
            dilepton_merged = lep_merged[:, 0] + lep_merged[:, 1]

            output["puppimet_pt_merged"].fill(met=puppimet_merged, weight=weights_valid)
            output["pt_ll_merged"].fill(ptll=dilepton_merged.pt, weight=weights_valid)
            output["mass_ll_merged"].fill(mll=dilepton_merged.mass, weight=weights_valid)
            output["mass_H_merged"].fill(m=ak.to_numpy(higgs_merged.mass), weight=weights_valid)
            output["pt_H_merged"].fill(pt=ak.to_numpy(higgs_merged.pt), weight=weights_valid)
       
            
        return output

    def postprocess(self, accumulator):
        return accumulator
