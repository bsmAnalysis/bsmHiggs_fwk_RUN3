import numpy as np
import awkward as ak
import os
import json
import argparse
import vector
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

class AK4_boost_Processor(processor.ProcessorABC):
    def __init__(self, xsec=0.89, lumi=41.5e3, nevts=3000, dataset_name=None):
        self.xsec = xsec
        self.lumi = lumi
        self.nevts = nevts
        self.dataset_name=dataset_name
        self._histograms = {
            
            "cutflow_2l": hist.Hist(
                hist.axis.StrCategory(["raw", "2lep", "Mll", "AK4_untag", "2_matched_jets"], name="cut"),
                storage=storage.Double()
            ),
            
            "upart_lead": Hist.new
            .Reg(100, 0., 1., name="upart", label="UParT bb score (leading jet)")
            .Weight(),

            "upart_lead_matched": Hist.new
            .Reg(100, 0., 1., name="upart", label="UParT bb score (matched leading jet)")
            .Weight(),

            "upart_sub": Hist.new
            .Reg(100, 0., 1., name="upart", label="UParT bb score (subleading jet)")
            .Weight(),

            "upart_sub_matched": Hist.new
            .Reg(100, 0., 1., name="upart", label="UParT bb score (matched subleading jet)")
            .Weight(),
            
            "upart_tot": Hist.new
            .Reg(100, 0., 1., name="upart", label="UParT bb score (all selected jets)")
            .Weight(),
            
            "upart_tot_matched": Hist.new
            .Reg(100, 0., 1., name="upart", label="UParT bb score (matched selected jets)")
            .Weight(),
            
            "pt_matched_ak4_lead": hist.Hist(hist.axis.Regular(100, 0, 1000, name="pt", label="lead Matched AK4 Jet pT [GeV]")),
            "pt_ak4_lead": hist.Hist(hist.axis.Regular(100, 0, 1000, name="pt", label="lead AK4 Jet pT [GeV]")),
            "pt_matched_ak4_tot": hist.Hist(hist.axis.Regular(100, 0, 1000, name="pt", label="tot Matched AK4 Jet pT [GeV]")),
            "pt_ak4_tot": hist.Hist(hist.axis.Regular(100, 0, 1000, name="pt", label="tot  AK4 Jet pT [GeV]")),
            "eta_matched_ak4_tot": hist.Hist(hist.axis.Regular(50, -2.5, 2.5, name="eta", label="tot Matched AK4 Jet eta")),
            "eta_ak4_tot": hist.Hist(hist.axis.Regular(50, -2.5, 2.5, name="eta", label="tot  AK4 Jet eta")),
            "dr_bb": hist.Hist(hist.axis.Regular(50, 0, 5, name="dr", label="ΔR(bb)")),
            "pt_bb": hist.Hist(hist.axis.Regular(100, 0, 1000, name="pt", label="pT(bb) [GeV]")),
            "pt_bb_04": hist.Hist(hist.axis.Regular(100, 0, 1000, name="pt", label="pT(bb)(dR<0.4) [GeV]")),
            "bb_pair_count": hist.Hist(hist.axis.Integer(0, 2, name="n", label="bb pair count category")),
            "bquark_multiplicity": hist.Hist(hist.axis.Integer(0, 6, name="n", label="Number of b quarks from A bosons ")),

            "dR_vs_pt_gen": Hist.new
            .Reg(50, 0, 500, name="pt", label="pT(bb) [GeV]")
            .Reg(50, 0, 5, name="dr", label="ΔR(bb)")
            .Weight(),
            
            "upart_bb_score": Hist.new
                .Reg(100, 0., 1., name="upart_bb", label="upart_bb_score")
                .Weight(),
            
            "event_pass_wp": Hist.new
            .Reg(101, 0., 1., name="upart_wp", label="uParT Working Point")
            .Weight(),
            
            "upart_vs_pt_matched": Hist.new
            .Reg(50, 0, 500, name="pt", label="Jet pT [GeV]")
            .Reg(50, 0, 1, name="upart", label="UParT bb Score")
            .Weight(),

            "upart_vs_eta_matched": Hist.new
            .Reg(50, -2.5, 2.5, name="eta", label="Jet eta ")
            .Reg(50, 0, 1, name="upart", label="UParT bb Score")
            .Weight(),

            "upart_vs_pt_lead_m":Hist.new
            .Reg(50, 0, 500, name="pt", label="Jet pT [GeV]")
            .Reg(50, 0, 1, name="upart", label="UParT bb Score")
            .Weight(),

            "upart_vs_pt_lead":Hist.new
            .Reg(50, 0, 500, name="pt", label="Jet pT [GeV]")
            .Reg(50, 0, 1, name="upart", label="UParT bb Score")
            .Weight(),

            "upart_vs_pt_sub_m":Hist.new
            .Reg(50, 0, 500, name="pt", label="Jet pT [GeV]")
            .Reg(50, 0, 1, name="upart", label="UParT bb Score")
            .Weight(),

            "upart_vs_pt_sub":Hist.new
            .Reg(50, 0, 500, name="pt", label="Jet pT [GeV]")
            .Reg(50, 0, 1, name="upart", label="UParT bb Score")
            .Weight(),
        }

        pt_bins = [(20, 50), (50, 100), (100, 1000)]
        self.pt_bins = pt_bins
        for ptmin, ptmax in pt_bins:
            self._histograms[f"upart_{ptmin}_{ptmax}"] = (
                Hist.new.Reg(100, 0, 1, name="upart", label="UParT Score")
                .Weight()
            )
            
            self._histograms[f"upart_mat_{ptmin}_{ptmax}"] = (
                Hist.new.Reg(100, 0, 1, name="upart", label="UParT Score")
                .Weight()
                
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

        double_jets =events.Jet[(events.Jet.pt > 20) & (np.abs(events.Jet.eta) < 2.5) &  tight_id & tight_lep_veto &
                                (events.Jet.svIdx1 != -1) & (events.Jet.svIdx2 != -1)          
                            ]
        double_jets = clean_by_dr(double_jets, leptons, 0.4)
        #double_jets= double_jets[double_jets.btagUParTAK4probbb > 0.38]
        double_jets = double_jets[ak.argsort(double_jets.pt, axis=-1, ascending=False)] 
        n_double_jets=ak.num(double_jets)
        
        

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
           
            full_mask_double = full_mask_mll & (n_double_jets >= 2)
            output["cutflow_2l"].fill(cut="AK4_untag", weight=np.sum(weights.weight()[full_mask_double]))
            selected_double_jets = double_jets[full_mask_double]
            selected_weights = ak.broadcast_arrays(weights.weight()[full_mask_double], selected_double_jets.pt)[0]
            
            
            n_double_jets_selected = n_double_jets[full_mask_double]
            weights_selected = weights.weight()[full_mask_double]
            
            # For signal matching:
            is_signal = any(x in self.dataset_name for x in ["ZH", "WH"])
            if is_signal:
                
                # Apply mask
                events = events[full_mask_double]
                bb_pairs, n_bquark_list = extract_gen_bb_pairs(events)
                for iev in range(len(events)):
                    event_weight = weights.weight()[full_mask_double][iev]
                    #gen level bb pairs
                    vec_genbb1_truth, vec_genbb2_truth = bb_pairs[iev]
                    n_bquarks = n_bquark_list[iev]
                    output["bquark_multiplicity"].fill(n=n_bquarks)
                    
                    if len(vec_genbb1_truth) == 2:
                        output["dr_bb"].fill(dr=vec_genbb1_truth[0].deltaR(vec_genbb1_truth[1]))
                        output["pt_bb"].fill(pt=(vec_genbb1_truth[0] + vec_genbb1_truth[1]).pt)
                        output["dR_vs_pt_gen"].fill(pt=(vec_genbb1_truth[0] + vec_genbb1_truth[1]).pt,dr=vec_genbb1_truth[0].deltaR(vec_genbb1_truth[1]))
                        if vec_genbb1_truth[0].deltaR(vec_genbb1_truth[1])< 0.4:
                            output["pt_bb_04"].fill(pt=(vec_genbb1_truth[0] + vec_genbb1_truth[1]).pt)
                    if len(vec_genbb2_truth) == 2:
                        output["dr_bb"].fill(dr=vec_genbb2_truth[0].deltaR(vec_genbb2_truth[1]))
                        output["pt_bb"].fill(pt=(vec_genbb2_truth[0] + vec_genbb2_truth[1]).pt)
                        output["dR_vs_pt_gen"].fill(pt=(vec_genbb2_truth[0] + vec_genbb2_truth[1]).pt,dr=vec_genbb2_truth[0].deltaR(vec_genbb2_truth[1]))
                        if vec_genbb2_truth[0].deltaR(vec_genbb2_truth[1])< 0.4:
                            output["pt_bb_04"].fill(pt=(vec_genbb2_truth[0] + vec_genbb2_truth[1]).pt)
                    if len(vec_genbb1_truth) == 2 and len(vec_genbb2_truth) == 2:
                        output["bb_pair_count"].fill(n=1)
                    elif len(vec_genbb1_truth) == 2 or len(vec_genbb2_truth) == 2:
                        output["bb_pair_count"].fill(n=0)
                    # move to matching each jet to a bb pair
                    if len(selected_double_jets[iev]) > 0:
                        jet1=selected_double_jets[iev][0]
                        leading_ak4 = vector.obj(
                            pt=jet1["pt"],
                            eta=jet1["eta"],
                            phi=jet1["phi"],
                            mass=jet1["mass"])
                       
                        output["pt_ak4_lead"].fill(pt=leading_ak4.pt, weight=event_weight)
                        output["pt_ak4_tot"].fill(pt=leading_ak4.pt, weight=event_weight)
                        output["eta_ak4_tot"].fill(eta=leading_ak4.eta, weight=event_weight)
                        upart_score1 = jet1.btagUParTAK4probbb
                        output["upart_lead"].fill(upart=upart_score1, weight=event_weight)
                        output["upart_tot"].fill(upart=upart_score1, weight=event_weight)
                        output["upart_vs_pt_lead"].fill(pt=leading_ak4.pt, upart=upart_score1, weight=event_weight)
                        for ptmin, ptmax in self.pt_bins:
                            if ptmin <= leading_ak4.pt < ptmax:
                                output[f"upart_{ptmin}_{ptmax}"].fill(upart=upart_score1, weight=event_weight)
                                break
                        if is_jet_matched_to_bquark_pair(vec_genbb1_truth, leading_ak4, dr_threshold=0.4) or is_jet_matched_to_bquark_pair(vec_genbb2_truth, leading_ak4, dr_threshold=0.4):
                            for ptmin, ptmax in self.pt_bins:
                                if ptmin <= leading_ak4.pt < ptmax:
                                    output[f"upart_mat_{ptmin}_{ptmax}"].fill(upart=upart_score1, weight=event_weight)
                                    break
                            output["eta_matched_ak4_tot"].fill(eta=leading_ak4.eta, weight=event_weight)
                            output["upart_vs_eta_matched"].fill(eta=leading_ak4.eta, upart=upart_score1, weight=event_weight)
                            output["pt_matched_ak4_lead"].fill(pt=leading_ak4.pt, weight=event_weight)
                            output["pt_matched_ak4_tot"].fill(pt=leading_ak4.pt, weight=event_weight)
                            output["upart_lead_matched"].fill(upart=upart_score1, weight=event_weight)
                            output["upart_tot_matched"].fill(upart=upart_score1, weight=event_weight)
                            output["upart_vs_pt_matched"].fill(pt=leading_ak4.pt, upart=upart_score1, weight=event_weight)
                            output["upart_vs_pt_lead_m"].fill(pt=leading_ak4.pt, upart=upart_score1, weight=event_weight)

                    if len(selected_double_jets[iev]) > 1:
                        jet2=selected_double_jets[iev][1]  #  subleading
                        vec_jet = vector.obj(
                            pt=jet2["pt"], eta=jet2["eta"], phi=jet2["phi"], mass=jet2["mass"]
                        )
                        output["pt_ak4_tot"].fill(pt=vec_jet.pt, weight=event_weight)
                        output["eta_ak4_tot"].fill(eta=vec_jet.eta, weight=event_weight)
                        upart_score2 = jet2.btagUParTAK4probbb
                        output["upart_tot"].fill(upart=upart_score2, weight=event_weight)
                        output["upart_sub"].fill(upart=upart_score2, weight=event_weight)
                        output["upart_vs_pt_sub"].fill(pt=vec_jet.pt, upart=upart_score2, weight=event_weight)
                        for ptmin, ptmax in self.pt_bins:
                            if ptmin <= vec_jet.pt < ptmax:
                                output[f"upart_{ptmin}_{ptmax}"].fill(upart=upart_score2, weight=event_weight)
                                break
                        if is_jet_matched_to_bquark_pair(vec_genbb1_truth, vec_jet, dr_threshold=0.4) or is_jet_matched_to_bquark_pair(vec_genbb2_truth, vec_jet, dr_threshold=0.4):
                            for ptmin, ptmax in self.pt_bins:
                                if ptmin <= vec_jet.pt < ptmax:
                                    output[f"upart_mat_{ptmin}_{ptmax}"].fill(upart=upart_score2, weight=event_weight)
                                    break
                            output["eta_matched_ak4_tot"].fill(eta=vec_jet.eta, weight=event_weight)
                            output["upart_vs_eta_matched"].fill(eta=vec_jet.eta, upart=upart_score2, weight=event_weight)
                            output["upart_tot_matched"].fill(upart=upart_score2, weight=event_weight)
                            output["upart_sub_matched"].fill(upart=upart_score2, weight=event_weight)
                            output["pt_matched_ak4_tot"].fill(pt=vec_jet.pt, weight=event_weight)
                            output["upart_vs_pt_sub_m"].fill(pt=vec_jet.pt, upart=upart_score2, weight=event_weight)
                            output["upart_vs_pt_matched"].fill(pt=vec_jet.pt, upart=upart_score2, weight=event_weight)
            #background matching  to single q/g  
            if not is_signal:
                events = events[full_mask_double]                                                                                                 
                selected_double_jets = double_jets[full_mask_double]                                                                              
                weights_selected = weights.weight()[full_mask_double]                                                                             
                matched_mask = match_jets_to_single_qg(selected_double_jets, events.GenPart, dr_threshold=0.4)                      
                matched_jets = selected_double_jets[matched_mask]                                                                                 
                                                                                                                                                  
                # Broadcast weights to jet level                                                                                                  
                selected_weights = ak.broadcast_arrays(weights.weight()[full_mask_double], selected_double_jets.pt)[0]                            
                matched_weights = selected_weights[matched_mask]            
                matched_mask = match_jets_to_single_qg(selected_double_jets, events.GenPart, dr_threshold=0.4)
                
                for iev in range(len(selected_double_jets)):
                    event_weight = weights.weight()[full_mask_double][iev]
                    jets_event = selected_double_jets[iev]
                    matched_mask_event = matched_mask[iev]
                    
                    
                    for ij, jet in enumerate(jets_event):
                        upart_score = jet.btagUParTAK4probbb
                        output["upart_tot"].fill(upart=upart_score, weight=event_weight)
                        output["pt_ak4_tot"].fill(pt=jet.pt, weight=event_weight)
                        output["eta_ak4_tot"].fill(eta=jet.eta, weight=event_weight)
                        for ptmin, ptmax in self.pt_bins:
                            if ptmin <= jet.pt < ptmax:
                                output[f"upart_{ptmin}_{ptmax}"].fill(upart=upart_score, weight=event_weight)
                                if  matched_mask_event[ij]:
                                    output[f"upart_mat_{ptmin}_{ptmax}"].fill(upart=upart_score, weight=event_weight)
                                break
                        if ij == 0:
                            output["upart_lead"].fill(upart=upart_score, weight=event_weight)
                            output["pt_ak4_lead"].fill(pt=jet.pt, weight=event_weight)
                            output["upart_vs_pt_lead"].fill(pt=jet.pt, upart=upart_score, weight=event_weight)
                            
                        if ij == 1:
                            output["upart_sub"].fill(upart=upart_score, weight=event_weight)
                            output["upart_vs_pt_sub"].fill(pt=jet.pt, upart=upart_score, weight=event_weight)
                        if  matched_mask_event[ij]:
                            output["upart_tot_matched"].fill(upart=upart_score, weight=event_weight)
                            output["pt_matched_ak4_tot"].fill(pt=jet.pt, weight=event_weight)
                            output["eta_matched_ak4_tot"].fill(eta=jet.eta, weight=event_weight)
                            output["upart_vs_pt_matched"].fill(pt=jet.pt, upart=upart_score, weight=event_weight)
                            output["upart_vs_eta_matched"].fill(eta=jet.eta, upart=upart_score, weight=event_weight)
                            if ij == 0:
                                output["upart_lead_matched"].fill(upart=upart_score, weight=event_weight)
                                output["pt_matched_ak4_lead"].fill(pt=jet.pt, weight=event_weight)
                                output["upart_vs_pt_lead_m"].fill(pt=jet.pt, upart=upart_score, weight=event_weight)
                            if ij == 1:
                                output["upart_sub_matched"].fill(upart=upart_score, weight=event_weight)
                                output["upart_vs_pt_sub_m"].fill(pt=jet.pt, upart=upart_score, weight=event_weight)
            # event based  WP for optimization
                                
            wps = np.linspace(0.0, 1.0, 101)
            
            for wp in wps:
                double_jets_wp = double_jets[double_jets.btagUParTAK4probbb > wp]
                
                n_double = ak.num(double_jets_wp)
                # Full event selection                                                                                                                                              
                event_mask = full_mask_mll & (n_double >= 2)
                # Only fill passing events                                                                                                                                          
                if ak.any(event_mask):
                    weights_sel = weights.weight()[event_mask]
                    wp_array = np.full(len(weights_sel), wp)

                    output["event_pass_wp"].fill(
                        upart_wp=wp_array,
                         weight=weights_sel
                    )

        return output

    def postprocess(self, accumulator):
        return accumulator
