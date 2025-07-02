import numpy as np
import awkward as ak
import vector
from coffea import processor
from coffea.analysis_tools import Weights
import hist
from hist import Hist
from collections import defaultdict
from boost_histogram import storage
from utils.matching import (
    clean_by_dr,
    extract_gen_bb_pairs,
    is_jet_matched_to_bquark_pair,
    match_jets_to_single_qg
)
from utils.jet_tight_id import compute_jet_id

class AK4_reg_Processor(processor.ProcessorABC):
    def __init__(self, xsec=1.0, lumi=41.5e3, nevts=1.0, dataset_name=None):
        self.xsec = xsec
        self.lumi = lumi
        self.nevts = nevts
        self.dataset_name = dataset_name
        self.roc_scores = {r: [] for r in ["tot", "lead", "sub", "tot_matched", "lead_matched", "sub_matched"]}
        self.roc_weights = {r: [] for r in ["tot", "lead", "sub", "tot_matched", "lead_matched", "sub_matched"]}
        self.roc_labels = {r: [] for r in ["tot", "lead", "sub", "tot_matched", "lead_matched", "sub_matched"]}
        self._histograms = {} 
    
        def add_upt_hists(base):
            self._histograms[f"{base}"] = Hist.new.Reg(50, 0, 1, name="upart", label="UParT Score").Weight()
            self._histograms[f"{base}_pt"] = Hist.new.Reg(100, 0, 1000, name="pt", label="Jet pT [GeV]").Weight()
            self._histograms[f"{base}_eta"] = Hist.new.Reg(25, -2.5, 2.5, name="eta", label="Jet η").Weight()
            self._histograms[f"{base}_2d"] = (
                Hist.new.Reg(50, 0, 500, name="pt", label="Jet pT [GeV]")
                .Reg(50, 0, 1, name="upart", label="UParT Score")
                .Weight()
            )
            self._histograms[f"{base}_2d_eta"] = (
                Hist.new.Reg(25, -2.5, 2.5, name="eta", label="Jet η")
                .Reg(50, 0, 1, name="upart", label="UParT Score")
                .Weight()
            )

        for suffix in ["", "_matched", "_reg", "_reg_matched"]:
            for region in ["lead", "sub", "tot"]:
                add_upt_hists(f"upart_{region}{suffix}")

        for suffix in ["", "_reg"]:
            #self._histograms[f"event_pass_wp{suffix}"] = Hist.new.Reg(50, 0.0, 1.0, name="wp", label="UParT WP").Weight()
            self._histograms[f"mass_H{suffix}"] = Hist.new.Reg(80, 0, 400, name="m", label=f"Higgs mass {suffix}").Weight()
            self._histograms[f"pt_H{suffix}"] = Hist.new.Reg(100, 0, 1000, name="pt", label=f"Higgs pt {suffix}").Weight()
            self._histograms[f"mass_jj{suffix}"] = Hist.new.Reg(80, 0, 400, name="m", label=f"JJ mass {suffix}").Weight()
            self._histograms[f"pt_jj{suffix}"] = Hist.new.Reg(100, 0, 1000, name="pt", label=f"JJ  pt {suffix}").Weight()
        self._histograms["cutflow_2l"] = hist.Hist(
            hist.axis.StrCategory(["raw", "2lep", "Mll", "2AK4_std", "2AK4_reg", "2AK4_std_mat", "2AK4_reg_mat"], name="cut"),
            storage=storage.Double()
        )
        self._histograms["dr_bb"] = Hist.new.Reg(50, 0, 5, name="dr", label="ΔR(bb)").Weight()
        self._histograms["pt_bb"] = Hist.new.Reg(100, 0, 1000, name="pt", label="pT(bb) [GeV]").Weight()
        self._histograms["pt_bb_04"] = Hist.new.Reg(100, 0, 1000, name="pt", label="pT(bb) ΔR<0.4 [GeV]").Weight()
        self._histograms["dR_vs_pt_gen"] = (
            Hist.new.Reg(50, 0, 500, name="pt", label="pT(bb) [GeV]")
            .Reg(50, 0, 5, name="dr", label="ΔR(bb)")
            .Weight()
        )
      
        self._histograms["mass_H_gen"] = Hist.new.Reg(70, 0, 350, name="mass", label="Gen Higgs mass [GeV]").Weight()
        self._histograms["pt_H_gen"] = Hist.new.Reg(100, 0, 1000, name="pt", label="Gen Higgs PT [GeV]").Weight()
        self._histograms["pt_std_vs_reg"] = (
            Hist.new
            .Reg(25, 0, 500, name="pt_std", label="Jet pT (standard) [GeV]")
            .Reg(25, 0, 500, name="pt_reg", label="Jet pT (regressed) [GeV]")
            .Weight()
        )
        
        self._histograms["event_pass_wp_20"] = Hist.new.Reg(20, 0.0, 1.0, name="wp", label="UParT WP").Weight()
        self._histograms["event_pass_wp_25"] = Hist.new.Reg(25, 0.0, 1.0, name="wp", label="UParT WP").Weight()
        self._histograms["event_pass_wp_50"] = Hist.new.Reg(50, 0.0, 1.0, name="wp", label="UParT WP").Weight()
        self._histograms["event_pass_wp_100"] = Hist.new.Reg(100, 0.0, 1.0, name="wp", label="UParT WP").Weight()
        self._histograms[f"event_pass_wp_matched"] = Hist.new.Reg(50, 0.0, 1.0, name="wp", label="UParT WP").Weight()
    @property
    def histograms(self):
        return self._histograms

    def process(self, events):
        try:
            events = events.compute()
        except Exception:
            pass

        n = len(events)
        original_weights = Weights(n)
        original_weights.add("norm", np.ones(n) * (self.lumi * self.xsec / self.nevts))
        output = {k: h.copy() for k, h in self._histograms.items()}
        
        output["cutflow_2l"].fill(cut="raw", weight=np.sum(original_weights.weight()))
        muons = events.Muon[(events.Muon.pt > 10) & (abs(events.Muon.eta) < 2.5) & events.Muon.tightId & (events.Muon.pfRelIso03_all < 0.15)]
        electrons = events.Electron[(events.Electron.pt > 15) & (abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased >= 4)]
        leptons = ak.concatenate([muons, electrons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]

        mask_2l = ak.num(leptons) >= 2
        events = events[mask_2l]
        leptons = leptons[mask_2l]
        weights=original_weights.weight()[mask_2l]
        output["cutflow_2l"].fill(cut="2lep", weight=np.sum(original_weights.weight()[mask_2l]))
        lead = leptons[:, 0]
        sub= leptons[:, 1]

        pt_sel = (((abs(lead.pdgId) == 11) & (lead.pt > 25) & (sub.pt > 15)) |
                  ((abs(lead.pdgId) == 13) & (lead.pt > 15) & (sub.pt > 10)))
        z_window = ((lead + sub).mass > 80) & ((lead + sub).mass < 100)
        lep_mask = pt_sel & (lead.charge != sub.charge) & (abs(lead.pdgId) == abs(sub.pdgId)) & z_window
        events = events[lep_mask]
        leptons = leptons[lep_mask]
        weights=original_weights.weight()[mask_2l][lep_mask]
        output["cutflow_2l"].fill(cut="Mll", weight=np.sum(original_weights.weight()[mask_2l][lep_mask]))
        tight_id, tight_lep_veto = compute_jet_id(events.Jet)
        jets = events.Jet
        
        jets = jets[(jets.pt > 20) & (abs(jets.eta) < 2.5) & tight_id & tight_lep_veto &  (jets.svIdx1 != -1) & (jets.svIdx2 != -1) ]
        raw_pt = jets.pt * (1 - jets.rawFactor)
        pt_reg = raw_pt * jets.PNetRegPtRawCorr * jets.PNetRegPtRawCorrNeutrino

        jets = ak.with_field(jets, pt_reg, "pt_regressed")
        # Flatten for histogram filling
        pt_std_flat = ak.flatten(jets.pt)
        pt_reg_flat = ak.flatten(jets.pt_regressed)
        weights_flat = np.repeat(weights, ak.num(jets))

        output["pt_std_vs_reg"].fill(
            pt_std=pt_std_flat,
            pt_reg=pt_reg_flat,
            weight=weights_flat
        )
     
        jets= clean_by_dr(jets, leptons, 0.4)
        
        jets_reg = jets[ak.argsort(jets.pt , axis=-1, ascending=False)]
        mask_reg = ak.num(jets_reg) >= 2

        events_reg = events[mask_reg]
        jets_reg = jets_reg[mask_reg]
        
        weights_reg = original_weights.weight()[mask_2l][lep_mask][mask_reg]
        output["cutflow_2l"].fill(cut="2AK4_reg", weight=np.sum(weights_reg))
        is_signal = any(x in self.dataset_name for x in ["ZH", "WH"])
        def fill_jets(jets, pt_arr, events, w_evt, suffix, do_match):
            if do_match:
                bb_pairs, _ = extract_gen_bb_pairs(events)
            else:
                matched_mask = match_jets_to_single_qg(jets, events.GenPart, dr_threshold=0.4)

            for iev in range(len(jets)):
                jets_evt = jets[iev]
                weights_evt = w_evt[iev]
                pt_evt = pt_arr[iev]

                if do_match:
                    vec_genbb1, vec_genbb2 = bb_pairs[iev]
                matched_jets = []
                for ij, j in enumerate(jets_evt):
                    pt = pt_evt[ij]
                    eta = j.eta
                    score = j.btagUParTAK4probbb
                    vec_j = vector.obj(pt=pt, eta=eta, phi=j.phi, mass=j.mass)
                    # Regular histograms (not matched)
                    for r in ["tot", "lead", "sub"]:
                        if (r == "lead" and ij != 0) or (r == "sub" and ij != 1):
                            continue
                        base = f"upart_{r}{suffix}"
                        output[base].fill(upart=score, weight=weights_evt)
                        output[f"{base}_pt"].fill(pt=pt, weight=weights_evt)
                        output[f"{base}_eta"].fill(eta=eta, weight=weights_evt)
                        output[f"{base}_2d"].fill(pt=pt, upart=score, weight=weights_evt)
                        output[f"{base}_2d_eta"].fill(eta=eta, upart=score, weight=weights_evt)
                        
                    # ROC data collection (non-matched)
                    if suffix == "":
                        for r in ["tot", "lead", "sub"]:
                            if (r == "lead" and ij != 0) or (r == "sub" and ij != 1):
                                continue
                            self.roc_scores[r].append(score)
                            self.roc_weights[r].append(weights_evt)
                            self.roc_labels[r].append(1 if is_signal else 0)
                        # Matched logic
                    matched = False
                    if do_match:
                        vec_j = vector.obj(pt=pt, eta=eta, phi=j.phi, mass=j.mass)
                        matched = (
                        is_jet_matched_to_bquark_pair(vec_genbb1, vec_j, 0.4) or
                        is_jet_matched_to_bquark_pair(vec_genbb2, vec_j, 0.4)
                        )
                    else:
                        matched = matched_mask[iev][ij]

                    if matched:
                        matched_jets.append(vec_j)
                        
                        # ROC data collection (matched jets only)
                        if suffix == "":
                            for r in ["tot", "lead", "sub"]:
                                if (r == "lead" and ij != 0) or (r == "sub" and ij != 1):
                                    continue
                                roc_key = f"{r}_matched"
                                self.roc_scores[roc_key].append(score)
                                self.roc_weights[roc_key].append(weights_evt)
                                self.roc_labels[roc_key].append(1 if is_signal else 0)
                        matched_suffix = "_matched" if not suffix else f"{suffix}_matched"
                        for r in ["tot", "lead", "sub"]:
                            if (r == "lead" and ij != 0) or (r == "sub" and ij != 1):
                                continue
                            base = f"upart_{r}{matched_suffix}"
                            output[base].fill(upart=score, weight=weights_evt)
                            output[f"{base}_pt"].fill(pt=pt, weight=weights_evt)
                            output[f"{base}_eta"].fill(eta=eta, weight=weights_evt)
                            output[f"{base}_2d"].fill(pt=pt, upart=score, weight=weights_evt)
                            output[f"{base}_2d_eta"].fill(eta=eta, upart=score, weight=weights_evt)
                            
                # dijet  reconstruction
                if do_match and len(jets_evt) >= 2:
                    j1 = vector.obj(pt=pt_evt[0], eta=jets_evt[0].eta, phi=jets_evt[0].phi, mass=jets_evt[0].mass)
                    j2 = vector.obj(pt=pt_evt[1], eta=jets_evt[1].eta, phi=jets_evt[1].phi, mass=jets_evt[1].mass)
                    jj = j1 + j2
                    output[f"mass_jj{suffix}"].fill(m=jj.mass, weight=weights_evt)
                    output[f"pt_jj{suffix}"].fill(pt=jj.pt, weight=weights_evt)
                # Higgs reconstruction from matched jets
                if do_match and len(matched_jets) >= 2:
                    # Optionally: sort by pt to ensure consistent ordering
                    #matched_jets = sorted(matched_jets, key=lambda v: v.pt, reverse=True)
                    h = matched_jets[0] + matched_jets[1]
                    output[f"mass_H{suffix}"].fill(m=h.mass, weight=weights_evt)
                    output[f"pt_H{suffix}"].fill(pt=h.pt, weight=weights_evt)
                # INSIDE event loop — matched jets scan
                wps = np.linspace(0.0, 1.0, 50)
                matched_scores = []

                for ij, j in enumerate(jets_evt):
                    pt = pt_evt[ij]
                    eta = j.eta
                    score = j.btagUParTAK4probbb

                    if do_match:
                        vec_j = vector.obj(pt=pt, eta=eta, phi=j.phi, mass=j.mass)
                        matched = (
                            is_jet_matched_to_bquark_pair(vec_genbb1, vec_j, 0.4) or
                            is_jet_matched_to_bquark_pair(vec_genbb2, vec_j, 0.4)
                        )
                    else:
                        matched = matched_mask[iev][ij]

                    if matched:
                        matched_scores.append(score)

                    for wp in wps:
                        if sum(s > wp for s in matched_scores) >= 2:
                            output[f"event_pass_wp_matched"].fill(
                                wp=wp,
                                weight=weights_evt
                            )
            # WP scan
            
            for nbins, name in zip([20, 25, 50, 100], ["20", "25", "50", "100"]):
                WPs = np.linspace(0.0, 1.0, nbins)
                scores = ak.fill_none(jets.btagUParTAK4probbb, 0.0)
                for wp in WPs:
                    jets_wp = jets[scores > wp]
                    mask = ak.num(jets_wp) >= 2
                    if ak.any(mask):
                        output[f"event_pass_wp_{name}"].fill(
                            wp=np.full(ak.sum(mask), wp),
                            weight=w_evt[mask]
                        )
        fill_jets(jets_reg, jets_reg.pt, events_reg, weights_reg, "", is_signal)
        fill_jets(jets_reg, jets_reg.pt_regressed, events_reg, weights_reg, "_reg", is_signal)
        events_final = events_reg
        weights_final = original_weights.weight()[mask_2l][lep_mask][mask_reg]
        if is_signal:
            bb_pairs, n_bquark_list = extract_gen_bb_pairs(events_final)

            for iev in range(len(events_final)):
                weight = weights_final[iev]
                vec_genbb1, vec_genbb2 = bb_pairs[iev]
                n_bquarks = n_bquark_list[iev]

                def fill_pair(pair):
                    if len(pair) == 2:
                        vec1, vec2 = pair[0], pair[1]
                        pt_bb = (vec1 + vec2).pt
                        dr_bb = vec1.deltaR(vec2)
                        output["pt_bb"].fill(pt=pt_bb)
                        output["dr_bb"].fill(dr=dr_bb)
                        output["dR_vs_pt_gen"].fill(pt=pt_bb, dr=dr_bb)
                        if dr_bb < 0.4:
                            output["pt_bb_04"].fill(pt=pt_bb)

                fill_pair(vec_genbb1)
                fill_pair(vec_genbb2)

                # Count bb pair presence
                has_bb1 = len(vec_genbb1) == 2
                has_bb2 = len(vec_genbb2) == 2

                # Reconstruct H from 4 b-quarks
                if has_bb1 and has_bb2:
                    vec_H = vec_genbb1[0] + vec_genbb1[1] + vec_genbb2[0] + vec_genbb2[1]
                    output["mass_H_gen"].fill(mass=vec_H.mass)
                    output["pt_H_gen"].fill(pt=vec_H.pt)

        output["roc_data"] = {}
        for r in ["tot", "lead", "sub", "tot_matched", "lead_matched", "sub_matched"]:
            output["roc_data"][r] = {
                "score": np.array(self.roc_scores[r]),
                "weight": np.array(self.roc_weights[r]),
                "label": np.array(self.roc_labels[r]),
            }
        return output

    def postprocess(self, accumulator):
        return accumulator
