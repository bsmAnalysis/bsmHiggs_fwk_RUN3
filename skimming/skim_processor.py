import awkward as ak
import numpy as np
import fnmatch
from coffea import processor
import os
import correctionlib

#------------------------------- helpers --------------------------------#

def _unflatten_like(flat, counts):
    return ak.unflatten(ak.Array(flat), counts)

def _mask_lepton_overlap(jets, leptons, dr=0.4):
    if (leptons is None) or (ak.num(leptons, axis=1).layout is None):
        return ak.ones_like(jets.pt, dtype=bool)
    pairs = ak.cartesian({"j": jets, "l": leptons}, axis=1, nested=True)
    dphi = np.arctan2(np.sin(pairs.j.phi - pairs.l.phi),
                      np.cos(pairs.j.phi - pairs.l.phi))
    deta = pairs.j.eta - pairs.l.eta
    dr2  = dphi * dphi + deta * deta
    return ak.all(dr2 > (dr * dr), axis=2)

#----------------------------------------------------------------------------------------------------------------------------#

class NanoAODSkimmer(processor.ProcessorABC):
    def __init__(self, branches_to_keep, trigger_groups, met_filter_flags, dataset_name=None, corrections_dir=None):
        self.branches_to_keep = branches_to_keep
        self.trigger_groups = trigger_groups
        self.met_filter_flags = met_filter_flags
        self.dataset_name = dataset_name
        self.corrections_dir = corrections_dir or os.path.join(os.path.dirname(__file__), "corrections")
        self._jet_veto = None
        self._jet_veto_type = "jetvetomap"
        self._loaded_veto = False
            
#----------------------------------------------------------------------------------------------------------------------------#

    # -- load veto map (2024)-- #
    # pt_raw = pt * (1 − rawFactor); select jets with pT_raw>15, tight ID, (chEmEF+neEmEF)<0.9, ΔR>0.2 from all leptons.
    # If any selected jet falls in a vetoed region, reject the whole event.
    # https://cms-jerc.web.cern.ch/Recommendations/#2024_1
    # Official Run-3 2024 requirement to remove detector “hot/cold”.
    
    def _ensure_veto_loaded(self):
         if self._loaded_veto:
             return
         path = os.path.join(self.corrections_dir, "jetvetomaps.json.gz")
         if os.path.exists(path):
             try:
                 cset = correctionlib.CorrectionSet.from_file(path)
                 picked = None
                 for k in cset:
                     if "Summer24Prompt24_RunBCDEFGHI_V1" in str(k):
                         picked = cset[k]; break
                 if picked is None:
                     for k in cset:
                         try:
                             ins = [i.name for i in cset[k].inputs]
                             if {"type","eta","phi"}.issubset(set(ins)):
                                 picked = cset[k]; break
                         except Exception:
                             pass
                 self._jet_veto = picked
                 if self._jet_veto is None:
                     print("[Skim:JetVeto] No suitable correction found; veto disabled.")
             except Exception as e:
                 print(f"[Skim:JetVeto] Failed to load veto maps: {e}")
         else:
             print("[Skim:JetVeto] jetvetomaps.json.gz not found; veto disabled.")
         self._loaded_veto = True
 
#----------------------------------------------------------------------------------------------------------------------------#    
 
    def select_fields(self, collection, fields):
        if not fields:
            return collection
        all_fields = ak.fields(collection)
        selected_fields = set()
        for field in fields:
            if "*" in field:
                selected_fields.update(fnmatch.filter(all_fields, field))
            elif field in all_fields:
                selected_fields.add(field)
        return collection[list(selected_fields)]
    
#----------------------------------------------------------------------------------------------------------------------------#

    def process(self, events):
        self._ensure_veto_loaded()
        n_before = len(events)
        
        jets_full = events.Jet        
        eta_abs   = np.abs(jets_full.eta)
        chMult    = jets_full.chMultiplicity
        neMult    = jets_full.neMultiplicity
        neHEF     = jets_full.neHEF
        neEmEF    = jets_full.neEmEF
        chHEF     = jets_full.chHEF
        muEF      = getattr(jets_full, "muEF", ak.zeros_like(eta_abs))
        chEmEF    = getattr(jets_full, "chEmEF", ak.zeros_like(eta_abs))
    
        passJetIdTight = (
            ((eta_abs <= 2.6) & (neHEF < 0.99) & (neEmEF < 0.9) & ((chMult + neMult) > 1) & (chHEF > 0.01) & (chMult > 0))
            | ((eta_abs > 2.6) & (eta_abs <= 2.7) & (neHEF < 0.90) & (neEmEF < 0.99))
            | ((eta_abs > 2.7) & (eta_abs <= 3.0) & (neHEF < 0.99))
            | ((eta_abs > 3.0) & (neMult >= 2) & (neEmEF < 0.4))
        )
        passJetIdTightLepVeto = ak.where(
            eta_abs <= 2.7, passJetIdTight & (muEF < 0.8) & (chEmEF < 0.8), passJetIdTight
        )
        jets_full = ak.with_field(jets_full, passJetIdTight, "passJetIdTight")
        jets_full = ak.with_field(jets_full, passJetIdTightLepVeto, "passJetIdTightLepVeto")
    
        rawFactor_full = ak.fill_none(getattr(jets_full, "rawFactor", ak.zeros_like(jets_full.pt)), 0.0)
        pt_raw_full    = jets_full.pt * (1.0 - rawFactor_full)
    
        mask_mu = _mask_lepton_overlap(jets_full, getattr(events, "Muon", None),     dr=0.2)
        mask_el = _mask_lepton_overlap(jets_full, getattr(events, "Electron", None), dr=0.2)
    
        jet_min = (
            (pt_raw_full > 15.0)
            & ak.values_astype(jets_full.passJetIdTight, bool)
            & ((jets_full.chEmEF + jets_full.neEmEF) < 0.9)
            & mask_mu & mask_el
        )


        if self._jet_veto is not None:
            counts    = ak.num(jets_full.pt, axis=1)
            eta_flat  = ak.to_numpy(ak.flatten(jets_full.eta))
            phi_flat  = ak.to_numpy(ak.flatten(jets_full.phi))
            vflat     = self._jet_veto.evaluate(self._jet_veto_type, eta_flat, phi_flat)
            veto_j    = _unflatten_like(vflat, counts) > 0.5
            veto_ev   = ak.any(jet_min & veto_j, axis=1)
            events    = events[~veto_ev]   # events are vetoed here
            jets_full = jets_full[~veto_ev]
                   
            n_after = len(events)
            print(f"[JetVeto] dropped {n_before - n_after} / {n_before} events")

        out = {}
        out["run"] = events.run
        out["event"] = events.event
        out["luminosityBlock"] = events.luminosityBlock
    
        if hasattr(events, "Rho") and hasattr(events.Rho, "fixedGridRhoFastjetAll"):
            out["fixedGridRhoFastjetAll"] = ak.values_astype(events.Rho.fixedGridRhoFastjetAll, "float32")
        if hasattr(events, "genWeight"):
            out["genWeight"] = events.genWeight
        if hasattr(events, "bunchCrossing"):
            out["bunchCrossing"] = events.bunchCrossing
        if hasattr(events, "LHEPdfWeight"):
            out["LHEPdfWeight"] = events.LHEPdfWeight

        #======================================================# 
        # ------------------- Trigger logic -------------------# 
        #======================================================#
        trigger_mask = ak.zeros_like(events.event, dtype=bool)
        trigger_type = ak.zeros_like(events.event, dtype=int)
        available_hlt = dir(events.HLT)
        for bit, patterns in self.trigger_groups.items():
            group_fired = ak.zeros_like(events.event, dtype=bool)
            for pattern in patterns:
                patt = pattern.replace("HLT_", "")
                for trig in fnmatch.filter(available_hlt, patt):
                    group_fired = group_fired | events.HLT[trig]
            trigger_mask = trigger_mask | group_fired
            trigger_type = trigger_type | (group_fired * (1 << bit))

        #======================================================# 
        # ------------------ MET filter logic -----------------# 
        #======================================================#
        met_filter_mask = ak.ones_like(events.event, dtype=bool)
        for flag in self.met_filter_flags:
            if hasattr(events, flag):
                met_filter_mask &= getattr(events, flag)
      
        out["has_trigger"]    = trigger_mask & met_filter_mask
        out["trigger_type"]   = trigger_type
    
        #======================================================# 
        # ------------------ Object selection -----------------# 
        #======================================================#
        for obj in self.branches_to_keep:
            if not hasattr(events, obj):
                continue
            collection = getattr(events, obj)
            
            # ---------------- MUON ---------------- #
            if obj == "Muon":
                collection = collection[(collection.pt > 10) & (np.abs(collection.eta) < 2.5)]

            # -------------- ELECTRON -------------- #
            elif obj == "Electron":
                collection = collection[(collection.pt > 15) & (np.abs(collection.eta) < 2.5)]

            # ---------------- JET ----------------- #
            elif obj == "Jet":
                rawFactor = ak.fill_none(getattr(collection, "rawFactor", ak.zeros_like(collection.pt)), 0.0)
                pt_raw    = collection.pt * (1 - rawFactor)
            
                sel        = (pt_raw > 15) & (np.abs(collection.eta) < 4.8)
                collection = collection[sel]
                rawFactor  = rawFactor[sel]
                pt_raw     = pt_raw[sel]  
            
                # recompute tight lep-veto id for filtered jets
                eta    = np.abs(collection.eta)
                chMult = collection.chMultiplicity
                neMult = collection.neMultiplicity
                neHEF  = collection.neHEF
                neEmEF = collection.neEmEF
                chHEF  = collection.chHEF
                muEF   = getattr(collection, "muEF", ak.zeros_like(eta))
                chEmEF = getattr(collection, "chEmEF", ak.zeros_like(eta))
                passJetIdTight = (
                    ((eta <= 2.6) & (neHEF < 0.99) & (neEmEF < 0.9) & ((chMult + neMult) > 1) & (chHEF > 0.01) & (chMult > 0))
                    | ((eta > 2.6) & (eta <= 2.7) & (neHEF < 0.90) & (neEmEF < 0.99))
                    | ((eta > 2.7) & (eta <= 3.0) & (neHEF < 0.99))
                    | ((eta > 3.0) & (neMult >= 2) & (neEmEF < 0.4))
                )
                passJetIdTightLepVeto = ak.where(eta <= 2.7,
                                                 passJetIdTight & (muEF < 0.8) & (chEmEF < 0.8),
                                                 passJetIdTight)
                collection = ak.with_field(collection, passJetIdTightLepVeto, "passJetIdTightLepVeto")
                collection = ak.with_field(collection, rawFactor, "rawFactor")
                if hasattr(collection, "genJetIdx"):
                    collection = ak.with_field(collection, collection.genJetIdx, "genJetIdx")
            
                # --- PNet regressed pT (uses RAW pT) ---
                pnet_cor     = getattr(collection, "PNetRegPtRawCorr", ak.ones_like(collection.pt))
                pnet_cor_net = getattr(collection, "PNetRegPtRawCorrNeutrino", ak.ones_like(collection.pt))
            
                pt_regressed = pt_raw * pnet_cor * pnet_cor_net
                collection   = ak.with_field(collection, ak.values_astype(pt_regressed, "float32"), "pt_regressed")
                
                # Save GenJet pT for matched jets; NaN if no match or on data
                if hasattr(events, "GenJet") and hasattr(collection, "genJetIdx"):
                    genIdx  = collection.genJetIdx
                    has_gen = ak.values_astype(genIdx >= 0, bool)
                    safe    = ak.mask(genIdx, has_gen)
                    pt_gen  = events.GenJet[safe].pt
                    # fill None (from masked) with NaN and cast to float32
                    pt_gen  = ak.values_astype(ak.fill_none(pt_gen, np.nan), "float32")
                    collection = ak.with_field(collection, pt_gen, "pt_genMatched")
                else:
                    # data or missing GenJet: make an array of NaNs with the right shape/dtype
                    nan_arr = ak.values_astype(ak.ones_like(collection.pt), "float32") * np.nan
                    collection = ak.with_field(collection, nan_arr, "pt_genMatched")

    
                        
            out[obj] = self.select_fields(collection, self.branches_to_keep[obj])
        
        return out

    def postprocess(self, accumulator):
        return accumulator