from coffea import processor
import correctionlib
import awkward as ak
import numpy as np
import fnmatch
import os

#------------------------------- helpers --------------------------------#

def _unflatten_like(flat, counts):
    return ak.unflatten(ak.Array(flat), counts)

#----------------------------------------------------------------------------------------------------------------------------#

class NanoAODSkimmer(processor.ProcessorABC):
    def __init__(self, branches_to_keep, trigger_groups, met_filter_flags, dataset_name=None, corrections_dir=None):
        self.branches_to_keep = branches_to_keep
        self.trigger_groups   = trigger_groups
        self.met_filter_flags = met_filter_flags
        self.dataset_name     = dataset_name
        self.corrections_dir  = corrections_dir or os.path.join(os.path.dirname(__file__), "corrections")
        self._jet_veto        = None
        self._jet_veto_type   = "jetvetomap"
        self._loaded_veto     = False
            
#----------------------------------------------------------------------------------------------------------------------------#

    # -- load veto map (2024)-- #
    # pt_raw = pt * (1 − rawFactor); select jets with pT_raw>15, tight ID, (chEmEF+neEmEF)<0.9.
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
         # --- PV selection ---
        if ("PV" in events.fields) and ("npvsGood" in ak.fields(events.PV)):
            pv_mask    = events.PV.npvsGood >= 1
            n_after_pv = ak.sum(pv_mask)
            print(f"selected {n_after_pv} / {n_before} events with npvsGood>=1")
        else:
            print(" No PV.npvsGood found skipping PV preselection.")
            #pv_mask = ak.ones_like(events.event, dtype=bool)
            #n_after_pv = n_before

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

        jet_min = (
            (pt_raw_full > 15.0)
            & ak.values_astype(jets_full.passJetIdTightLepVeto, bool)
            & ((jets_full.chEmEF + jets_full.neEmEF) < 0.9)
        )


        # --- evaluate veto map per jet (True => jet is inside vetoed region) ---
        if self._jet_veto is not None:
            counts    = ak.num(jets_full.pt, axis=1)
            eta_flat  = ak.to_numpy(ak.flatten(jets_full.eta))
            phi_flat  = ak.to_numpy(ak.flatten(jets_full.phi))
            vflat     = self._jet_veto.evaluate(self._jet_veto_type, eta_flat, phi_flat)
            veto_j    = _unflatten_like(vflat, counts) > 0.5
        else:
            veto_j    = ak.zeros_like(jets_full.pt, dtype=bool)

        # --- define good/bad jets among the selected ones ---
        good_j   = jet_min & ~veto_j
        bad_j    = jet_min &  veto_j

        num_good = ak.sum(good_j, axis=1)           # per-event int
        has_bad  = ak.any(bad_j,  axis=1)           # per-event bool
        print("[DEBUG] num_good: " + str(num_good))
        print("[DEBUG] has_bad:  " + str(has_bad))
        
        # Require ≥2 good jets AND no bad selected jets
        event_mask = (num_good >= 2) & (~has_bad)
        event_mask = ak.values_astype(event_mask, bool)

        # Quick sanity prints before indexing
        print("[DEBUG] len(events) =", len(events), "len(event_mask) =", len(event_mask))
        
        n_total         = len(events)
        n_good_ge2      = int(ak.count_nonzero(num_good >= 2))
        n_bad_any       = int(ak.count_nonzero(has_bad))
        n_ge2_and_bad   = int(ak.count_nonzero((num_good >= 2) &  has_bad))
        n_ge2_and_nobad = int(ak.count_nonzero((num_good >= 2) & ~has_bad))  # == kept
        n_lt2_and_nobad = int(ak.count_nonzero((num_good <  2) & ~has_bad))
        n_lt2_and_bad   = int(ak.count_nonzero((num_good <  2) &  has_bad))

        print(f"[DEBUG] total={n_total} "
            f"ge2_good={n_good_ge2} any_bad={n_bad_any} "
            f"ge2_good&bad={n_ge2_and_bad} ge2_good&no_bad={n_ge2_and_nobad} "
            f"<2_good&no_bad={n_lt2_and_nobad} <2_good&bad={n_lt2_and_bad}")
        
        
        #=======================================================# 
        # ------------------ Pile up Info 2024 -----------------# 
        #=======================================================#
        # TBD : no pileup txt file for 2024 yet
        # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData
        
        #======================================================# 
        # ------------------ MET filter logic -----------------# 
        #======================================================#
        met_filter_mask = ak.ones_like(events.event, dtype=bool)
        
        for flag in self.met_filter_flags:
            flag = flag.replace("Flag_", "")
            if hasattr(events.Flag, flag):
                met_filter_mask = met_filter_mask & getattr(events.Flag, flag)

        kept_jet = int(ak.count_nonzero(event_mask))
        print(f"[DEBUG: JetVeto/Count] would keep {kept_jet} / {len(event_mask)} (≥2 good & no bad)")
        kept_met = int(ak.count_nonzero(met_filter_mask))
        print(f"[DEBUG: MET filter/Count]  would keep {kept_met} / {len(met_filter_mask)}")
        
        # Final mask
        final_mask = pv_mask & event_mask & met_filter_mask
         
        events = events[final_mask]
               
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

        
        # ---------------- Fill outputs when final mask applied --------- #
        out = {}
        out["run"]   = events.run
        out["event"] = events.event
        out["luminosityBlock"] = events.luminosityBlock
        out["has_trigger"]  = trigger_mask
        out["trigger_type"] = trigger_type
  
        if hasattr(events, "Rho") and hasattr(events.Rho, "fixedGridRhoFastjetAll"):
            out["fixedGridRhoFastjetAll"] = ak.values_astype(events.Rho.fixedGridRhoFastjetAll, "float32")
        if hasattr(events, "genWeight"):
            out["genWeight"] = events.genWeight
    
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
                collection = ak.with_field(collection, passJetIdTightLepVeto[final_mask], "passJetIdTightLepVeto")
                collection = ak.with_field(collection, passJetIdTight[final_mask], "passJetIdTight")
                collection = ak.with_field(collection, rawFactor_full[final_mask], "rawFactor")
                if hasattr(collection, "genJetIdx"):
                     collection = ak.with_field(collection, collection.genJetIdx, "genJetIdx")
                     
                # --- UParT regressed pT (uses RAW pT) -- #
                upart_cor     = getattr(collection, "UParTAK4RegPtRawCorr", ak.ones_like(collection.pt))
                upart_cor_net = getattr(collection, "UParTAK4RegPtRawCorrNeutrino", ak.ones_like(collection.pt))
            
                upart_pt_reg = pt_raw_full[final_mask] * upart_cor * upart_cor_net
                collection   = ak.with_field(collection, ak.values_astype(upart_pt_reg, "float32"), "upart_pt_reg")
                
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
                    # data: make an array of NaNs with the right shape/dtype
                    nan_arr = ak.values_astype(ak.ones_like(collection.pt), "float32") * np.nan
                    collection = ak.with_field(collection, nan_arr, "pt_genMatched")  
                        
            out[obj] = self.select_fields(collection, self.branches_to_keep[obj])
        
        return out

    def postprocess(self, accumulator):
        return accumulator
