import awkward as ak
import numpy as np
import fnmatch
from coffea import processor

#----------------------------------------------------------------------------------------------------------------------------#

class NanoAODSkimmer(processor.ProcessorABC):
    def __init__(self, branches_to_keep, trigger_groups, met_filter_flags, dataset_name=None):
        self.branches_to_keep = branches_to_keep
        self.trigger_groups = trigger_groups
        self.met_filter_flags = met_filter_flags
        self.dataset_name = dataset_name
            
            
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
        out = {}
        out["run"] = events.run
        out["event"] = events.event
        out["luminosityBlock"] = events.luminosityBlock
        out["fixedGridRhoFastjetAll"] = ak.values_astype(events.Rho.fixedGridRhoFastjetAll, "float32")
        
        if hasattr(events, "genWeight"):
            out["genWeight"] = events.genWeight
        if hasattr(events, "bunchCrossing"):
            out["bunchCrossing"] = events.bunchCrossing        
        
        # if not hasattr(self, "_printed_schema"):
        #     self._printed_schema = True
        #     # Top-level branches in Events
        #     print("[DEBUG] top-level fields:", ak.fields(events))
        #     print("[DEBUG] HLT fields:", ak.fields(events.HLT))
        #     print("[DEBUG] Jet fields:", ak.fields(events.Jet))
        #     print("[DEBUG] Electron fields:", ak.fields(events.Electron))


        #======================================================#
        # ------------------- Trigger logic -------------------#
        #======================================================#
        
        trigger_mask = ak.zeros_like(events.event, dtype=bool)
        trigger_type = ak.zeros_like(events.event, dtype=int)
        available_hlt = dir(events.HLT)  

        for bit, patterns in self.trigger_groups.items():
            group_fired = ak.zeros_like(events.event, dtype=bool)

            for pattern in patterns:
                pattern_no_prefix = pattern.replace("HLT_", "")
                matched_triggers = fnmatch.filter(available_hlt, pattern_no_prefix)

                for trig in matched_triggers:
                    group_fired = group_fired | events.HLT[trig]

            trigger_mask = trigger_mask | group_fired
            trigger_type = trigger_type | (group_fired * (1 << bit))

        #======================================================#
        # ------------------ MET filter logic -----------------#
        #======================================================#
        met_filter_mask = ak.ones_like(events.event, dtype=bool)
        
        for idx, flag in enumerate(self.met_filter_flags):
            if hasattr(events, flag):
                passed = getattr(events, flag)
                met_filter_mask &= passed  # combine filters
               
        
        out["passMETFilters"] = met_filter_mask

        # Combine trigger and MET filter results into `has_trigger`
        combined_trigger_mask = trigger_mask & met_filter_mask
        
        out["has_trigger"]  = combined_trigger_mask
        out["trigger_type"] = trigger_type
        
        #======================================================#
        # ------------------ Object selection -----------------#
        #======================================================#
        
        for obj in self.branches_to_keep:
            if not hasattr(events, obj):
                continue
            collection = getattr(events, obj)
            
            
            # ---------------- MUON ----------------
            if obj == "Muon":
                collection = collection[(collection.pt > 10) & (abs(collection.eta) < 2.5)]

            # -------------- ELECTRON --------------
            elif obj == "Electron":
                ele = collection
                collection = ele[(ele.pt > 15) & (abs(ele.eta) < 2.5)]              
                
            # ---------------- JET -----------------
            elif obj == "Jet":
                if hasattr(events, "Jet"):
                    collection = events.Jet

                    # Step 1: Compute raw_pt *before* filtering
                    rawFactor = getattr(collection, "rawFactor", ak.zeros_like(collection.pt))
                    pt_raw    = collection.pt   * (1 - rawFactor)
                    mass_raw  = collection.mass * (1 - rawFactor)

                    # Step 2: Apply cut on raw_pt
                    selection_mask = (pt_raw > 15) & (abs(collection.eta) < 4.8)
                    collection = collection[selection_mask]
                    pt_raw     = pt_raw[selection_mask]
                    mass_raw   = mass_raw[selection_mask]
                    rawFactor  = rawFactor[selection_mask]

                    # Step 3: Now extract fields from filtered collection
                    eta = abs(collection.eta)
                    chMult = collection.chMultiplicity
                    neMult = collection.neMultiplicity
                    neHEF = collection.neHEF
                    neEmEF = collection.neEmEF
                    chHEF = collection.chHEF
                    muEF = getattr(collection, "muEF", ak.zeros_like(eta))
                    chEmEF = getattr(collection, "chEmEF", ak.zeros_like(eta))
                    
                    passJetIdTight = (
                        ((eta <= 2.6) & (neHEF < 0.99) & (neEmEF < 0.9) & ((chMult + neMult) > 1) & (chHEF > 0.01) & (chMult > 0))
                        |
                        ((eta > 2.6) & (eta <= 2.7) & (neHEF < 0.90) & (neEmEF < 0.99))
                        |
                        ((eta > 2.7) & (eta <= 3.0) & (neHEF < 0.99))
                        |
                        ((eta > 3.0) & (neMult >= 2) & (neEmEF < 0.4))
                    )
                    
                    passJetIdTightLepVeto = ak.where(
                        eta <= 2.7,
                        passJetIdTight & (muEF < 0.8) & (chEmEF < 0.8),
                        passJetIdTight
                    )
                    
                    collection = ak.with_field(collection, passJetIdTight, "passJetIdTight")
                    collection = ak.with_field(collection, passJetIdTightLepVeto, "passJetIdTightLepVeto")
                    collection = ak.with_field(collection, rawFactor, "rawFactor")
                    if hasattr(collection, "genJetIdx"):
                        collection = ak.with_field(collection, collection.genJetIdx, "genJetIdx")
                    
                    # Step 4: Access regressed quantities from filtered collection
                    pnet_cor = getattr(collection, "PNetRegPtRawCorr", ak.ones_like(collection.pt))
                    pnet_cor_net = getattr(collection, "PNetRegPtRawCorrNeutrino", ak.ones_like(collection.pt))
                    pnet_resol = getattr(collection, "PNetRegPtRawRes", ak.ones_like(collection.pt))

                    # Step 5: Recompute pt_regressed using filtered raw_pt and corrections
                    collection = ak.with_field(collection, pt_raw * pnet_cor * pnet_cor_net, "pt_regressed")
           
            out[obj] = self.select_fields(collection, self.branches_to_keep[obj])
        return out

    def postprocess(self, accumulator):
        return accumulator
