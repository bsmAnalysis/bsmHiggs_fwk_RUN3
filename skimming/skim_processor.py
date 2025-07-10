import awkward as ak
import fnmatch
from coffea import processor

class NanoAODSkimmer(processor.ProcessorABC):
    def __init__(self, branches_to_keep, trigger_groups, dataset_name= None):
        self.branches_to_keep = branches_to_keep
        self.trigger_groups = trigger_groups
        self.dataset_name = dataset_name 
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

    def process(self, events):
        out = {}
        out["run"] = events.run
        out["event"] = events.event
        out["luminosityBlock"] = events.luminosityBlock
        '''
        if hasattr(events, "genWeight"):
            out["genWeight"] = events.genWeight
        if hasattr(events, "bunchCrossing"):
            out["bunchCrossing"] = events.bunchCrossing
        '''
        # Trigger logic
        
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

        out["has_trigger"] = trigger_mask
        out["trigger_type"] = trigger_type
        '''
        for bit, triggers in self.trigger_groups.items():
            group_fired = ak.zeros_like(events.event, dtype=bool)

            for trig in triggers:
                if hasattr(events.HLT, trig):
                    group_fired = group_fired | events.HLT[trig]
            trigger_mask = trigger_mask | group_fired
            trigger_type = trigger_type | (group_fired * (1 << bit))

        out["has_trigger"] = trigger_mask
        out["trigger_type"] = trigger_type
        
        # Get all available trigger names from the file
        available_hlt = dir(events.HLT)

        for bit, patterns in self.trigger_groups.items():
            group_fired = ak.zeros_like(events.event, dtype=bool)
            
            for pattern in patterns:
                # Match all triggers in the file that match this pattern (e.g., with *)
                matched_triggers = fnmatch.filter(available_hlt, pattern)

                for trig in matched_triggers:
                    group_fired = group_fired | events.HLT[trig]
                trigger_mask = trigger_mask | group_fired
                trigger_type = trigger_type | (group_fired * (1 << bit))

            out["has_trigger"] = trigger_mask
            out["trigger_type"] = trigger_type
        '''
        # Object selections
        for obj in self.branches_to_keep:
            if not hasattr(events, obj):
                continue
            collection = getattr(events, obj)
            
            if obj == "Muon":
                collection = collection[(collection.pt > 10) & (abs(collection.eta) < 2.5)]
            elif obj == "Electron":

                collection = collection[(collection.pt > 15) & (abs(collection.eta) < 2.5)]
            
            elif obj == "Jet":
                if hasattr(events, "Jet"):
                    collection = events.Jet

                    # Step 1: Compute raw_pt *before* filtering
                    rawFactor = getattr(collection, "rawFactor", ak.zeros_like(collection.pt))
                    raw_pt = collection.pt * (1 - rawFactor)

                    # Step 2: Apply cut on raw_pt
                    selection_mask = (raw_pt > 15) & (abs(collection.eta) < 2.5)
                    collection = collection[selection_mask]
                    raw_pt = raw_pt[selection_mask]  # update raw_pt to match

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
                    
                    collection["passJetIdTight"] = passJetIdTight
                    collection["passJetIdTightLepVeto"] = passJetIdTightLepVeto
                    
                    # Step 4: Access regressed quantities from filtered collection
                    pnet_cor = getattr(collection, "PNetRegPtRawCorr", ak.ones_like(collection.pt))
                    pnet_cor_net = getattr(collection, "PNetRegPtRawCorrNeutrino", ak.ones_like(collection.pt))
                    pnet_resol = getattr(collection, "PNetRegPtRawRes", ak.ones_like(collection.pt))

                    # Step 5: Recompute pt_regressed using filtered raw_pt and corrections
                    pt_regressed = raw_pt * pnet_cor * pnet_cor_net
                    collection["pt_regressed"] = pt_regressed
            elif obj == "FatJet":
                if hasattr(events, "FatJet"):
                    #collection = events.FatJet
                    collection = collection[(collection.pt > 100) & (abs(collection.eta) < 2.5)]
                    eta_f = abs(collection.eta)
                    chMult = collection.chMultiplicity
                    neMult = collection.neMultiplicity
                    neHEF = collection.neHEF
                    neEmEF = collection.neEmEF
                    chHEF = collection.chHEF
                    muEF = getattr(collection, "muEF", ak.zeros_like(eta_f))
                    chEmEF = getattr(collection, "chEmEF", ak.zeros_like(eta_f))
                
                    passJetIdTight = (
                        ((eta_f <= 2.6) & (neHEF < 0.99) & (neEmEF < 0.9) & ((chMult + neMult) > 1) & (chHEF > 0.01) & (chMult > 0))
                        |
                        ((eta_f > 2.6) & (eta_f <= 2.7) & (neHEF < 0.90) & (neEmEF < 0.99))
                        |
                        ((eta_f > 2.7) & (eta_f <= 3.0) & (neHEF < 0.99))
                        |
                        ((eta_f > 3.0) & (neMult >= 2) & (neEmEF < 0.4))
                    )
                    
                    passJetIdTightLepVeto = ak.where(
                        eta_f <= 2.7,
                        passJetIdTight & (muEF < 0.8) & (chEmEF < 0.8),
                        passJetIdTight
                    )
                
                    collection["passJetIdTight"] = passJetIdTight
                    collection["passJetIdTightLepVeto"] = passJetIdTightLepVeto
                    #out[obj] = self.select_fields(collection, self.branches_to_keep[obj])
            out[obj] = self.select_fields(collection, self.branches_to_keep[obj])
        return out

    def postprocess(self, accumulator):
        return accumulator
