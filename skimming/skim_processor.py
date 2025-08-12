import awkward as ak
import numpy as np
import fnmatch
import gzip
import json
import os
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
import correctionlib

#----------------------------------------------------------------------------------------------------------------------------#

class NanoAODSkimmer(processor.ProcessorABC):
    def __init__(
            self, 
            branches_to_keep, 
            trigger_groups, 
            dataset_name= None,
            # ---- EGM configuration / eT-dependent correction---# 
            # https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#2022_2023_and_2024_Scale_and_Sme
            # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/egmScaleAndSmearingExample.py
            # egm_json_path = "electronSS_EtDependent.json.gz",     # local test : xrdcp root://eoscms.cern.ch/<egm_json_path> . 
            egm_json_path  = "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2024/SS/electronSS_EtDependent.json.gz",
            egm_scale_name = "EGMScale_Compound_Ele_2024",          # DATA
            egm_smear_name = "EGMSmearAndSyst_ElePTsplit_2024",     # MC
            save_mc_variations = False
            ):
        
        self.branches_to_keep = branches_to_keep
        self.trigger_groups = trigger_groups
        self.dataset_name = dataset_name 
        # --- Load EGM corrections ---#
        self.save_mc_variations = save_mc_variations
        if egm_json_path is not None:
            if not os.path.exists(egm_json_path) and os.path.exists(egm_json_path + ".gz"):
                os.system(f"gunzip -k {egm_json_path}.gz")
            
            cset = correctionlib.CorrectionSet.from_file(egm_json_path)
            self.scale_eval = cset.compound[egm_scale_name]
            self.smear_eval = cset[egm_smear_name]
        self._rng = np.random.default_rng(12345)     
        
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
        
        is_data = not hasattr(events, "genWeight")
        
        if hasattr(events, "genWeight"):
            out["genWeight"] = events.genWeight
        if hasattr(events, "bunchCrossing"):
            out["bunchCrossing"] = events.bunchCrossing

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

        out["has_trigger"] = trigger_mask
        out["trigger_type"] = trigger_type
        
        #======================================================#
        # ------------------ Object selection -----------------#
        #======================================================#
        
        for obj in self.branches_to_keep:
            if not hasattr(events, obj):
                continue
            collection = getattr(events, obj)
            
            ###-- LEPTON SELECTION --###
            
            if obj == "Muon":
                collection = collection[(collection.pt > 10) & (abs(collection.eta) < 2.5)]
            elif obj == "Electron":
                ele = collection
                
                #####################################
                # Apply Scale Correction (for Data) #
                #####################################

                if is_data:               
                    run_b, scEta_b, r9_b, pt_b, seedGain_b = ak.broadcast_arrays(events.run, ele.superclusterEta, ele.r9, ele.pt, ele.seedGain)
                    counts = ak.num(ele.pt)
                    
                    scale_flat = self.scale_eval.evaluate(
                        "scale",
                        ak.to_numpy(ak.flatten(run_b)),
                        ak.to_numpy(ak.flatten(scEta_b)),
                        ak.to_numpy(ak.flatten(r9_b)),
                        ak.to_numpy(np.abs(ak.flatten(scEta_b))),
                        ak.to_numpy(ak.flatten(pt_b)),
                        ak.to_numpy(ak.flatten(seedGain_b)),
                    )
                    scale = ak.unflatten(ak.Array(scale_flat), counts)
                    print(f"Multiplicative scale (for data): mean={ak.mean(ak.flatten(scale)):.6f}")
                    
                    ele = ak.with_field(ele, ele.pt * scale, "pt")
            
                
                ######################################
                # Apply Smearing Correction (for MC) #
                ######################################    
                
                else:
                    ele = ak.with_field(ele, ele.pt, "pt_raw")                  
                    counts = ak.num(ele.pt_raw)
                    
                    smear_flat = self.smear_eval.evaluate(
                        "smear",
                        ak.to_numpy(ak.flatten(ele.pt_raw)),
                        ak.to_numpy(ak.flatten(ele.r9)),
                        ak.to_numpy(np.abs(ak.flatten(ele.superclusterEta))),
                    )
                    smear = ak.unflatten(ak.Array(smear_flat), counts)                   
                    print(f"Smearing width (for MC): mean={ak.mean(ak.flatten(smear)):.6f}")

                    n_flat = self._rng.normal(size=len(ak.flatten(ele.pt_raw)))
                    n = ak.unflatten(ak.Array(n_flat), counts)

                    corr_nom = 1.0 + smear * n
                    ele = ak.with_field(ele, ele.pt_raw * corr_nom, "pt")
                    
                    if self.save_mc_variations:
                        #--- smearing uncertainty ---#
                        unc_smear_flat = self.smear_eval.evaluate(
                            "esmear",
                            ak.to_numpy(ak.flatten(ele.pt_raw)),
                            ak.to_numpy(ak.flatten(ele.r9)),
                            ak.to_numpy(np.abs(ak.flatten(ele.superclusterEta))),
                            )
                        unc_smear = ak.unflatten(ak.Array(unc_smear_flat), counts)
                        print(f"Smearing uncertainties: {unc_smear}")
                        
                        corr_smear_up   = 1.0 + (smear + unc_smear) * n
                        corr_smear_down = 1.0 + (smear - unc_smear) * n
                        ele = ak.with_field(ele, ele.pt_raw * corr_smear_up,   "pt_smearUp")
                        ele = ak.with_field(ele, ele.pt_raw * corr_smear_down, "pt_smearDown")
                        
                        #--- scale uncertainty ---#
                        unc_scale_flat = self.smear_eval.evaluate(
                            "escale",
                            ak.to_numpy(ak.flatten(ele.pt_raw)),
                            ak.to_numpy(ak.flatten(ele.r9)),
                            ak.to_numpy(np.abs(ak.flatten(ele.superclusterEta))),
                            )
                        unc_scale = ak.unflatten(ak.Array(unc_scale_flat), counts)
                        ele = ak.with_field(ele, ele.pt * (1.0 + unc_scale), "pt_scaleUp")
                        ele = ak.with_field(ele, ele.pt * (1.0 - unc_scale), "pt_scaleDown")
                    
                collection = ele[(ele.pt > 15) & (abs(ele.eta) < 2.5)]
                
                
            ###-- JET SELECTION --###
            elif obj == "Jet":
                if hasattr(events, "Jet"):
                    collection = events.Jet

                    # Step 1: Compute raw_pt *before* filtering
                    rawFactor = getattr(collection, "rawFactor", ak.zeros_like(collection.pt))
                    raw_pt = collection.pt * (1 - rawFactor)

                    # Step 2: Apply cut on raw_pt
                    selection_mask = (raw_pt > 15) & (abs(collection.eta) < 4.8)
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
           
            out[obj] = self.select_fields(collection, self.branches_to_keep[obj])
        return out

    def postprocess(self, accumulator):
        return accumulator
