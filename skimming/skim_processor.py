import awkward as ak
import numpy as np
import fnmatch
import gzip
import json
import os
import uproot
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
            
            # https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3#2022_2023_and_2024_Scale_and_Sme
          
            # ---- EGM configuration / eT-dependent correction---# 
            # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/egmScaleAndSmearingExample.py   
            # egm_json_path = "electronSS_EtDependent.json.gz", # local test : xrdcp root://eoscms.cern.ch/<> . 
            egm_json_path  = "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2024/SS/electronSS_EtDependent.json.gz",
            egm_scale_name = "EGMScale_Compound_Ele_2024",          # DATA
            egm_smear_name = "EGMSmearAndSyst_ElePTsplit_2024",     # MC
            save_mc_variations = True,
            
            #--- ID Tight SFs (2024 combined egamma) ---#
            # https://twiki.cern.ch/twiki/bin/view/CMS/EgammSFandSSRun3
            # id_sf_merged_path = "merged_EGamma_SF2D_Tight.root",
            id_sf_merged_path = "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2024/EleID/passingCutBasedTight122XV1/merged_EGamma_SF2D_Tight.root",
            
            #--- HLT Ele30 TightID (2023D proxy) / Need to change later (expected at the beginning of September) ---#
            # hlt_ele30_path = "egammaEffi.txt_EGM2D.root",
            hlt_ele30_path = "/eos/cms/store/group/phys_egamma/ScaleFactors/Data2023/ForPrompt23D/tnpEleHLT/HLT_SF_Ele30_TightID/egammaEffi.txt_EGM2D.root",
            hlt_use_2023_proxy = True,
            ):
        
        self.branches_to_keep = branches_to_keep
        self.trigger_groups = trigger_groups
        self.dataset_name = dataset_name 
        self._rng = np.random.default_rng(12345)   
        
        #--- Load EGM corrections ---#
        self.save_mc_variations = save_mc_variations
        if egm_json_path is not None:
            if not os.path.exists(egm_json_path) and os.path.exists(egm_json_path + ".gz"):
                os.system(f"gunzip -k {egm_json_path}.gz")
            
            cset = correctionlib.CorrectionSet.from_file(egm_json_path)
            self.scale_eval = cset.compound[egm_scale_name]
            self.smear_eval = cset[egm_smear_name]
            
        #---Load ID Tight TH2D 2024 (eta on x, pT on y) ---#
        with uproot.open(id_sf_merged_path) as f:
            h = f["EGamma_SF2D"]
            self.id_xedges = h.axes[0].edges()   
            self.id_yedges = h.axes[1].edges()   
            self.id_vals   = h.values()     
            
        #--- Load HLT Ele30 TightID ---#   
        self.hlt_has = True
        self.hlt_xedges = self.hlt_yedges = None
        self.hlt_sf2d = None
        
        if hlt_ele30_path and os.path.exists(hlt_ele30_path):
            with uproot.open(hlt_ele30_path) as f:
                hS = f["EGamma_SF2D"]
                self.hlt_xedges = hS.axes[0].edges()   # x = SuperCluster eta
                self.hlt_yedges = hS.axes[1].edges()   # y = pT [GeV]
                self.hlt_sf2d   = hS.values()
                self.hlt_has = True
        else:
            print(f"[HLT] File not found: {hlt_ele30_path} — HLT SF disabled.")
        
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

    def _lookup_th2(self, x, y, xedges, yedges, vals):
        """
        Vectorized TH2 lookup with clipping to edge bins.
        x: eta (jagged), y: pt (jagged)
        xedges: bin edges for x axis, yedges: bin edges for y axis
        """
        xf = ak.to_numpy(ak.flatten(x)).astype(np.float64, copy=False)
        yf = ak.to_numpy(ak.flatten(y)).astype(np.float64, copy=False)
        
        nx = len(xedges) - 1
        ny = len(yedges) - 1
        
        x_min = np.nextafter(xedges[0], xedges[1])
        x_max = np.nextafter(xedges[-1], xedges[0])  
        y_min = np.nextafter(yedges[0], yedges[1])
        y_max = np.nextafter(yedges[-1], yedges[0])
        
        xf = np.clip(xf, x_min, x_max)
        yf = np.clip(yf, y_min, y_max)
        
        ix = np.digitize(xf, xedges, right=False) - 1  
        iy = np.digitize(yf, yedges, right=False) - 1  
        
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        
        shape = vals.shape
        if shape == (ny, nx):         
            sf_flat = vals[iy, ix]
        elif shape == (nx, ny):      
            sf_flat = vals[ix, iy]
        else:
            raise RuntimeError(f"Unexpected TH2 shape {shape}; expected (ny,nx) or (nx,ny).")

        return ak.unflatten(ak.Array(sf_flat), ak.num(x))

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
                
                # The Scale and Smearing corrections should not be used for electrons 
                # below ~15 GeV and should be used with caution between 15 and 20 GeV, 
                # as they were not tuned for this pT range.
                collection = ele[(ele.pt > 20) & (abs(ele.eta) < 2.5)]
                
                #########################################
                # Apply cut-based Tight ID SFs (for MC) #
                #########################################  
                
                if not is_data:
                    ele_sel = collection
                    
                    eta_for_sf = ele_sel.eta
                    pt_for_sf  = ele_sel.pt
                    
                    sf_ele = self._lookup_th2(eta_for_sf, pt_for_sf, self.id_xedges, self.id_yedges, self.id_vals)
                    
                    w_ele_id_tight = ak.prod(sf_ele, axis=1, mask_identity=True)
                    w_ele_id_tight = ak.fill_none(w_ele_id_tight, 1.0)
                    out["weight_ele_id_tight"] = w_ele_id_tight
                    
                    
                ############################################
                # HLT Ele30 TightID event weight (for MC)  #
                ############################################
                if not is_data:
                    if self.hlt_has:
                        
                        x = collection.superclusterEta
                        y = collection.pt
                        
                        x_lead = x[:, :1] 
                        y_lead = y[:, :1]
                        
                        sf_lead = self._lookup_th2(x_lead, y_lead, self.hlt_xedges, self.hlt_yedges, self.hlt_sf2d)
                        
                        w_hlt = ak.fill_none(ak.firsts(sf_lead), 1.0)
                    else:
                        w_hlt = ak.ones_like(out["run"], dtype=float)

                    out["weight_ele_hlt_Ele30_TightID"] = w_hlt
                                         
                
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
