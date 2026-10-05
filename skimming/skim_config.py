branches_to_keep = {
            "Muon": ["pt","eta","phi","charge","tightId","looseId","mass","pfRelIso04_all"],
            "Electron": ["pt","eta","phi","charge","cutBased","mass","pfRelIso03_all","superclusterEta","mvaIso_WP90","mvaIso_WP80"],   
            "Jet": ["pt","eta","phi","mass",
                    "btagUParTAK4probbb","btagUParTAK4B","hadronFlavour"],                  
            "PuppiMET": ["pt","phi"],
            "Pileup":["nTrueInt"],
            "PV": ["npvsGood"],
        }


trigger_groups = {
    0: ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"],
    1: ["IsoMu24"],
    2: ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
    3: ["Ele30_WPTight_Gsf"],
    4: ["Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
    5: ["PFMET120_PFMHT120_IDTight","PFMET120_PFMHT120_IDTight_PFHT60"],
    6:["PFMETNoMu120_PFMHTNoMu120_IDTight","PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF", "PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60"],
    7: ["BTagMu_AK4DiJet20_Mu5"],        
}


met_filter_flags = [
    "goodVertices",
    "eeBadScFilter",
    "globalSuperTightHalo2016Filter",
    "EcalDeadCellTriggerPrimitiveFilter",
    "BadPFMuonFilter",
    "BadPFMuonDzFilter",
    "ecalBadCalibFilter",
    "hfNoisyHitsFilter"
]
