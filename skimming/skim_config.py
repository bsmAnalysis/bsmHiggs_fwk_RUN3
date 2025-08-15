
branches_to_keep = {
            "Muon": ["pt","eta","phi","charge","pdgId","tightId","looseId","mass","pfRelIso03_all","pfRelIso04_all","ptErr",
                     "nTrackerLayers","genPartIdx","genPartFlav"],
            "Electron": ["pt","eta","phi","charge","pdgId","cutBased","mass","pfRelIso03_all","pfRelIso04_all",
                         "seedGain","r9","superclusterEta","mvaIso_WP90","mvaIso_WP80"],   
            "Jet": ["pt","eta","phi","mass","rawFactor","area","genJetIdx",
                    "btagUParTAK4probbb","svIdx1","svIdx2","btagUParTAK4B",
                    "passJetIdTight","passJetIdTightLepVeto","chEmEF","neEmEF","chMultiplicity",
                    "pt_regressed","hadronFlavour","partonFlavour"],
            "PFMET": ["pt","phi","sumEt"],
            "PuppiMET": ["pt","phi","sumEt","phiUnclusteredDown","phiUnclusteredUp","ptUnclusteredDown","ptUnclusteredUp"],
            "Pileup":["nTrueInt","nPU"],
            "LHE": ["HT","Njets"],
            "PV": ["npvsGood","npvs"],
            "GenJet": ["pt"],
        }


trigger_groups = {
    0: ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"],
    1: ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8"],
    2: ["IsoMu24"],
    3: ["Mu50"],
    4: ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL","Ele32_WPTight_Gsf"],
    5: ["Ele30_WPTight_Gsf"],
    6: ["Ele32_WPTight_Gsf"],
    7: ["Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
    8: ["PFMET120_PFMHT120_IDTight","PFMET120_PFMHT120_IDTight_PFHT60","PFMETNoMu120_PFMHTNoMu120_IDTight","PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF", "PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60"],
    9: ["QuadPFJet105_88_76_15"],
    11: ["PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_2p0"],
    12: ["PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_4p3"],
    13: ["PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5"],
    14: ["PFHT400_FivePFJet_120_120_60_30_30"],
    15: ["PFHT400_FivePFJet_120_120_60_30_30_PNet2BTag_4p3"],
    16: ["PFHT400_FivePFJet_120_120_60_30_30_PNet2BTag_5p6"],
    17: ["QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1"],
    18: ["QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2"],
    19: ["QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1"],
    20: ["QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2"],
    21: ["QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1"],
    22: ["QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2"],
    23: ["PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70"],
    24: ["BTagMu_AK4DiJet20_Mu5"],        
}


met_filter_flags = [
    "Flag_goodVertices",
    "Flag_eeBadScFilter",
    "Flag_HBHENoiseFilter",
    "Flag_HBHENoiseIsoFilter",
    "Flag_globalSuperTightHalo2016Filter",
    "Flag_EcalDeadCellTriggerPrimitiveFilter",
    "Flag_BadPFMuonFilter",
    "Flag_BadPFMuonDzFilter",
    "Flag_ecalBadCalibFilter",
    "Flag_hfNoisyHitsFilter"
]
