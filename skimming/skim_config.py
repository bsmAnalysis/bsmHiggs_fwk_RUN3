branches_to_keep = {
            "Muon": ["pt","eta","phi","charge","tightId","looseId","mass","pfRelIso04_all"],
            "Electron": ["pt","eta","phi","charge","cutBased","mass","pfRelIso03_all",
                         "seedGain","r9","superclusterEta","mvaIso_WP90"],   
            "Jet": ["pt","eta","phi","mass","rawFactor","area","pt_genMatched",
                    "btagUParTAK4probbb","btagUParTAK4B","passJetIdTight","passJetIdTightLepVeto",
                    "pt_regressed","hadronFlavour","partonFlavour"],                  
            "PuppiMET": ["pt","phi"],
            "Pileup":["nTrueInt","nPU"],
            "PV": ["npvsGood","npvs"],
        }


trigger_groups = {
    0: ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"],
    1: ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8"],
    2: ["IsoMu24"],
    3: ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL","Ele32_WPTight_Gsf"],
    4: ["Ele30_WPTight_Gsf"],
    5: ["Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
    6: ["PFMET120_PFMHT120_IDTight","PFMET120_PFMHT120_IDTight_PFHT60","PFMETNoMu120_PFMHTNoMu120_IDTight","PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF", "PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60"],
    7: ["QuadPFJet105_88_76_15"],
    8: ["PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_2p0"],
    9: ["PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_4p3"],
    10: ["PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5"],
    11: ["PFHT400_FivePFJet_120_120_60_30_30"],
    12: ["PFHT400_FivePFJet_120_120_60_30_30_PNet2BTag_4p3"],
    13: ["PFHT400_FivePFJet_120_120_60_30_30_PNet2BTag_5p6"],
    14: ["QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1"],
    15: ["QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2"],
    16: ["QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1"],
    17: ["QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2"],
    18: ["QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1"],
    19: ["QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2"],
    20: ["PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70"],
    21: ["BTagMu_AK4DiJet20_Mu5"],        
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
