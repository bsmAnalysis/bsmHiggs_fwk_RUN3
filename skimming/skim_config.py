
branches_to_keep = {
            "Muon": ["pt", "eta", "phi","charge","pdgId","tightId","mass","pfRelIso03_all","pfRelIso04_all"],                                                                                       
            "Electron": ["pt", "eta", "phi","charge","pdgId","cutBased","mass","pfRelIso03_all","pfRelIso04_all"],                                                                                  
            "Jet": ["pt", "eta", "phi","btagUParTAK4probbb","svIdx1","svIdx2","mass","btagDeepFlavB","btagUParTAK4","passJetIdTight", "passJetIdTightLepVeto","pt_regressed","hadronFlavour","partonFlavour","pnet_resol"],
            "PFMET": ["pt", "phi","sumEt"],
            "PuppiMET": ["pt", "phi","sumEt"],
            "Pileup":["nTrueInt","nPU"],
            "LHE": ["HT","NJets"],
            "PV": ["npvsGood","npvs"],
            
            
        }


trigger_groups = {
    0: ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"],
    1: ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8"],
    2: ["IsoMu24"],
    3: ["Mu50"],
    4: ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL","DoubleEle25_CaloIdL_MW"],
    5: ["Ele32_WPTight_Gsf"],
    6: ["Ele35_WPTight_Gsf"],
    7: ["Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
    8: ["PFMET120_PFMHT120_IDTight","PFMET120_PFMHT120_IDTight_PFHT60"],
    9: ["QuadPFJet105_88_76_15_PFBTagDeepCSV_1p3_VBF2"],
    10: ["QuadPFJet105_90_76_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1", "QuadPFJet105_88_76_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1"],
    11: ["QuadPFJet105_88_76_15"],
    12: ["DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71"],
    13: ["PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_2p0"],
    14: ["PFHT330PT30_QuadPFJet_75_60_45_40_PNet3BTag_4p3"],
    15: ["PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5"],
    16: ["PFHT400_FivePFJet_120_120_60_30_30"],
    17: ["PFHT400_FivePFJet_120_120_60_30_30_PNet2BTag_4p3"],
    18: ["PFHT400_FivePFJet_120_120_60_30_30_PNet2BTag_5p6"],
    19: ["QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1"],
    20: ["QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2"],
    21: ["QuadPFJet105_88_76_15_PNet2BTag_0p4_0p12_VBF1"],
    22: ["QuadPFJet105_88_76_15_PNetBTag_0p4_VBF2"],
    23: ["QuadPFJet111_90_80_15_PNet2BTag_0p4_0p12_VBF1"],
    24: ["QuadPFJet111_90_80_15_PNetBTag_0p4_VBF2"],
    25: ["PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70"],
   
}
