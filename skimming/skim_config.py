
branches_to_keep = {
            "Muon": ["pt", "eta", "phi","charge","pdgId","tightId","mass","pfRelIso03_all"],  # keep all fields                                                                                       
            "Electron": ["pt", "eta", "phi","charge","pdgId","cutBased","mass","pfRelIso04_all"],  # keep all fields                                                                                   
            "Jet": ["pt", "eta", "phi","btagUParTAK4probbb","svIdx1","svIdx2","mass","btagDeepFlavB","btagUParTAK4","passJetIdTight", "passJetIdTightLepVeto","pt_regressed","hadronFlavour","partonFlavour","pnet_resol"],
            "FatJet": ["pt", "eta", "phi", "msoftdrop","globalParT3_Xbb","globalParT3_QCD","mass","particleNet_XbbVsQCD", "subJetIdx1", "subJetIdx2","passJetIdTight", "passJetIdTightLepVeto"],
            #"GenPart": ["pt", "eta", "phi", "pdgId","statusFlags", "status", "genPartIdxMother"],
            "PFMET": ["pt", "phi","sumEt"],
            "PuppiMET": ["pt", "phi","sumEt"],
            "Pileup":["nTrueInt","nPU"],
            "LHE": ["HT","NJets"],
            "PV": ["npvsGood","npvs"],
            #"SubJet":["btagDeepFlavB","btagUParTAK4B"],
            
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
}
