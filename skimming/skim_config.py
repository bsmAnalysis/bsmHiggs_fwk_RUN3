# Nominal standard / UParT-visible / UParT-plus-neutrino momenta only.
# The processor appends nine Jet reconstruction inputs and event rho.
# METCorrectionJet and RawPuppiMET px/py are saved separately for MET replay.
# Default production computes and writes NO JES/JER up/down arrays.
# All saved Jet candidates pass |eta|<2.5 and tight-lepton-veto ID, with NO pT cut.
branches_to_keep = {
    "Muon": ["pt", "eta", "phi", "charge", "tightId", "looseId", "mass", "pfRelIso04_all"],
    "Electron": ["pt", "eta", "phi", "charge", "mass", "pfRelIso03_all", "superclusterEta", "mvaIso_WP90", "mvaIso_WP80"],
    "Jet": [
        "pt", "eta", "phi", "mass", "btagUParTAK4probbb", "btagUParTAK4B",
        # Every saved analysis candidate already passes tight-lepton-veto ID.
        "hadronFlavour",
        "pt_upart", "mass_upart",
        "pt_upartnu", "mass_upartnu",
        # Mandatory unsmeared JEC stages/gen candidates are appended by the
        # nominal-storage processor. These are INPUTS, not up/down branches.
    ],
    "PuppiMET": ["pt", "phi"],
    "Pileup": ["nTrueInt"],
}

trigger_groups = {
    0: ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"],
    1: ["IsoMu24"],
    2: ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
    3: ["Ele30_WPTight_Gsf"],
    4: ["Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ"],
    5: ["PFMET120_PFMHT120_IDTight", "PFMET120_PFMHT120_IDTight_PFHT60",
        "PFMETNoMu120_PFMHTNoMu120_IDTight", "PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF",
        "PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60"],
}

met_filter_flags = [
    "goodVertices", "eeBadScFilter", "globalSuperTightHalo2016Filter",
    "EcalDeadCellTriggerPrimitiveFilter", "BadPFMuonFilter", "BadPFMuonDzFilter",
    "ecalBadCalibFilter", "hfNoisyHitsFilter",
]
