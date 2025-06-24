import awkward as ak
import numpy as np

def compute_jet_id(jets):
    eta = abs(jets.eta)
    chMult = jets.chMultiplicity
    neMult = jets.neMultiplicity
    neHEF = jets.neHEF
    neEmEF = jets.neEmEF
    chHEF = jets.chHEF
    muEF = getattr(jets, "muEF", ak.zeros_like(eta))
    chEmEF = getattr(jets, "chEmEF", ak.zeros_like(eta))

    passTight = (
        ((eta <= 2.6) & (neHEF < 0.99) & (neEmEF < 0.9) & ((chMult + neMult) > 1) & (chHEF > 0.01) & (chMult > 0))
        |
        ((eta > 2.6) & (eta <= 2.7) & (neHEF < 0.90) & (neEmEF < 0.99))
        |
        ((eta > 2.7) & (eta <= 3.0) & (neHEF < 0.99))
        |
        ((eta > 3.0) & (neMult >= 2) & (neEmEF < 0.4))
    )

    passTightLepVeto = ak.where(
        eta <= 2.7,
        passTight & (muEF < 0.8) & (chEmEF < 0.8),
        passTight
    )

    return passTight, passTightLepVeto
