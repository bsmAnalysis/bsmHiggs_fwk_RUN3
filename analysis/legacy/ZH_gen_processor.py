import numpy as np
import awkward as ak
import vector
from coffea import processor
from hist import Hist, axis
from coffea.analysis_tools import Weights

# Vector construction

def make_vector_old(jets):
    return vector.awkward.zip({
        "pt": jets.pt,
        "eta": jets.eta,
        "phi": jets.phi,
        "mass": jets.mass
    }, with_name="Momentum4D")

# ∆phi and ∆R helpers
def delta_phi(obj1, obj2):
    dphi = obj1.phi - obj2.phi
    return (dphi + np.pi) % (2 * np.pi) - np.pi

def delta_r(obj1, obj2):
    deta = obj1.eta - obj2.eta
    dphi = delta_phi(obj1, obj2)
    return np.sqrt(deta**2 + dphi**2)

def extract_gen_bb_pairs(genparts):
    all_vec_genbb1 = []
    all_vec_genbb2 = []

    for parts in genparts:
        a_indices = [i for i, p in enumerate(parts) if p.pdgId == 36]
        if len(a_indices) < 2:
            all_vec_genbb1.append([])
            all_vec_genbb2.append([])
            continue

        idx_A1, idx_A2 = a_indices[0], a_indices[1]
        b_quarks = parts[(abs(parts.pdgId) == 5) & (parts.status == 23)]

        groups = {}
        for b in b_quarks:
            mom_idx = b["genPartIdxMother"]
            groups.setdefault(mom_idx, []).append(b)

        def to_vec(bq):
            return vector.obj(pt=bq.pt, eta=bq.eta, phi=bq.phi, mass=bq.mass)

        bb1 = [to_vec(b) for b in groups.get(idx_A1, [])[:2]]
        bb2 = [to_vec(b) for b in groups.get(idx_A2, [])[:2]]

        all_vec_genbb1.append(bb1)
        all_vec_genbb2.append(bb2)

    return all_vec_genbb1, all_vec_genbb2

class TOTAL_Processor(processor.ProcessorABC):
    def __init__(self, xsec=0.89, nevts=3000, isMC=True, dataset_name=None, is_MVA=False, run_eval=True):
        self.xsec = xsec
        self.nevts = nevts
        self.isMC = isMC
        self.dataset_name = dataset_name
        self.isMC = isMC
        self.is_MVA= is_MVA
        self._trees = None
        self.run_eval= run_eval

        self._histograms = {}

        for i in range(1, 5):
            self._histograms[f"pT_gen:b{i}"] = Hist.new.Reg(50, 0, 300, name="pT").Double()
            self._histograms[f"eta_gen:b{i}"] = Hist.new.Reg(50, -5, 5, name="eta").Double()

        self._histograms["pt_gen:bb"] = Hist.new.Reg(50, 0, 300, name="pT").Double()
        self._histograms["eta_gen:bb"] = Hist.new.Reg(50, -5, 5, name="eta").Double()
        self._histograms["deltaR_gen:bb"] = Hist.new.Reg(60, 0, 6, name="deltaR").Double()

        self._histograms["mean_dR_vs_ptbb"] = Hist(
            axis.Regular(50, 0, 300, name="ptbb", label="pT(bb) [GeV]"),
            axis.Regular(80, 0, 8,name="dR", label="<ΔR>")
        )

        
        self._histograms["eff_denom_dRlt04_vs_ptbb"] = (
            Hist.new
            .Reg(50, 0, 300, name="ptbb", label="pT(bb) [GeV]")
            .Double()
)

        self._histograms["eff_num_dRlt04_vs_ptbb"] = (
            Hist.new
            .Reg(50, 0, 300, name="ptbb", label="pT(bb) [GeV]")
            .Double()
        )


        self._histograms["deltaR_gen:AA"] = Hist.new.Reg(80, 0, 8, name="deltaR").Double()
        self._histograms["deltaR_gen:ll"] = Hist.new.Reg(50, 0, 5, name="deltaR").Double()
        self._histograms["deltaR_gen:ZH"] = Hist.new.Reg(80, 0, 8, name="deltaR").Double()
        self._histograms["deltaPhi_gen:ZH"] = Hist.new.Reg(50, -np.pi, np.pi, name="deltaPhi").Double()
        self._histograms["deltaEta_gen:ZH"] = Hist.new.Reg(50, -5, 5, name="deltaEta").Double()

        self._histograms["pt_gen:ll"] = Hist.new.Reg(50, 0, 300, name="pT").Double()
        self._histograms["eta_gen:ll"] = Hist.new.Reg(50, -5, 5, name="eta").Double()
        self._histograms["pt_gen:nunu"] = Hist.new.Reg(50, 0, 300, name="pT").Double()

        self._histograms["eta_gen:higgs"] = Hist.new.Reg(50, -5, 5, name="eta").Double()
        self._histograms["pt_gen:higgs"] = Hist.new.Reg(500, 0, 500, name="pT").Double()
    def process(self, events):
        output = {key: hist.copy() for key, hist in self._histograms.items()}
        genparts = events.GenPart
        isHard = (((genparts.statusFlags >> 7) & 1) == 1) | (((genparts.statusFlags >> 8) & 1) == 1)
        genparts = genparts[isHard]

        genH = genparts[(genparts.pdgId == 25) & (genparts.status == 22)][:, 0]
        output["eta_gen:higgs"].fill(eta=make_vector_old(genH).eta)
        output["pt_gen:higgs"].fill(eta=make_vector_old(genH).pt)
        genA = genparts[genparts.pdgId == 36]
        if ak.num(genA) >= 2:
            output["deltaR_gen:AA"].fill(deltaR=delta_r(genA[:, 0], genA[:, 1]))

        bb1_list, bb2_list = extract_gen_bb_pairs(genparts)

        for bpair1, bpair2 in zip(bb1_list, bb2_list):
            for pair in [bpair1, bpair2]:
                if len(pair) == 2:
                    b1, b2 = pair
                    vec_bb = b1 + b2
                    pt_bb = vec_bb.pt
                    dR_bb = b1.deltaR(b2)
                    output["pt_gen:bb"].fill(pT=pt_bb)
                    output["eta_gen:bb"].fill(eta=vec_bb.eta)
                    output["deltaR_gen:bb"].fill(deltaR=dR_bb)
                    output["mean_dR_vs_ptbb"].fill(ptbb=pt_bb, dR=dR_bb)
                    output["eff_denom_dRlt04_vs_ptbb"].fill(ptbb=pt_bb)
                    if dR_bb < 0.4:
                        output["eff_num_dRlt04_vs_ptbb"].fill(ptbb=pt_bb)

            for idx, pair in enumerate([bpair1, bpair2], 1):
                if len(pair) == 2:
                    output[f"pT_gen:b{2*idx-1}"].fill(pT=pair[0].pt)
                    output[f"pT_gen:b{2*idx}"].fill(pT=pair[1].pt)
                    output[f"eta_gen:b{2*idx-1}"].fill(eta=pair[0].eta)
                    output[f"eta_gen:b{2*idx}"].fill(eta=pair[1].eta)

        isZ = genparts.pdgId == 23
        isLep = ((abs(genparts.pdgId) == 11) | (abs(genparts.pdgId) == 13)) & (genparts.status == 23)
        leptons = genparts[isLep & isZ[genparts.genPartIdxMother]]
        leptons = leptons[ak.num(leptons) == 2]
        if len(leptons) ==2:
            p4_leps = make_vector_old(leptons)
            ll = p4_leps[:, 0] + p4_leps[:, 1]
            output["pt_gen:ll"].fill(pT=ll.pt)
            output["eta_gen:ll"].fill(eta=ll.eta)
            output["deltaR_gen:ll"].fill(deltaR=delta_r(p4_leps[:, 0], p4_leps[:, 1]))
            output["deltaR_gen:ZH"].fill(deltaR=delta_r(ll, make_vector_old(genH)))
            output["deltaPhi_gen:ZH"].fill(deltaPhi=delta_phi(ll, make_vector_old(genH)))
            output["deltaEta_gen:ZH"].fill(deltaEta=ll.eta - make_vector_old(genH).eta)

        isNu = ((abs(genparts.pdgId) == 12) | (abs(genparts.pdgId) == 14) | (abs(genparts.pdgId) == 16)) & (genparts.status == 23)
        neutrinos = genparts[isNu & isZ[genparts.genPartIdxMother]]
        neutrinos = neutrinos[ak.num(neutrinos) == 2]
        if len(neutrinos) ==2:
            p4_nus = make_vector_old(neutrinos)
            p4_nunu = p4_nus[:, 0] + p4_nus[:, 1]
            output["pt_gen:nunu"].fill(pT=p4_nunu.pt)

        return output

    def postprocess(self, accumulator):
        return accumulator
