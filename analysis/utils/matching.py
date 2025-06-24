import numpy as np
import awkward as ak
import os
import vector
import hist
import coffea.util


def delta_r(obj1, obj2):
    deta = obj1.eta - obj2.eta
    dphi = np.abs(obj1.phi - obj2.phi)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)

def clean_by_dr(objects, others, drmin):
    pairs = ak.cartesian([objects, others], nested=True)
    dr = delta_r(pairs['0'], pairs['1'])
    mask = ak.all(dr > drmin, axis=-1)
    return objects[mask]

def extract_gen_bb_pairs(events):
    genparts = events.GenPart
    bb_pairs = []
    n_bquarks_list = []

    for iev in range(len(events)):
        parts = genparts[iev]
        a_indices = [i for i, p in enumerate(parts) if p.pdgId == 36]
        if len(a_indices) < 2:
            bb_pairs.append(([], []))
            n_bquarks_list.append(0)
            continue

        idx_A1, idx_A2 = a_indices[0], a_indices[1]

        is_qg = ((abs(parts.pdgId) >= 1) & (abs(parts.pdgId) <= 5)) | (abs(parts.pdgId) == 21)
        has_good_status = (parts.status == 1) | (parts.status == 23)
        qg_particles = parts[is_qg & has_good_status]

        b_quarks = qg_particles[(abs(qg_particles.pdgId) == 5)]
        b_quarks = b_quarks[(b_quarks.pt > 0) & (abs(b_quarks.eta) < 2.5)]
        

        groups = {}
        for b in b_quarks:
            mom_idx = b["genPartIdxMother"]
            if mom_idx not in groups:
                groups[mom_idx] = []
            groups[mom_idx].append(b)

        def to_vec(bq):
            return vector.obj(pt=bq["pt"], eta=bq["eta"], phi=bq["phi"], mass=bq["mass"])

        vec_genbb1_truth = [to_vec(b) for b in groups.get(idx_A1, [])[:2]]
        vec_genbb2_truth = [to_vec(b) for b in groups.get(idx_A2, [])[:2]]

        n_bquarks_list.append(len(groups.get(idx_A1, [])) + len(groups.get(idx_A2, [])))
        bb_pairs.append((vec_genbb1_truth, vec_genbb2_truth))

    return bb_pairs, n_bquarks_list

def is_jet_matched_to_bquark_pair(bb_pair, reco_jet, dr_threshold=0.8):
    if len(bb_pair) != 2:
        return False
    return all(reco_jet.deltaR(b) < dr_threshold for b in bb_pair)

def match_jets_to_single_qg(jets, genparts, dr_threshold=0.4):
    matched_mask = []
    for jets_event, gen_event in zip(jets, genparts):
        is_qg = ((abs(gen_event.pdgId) <= 5) | (gen_event.pdgId == 21))
        status_mask = (gen_event.status == 1) | (gen_event.status == 23)
        valid_partons = gen_event[is_qg & status_mask]

        matched_event_mask = []
        for jet in jets_event:
            n_matches = sum(jet.delta_r(p) < dr_threshold for p in valid_partons)
            matched_event_mask.append(n_matches == 1)  # Only exactly 1 match
        matched_mask.append(matched_event_mask)

    return ak.Array(matched_mask)
