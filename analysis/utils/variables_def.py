import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector
import itertools

def make_vector(objs):
    return ak.zip({
        "pt": objs.pt,
        "eta": objs.eta,
        "phi": objs.phi,
        "mass": objs.mass
    }, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)

def make_regressed_vector(objs):
    return ak.zip({
        "pt": objs.regressed,
        "eta": objs.eta,
        "phi": objs.phi,
        "mass": objs.mass
    }, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)

def dr_bb_avg(bjets):
    output = []

    vec_bjets = make_vector(bjets)  # Vectorize the full jagged array

    for jets in vec_bjets:
        if len(jets) < 2:
            output.append(np.nan)
            continue

        drs = [j1.delta_r(j2) for j1, j2 in itertools.combinations(jets, 2)]
        avg_dr = np.mean(drs)
        output.append(avg_dr)

    return ak.Array(output)

def min_dm_bb_bb(bjets, all_jets=None, btag_name="btagUParTAK4B"):
    '''
    Computes minimum |m(bb) - m(bb)| from valid 2+2 b-jet combinations.
    Handles:
      - ≥4 b-jets: 3 unique 2+2 pairings
      - 3 b-jets + ≥1 untagged: select best untagged jet as 4th
      - 3 b-jets only: reuse jets to form fake pair
    '''
    output = []

    for jets_b, jets_all in zip(bjets, all_jets if all_jets is not None else bjets):
        jets_b = list(jets_b)
        jets_all = list(jets_all)
        combos = []

        if len(jets_b) >= 4:
            jets_use = jets_b[:4]
            combos = [
                ((jets_use[0], jets_use[1]), (jets_use[2], jets_use[3])),
                ((jets_use[0], jets_use[2]), (jets_use[1], jets_use[3])),
                ((jets_use[0], jets_use[3]), (jets_use[1], jets_use[2]))
            ]

        elif len(jets_b) == 3:
            # Try using best untagged
            untagged = [j for j in jets_all if all(j is not jb for jb in jets_b)]
            if len(untagged) >= 1:
                best_untagged = max(untagged, key=lambda j: getattr(j, btag_name, 0))
                jets_use = jets_b + [best_untagged]
                combos = [
                    ((jets_use[0], jets_use[1]), (jets_use[2], jets_use[3])),
                    ((jets_use[0], jets_use[2]), (jets_use[1], jets_use[3])),
                    ((jets_use[0], jets_use[3]), (jets_use[1], jets_use[2]))
                ]

            # Fallback: reuse jets to fake pairs
            if not combos:
                for i in range(3):
                    for j in range(i + 1, 3):
                        k = [x for x in range(3) if x != i and x != j][0]
                        bb1 = (jets_b[i], jets_b[j])
                        bb2a = (jets_b[i], jets_b[k])
                        bb2b = (jets_b[j], jets_b[k])
                        combos.append((bb1, bb2a))
                        combos.append((bb1, bb2b))

        else:
            output.append(np.nan)
            continue

        min_dm = float("inf")
        for (j1a, j1b), (j2a, j2b) in combos:
            m1 = (j1a + j1b).mass
            m2 = (j2a + j2b).mass
            dm = abs(m1 - m2)
            if dm < min_dm:
                min_dm = dm

        output.append(min_dm)

    return ak.Array(output)

def dr_bb_bb_avg(bjets, all_jets=None, btag_name="btagDeepFlavB"):
    '''
    Computes average ΔR between two bb pairs.
    Same pairing logic as min_dm_bb_bb.
    '''
    output = []

    for jets_b, jets_all in zip(bjets, all_jets if all_jets is not None else bjets):
        jets_b = list(jets_b)
        jets_all = list(jets_all)
        combos = []

        if len(jets_b) >= 4:
            jets_use = jets_b[:4]
            combos = [
                ((jets_use[0], jets_use[1]), (jets_use[2], jets_use[3])),
                ((jets_use[0], jets_use[2]), (jets_use[1], jets_use[3])),
                ((jets_use[0], jets_use[3]), (jets_use[1], jets_use[2]))
            ]

        elif len(jets_b) == 3:
            untagged = [j for j in jets_all if all(j is not jb for jb in jets_b)]
            if len(untagged) >= 1:
                best_untagged = max(untagged, key=lambda j: getattr(j, btag_name, 0))
                jets_use = jets_b + [best_untagged]
                combos = [
                    ((jets_use[0], jets_use[1]), (jets_use[2], jets_use[3])),
                    ((jets_use[0], jets_use[2]), (jets_use[1], jets_use[3])),
                    ((jets_use[0], jets_use[3]), (jets_use[1], jets_use[2]))
                ]

            if not combos:
                for i in range(3):
                    for j in range(i + 1, 3):
                        k = [x for x in range(3) if x != i and x != j][0]
                        bb1 = (jets_b[i], jets_b[j])
                        bb2a = (jets_b[i], jets_b[k])
                        bb2b = (jets_b[j], jets_b[k])
                        combos.append((bb1, bb2a))
                        combos.append((bb1, bb2b))

        else:
            output.append(np.nan)
            continue

        min_avg_dr = float("inf")
        for (j1a, j1b), (j2a, j2b) in combos:
            dr1 = j1a.delta_r(j1b)
            dr2 = j2a.delta_r(j2b)
            avg_dr = 0.5 * (dr1 + dr2)
            if avg_dr < min_avg_dr:
                min_avg_dr = avg_dr

        output.append(min_avg_dr)

    return ak.Array(output)

def dr_doubleb_bb(double_bjets, single_bjets):
    '''
    Computes ΔR between:
      - leading double-b jet and bb pair (if ≥2 single-b jets)
      - leading double-b jet and single-b jet (if only 1)
    '''
    output = []

    for dbs, sbs in zip(double_bjets, single_bjets):
        dbs = list(dbs)
        sbs = list(sbs)

        if len(dbs) == 0 or len(sbs) == 0:
            output.append(np.nan)
            continue

        db = dbs[0]

        if len(sbs) >= 2:
            bb = sbs[0] + sbs[1]
            dr = db.delta_r(bb)
        else:
            dr = db.delta_r(sbs[0])

        output.append(dr)

    return ak.Array(output)

def min_dm_doubleb_bb(double_bjets, single_bjets, all_jets=None, btag_name="btagDeepFlavB"):
    '''
    Computes |m(doubleb) - m(bb)| for:
      - 2 single-b jets
      - 1 single-b + best untagged
      - 1 single-b used twice
    '''
    output = []

    for dbs, sbs, jets_all in zip(double_bjets, single_bjets, all_jets if all_jets is not None else single_bjets):
        dbs = list(dbs)
        sbs = list(sbs)
        jets_all = list(jets_all)

        if len(dbs) == 0 or len(sbs) == 0:
            output.append(np.nan)
            continue

        db = dbs[0]  # leading double-b jet
        bb_combos = []

        if len(sbs) >= 2:
            bb_combos.append((sbs[0], sbs[1]))

        elif len(sbs) == 1:
            sb = sbs[0]
            untagged = [j for j in jets_all if j is not sb and j not in dbs]
            if len(untagged) >= 1:
                best = max(untagged, key=lambda j: getattr(j, btag_name, 0))
                bb_combos.append((sb, best))
            # fallback: reuse same jet
            bb_combos.append((sb, sb))

        else:
            output.append(np.nan)
            continue

        min_dm = float("inf")
        for b1, b2 in bb_combos:
            dm = abs((b1 + b2).mass - db.mass)
            if dm < min_dm:
                min_dm = dm

        output.append(min_dm)

    return ak.Array(output)

def higgs_kin(bjets, all_jets):
    '''
    Computes Higgs candidate 4-vector (mass, pt, phi) depending on:
        - Case 1: len(all_jets) == 3 -> use the 3 b-jets (if ≥3 available)
        - Case 2: len(all_jets) >= 4 -> use top 4 all_jets by pt
    Returns:
        Tuple of ak.Arrays: (mass, pt,ETA, phi)
    '''
    mass_list = []
    pt_list = []
    phi_list = []
    eta_list = []
    for bjs, jets in zip(bjets, all_jets):
        bjs = list(bjs)
        jets = list(jets)

        vec = None

        # Case 1: Exactly 3 jets, use 3 b-jets if available
        if len(jets) == 3 and len(bjs) == 3:
            vec = bjs[0] + bjs[1] + bjs[2]

        # Case 2: At least 4 jets, use top 4 by pt
        elif len(jets) >= 4:
            sorted_jets = sorted(jets, key=lambda j: j.pt, reverse=True)
            vec = sorted_jets[0] + sorted_jets[1] + sorted_jets[2] + sorted_jets[3]

        # Fallback: set dummy values (e.g. 0)
        if vec is None:
            mass_list.append(0.0)
            pt_list.append(0.0)
            phi_list.append(0.0)
            eta_list.append(0.0)
        else:
            mass_list.append(vec.mass)
            pt_list.append(vec.pt)
            phi_list.append(vec.phi)
            eta_list.append(vec.eta)
    return ak.Array(mass_list), ak.Array(pt_list), ak.Array(phi_list), ak.Array(eta_list)


def m_bbj(bjets, all_jets):
    '''
    Computes mbbj for each event in these cases:
      - 3 b-jets & 3 jets: invariant mass of the 3 b-jets
      - ≥4 b-jets & ≥4 jets & no untagged jets: 
            bb pair with min ΔR + 3rd b with lowest btag
      - if untagged jets exist: 
            bb pair with min ΔR + highest-pt untagged jet
      - fallback: 3 highest-pt b-jets
    Returns:
        ak.Array of masses with same length as input.
    '''
    out = []

    for bjs, jets in zip(bjets, all_jets):
        bjs = list(bjs)
        jets = list(jets)

        # Case 1: 3 b-jets and 3 jets
        if len(bjs) == 3 and len(jets) == 3:
            out.append((bjs[0] + bjs[1] + bjs[2]).mass)
            continue

        # Check untagged jets
        untagged = [j for j in jets if all(j is not bj for bj in bjs)]

        # Case 2: untagged jets exist
        if untagged:
            bb_pairs = list(itertools.combinations(bjs, 2))
            drs = [b1.delta_r(b2) for b1, b2 in bb_pairs]
            min_idx = drs.index(min(drs))
            b1, b2 = bb_pairs[min_idx]
            j = max(untagged, key=lambda jet: jet.pt)
            out.append((b1 + b2 + j).mass)
            continue

        # Case 3: ≥4 b-jets, no untagged jets
        if len(bjs) >= 4:
            bb_pairs = list(itertools.combinations(bjs, 2))
            drs = [b1.delta_r(b2) for b1, b2 in bb_pairs]
            min_idx = drs.index(min(drs))
            b1, b2 = bb_pairs[min_idx]
            # find a third distinct b-jet with lowest btag
            third_bs = [b for b in bjs if b is not b1 and b is not b2]
            if third_bs:
                third_b = min(third_bs, key=lambda b: b.btagUParTAK4B)
                out.append((b1 + b2 + third_b).mass)
                continue

        # Fallback: take 3 highest-pt b-jets
        top3_bjets = sorted(bjs, key=lambda b: b.pt, reverse=True)[:3]
        m = (top3_bjets[0] + top3_bjets[1] + top3_bjets[2]).mass
        out.append(m)

    return ak.Array(out)
