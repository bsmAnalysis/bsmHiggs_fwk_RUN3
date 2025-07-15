import numpy as np
import awkward as ak
import os
import json
import argparse
import vector
vector.register_awkward()
from coffea import processor
from coffea.analysis_tools import Weights
import hist
from hist import Hist
import coffea.util
import itertools

def min_dm_bb_bb(bjets, all_jets=None, btag_name="btagDeepFlavB"):
    '''
    Computes minimum |m(bb) - m(bb)| for valid 2+2 combinations.
    Handles:
      - ≥4 b-jets: all unique 2+2 pairings
      - 3 b-jets + ≥1 untagged: use best untagged jet as 4th
      - 3 b-jets: reuse one jet to create two pairs
    '''
    output = []

    for jets_b, jets_all in zip(bjets, all_jets if all_jets is not None else bjets):
        jets_b = list(jets_b)
        combos = []

        if len(jets_b) >= 4:
            jets_use = jets_b[:4]
            combos = [
                ((jets_use[0], jets_use[1]), (jets_use[2], jets_use[3])),
                ((jets_use[0], jets_use[2]), (jets_use[1], jets_use[3])),
                ((jets_use[0], jets_use[3]), (jets_use[1], jets_use[2]))
            ]

        elif len(jets_b) == 3:
            if len(jets_all) >= 4:
                untagged = [j for j in jets_all if j not in jets_b]
                if len(untagged) >= 1:
                    best_untagged = max(untagged, key=lambda j: getattr(j, btag_name, 0))
                    jets_use = jets_b + [best_untagged]
                    combos = [
                        ((jets_use[0], jets_use[1]), (jets_use[2], jets_use[3])),
                        ((jets_use[0], jets_use[2]), (jets_use[1], jets_use[3])),
                        ((jets_use[0], jets_use[3]), (jets_use[1], jets_use[2]))
                    ]

            if not combos:
                # Reuse logic
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
    Structure matches min_dm_bb_bb:
      - ≥4 b-jets: 3 unique (bb, bb) combinations
      - 3 b-jets + ≥1 untagged: use best untagged as 4th
      - 3 b-jets: reuse one jet to make two pairs
    Returns the average ΔR for the combination.
    '''
    output = []

    for jets_b, jets_all in zip(bjets, all_jets if all_jets is not None else bjets):
        jets_b = list(jets_b)
        combos = []

        if len(jets_b) >= 4:
            jets_use = jets_b[:4]
            combos = [
                ((jets_use[0], jets_use[1]), (jets_use[2], jets_use[3])),
                ((jets_use[0], jets_use[2]), (jets_use[1], jets_use[3])),
                ((jets_use[0], jets_use[3]), (jets_use[1], jets_use[2]))
            ]

        elif len(jets_b) == 3:
            if len(jets_all) >= 4:
                untagged = [j for j in jets_all if j not in jets_b]
                if len(untagged) >= 1:
                    best_untagged = max(untagged, key=lambda j: getattr(j, btag_name, 0))
                    jets_use = jets_b + [best_untagged]
                    combos = [
                        ((jets_use[0], jets_use[1]), (jets_use[2], jets_use[3])),
                        ((jets_use[0], jets_use[2]), (jets_use[1], jets_use[3])),
                        ((jets_use[0], jets_use[3]), (jets_use[1], jets_use[2]))
                    ]

            if not combos:
                # Reuse logic
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

        min_dr_avg = float("inf")
        for (j1a, j1b), (j2a, j2b) in combos:
            dr1 = j1a.deltaR(j1b)
            dr2 = j2a.deltaR(j2b)
            dr_avg = 0.5 * (dr1 + dr2)
            if dr_avg < min_dr_avg:
                min_dr_avg = dr_avg

        output.append(min_dr_avg)

    return ak.Array(output)
def min_dm_doubleb_bb(double_bjets, single_bjets, all_jets=None, btag_name="btagDeepFlavB"):
    '''
    Computes |m(doubleb) - m(bb)| using:
      - 2 single-b jets (best)
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

        db = dbs[0]  # use leading double-b jet
        bb_combos = []

        if len(sbs) >= 2:
            bb_combos.append((sbs[0], sbs[1]))

        elif len(sbs) == 1:
            sb = sbs[0]
            untagged = [j for j in jets_all if j not in sbs and j not in dbs]
            if len(untagged) >= 1:
                best_untagged = max(untagged, key=lambda j: getattr(j, btag_name, 0))
                bb_combos.append((sb, best_untagged))
            bb_combos.append((sb, sb))

        else:
            output.append(np.nan)
            continue

        min_dm = float("inf")
        for b1, b2 in bb_combos:
            m_bb = (b1 + b2).mass
            m_db = db.mass
            dm = abs(m_bb - m_db)
            if dm < min_dm:
                min_dm = dm

        output.append(min_dm)

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
            dr = db.deltaR(bb)
        else:
            dr = db.deltaR(sbs[0])

        output.append(dr)

    return ak.Array(output)

def m_bbj(bjets, all_jets):
    '''
    Computes m(b1 + b2 + j), where:
      - (b1, b2) is the bb pair with minimum ΔR among b-tagged jets
      - j is the highest-pt untagged jet
    '''
    output = []

    for bjs, jets in zip(bjets, all_jets):
        bjs = list(bjs)
        jets = list(jets)

        if len(bjs) < 2:
            output.append(np.nan)
            continue

        # All bb pairs and ΔRs
        bb_pairs = list(itertools.combinations(bjs, 2))
        drs = [b1.deltaR(b2) for b1, b2 in bb_pairs]

        if not drs:
            output.append(np.nan)
            continue

        min_idx = np.argmin(drs)
        b1, b2 = bb_pairs[min_idx]

        # Highest-pt untagged jet
        untagged = [j for j in jets if j not in bjs]
        if not untagged:
            output.append(np.nan)
            continue

        j = max(untagged, key=lambda jet: jet.pt)

        m = (b1 + b2 + j).mass
        output.append(m)

    return ak.Array(output)

