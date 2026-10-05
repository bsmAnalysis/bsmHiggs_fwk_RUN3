import awkward as ak
import hist
import numpy as np
from coffea import processor

DR_AXIS = hist.axis.Regular(
    160, 0.0, 8.0,
    name="dr_bb",
    label=r"$\Delta R(b,\bar{b})$",
    underflow=True,
    overflow=True,
)

PT_AXIS = hist.axis.Regular(
    50, 0.0, 500.0,
    name="pt_bb",
    label=r"$p_{\mathrm{T}}(b\bar{b})$ [GeV]",
    underflow=True,
    overflow=True,
)

B_PT_AXIS = hist.axis.Regular(
    20, 0.0, 200.0,
    name="pt_b",
    label=r"$p_{\mathrm{T}}(b)$ [GeV]",
    underflow=True,
    overflow=True,
)

HIGGS_PT_AXIS = hist.axis.Regular(
    60, 0.0, 600.0,
    name="pt_h",
    label=r"$p_{\mathrm{T}}(H_{4b})$ [GeV]",
    underflow=True,
    overflow=True,
)

HIGGS_ETA_AXIS = hist.axis.Regular(
    100, -5.0, 5.0,
    name="eta_h",
    label=r"$\eta(H_{4b})$",
    underflow=True,
    overflow=True,
)

DPHI_AXIS = hist.axis.Regular(
    64, 0.0, np.pi,
    name="abs_dphi_bb",
    label=r"$|\Delta\phi(b,\bar{b})|$",
    underflow=True,
    overflow=True,
)

DETA_AXIS = hist.axis.Regular(
    60, 0.0, 6.0,
    name="abs_deta_bb",
    label=r"$|\Delta\eta(b,\bar{b})|$",
    underflow=True,
    overflow=True,
)

def _weighted_hist(*axes):
    return hist.Hist(*axes, storage=hist.storage.Weight())


def _find_ancestor_index(genparts, start_index, abs_pdgid, max_depth=20):
    #Find the nearest ancestor with this PDG ID, returning -1 if absent
    alive = start_index >= 0
    index = ak.where(alive, start_index, 0)
    found_index = ak.full_like(start_index, -1)

    for _ in range(max_depth):
        ancestor = genparts[index]

        matched = alive & (abs(ancestor.pdgId) == abs(abs_pdgid))
        found_index = ak.where(matched, index, found_index)

        next_index = ancestor.genPartIdxMother
        alive = alive & (~matched) & (next_index >= 0)
        index = ak.where(alive, next_index, 0)

    return found_index


def _has_ancestor(genparts, start_index, abs_pdgid, max_depth=20):
    return _find_ancestor_index(
        genparts, start_index, abs_pdgid, max_depth=max_depth,
    ) >= 0


def _find_decay_a_index(genparts, b_mother_index, max_depth=20):
    #Follow b copies to their parent a, stop at any other particle.

    alive = b_mother_index >= 0
    index = ak.where(alive, b_mother_index, 0)
    a_index = ak.full_like(b_mother_index, -1)

    for _ in range(max_depth):
        parent = genparts[index]

        is_a = alive & (abs(parent.pdgId) == 36)
        a_index = ak.where(is_a, index, a_index)

        is_b_copy = alive & (abs(parent.pdgId) == 5)
        next_index = parent.genPartIdxMother

        alive = is_b_copy & (next_index >= 0)
        index = ak.where(alive, next_index, 0)

    return a_index


def _build_bb_pairs(genparts):
    #Build bb pairs from H -> aa, ordered by the parent-a GenPart index.

    flags = genparts.statusFlags

    is_hard = (
        (((flags >> 7) & 1) == 1)
        | (((flags >> 8) & 1) == 1)
    )

    is_last_copy = (((flags >> 13) & 1) == 1)

    # Select last-copy b quarks linked to the hard process.
    bquarks = genparts[(abs(genparts.pdgId) == 5) & is_hard & is_last_copy]

    # Find the parent a boson while allowing intermediate b-copy links.
    a_ancestor_index = _find_decay_a_index(genparts, bquarks.genPartIdxMother)

    bquarks = ak.with_field(bquarks, a_ancestor_index, 'aAncestorIdx')

    bquarks = bquarks[bquarks.aAncestorIdx >= 0]

    # Pair a b and bbar only when they share the same parent a.
    candidates = ak.combinations(bquarks, 2, fields=['b1', 'b2'])

    same_a_ancestor = candidates.b1.aAncestorIdx == candidates.b2.aAncestorIdx

    opposite_flavour = (candidates.b1.pdgId * candidates.b2.pdgId) == -25

    a_index = candidates.b1.aAncestorIdx
    parent_a = genparts[a_index]

    parent_is_a = abs(parent_a.pdgId) == 36

    # The a may have intermediate copies, but must descend from a Higgs.
    a_from_higgs = _has_ancestor(genparts, parent_a.genPartIdxMother, abs_pdgid=25)

    valid_pair = (
        same_a_ancestor
        & opposite_flavour
        & parent_is_a
        & a_from_higgs
    )

    pairs = candidates[valid_pair]
    parent_a = parent_a[valid_pair]
    a_index = a_index[valid_pair]

    deta = pairs.b1.eta - pairs.b2.eta

    dphi = (pairs.b1.phi - pairs.b2.phi + np.pi) % (2.0 * np.pi) - np.pi

    dr_bb = np.sqrt(deta * deta + dphi * dphi)

    # GenPart objects have Lorentz-vector behaviour under NanoAODSchema.
    bb_p4 = pairs.b1 + pairs.b2
    pt_bb = bb_p4.pt

    pairs = ak.with_field(pairs, dr_bb, "dr_bb")
    pairs = ak.with_field(pairs, np.abs(dphi), 'abs_dphi_bb')
    pairs = ak.with_field(pairs, np.abs(deta), 'abs_deta_bb')
    pairs = ak.with_field(pairs, pt_bb, "pt_bb")
    pairs = ak.with_field(pairs, parent_a.pt, "a_pt")
    pairs = ak.with_field(pairs, a_index, "a_index")

    # Deterministic ordering by parent-a GenPart index.
    return pairs[ak.argsort(pairs.a_index, axis=1, ascending=True)]


class GenHaa4bProcessor(processor.ProcessorABC):
    """Fill generator-level H -> aa -> 4b kinematics.

    Events have unit weight unless use_genweight enables the raw genWeight.
    Neither mode applies cross-section or luminosity normalization.
    """

    def __init__(self, use_genweight=False):
        self.use_genweight = use_genweight

    @staticmethod
    def _empty_output():
        return {
            # bb1 and bb2 follow the parent-a GenPart order.
            "dr_gen_bb1": _weighted_hist(DR_AXIS),
            "dr_gen_bb2": _weighted_hist(DR_AXIS),

            "pt_gen_bb1": _weighted_hist(PT_AXIS),
            "pt_gen_bb2": _weighted_hist(PT_AXIS),

            "dr_vs_pt_gen_bb1": _weighted_hist(PT_AXIS, DR_AXIS),
            "dr_vs_pt_gen_bb2": _weighted_hist(PT_AXIS, DR_AXIS),

            "abs_dphi_gen_bb1": _weighted_hist(DPHI_AXIS),
            "abs_dphi_gen_bb2": _weighted_hist(DPHI_AXIS),

            "abs_deta_gen_bb1": _weighted_hist(DETA_AXIS),
            "abs_deta_gen_bb2": _weighted_hist(DETA_AXIS),

            "abs_dphi_vs_abs_deta_gen_bb1": _weighted_hist(DETA_AXIS, DPHI_AXIS),
            "abs_dphi_vs_abs_deta_gen_bb2": _weighted_hist(DETA_AXIS, DPHI_AXIS),
            # Order bb1 and bb2 by their parent a's pT.
            "dr_gen_bb1_pt_sort": _weighted_hist(DR_AXIS),
            "dr_gen_bb2_pt_sort": _weighted_hist(DR_AXIS),

            "pt_gen_bb1_pt_sort": _weighted_hist(PT_AXIS),
            "pt_gen_bb2_pt_sort": _weighted_hist(PT_AXIS),

            "dr_vs_pt_gen_bb1_pt_sort": _weighted_hist(PT_AXIS, DR_AXIS),
            "dr_vs_pt_gen_bb2_pt_sort": _weighted_hist(PT_AXIS, DR_AXIS),

            "abs_dphi_gen_bb1_pt_sort": _weighted_hist(DPHI_AXIS),
            "abs_dphi_gen_bb2_pt_sort": _weighted_hist(DPHI_AXIS),

            "abs_deta_gen_bb1_pt_sort": _weighted_hist(DETA_AXIS),
            "abs_deta_gen_bb2_pt_sort": _weighted_hist(DETA_AXIS),

            "abs_dphi_vs_abs_deta_gen_bb1_pt_sort": _weighted_hist(DETA_AXIS, DPHI_AXIS),
            "abs_dphi_vs_abs_deta_gen_bb2_pt_sort": _weighted_hist(DETA_AXIS, DPHI_AXIS),

            # Individual b quarks: b1 has the highest pT, b4 the lowest.
            "pt_gen_b1_pt_sort": _weighted_hist(B_PT_AXIS),
            "pt_gen_b2_pt_sort": _weighted_hist(B_PT_AXIS),
            "pt_gen_b3_pt_sort": _weighted_hist(B_PT_AXIS),
            "pt_gen_b4_pt_sort": _weighted_hist(B_PT_AXIS),

            # Higgs candidate reconstructed from the four b quarks.
            "pt_gen_H_from_4b": _weighted_hist(HIGGS_PT_AXIS),
            "eta_gen_H_from_4b": _weighted_hist(HIGGS_ETA_AXIS),

            # Number of reconstructed a -> bb pairs per event.
            "n_valid_a_to_bb": _weighted_hist(
                hist.axis.Integer(
                    0,
                    3,
                    name="n_pairs",
                    label="number of valid a->bb pairs",
                    underflow=True,
                    overflow=True,
                )
            ),
        }

    def process(self, events):
        # Allow use with both eager and dask-backed NanoEvents.
        try:
            events = events.eager_compute_divisions()
        except Exception:
            pass

        try:
            n_events = len(events)
        except TypeError:
            events = events.compute()
            n_events = len(events)

        output = self._empty_output()

        if self.use_genweight:
            if "genWeight" not in events.fields:
                raise RuntimeError('--use-genweight requested, but genWeight is absent')

            event_weight = ak.values_astype(events.genWeight, np.float64)
        else:
            event_weight = np.ones(n_events, dtype=np.float64)

        pairs = _build_bb_pairs(events.GenPart)
        n_pairs = ak.num(pairs, axis=1)

        output["n_valid_a_to_bb"].fill(
            n_pairs=ak.to_numpy(n_pairs),
            weight=ak.to_numpy(event_weight),
        )

        # Fill the distrib of the two a->bb systems

        def fill_two_pairs(ordered_pairs, suffix=""):
            # Fill each available pair; missing pairs contribute no entries.
            two_pairs = ak.pad_none(ordered_pairs, 2, axis=1, clip=True)

            for i, label in enumerate(("bb1", "bb2")):
                pair = two_pairs[:, i]
                valid = ~ak.is_none(pair)

                dr = ak.to_numpy(pair[valid].dr_bb)
                pt = ak.to_numpy(pair[valid].pt_bb)
                abs_dphi = ak.to_numpy(pair[valid].abs_dphi_bb)
                abs_deta = ak.to_numpy(pair[valid].abs_deta_bb)
                weight = ak.to_numpy(event_weight[valid])

                output[f'dr_gen_{label}{suffix}'].fill(dr_bb=dr, weight=weight)

                output[f'pt_gen_{label}{suffix}'].fill(pt_bb=pt, weight=weight)

                output[f'dr_vs_pt_gen_{label}{suffix}'].fill(
                    pt_bb=pt,
                    dr_bb=dr,
                    weight=weight,
                )

                output[f'abs_dphi_gen_{label}{suffix}'].fill(
                    abs_dphi_bb=abs_dphi,
                    weight=weight,
                )

                output[f'abs_deta_gen_{label}{suffix}'].fill(
                    abs_deta_bb=abs_deta,
                    weight=weight,
                )

                output[f'abs_dphi_vs_abs_deta_gen_{label}{suffix}'].fill(
                    abs_deta_bb=abs_deta,
                    abs_dphi_bb=abs_dphi,
                    weight=weight,
                )

        # First fill pairs in their parent-a GenPart order.
        fill_two_pairs(pairs)

        # Then fill bb1 and bb2 ordered by the parent-a pT.
        pairs_pt_sorted = pairs[ak.argsort(pairs.a_pt, axis=1, ascending=False)]

        fill_two_pairs(pairs_pt_sorted, suffix='_pt_sort')

        # Four individual b quarks and the reconstructed Higgs candidate.

        # Use events with exactly two valid pairs for the four-b observables.
        has_exactly_two_pairs = n_pairs == 2

        selected_pairs = pairs[has_exactly_two_pairs]
        selected_weight = event_weight[has_exactly_two_pairs]

        pair1 = selected_pairs[:, 0]
        pair2 = selected_pairs[:, 1]

        # Collect the four b quarks in each event.
        four_bquarks = ak.concatenate(
            [
                pair1.b1[:, np.newaxis],
                pair1.b2[:, np.newaxis],
                pair2.b1[:, np.newaxis],
                pair2.b2[:, np.newaxis],
            ],
            axis=1,
        )

        # Sort all four b quarks independently by decreasing individual pT.
        four_bquarks = four_bquarks[ak.argsort(four_bquarks.pt, axis=1, ascending=False)]

        selected_weight_numpy = ak.to_numpy(selected_weight)

        # b1 is the highest-pT b quark; b4 is the lowest-pT b quark.
        for i, label in enumerate(("b1", "b2", "b3", "b4")):
            output[f"pt_gen_{label}_pt_sort"].fill(
                pt_b=ak.to_numpy(four_bquarks[:, i].pt),
                weight=selected_weight_numpy,
            )

        # Reconstruct the Higgs four-vector from the same four b quarks.
        higgs_from_4b = (
            four_bquarks[:, 0]
            + four_bquarks[:, 1]
            + four_bquarks[:, 2]
            + four_bquarks[:, 3]
        )

        output["pt_gen_H_from_4b"].fill(
            pt_h=ak.to_numpy(higgs_from_4b.pt),
            weight=selected_weight_numpy,
        )

        output["eta_gen_H_from_4b"].fill(
            eta_h=ak.to_numpy(higgs_from_4b.eta),
            weight=selected_weight_numpy,
        )

        return output

    def postprocess(self, accumulator):
        return accumulator
