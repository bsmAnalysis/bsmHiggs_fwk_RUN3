from coffea import processor
import awkward as ak
import numpy as np

from utils.deltas_array import clean_by_dr


class BTagEfficiencyProcessor(processor.ProcessorABC):
    #Measure per-jet efficiency versus pT, |eta|, flavour and truth-b count

    def __init__(self, wp, tagger="btagUParTAK4B", tt_flavor=None):
        self.wp = wp
        self.tagger = tagger
        self.tt_flavor = tt_flavor

        self.pt_edges = np.array(
            [20, 30, 50, 70, 100, 140, 200, 300, 600, 1000],
            dtype=np.float64,
        )
        self.eta_edges = np.array([0.0, 1.5, 2.5], dtype=np.float64)
        # Flavour indices: b=0, c=1, light=2.
        self.flav_edges = np.array([0, 1, 2, 3], dtype=np.float64)
        # Selected truth-b counts: 0, 1, 2, 3, >=4.
        self.nbjet_edges = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)

    def empty_output(self):
        """Return zero counts with the same binning as a populated result."""
        shape = (
            len(self.pt_edges) - 1,
            len(self.eta_edges) - 1,
            3,
            len(self.nbjet_edges) - 1,
        )
        return {
            "pt_edges": self.pt_edges,
            "eta_edges": self.eta_edges,
            "flav_edges": self.flav_edges,
            "nbjet_edges": self.nbjet_edges,
            "BTagEff_Denom": np.zeros(shape, dtype=np.int64),
            "BTagEff_Num": np.zeros(shape, dtype=np.int64),
        }

    def process(self, events):
        jets = events.Jet

        # Restrict ttbar events to the requested heavy-flavour category.
        if self.tt_flavor is not None:
            gen_id = events.genTtbarId
            if self.tt_flavor == "ttLF":
                evt_tt_mask = (gen_id % 100) < 41
            elif self.tt_flavor == "ttCC":
                evt_tt_mask = ((gen_id % 100) >= 41) & ((gen_id % 100) <= 45)
            elif self.tt_flavor == "ttBB":
                evt_tt_mask = ((gen_id % 100) >= 51) & ((gen_id % 100) <= 55)
            else:
                raise ValueError(f"Unknown tt_flavor: {self.tt_flavor}")
            events = events[evt_tt_mask]
            jets = events.Jet

        # Loose leptons define the objects used for jet cleaning.
        muons = events.Muon[
            (events.Muon.pt > 10)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.looseId > 0.5)
            & (events.Muon.pfRelIso04_all < 0.25)
        ]
        electrons = events.Electron[
            (events.Electron.pt > 10)
            & (np.abs(events.Electron.superclusterEta) < 2.5)
            & (
                (np.abs(events.Electron.superclusterEta) < 1.4442)
                | (np.abs(events.Electron.superclusterEta) > 1.566)
            )
            & (events.Electron.mvaIso_WP90 > 0.5)
        ]
        muons = ak.with_field(muons, "mu", "lepton_type")
        electrons = ak.with_field(electrons, "e", "lepton_type")
        leptons = ak.concatenate([muons, electrons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]

        # Select jets before removing overlaps with leptons.
        good_jets = (jets.pt > 20.0) & (np.abs(jets.eta) < 2.5)
        # Apply the skimmed jet ID when the driver includes this field.
        if hasattr(jets, "passJetIdTightLepVeto"):
            good_jets = good_jets & (jets.passJetIdTightLepVeto > 0.5)
        jets = jets[good_jets]
        jets = clean_by_dr(jets, leptons, 0.4)

        # Measure efficiencies in events with at least three cleaned jets.
        evt_mask = ak.num(jets) >= 3
        jets = jets[evt_mask]
        if ak.sum(ak.num(jets)) == 0:
            return self.empty_output()

        # Count truth-b jets after selection and use one shared >=4 category.
        nbtruthb = ak.sum(np.abs(jets.hadronFlavour) == 5, axis=1)
        nbtruthb_cat = ak.where(nbtruthb >= 4, 4, nbtruthb)
        # Each jet inherits the truth-b count of its event.
        nbtruthb_perjet = ak.broadcast_arrays(nbtruthb_cat, jets.pt)[0]

        # Flatten the selected jets for the NumPy count arrays.
        pt = ak.to_numpy(ak.flatten(jets.pt))
        eta = ak.to_numpy(np.abs(ak.flatten(jets.eta)))
        flav = ak.to_numpy(ak.flatten(jets.hadronFlavour))
        btag = ak.to_numpy(ak.flatten(jets[self.tagger]))
        nbjet = ak.to_numpy(ak.flatten(nbtruthb_perjet)).astype(np.int32)

        # Use the same b/c/light ordering as the output maps.
        flav_bin = np.full_like(flav, 2, dtype=np.int32)
        flav_bin[np.abs(flav) == 5] = 0
        flav_bin[np.abs(flav) == 4] = 1

        # Keep jets inside the map range; upper edges are exclusive.
        pt_idx = np.digitize(pt, self.pt_edges) - 1
        eta_idx = np.digitize(eta, self.eta_edges) - 1
        valid = (
            (pt_idx >= 0)
            & (pt_idx < len(self.pt_edges) - 1)
            & (eta_idx >= 0)
            & (eta_idx < len(self.eta_edges) - 1)
            & (nbjet >= 0)
            & (nbjet < len(self.nbjet_edges) - 1)
        )
        pt_idx = pt_idx[valid]
        eta_idx = eta_idx[valid]
        flav_bin = flav_bin[valid]
        nbjet = nbjet[valid]
        tagged = (btag[valid] >= self.wp)

        # A selected b jet implies at least one truth-b jet in the event.
        bad = (flav_bin == 0) & (nbjet == 0)
        if np.any(bad):
            print("[ERROR] Found b-flavor jets in nb0 category:", np.sum(bad))

        shape = (
            len(self.pt_edges) - 1,
            len(self.eta_edges) - 1,
            3,
            len(self.nbjet_edges) - 1,
        )
        # Array axes: pT, |eta|, flavour, selected truth-b count.
        denom = np.zeros(shape, dtype=np.int64)
        num = np.zeros(shape, dtype=np.int64)
        np.add.at(denom, (pt_idx, eta_idx, flav_bin, nbjet), 1)
        np.add.at(num, (pt_idx, eta_idx, flav_bin, nbjet), tagged.astype(np.int64))
        return {
            "pt_edges": self.pt_edges,
            "eta_edges": self.eta_edges,
            "flav_edges": self.flav_edges,
            "nbjet_edges": self.nbjet_edges,
            "BTagEff_Denom": denom,
            "BTagEff_Num": num,
        }

    def postprocess(self, accumulator):
        return accumulator
