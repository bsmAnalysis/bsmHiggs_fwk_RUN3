from coffea import processor

from coffea.lumi_tools import LumiMask

import correctionlib

import awkward as ak

import numpy as np

import fnmatch

import os

import copy

import hashlib

import re

from typing import Optional, Tuple





JET_PATHS = (

    ("", "AK4PFPuppi", None),

    ("_upart", "AK4PFPuppiUParTRegression", "UParTAK4RegPtRawCorr"),

    ("_upartnu", "AK4PFPuppiUParTRegressionPlusNeutrino", "UParTAK4RegPtRawCorrNeutrino"),

)

JET_VARIATIONS = ("jesUp", "jesDown", "jerUp", "jerDown")

COMPACT_JET_FIELDS = tuple(

    f"{quantity}{suffix}_jec" for suffix, _, _ in JET_PATHS for quantity in ("pt", "mass")

) + ("genJetPtForJER", "genJetEtaForJER", "genJetPhiForJER")

MET_RECONSTRUCTION_FIELDS = ("pt_full", "pt_l1", "eta", "phi")

JET_RECIPE_VERSION = 2

# The 27 individual sources in the JERC full-source table linked by the user:

# https://docs.google.com/spreadsheets/d/1JZfk78_9SD225bcUuTWVo4i02vwI5FfeVKH-dwzUdhM/edit#gid=1345121349

# Do not select every non-Regrouped key: payloads also contain totals/subtotals

# and alternative decompositions, which must not become independent nuisances.

FULL_JES_SOURCES = (

    "AbsoluteMPFBias", "AbsoluteScale", "AbsoluteStat", "FlavorQCD", "Fragmentation",

    "PileUpDataMC", "PileUpPtBB", "PileUpPtEC1", "PileUpPtEC2", "PileUpPtHF", "PileUpPtRef",

    "RelativeFSR", "RelativeJEREC1", "RelativeJEREC2", "RelativeJERHF", "RelativePtBB",

    "RelativePtEC1", "RelativePtEC2", "RelativePtHF", "RelativeBal", "RelativeSample",

    "RelativeStatEC", "RelativeStatFSR", "RelativeStatHF", "SinglePionECAL", "SinglePionHCAL",

    "TimePtEta",

)





# ------------------------------- helpers -------------------------------- #



def _unflatten_like(flat, counts):

    return ak.unflatten(ak.Array(flat), counts)





def file_sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as stream:

        for chunk in iter(lambda: stream.read(1024 * 1024), b""):

            digest.update(chunk)

    return digest.hexdigest()





def _clip(x, lo, hi):

    return np.minimum(np.maximum(x, lo), hi)





def _as_np_flat(x):

    # Flatten jagged [event, obj] -> 1D numpy

    return ak.to_numpy(ak.flatten(x, axis=1))





def _phi_mpi_pi(dphi):

    return np.arctan2(np.sin(dphi), np.cos(dphi))





def _print_counts(label, mask):

    n = int(ak.count_nonzero(mask))

    print(f"[COUNT] {label}: {n} / {len(mask)}")





def _safe_get(obj, name, default):

    return getattr(obj, name) if hasattr(obj, name) else default





def _raw_ak4_kinematics(jets):

    # Only call on ORIGINAL NanoAOD jets, whose pt/mass match rawFactor.

    # After replacing pt with a newly corrected value, this formula no longer

    # recovers raw pT. Retain pt_raw separately throughout the correction chain.

    if not hasattr(jets, "rawFactor"):

        raise RuntimeError("Jet.rawFactor is required to recover the original raw four-vector")

    raw_factor = ak.fill_none(jets.rawFactor, 0.0)

    return jets.pt * (1.0 - raw_factor), jets.mass * (1.0 - raw_factor)





def _hist_counts(values_1d: np.ndarray, edges: np.ndarray) -> np.ndarray:

    c, _ = np.histogram(values_1d, bins=edges)

    return c.astype(np.int64)





def toppt_run2(pt):

    x = _clip(pt, 0.0, 2000.0)

    return 0.103 * np.exp(-0.0118 * x) - 0.000134 * x + 0.973





def toppt_extrap_13p6_over_13(pt):

    x = _clip(pt, 0.0, 2000.0)

    return 0.991 + 0.000075 * x





def toppt_run3(pt):

    return toppt_run2(pt) * toppt_extrap_13p6_over_13(pt)





def cms_event_id_u64(events, like=None):

    """

    Stable positive 31-bit JERSmear ID mixed from the full (run,lumi,event).



    Keep the historical helper name for compatibility. The old bit-packed

    uint64 could overflow correctionlib's scalar integer input; it also

    overlapped run/lumi bits and truncated event at 32 bits. Mix first, then

    reduce to the tutorial's positive-int32 domain. This is a random seed,

    not a unique event identifier. Stored event/run/lumi are unchanged.

    """

    def mix64(values):

        values = values + np.uint64(0x9E3779B97F4A7C15)

        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)

        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)

        return values ^ (values >> np.uint64(31))



    run = np.asarray(ak.to_numpy(events.run), dtype=np.uint64)

    lumi = np.asarray(ak.to_numpy(events.luminosityBlock), dtype=np.uint64)

    evt = np.asarray(ak.to_numpy(events.event), dtype=np.uint64)

    mixed = mix64(evt ^ mix64(run) ^ mix64(lumi ^ np.uint64(0xD1B54A32D192ED03)))

    key = ak.Array((mixed & np.uint64(0x7FFFFFFF)).astype(np.int64))

    if like is None:

        return key

    key_b, _ = ak.broadcast_arrays(key, like)

    return ak.values_astype(key_b, np.int64)





def _print_jet_pt_triplet(tag, jets, pt_raw, pt_full_jec, pt_full_jecjer, ev_idx=0, max_jets=8):

    nevt = len(jets)

    if nevt == 0:

        print(f"[DBG] {tag}: no events")

        return



    ev_idx = int(min(max(ev_idx, 0), nevt - 1))



    pt_nano_evt = jets.pt[ev_idx]

    pt_raw_evt  = pt_raw[ev_idx]

    pt_jec_evt  = pt_full_jec[ev_idx]

    pt_jjer_evt = pt_full_jecjer[ev_idx]



    nj = int(len(pt_nano_evt))

    take = min(nj, int(max_jets))



    print(f"\n[DBG] {tag} event_idx={ev_idx} njets={nj} (showing {take})")

    if take == 0:

        return



    nano = np.asarray(ak.to_numpy(pt_nano_evt[:take]), dtype=float)

    raw  = np.asarray(ak.to_numpy(pt_raw_evt[:take]),  dtype=float)

    jec  = np.asarray(ak.to_numpy(pt_jec_evt[:take]),  dtype=float)

    jjer = np.asarray(ak.to_numpy(pt_jjer_evt[:take]), dtype=float)



    print("   i     pt_raw       pt_nano      pt_jec      pt_jecjer   (jec/raw)  (jjer/jec)")

    for i in range(take):

        r = raw[i]

        j = jec[i]

        jj = jjer[i]

        print(

            f"{i:4d}"

            f" {r:11.3f}"

            f" {nano[i]:11.3f}"

            f" {j:11.3f}"

            f" {jj:11.3f}"

            f" {((j/r) if r>0 else 0.0):11.4f}"

            f" {((jj/j) if j>0 else 0.0):11.4f}"

        )





# ------------------------------- Processor -------------------------------- #



class NanoAODSkimmerAK4(processor.ProcessorABC):

    """

    Nominal-only standard/UParT-visible/UParT-plus-neutrino AK4 skim.



    Defaults: fixed central tight-lepton-veto candidates with raw pT>15 GeV,

    >=2 candidates, and nominal JER on MC jets after their dedicated JEC.

    Data receive JEC only. JES/JER variations are not calculated or saved.

    Nine Jet reconstruction fields plus event rho/identity reproduce later

    JES/JER with the same payloads and deterministic seed. Invalid regression

    momenta stay -1; mass scales with the same regression/JEC/JER factors.

    Nominal/compact storage omits the full GenJet output by default. Each
    saved reconstructed jet keeps its original NanoAOD generator candidate
    (pt/eta/phi), BEFORE the JER matching tests. Analysis must repeat those
    tests for each JES/JER input using the cached candidate. Input GenJet and
    Jet.genJetIdx remain available to the skimmer's correction calculations.
    Set save_full_genjets=True to retain configured GenJet/genJetIdx outputs.



    Golden JSON(data), PV, MET filters, and the configured trigger OR define

    common event requirements. A fixed event veto uses ALL original AK4 jets

    with nominal standard JEC and its prescribed minimal selection. It is

    independent of the saved central candidate selection.



    JEC-only Type-1 PUPPI MET uses complete Jet+CorrT1METJet inputs. Save

    METCorrectionJet(pt_full,pt_l1,eta,phi) for every statically eligible

    no-muon contributor BEFORE the corrected 15 GeV cut, plus raw MET px/py.

    No regression or JER is propagated to MET. The analysis can reconstruct

    JES and repeat threshold membership, pT/phi, selections and observables.



    Legacy full/compact storage and standard/union pT skims remain available

    for comparisons. Nominal storage requires fixed selection; legacy compact

    requires its union envelope. See COMPACT_README.txt for both schemas.

    Input normalization counters count events before ANY selection.

    """



    JET_PT_MIN = 15.0  # final corrected jet pT, no muon subtraction

    JET_RAW_PT_MIN = 15.0  # original raw pT; fixed under JES/JER variations

    JET_ETA_MAX = 2.5  # fixed saved-candidate requirement

    MIN_SELECTED_JETS = 2

    MET_JET_PT_MIN = 15.0  # corrected, muon-subtracted pT

    # Preserve the supplied processor/tutorial cap. The MET slides do not

    # specify this cap; CMSSW's generic default is 9.9. Check forward-jet

    # closure before changing this analysis-level setting.

    MET_JET_ETA_MAX = 5.2

    JEC_TAG = "Summer24Prompt24_V5"

    JER_TAG = "Summer24Prompt24_JRV2_MC"

    def __init__(

        self,

        branches_to_keep: dict,

        trigger_groups: dict,

        met_filter_flags: list,

        dataset_name: Optional[str] = None,

        corrections_dir: Optional[str] = None,

        golden_json_dir: Optional[str] = None,

        do_jer: bool = True,          # MC output jets include nominal JER when available

        debug: bool = False,

        debug_event_index: int = 10,

        require_trigger: bool = True,

        jet_selection_mode: str = "fixed",  # "fixed", "standard" or "union"

        met_jes_selection: str = "varied",     # "nominal" or "varied"

        jes_scheme: str = "reduced",          # "reduced", "full" or "total"

        jet_systematics_storage: str = "nominal",  # "nominal", "full" or "compact"

        save_full_genjets: bool = False,  # nominal/compact: use per-reco candidate cache

    ):

        self.branches_to_keep = copy.deepcopy(branches_to_keep)

        self.trigger_groups = trigger_groups

        self.met_filter_flags = met_filter_flags

        self.dataset_name = dataset_name



        self.do_jer = bool(do_jer)

        self.debug = bool(debug)

        self.debug_event_index = int(debug_event_index)

        # Set False only when trigger decisions should be recorded without a cut.

        # Existing drivers need no change: the requested trigger cut is on by default.

        self.require_trigger = bool(require_trigger)

        if jet_selection_mode not in ("fixed", "standard", "union"):

            raise ValueError("jet_selection_mode must be 'fixed', 'standard' or 'union'")

        self.jet_selection_mode = jet_selection_mode

        if met_jes_selection not in ("varied", "nominal"):

            raise ValueError("met_jes_selection must be 'varied' or 'nominal'")

        self.met_jes_selection = met_jes_selection

        if jes_scheme not in ("reduced", "full", "total"):

            raise ValueError("jes_scheme must be 'reduced', 'full' or 'total'")

        self.jes_scheme = jes_scheme

        if jet_systematics_storage not in ("nominal", "full", "compact"):

            raise ValueError("jet_systematics_storage must be 'nominal', 'full' or 'compact'")

        if jet_systematics_storage == "nominal" and jet_selection_mode != "fixed":

            raise ValueError("Nominal-only storage requires fixed selection without a corrected-pT cut")

        if jet_systematics_storage == "compact" and jet_selection_mode != "union":

            raise ValueError("Compact storage requires jet_selection_mode='union' to preserve migrations")

        self.jet_systematics_storage = jet_systematics_storage

        self.save_full_genjets = bool(save_full_genjets)



        self.corrections_dir = corrections_dir or os.path.join(os.path.dirname(__file__), "corrections")

        self.golden_json_dir = golden_json_dir or os.path.join(os.path.dirname(__file__), "golden_json")



        # Load the certified run/lumisection list once; process() applies it to data.

        # MC does not use the lumi mask; data requires it in process().

        self._lumi_mask = None

        golden_json_path = os.path.join(

            self.golden_json_dir,

            "Cert_Collisions2024_378981_386951_Golden.json",

        )

        if os.path.exists(golden_json_path):

            print(f"[INIT] Loading golden JSON from {golden_json_path}")

            self._lumi_mask = LumiMask(golden_json_path)

        else:

            print(f"[INIT] golden_json_path='{golden_json_path}' absent; data will fail the required lumi-mask check")



        # Jet veto map

        self._loaded_veto = False

        self._jet_veto = None

        self._jet_veto_key = "Summer24Prompt24_RunBCDEFGHI_V1"

        self._jet_veto_type = "jetvetomap"

        self._veto_sha256 = None



        # JERC

        self._loaded_jerc = False

        self._cset_jerc = None



        # Full factorized JEC chains and matching JER/JES evaluators by algorithm.

        self._jec = {}

        self._jer_resolution = {}

        self._jer_sf_nominal = {}

        self._jer_sf_uncertainty = {}

        self._jes_total = {}

        self._jes_corrections = {}

        self._analysis_jes_sources = []

        self._jerc_sha256 = None



        # Standard L1 is also needed by Type-1 MET.

        self._cL1_data = None

        self._cL1_mc = None



        # JER smearing tool from jer_smear.json.gz (jets only)

        self._loaded_jersmear = False

        self._cJerSmear = None

        self._jersmear_sha256 = None



    # ---------------- loaders ---------------- #



    def _ensure_veto_loaded(self):

        if self._loaded_veto:

            return

        path = os.path.join(self.corrections_dir, "jetvetomaps.json.gz")

        if not os.path.exists(path):

            raise FileNotFoundError(f"Required jet-veto payload is missing: {path}")

        cset = correctionlib.CorrectionSet.from_file(path)

        if self._jet_veto_key not in set(cset):

            raise KeyError(f"Jet-veto key {self._jet_veto_key!r} is absent from {path}")

        self._jet_veto = cset[self._jet_veto_key]

        self._veto_sha256 = file_sha256(path)

        print(f"[Skim:JetVeto] Loaded '{self._jet_veto_key}' from {path}")

        self._loaded_veto = True



    def _ensure_jerc_loaded(self):

        if self._loaded_jerc:

            return

        path = os.path.join(self.corrections_dir, "regJet_jerc.json.gz")

        if not os.path.exists(path):

            raise FileNotFoundError(f"Required 2024 JERC payload not found: {path}")

        cset = correctionlib.CorrectionSet.from_file(path)

        available_keys = set(cset)



        def check_inputs(correction, names):

            actual = [(i.name, i.type) for i in correction.inputs]

            expected = [(name, "real") for name in names]

            if actual != expected:

                raise RuntimeError(f"Unexpected inputs for {correction.name}: {actual}; expected {expected}")



        for _, algo, _ in JET_PATHS:

            self._jec[algo] = {}

            for sample in ("MC", "DATA"):

                key = f"{self.JEC_TAG}_{sample}_L1L2L3Res_{algo}"

                corr = cset.compound[key]

                names = ["JetA", "JetEta", "JetPt", "Rho"]

                if algo == "AK4PFPuppi":

                    names.append("JetPhi")

                if sample == "DATA":

                    names.append("run")

                check_inputs(corr, names)

                self._jec[algo][sample] = corr

            for component, store, names in (

                ("PtResolution", self._jer_resolution, ("JetEta", "JetPt", "Rho")),

                ("ScaleFactor", self._jer_sf_nominal, ("JetEta", "JetPt")),

                ("SFUncertainty", self._jer_sf_uncertainty, ("JetEta", "JetPt")),

            ):

                corr = cset[f"{self.JER_TAG}_{component}_{algo}"]

                check_inputs(corr, names)

                store[algo] = corr

            corr = cset[f"{self.JEC_TAG}_MC_Total_{algo}"]

            check_inputs(corr, ("JetEta", "JetPt"))

            self._jes_total[algo] = corr



        self._cL1_data = cset[f"{self.JEC_TAG}_DATA_L1FastJet_AK4PFPuppi"]

        self._cL1_mc = cset[f"{self.JEC_TAG}_MC_L1FastJet_AK4PFPuppi"]

        self._cset_jerc = cset

        check_inputs(self._cL1_data, ("JetA", "JetEta", "JetPt", "Rho"))

        check_inputs(self._cL1_mc, ("JetA", "JetEta", "JetPt", "Rho"))

        self._jes_corrections = {"Total": self._jes_total}

        self._analysis_jes_sources = ["Total"]

        if self.jes_scheme == "reduced":

            prefix = f"{self.JEC_TAG}_MC_Regrouped_"

            by_algo = {}

            for _, algo, _ in JET_PATHS:

                ending = "_" + algo

                sources = {}

                for key in cset:

                    if key.startswith(prefix) and key.endswith(ending):

                        # Keep the official source spelling, including year.

                        source = key[len(f"{self.JEC_TAG}_MC_"):-len(ending)]

                        if source == "Regrouped_Total":

                            continue  # a total is not an individual source

                        if not re.fullmatch(r"[A-Za-z0-9_]+", source):

                            raise RuntimeError(f"Unsupported JES source name {source!r}")

                        check_inputs(cset[key], ("JetEta", "JetPt"))

                        sources[source] = cset[key]

                if not sources:

                    raise RuntimeError(

                        f"Reduced JES requested, but no individual {prefix}*_{algo} "

                        "sources are available. Regrouped_Total alone is insufficient. "

                        "Use a payload containing the reduced sources, or explicitly "

                        "select --jes-scheme total for the Total-only workflow."

                    )

                by_algo[algo] = sources

            reference = set(by_algo["AK4PFPuppi"])

            if len(reference) != 11:

                raise RuntimeError(

                    f"Reduced JES expects 11 individual sources, found {len(reference)}: "

                    f"{sorted(reference)}. Check the campaign's recommended reduced set; "

                    "do not silently combine alternative decompositions."

                )

            for algo, sources in by_algo.items():

                if set(sources) != reference:

                    raise RuntimeError(

                        f"Reduced JES source set differs for {algo}: "

                        f"missing={sorted(reference-set(sources))}, "

                        f"extra={sorted(set(sources)-reference)}"

                    )

            self._analysis_jes_sources = sorted(reference)

            for source in self._analysis_jes_sources:

                self._jes_corrections[source] = {

                    algo: sources[source] for algo, sources in by_algo.items()

                }

        elif self.jes_scheme == "full":

            self._analysis_jes_sources = list(FULL_JES_SOURCES)

            for source in self._analysis_jes_sources:

                self._jes_corrections[source] = {}

                for _, algo, _ in JET_PATHS:

                    key = f"{self.JEC_TAG}_MC_{source}_{algo}"

                    if key not in available_keys:

                        raise RuntimeError(f"Full JES requested, but required source {key} is missing")

                    check_inputs(cset[key], ("JetEta", "JetPt"))

                    self._jes_corrections[source][algo] = cset[key]

        self._extend_source_output_fields()

        self._jerc_sha256 = file_sha256(path)

        self._loaded_jerc = True

        print(f"[Skim:JERC] Loaded {self.JEC_TAG} / {self.JER_TAG} for all three jet definitions from {path}")

        print(f"[Skim:JERC] SHA256: {self._jerc_sha256}")

        print(f"[Skim:JES] scheme={self.jes_scheme}; analysis sources={self._analysis_jes_sources}")



    def _jes_variation_specs(self):

        """(branch stage, payload source, sign), with Total kept for monitoring."""

        specs = [("jesUp", "Total", 1.0), ("jesDown", "Total", -1.0)]

        for source in self._analysis_jes_sources:

            if source != "Total":

                specs.extend((

                    ("jes" + source + "Up", source, 1.0),

                    ("jes" + source + "Down", source, -1.0),

                ))

        return specs



    def _jet_variation_labels(self):

        return tuple(stage for stage, _, _ in self._jes_variation_specs()) + ("jerUp", "jerDown")



    def _extend_source_output_fields(self):

        """Append only source-specific output kinematics to local config copies."""

        # This dictionary controls OUTPUT only. Original events.GenJet and
        # Jet.genJetIdx still supply the candidate used during skimming.
        # The per-reco cache below replaces the need to save that collection.
        if (self.jet_systematics_storage in ("nominal", "compact")
                and not self.save_full_genjets):
            self.branches_to_keep.pop("GenJet", None)
            self.branches_to_keep["Jet"] = [
                field for field in self.branches_to_keep.get("Jet", [])
                if field != "genJetIdx"
            ]

        if self.jet_systematics_storage == "nominal":

            # Also sanitize an old skim_config.py: removing branches alone

            # would not be enough, but this mode additionally skips their

            # calculations below. Wildcards could leak input NanoAOD shifts.

            for obj in ("Jet", "PuppiMET"):

                fields = self.branches_to_keep.get(obj, [])

                if any("*" in field for field in fields):

                    raise ValueError("Nominal-only Jet/PuppiMET config must use explicit field names")

                self.branches_to_keep[obj] = [

                    field for field in fields if not re.search(r"_(jes|jer)", field, re.IGNORECASE)

                ]

            # All saved fixed candidates pass this ID; no redundant ID flag.

            self.branches_to_keep["Jet"] = [

                field for field in self.branches_to_keep["Jet"]

                if field != "passJetIdTightLepVeto"

            ]

            for field in COMPACT_JET_FIELDS:

                if field not in self.branches_to_keep["Jet"]:

                    self.branches_to_keep["Jet"].append(field)

            self.branches_to_keep.setdefault("Rho", [])

            if "fixedGridRhoFastjetAll" not in self.branches_to_keep["Rho"]:

                self.branches_to_keep["Rho"].append("fixedGridRhoFastjetAll")

            return

        # Legacy comparison modes should work with the new nominal config too.

        for obj, fields in (

            ("Jet", [f"{q}{suffix}_{stage}" for suffix, _, _ in JET_PATHS

                     for stage in JET_VARIATIONS for q in ("pt", "mass")]),

            ("PuppiMET", [f"{q}_{stage}" for stage in ("jesUp", "jesDown") for q in ("pt", "phi")]),

        ):

            selected = self.branches_to_keep.get(obj, [])

            selected.extend(field for field in fields if field not in selected)

        stages = [stage for stage, source, _ in self._jes_variation_specs() if source != "Total"]

        individual_jet_fields = [f"{quantity}{suffix}_{stage}" for suffix, _, _ in JET_PATHS

                                 for stage in stages for quantity in ("pt", "mass")]

        if self.jet_systematics_storage == "compact":

            # Calculate all source branches in memory for the skim envelope,

            # but write only the inputs needed to reproduce them in analysis.

            self.branches_to_keep["Jet"] = [field for field in self.branches_to_keep.get("Jet", [])

                                            if field not in individual_jet_fields]

            jet_fields = list(COMPACT_JET_FIELDS)

            self.branches_to_keep.setdefault("Rho", [])

            if "fixedGridRhoFastjetAll" not in self.branches_to_keep["Rho"]:

                self.branches_to_keep["Rho"].append("fixedGridRhoFastjetAll")

        else:

            jet_fields = individual_jet_fields

        for obj, fields in (

            ("Jet", jet_fields),

            ("PuppiMET", [f"{quantity}_{stage}" for stage in stages for quantity in ("pt", "phi")]),

        ):

            selected = self.branches_to_keep.get(obj)

            if selected:

                selected.extend(field for field in fields if field not in selected)



    def _systematics_metadata(self):

        """Record exact keys/branch names; this does not set fit correlations."""

        nominal_only = self.jet_systematics_storage == "nominal"

        cached = self.jet_systematics_storage in ("nominal", "compact")

        return {

            "jec_tag": self.JEC_TAG, "jer_tag": self.JER_TAG,

            "jerc_sha256": self._jerc_sha256, "jes_scheme": self.jes_scheme,

            "jersmear_sha256": self._jersmear_sha256,

            "jet_systematics_storage": self.jet_systematics_storage,

            "jet_reconstruction_schema": 2 if nominal_only else (1 if cached else None),

            "jet_recipe_version": JET_RECIPE_VERSION,

            "jet_reconstruction_fields": list(COMPACT_JET_FIELDS) if cached else [],

            "full_genjet_collection_saved": "GenJet" in self.branches_to_keep,

            "jer_generator_reference": (
                "per_reco_NanoAOD_candidate_before_matching" if cached
                else "production_NanoAOD_GenJet_genJetIdx"
            ),

            "jet_selection_mode": self.jet_selection_mode,

            "jet_pt_min": None if self.jet_selection_mode == "fixed" else self.JET_PT_MIN,

            "jet_raw_pt_min": self.JET_RAW_PT_MIN if self.jet_selection_mode == "fixed" else None,

            "jet_eta_max": self.JET_ETA_MAX if self.jet_selection_mode == "fixed" else None,

            "saved_jets_pass_tight_lepton_veto": self.jet_selection_mode == "fixed",

            "min_selected_jets": self.MIN_SELECTED_JETS,

            "analysis_jes_sources": self._analysis_jes_sources,

            "production_variations_computed": not nominal_only,

            "total_branches_for_monitoring": not nominal_only and self.jes_scheme != "total",

            "met_jes_selection": self.met_jes_selection,

            "met_jet_eta_max": self.MET_JET_ETA_MAX,

            "met_jet_pt_min": self.MET_JET_PT_MIN,

            "jer_propagated_to_met": False,

            "met_reconstruction_schema": 1 if nominal_only else None,

            "met_reconstruction_collection": "METCorrectionJet" if nominal_only else None,

            "met_reconstruction_fields": list(MET_RECONSTRUCTION_FIELDS) if nominal_only else [],

            "raw_met_vector_fields": ["px", "py"] if nominal_only else [],

            "met_cache_selection": "static eta/EM/positive no-muon raw pT; no corrected-pT cut",

            "met_jec_lookup": "full raw pT; factors multiply no-muon vector",

            "met_jes_lookup": "corrected no-muon pT; shift full JEC, keep L1 nominal",

            "jet_veto_treatment": "fixed_nominal_standard_jec_event_veto",

            "jet_veto_key": self._jet_veto_key, "jet_veto_category": self._jet_veto_type,

            "jet_veto_sha256": self._veto_sha256,

            "jet_veto_selection": "standard JEC pT>=15; abs(eta)<5.191; tight-lepton-veto ID; EM<=0.9",

            "jer_gen_target": "NanoAOD GenJet candidate; plus-neutrino convention requires validation",

            "jer_seed_recipe": "mixed_run_luminosityBlock_event_positive31bit_v1",

            "jer_scheme": "inclusive", "mc_jet_jer_requested": self.do_jer,

            "jes_variations": [

                {"stage": stage, "source": source, "sign": sign,

                 "correction_keys": {algo: correction.name for algo, correction

                                     in self._jes_corrections[source].items()},

                 "jet_fields": ([f"{q}{suffix}_{stage}" for suffix, _, _ in JET_PATHS

                                 for q in ("pt", "mass")]

                                if not nominal_only and (self.jet_systematics_storage == "full" or source == "Total") else []),

                 "met_fields": [] if nominal_only else [f"{q}_{stage}" for q in ("pt", "phi")]}

                for stage, source, sign in self._jes_variation_specs()

            ],

        }



    def _ensure_jersmear_loaded(self):

        if self._loaded_jersmear:

            return

        path = os.path.join(self.corrections_dir, "jer_smear.json.gz")

        if not os.path.exists(path):

            raise FileNotFoundError(f"MC JER was requested, but {path} is missing")

        cset = correctionlib.CorrectionSet.from_file(path)

        self._cJerSmear = cset["JERSmear"]

        self._jersmear_sha256 = file_sha256(path)

        expected = [("JetPt", "real"), ("JetEta", "real"), ("GenPt", "real"),

                    ("Rho", "real"), ("EventID", "int"), ("JER", "real"), ("JERSF", "real")]

        actual = [(i.name, i.type) for i in self._cJerSmear.inputs]

        if actual != expected:

            raise RuntimeError(f"Unexpected JERSmear inputs: {actual}")

        self._loaded_jersmear = True

        print(f"[Skim:JERSmear] Loaded JERSmear from {path}")



    # ---------------- small utils ---------------- #



    def select_fields(self, collection, fields):

        # Project requested fields only after all selections/calculations. Fields

        # such as pt_raw can be used internally even when not saved in the config.

        if not fields:

            return collection

        all_fields = ak.fields(collection)

        selected = []

        for f in fields:

            if "*" in f:

                selected.extend([x for x in all_fields if fnmatch.fnmatch(x, f)])

            elif f in all_fields:

                selected.append(f)

        out = []

        seen = set()

        for x in selected:

            if x not in seen:

                out.append(x)

                seen.add(x)

        return collection[out]



    # ---------------- JEC on AK4 jets ---------------- #



    def _jec_factor_flat(self, *, algo, isData, area, eta, pt, rho, phi, run):

        """Evaluate the complete chain in its verified positional input order.



        The compound updates JetPt between levels. Standard jets need phi;

        the UParT chains do not. DATA selects the correct nib internally by run.

        """

        if len(pt) == 0:

            return np.empty(0, dtype=np.float64)

        args = [area, eta, pt, rho]

        if algo == "AK4PFPuppi":

            args.append(phi)

        if isData:

            args.append(run)

        correction = self._jec[algo]["DATA" if isData else "MC"]

        factor = np.asarray(correction.evaluate(*args), dtype=np.float64)

        return np.broadcast_to(factor, np.shape(pt))



    def _apply_jec_inputs(self, jets, rho, run, isData, pt_input, mass_input, algo):

        """Scale the supplied raw/regressed input four-vector with its JEC."""

        counts = ak.num(pt_input, axis=1)

        rho_b, _ = ak.broadcast_arrays(rho, pt_input)

        run_b, _ = ak.broadcast_arrays(run, pt_input)

        factor_flat = self._jec_factor_flat(

            algo=algo, isData=isData,

            area=_as_np_flat(jets.area).astype(np.float64),

            eta=_as_np_flat(jets.eta).astype(np.float64),

            pt=_as_np_flat(pt_input).astype(np.float64),

            rho=_as_np_flat(rho_b).astype(np.float64),

            phi=_as_np_flat(jets.phi).astype(np.float64),

            run=_as_np_flat(run_b).astype(np.float64),

        )

        factor = _unflatten_like(factor_flat, counts)

        return pt_input * factor, mass_input * factor



    def _apply_jec_ak4(self, jets, rho, run, isData: bool):

        """Recover raw values ONCE from ORIGINAL NanoAOD jets, then standard JEC."""

        pt_raw, mass_raw = _raw_ak4_kinematics(jets)

        pt_full, mass_full = self._apply_jec_inputs(

            jets, rho, run, isData, pt_raw, mass_raw, "AK4PFPuppi"

        )

        counts = ak.num(pt_raw, axis=1)

        rho_b, _ = ak.broadcast_arrays(rho, pt_raw)

        cL1 = self._cL1_data if isData else self._cL1_mc

        if int(ak.sum(counts)):

            l1 = cL1.evaluate(

                _as_np_flat(jets.area).astype(np.float64),

                _as_np_flat(jets.eta).astype(np.float64),

                _as_np_flat(pt_raw).astype(np.float64),

                _as_np_flat(rho_b).astype(np.float64),

            )

            pt_l1 = pt_raw * _unflatten_like(l1, counts)

        else:

            pt_l1 = pt_raw

        return pt_raw, mass_raw, pt_l1, pt_full, mass_full



    def _add_corrected_jet_fields(self, *, jets, events, rho, isData,

                                  pt_raw, mass_raw, pt_jec, mass_jec):

        """Compute independent definitions; never choose a b-tag WP here.



        The +neutrino factor is already complete relative to RAW pT. It is

        never multiplied by the visible factor. A uniform four-vector scale

        also defines the corresponding mass; eta/phi stay unchanged.

        JES branches use each selected source at unsmeared full-JEC pT, shift that pT,

        and then reevaluate nominal JER on the shifted input. This follows

        the supplied tutorial's JES -> JER order. JER up/down start from

        the unshifted full-JEC pT. Data branches are nominal copies.

        """

        corrected = ak.with_field(jets, jets.pt, "pt_nano")

        corrected = ak.with_field(corrected, jets.mass, "mass_nano")

        corrected = ak.with_field(corrected, pt_raw, "pt_raw")

        corrected = ak.with_field(corrected, mass_raw, "mass_raw")

        corrected = ak.with_field(corrected, ak.local_index(jets.pt, axis=1), "jet_index")



        if self.jet_systematics_storage in ("nominal", "compact"):

            # Save the NanoAOD candidate association, BEFORE the angular and

            # 3-resolution match test. A JES shift can change that match test.

            # Per-reco candidate coordinates avoid fragile GenJet indices after

            # selecting/reordering the output Jet rows.

            if not isData and hasattr(jets, "genJetIdx") and hasattr(events, "GenJet"):

                idx = ak.values_astype(jets.genJetIdx, np.int32)

                candidate = ak.mask(idx, (idx >= 0) & (idx < ak.num(events.GenJet.pt, axis=1)))

                gen_values = (ak.fill_none(events.GenJet.pt[candidate], -1.0),

                              ak.fill_none(events.GenJet.eta[candidate], 0.0),

                              ak.fill_none(events.GenJet.phi[candidate], 0.0))

            else:

                gen_values = (ak.zeros_like(jets.pt) - 1.0,

                              ak.zeros_like(jets.pt), ak.zeros_like(jets.pt))

            for name, values in zip(COMPACT_JET_FIELDS[-3:], gen_values):

                corrected = ak.with_field(corrected, values, name)



        smear_inputs = None

        if not isData and self.do_jer:

            if not hasattr(jets, "genJetIdx") or not hasattr(events, "GenJet"):

                raise RuntimeError("MC JER requested, but Jet.genJetIdx or GenJet is missing")

            smear_inputs = dict(

                eta=jets.eta, phi=jets.phi, rho=rho, events=events,

                gen_pt=events.GenJet.pt, gen_eta=events.GenJet.eta,

                gen_phi=events.GenJet.phi,

                gen_idx_for_each_reco=ak.values_astype(jets.genJetIdx, np.int32),

            )



        for suffix, algo, regression_branch in JET_PATHS:

            lookup_eta = jets.eta

            if regression_branch is None:

                valid = ak.ones_like(pt_raw, dtype=bool)

                path_pt_jec, path_mass_jec = pt_jec, mass_jec

            else:

                if not hasattr(jets, regression_branch):

                    raise RuntimeError(f"Required NanoAOD UParT branch is missing: Jet.{regression_branch}")

                regression = ak.fill_none(getattr(jets, regression_branch), -1.0)

                valid = np.isfinite(regression) & (regression > 0.0) & (pt_raw > 0.0)

                # NanoAOD uses -1 when the network output is unavailable. Use

                # harmless positive placeholders for lookup only, and mark all

                # saved momenta in this path -1 below. Do not silently substitute

                # standard momentum and label it as valid regression.

                reg_pt_raw = pt_raw * ak.where(valid, regression, 1.0)

                reg_mass_raw = mass_raw * ak.where(valid, regression, 1.0)

                lookup_pt = ak.where(valid, reg_pt_raw, 20.0)

                lookup_mass = ak.where(valid, reg_mass_raw, 0.0)

                lookup_eta = ak.where(valid, jets.eta, 0.0)

                lookup_jets = ak.with_field(jets, lookup_eta, "eta")

                path_pt_jec, path_mass_jec = self._apply_jec_inputs(

                    lookup_jets, rho, events.run, isData, lookup_pt, lookup_mass, algo

                )

                corrected = ak.with_field(corrected, valid, suffix[1:] + "_valid")

                corrected = ak.with_field(corrected, ak.where(valid, reg_pt_raw, -1.0), "pt" + suffix + "_raw")

                corrected = ak.with_field(corrected, ak.where(valid, reg_mass_raw, -1.0), "mass" + suffix + "_raw")



            momenta = {"jec": path_pt_jec}

            nominal_only = self.jet_systematics_storage == "nominal"

            if smear_inputs is not None:

                # Reuse the SAME unsmeared pT, resolution matching and event

                # identity for nominal/up/down; never smear a smeared jet.

                jer_tasks = (("", "nom"),) if nominal_only else (("", "nom"), ("jerUp", "up"), ("jerDown", "down"))

                for label, var in jer_tasks:

                    path_smear_inputs = dict(smear_inputs, eta=lookup_eta)

                    path_smear_inputs["gen_idx_for_each_reco"] = ak.where(

                        valid, smear_inputs["gen_idx_for_each_reco"], -1

                    )

                    momenta[label] = self._jer_smear_tool(

                        pt_in=path_pt_jec, algo=algo, var=var, **path_smear_inputs

                    )

            else:

                for label in (("",) if nominal_only else ("", "jerUp", "jerDown")):

                    momenta[label] = path_pt_jec



            counts = ak.num(path_pt_jec, axis=1)

            eta_flat = _as_np_flat(lookup_eta).astype(np.float64)

            pt_flat = _as_np_flat(path_pt_jec).astype(np.float64)

            uncertainties = {}

            for source, by_algo in (() if nominal_only else self._jes_corrections.items()):

                delta_flat = (by_algo[algo].evaluate(eta_flat, pt_flat)

                              if len(pt_flat) and not isData else np.zeros(len(pt_flat), dtype=np.float64))

                uncertainties[source] = _unflatten_like(delta_flat, counts)

            for stage, source, sign in (() if nominal_only else self._jes_variation_specs()):

                delta = uncertainties[source]

                shifted_jec_pt = path_pt_jec * (1.0 + sign * delta)

                if smear_inputs is not None:

                    # Nominal JER SF, evaluated at the JES-shifted pT.

                    momenta[stage] = self._jer_smear_tool(

                        pt_in=shifted_jec_pt, algo=algo, var="nom", **path_smear_inputs

                    )

                else:

                    momenta[stage] = shifted_jec_pt



            denominator = ak.where(path_pt_jec > 0.0, path_pt_jec, 1.0)

            for stage, path_pt in momenta.items():

                scale = ak.where(path_pt_jec > 0.0, path_pt / denominator, 1.0)

                path_mass = path_mass_jec * scale

                ending = "_" + stage if stage else ""

                corrected = ak.with_field(corrected, ak.where(valid, path_pt, -1.0), "pt" + suffix + ending)

                corrected = ak.with_field(corrected, ak.where(valid, path_mass, -1.0), "mass" + suffix + ending)

        return corrected



    # ---------------- JER smearing tool (jets only) ---------------- #



    def _jer_sf(self, JetEta_flat, JetPt_flat, cat: str, algo="AK4PFPuppi"):

        if cat not in ("nom", "up", "down"):

            raise ValueError(f"Unknown JER variation: {cat}")

        nominal = self._jer_sf_nominal[algo].evaluate(JetEta_flat, JetPt_flat)

        if cat == "nom":

            return nominal

        uncertainty = self._jer_sf_uncertainty[algo].evaluate(JetEta_flat, JetPt_flat)

        return nominal * (1.0 + uncertainty if cat == "up" else 1.0 - uncertainty)



    def _jer_smear_tool(

        self,

        *,

        pt_in: ak.Array,

        eta: ak.Array,

        phi: ak.Array,

        rho: ak.Array,

        events,

        gen_pt: ak.Array,

        gen_eta: ak.Array,

        gen_phi: ak.Array,

        gen_idx_for_each_reco: ak.Array,   # shape [event, jet] with -1 for no match

        mindr: float = 0.2,

        var: str = "nom",                  # "nom","up","down"

        algo: str = "AK4PFPuppi",

    ) -> ak.Array:

        """

        Apply jer_smear.json.gz tool to jets (NOT to MET).

        Retains the supplied processor's NanoAOD GenJet/genJetIdx matching

        for all three definitions. It does not construct a generator jet

        including decay neutrinos. Review this matching choice together with

        the experimental regression-JERC setup in the analysis validation.

        """

        if self._cJerSmear is None:

            raise RuntimeError("JERSmear must be loaded before requesting MC smearing")



        (pt_b, eta_b, phi_b, rho_b) = ak.broadcast_arrays(pt_in, eta, phi, rho)

        # Flatten for correctionlib and retain counts to recover event/jet rows.

        counts = ak.num(pt_b, axis=1)



        JetPt = _as_np_flat(pt_b).astype(np.float64)

        JetEta = _as_np_flat(eta_b).astype(np.float64)

        JetPhi = _as_np_flat(phi_b).astype(np.float64)

        Rho = _as_np_flat(rho_b).astype(np.float64)



        if len(JetPt) == 0:

            return pt_in

        reso = np.asarray(self._jer_resolution[algo].evaluate(JetEta, JetPt, Rho), dtype=np.float64)

        sf = np.asarray(self._jer_sf(JetEta, JetPt, var, algo), dtype=np.float64)



        idx = gen_idx_for_each_reco

        # Mask unmatched (-1) indices so they cannot select the last GenJet.

        idx_valid = (idx >= 0) & (idx < ak.num(gen_pt, axis=1))

        idx_masked = ak.mask(idx, idx_valid)



        gpt = ak.fill_none(gen_pt[idx_masked], -1.0)

        geta = ak.fill_none(gen_eta[idx_masked], 0.0)

        gphi = ak.fill_none(gen_phi[idx_masked], 0.0)



        dphi = _phi_mpi_pi(phi - gphi)

        dR = np.sqrt((eta - geta) ** 2 + dphi ** 2)



        reso_j = ak.unflatten(ak.Array(reso), counts)

        # Matched smearing requires both angular and resolution compatibility.

        # GenPt=-1 tells the tool to use stochastic smearing for unmatched jets.

        is_match = idx_valid & (dR < mindr) & (abs(pt_in - gpt) < 3.0 * reso_j * pt_in)

        genpt_for_tool = ak.where(is_match, gpt, ak.zeros_like(pt_in) - 1.0)

        GenPt = _as_np_flat(genpt_for_tool).astype(np.float64)



        # Seeds use event identity, not the row number, so early event filtering

        # and different chunk boundaries do not change surviving JER values.

        eid = cms_event_id_u64(events, like=pt_in)

        EventID = _as_np_flat(eid)



        JetPt_f = np.asarray(JetPt, dtype=np.float64)

        JetEta_f = np.asarray(JetEta, dtype=np.float64)

        GenPt_f = np.asarray(GenPt, dtype=np.float64)

        Rho_f = np.asarray(Rho, dtype=np.float64)

        JER_f = np.asarray(reso, dtype=np.float64)

        JERSF_f = np.asarray(sf, dtype=np.float64)

        EventID_i = np.asarray(EventID, dtype=np.int64)



        try:

            scale_flat = self._cJerSmear.evaluate(

                JetPt_f, JetEta_f, GenPt_f, Rho_f, EventID_i, JER_f, JERSF_f

            )

            scale_flat = np.asarray(scale_flat, dtype=np.float64)

        except Exception:

            scale_flat = np.fromiter(

                (

                    self._cJerSmear.evaluate(

                        float(a), float(b), float(c), float(d),

                        int(e), float(f), float(g)

                    )

                    for a, b, c, d, e, f, g in zip(JetPt_f, JetEta_f, GenPt_f, Rho_f, EventID_i, JER_f, JERSF_f)

                ),

                dtype=np.float64,

                count=len(JetPt_f),

            )



        if not np.all(np.isfinite(scale_flat)):

            raise RuntimeError("JERSmear returned a nonfinite factor")

        # The reference hybrid recipe truncates negative factors to zero.

        scale = ak.unflatten(ak.Array(np.maximum(scale_flat, 0.0)), counts)

        return pt_in * scale



    # ---------------- JetID (tight lepveto) ---------------- #



    def _jetid_tight_lepveto(self, jets):

        # JetID is needed for veto eligibility. It is not an extra requirement

        # in the corrected-pT-selected output jet definition.

        eta_abs = abs(jets.eta)

        chMult = jets.chMultiplicity

        neMult = jets.neMultiplicity

        neHEF = jets.neHEF

        neEmEF = jets.neEmEF

        chHEF = jets.chHEF

        muEF = _safe_get(jets, "muEF", ak.zeros_like(eta_abs))

        chEmEF = _safe_get(jets, "chEmEF", ak.zeros_like(eta_abs))



        pass_tight = (

            ((eta_abs <= 2.6) & (neHEF < 0.99) & (neEmEF < 0.9) & ((chMult + neMult) > 1) & (chHEF > 0.01) & (chMult > 0))

            | ((eta_abs > 2.6) & (eta_abs <= 2.7) & (neHEF < 0.90) & (neEmEF < 0.99))

            | ((eta_abs > 2.7) & (eta_abs <= 3.0) & (neHEF < 0.99))

            | ((eta_abs > 3.0) & (neMult >= 2) & (neEmEF < 0.4))

        )

        pass_tight_lv = ak.where(eta_abs <= 2.7, pass_tight & (muEF < 0.8) & (chEmEF < 0.8), pass_tight)

        return ak.values_astype(pass_tight_lv, bool)



    # ---------------- Type-1 MET updated recipe (NO JER->MET) ---------------- #



    def _type1_cache_for_collection(

        self, *, events, rho, isData, jet_eta, jet_phi, jet_area,

        jet_rawpt, jet_muSubFactor, jet_muSubDeltaPhi, pass_em_mask,

    ):

        """Cache every statically eligible Type-1 contributor BEFORE its pT cut.



        Retrieve L1/full JEC at FULL raw pT; apply factors to the no-muon

        raw vector (supplied slides 5-6). Eta/EM/positive-raw requirements

        are fixed. Do NOT discard entries whose corrected pT is below 15:

        they may enter under a later JES shift. No JER/regression is used.

        """

        counts = ak.num(jet_rawpt, axis=1)



        pt_noMuRaw = jet_rawpt * (1.0 - jet_muSubFactor)

        phi_noMuRaw = jet_phi + jet_muSubDeltaPhi

        # No RAW-pT >15 cut. Low-raw-pT jets can enter after correction.

        eligible = (

            (jet_rawpt > 0.0) & (pt_noMuRaw > 0.0)

            & np.isfinite(jet_rawpt) & np.isfinite(pt_noMuRaw)

            & np.isfinite(jet_eta) & np.isfinite(phi_noMuRaw)

            & (abs(jet_eta) < self.MET_JET_ETA_MAX) & pass_em_mask

        )

        rho_b, _ = ak.broadcast_arrays(rho, jet_rawpt)

        run_b, _ = ak.broadcast_arrays(events.run, jet_rawpt)

        flat = lambda x: _as_np_flat(x).astype(np.float64)

        area, eta, phi = flat(jet_area), flat(jet_eta), flat(jet_phi)

        rawpt, rho_flat, run_flat = flat(jet_rawpt), flat(rho_b), flat(run_b)

        keep = _as_np_flat(eligible).astype(bool)

        fac_L1 = np.ones(len(rawpt), dtype=np.float64)

        fac_full = np.ones(len(rawpt), dtype=np.float64)

        if np.any(keep):

            cL1 = self._cL1_data if isData else self._cL1_mc

            fac_L1[keep] = cL1.evaluate(area[keep], eta[keep], rawpt[keep], rho_flat[keep])

            fac_full[keep] = self._jec_factor_flat(

                algo="AK4PFPuppi", isData=isData,

                area=area[keep], eta=eta[keep], pt=rawpt[keep],

                rho=rho_flat[keep], phi=phi[keep], run=run_flat[keep],

            )

            if not (np.all(np.isfinite(fac_L1[keep]))

                    and np.all(np.isfinite(fac_full[keep]))

                    and np.all(fac_L1[keep] > 0.0)

                    and np.all(fac_full[keep] > 0.0)):

                raise RuntimeError("Invalid L1/full JEC factor for a Type-1 MET jet")



        # Zero unusable vectors before multiplication; NaN*0 is still NaN.

        noMuRaw_flat = flat(ak.where(eligible, pt_noMuRaw, 0.0))

        noMuFull_flat = noMuRaw_flat * fac_full

        pt_noMuL1 = _unflatten_like(noMuRaw_flat * fac_L1, counts)

        pt_noMuFull = _unflatten_like(noMuFull_flat, counts)

        return ak.zip({

            "pt_full": pt_noMuFull,

            "pt_l1": pt_noMuL1,

            "eta": ak.values_astype(jet_eta, np.float64),

            "phi": ak.values_astype(phi_noMuRaw, np.float64),

        })[eligible]



    def _type1_sum_from_cache(self, cache, *, isData, source=None, sign=0.0):

        """One Type-1 vector correction; shared by production and analysis.



        JES lookup uses corrected no-muon pT, as in the supplied tutorial.

        Only the full-JEC momentum shifts; L1 remains nominal. Threshold

        membership follows met_jes_selection. Static cuts were applied

        before caching. This sum is independent of saved Jet multiplicity.

        """

        pt_full = cache.pt_full

        delta = ak.zeros_like(pt_full)

        if source is not None and not isData:

            counts = ak.num(pt_full, axis=1)

            pt_flat = _as_np_flat(pt_full).astype(np.float64)

            eta_flat = _as_np_flat(cache.eta).astype(np.float64)

            if len(pt_flat):

                values = np.asarray(self._jes_corrections[source]["AK4PFPuppi"].evaluate(

                    eta_flat, pt_flat), dtype=np.float64)

                if not (np.all(np.isfinite(values)) and np.all(values >= 0) and np.all(values < 1)):

                    raise RuntimeError(f"Invalid {source} JES uncertainty for a Type-1 MET jet")

                delta = _unflatten_like(values, counts)

        shifted_pt = pt_full * (1.0 + sign * delta)

        selection_pt = shifted_pt if self.met_jes_selection == "varied" else pt_full

        dpt = ak.where(selection_pt > self.MET_JET_PT_MIN, shifted_pt - cache.pt_l1, 0.0)

        return (ak.sum(dpt * np.cos(cache.phi), axis=1),

                ak.sum(dpt * np.sin(cache.phi), axis=1))



    def _type1_sum_for_collection(self, **inputs):

        """Compatibility wrapper for the previous scalar-reference checks."""

        cache = self._type1_cache_for_collection(**inputs)

        stages = [("", None, 0.0)]

        if self.jet_systematics_storage != "nominal":

            stages += self._jes_variation_specs()

        return {stage: self._type1_sum_from_cache(

                    cache, isData=inputs["isData"], source=source, sign=sign)

                for stage, source, sign in stages}



    def _type1_met_from_ak4_and_corrt1(self, *, events, rho, isData, pt_raw_ak4, return_inputs=False):

        """Return nominal/selected-JES {stage: (pt, phi)} from RawPuppiMET.



        Use the complete Jet and CorrT1METJet inputs. Missing v15 inputs

        are errors rather than silent zero-muon/zero-EM substitutions.

        MET is JEC-only, regardless of self.do_jer.

        """

        for collection in ("RawPuppiMET", "Jet", "CorrT1METJet"):

            if not hasattr(events, collection):

                raise RuntimeError(f"{collection} is required for complete Type-1 PUPPI MET")

        jets, ct1 = events.Jet, events.CorrT1METJet

        requirements = (

            ("RawPuppiMET", events.RawPuppiMET, ("pt", "phi")),

            ("Jet", jets, ("eta", "phi", "area", "muonSubtrFactor",

                           "muonSubtrDeltaPhi", "chEmEF", "neEmEF")),

            ("CorrT1METJet", ct1, ("rawPt", "eta", "phi", "area",

                                   "muonSubtrFactor", "muonSubtrDeltaPhi", "EmEF")),

        )

        for name, collection, fields in requirements:

            missing = set(fields) - set(ak.fields(collection))

            if missing:

                raise RuntimeError(f"Missing Type-1 MET fields in {name}: {sorted(missing)}")



        # Compute/cache Cartesian raw MET in float64. A float32 nominal

        # polar MET alone is insufficient to reconstruct absolute shifts.

        raw_pt = ak.values_astype(events.RawPuppiMET.pt, np.float64)

        raw_phi = ak.values_astype(events.RawPuppiMET.phi, np.float64)

        met_px, met_py = raw_pt * np.cos(raw_phi), raw_pt * np.sin(raw_phi)

        ak4_cache = self._type1_cache_for_collection(

            events=events, rho=rho, isData=isData,

            jet_eta=jets.eta, jet_phi=jets.phi, jet_area=jets.area,

            jet_rawpt=pt_raw_ak4, jet_muSubFactor=jets.muonSubtrFactor,

            jet_muSubDeltaPhi=jets.muonSubtrDeltaPhi,

            pass_em_mask=(jets.chEmEF + jets.neEmEF) < 0.9,

        )

        ct1_cache = self._type1_cache_for_collection(

            events=events, rho=rho, isData=isData,

            jet_eta=ct1.eta, jet_phi=ct1.phi, jet_area=ct1.area,

            jet_rawpt=ct1.rawPt, jet_muSubFactor=ct1.muonSubtrFactor,

            jet_muSubDeltaPhi=ct1.muonSubtrDeltaPhi,

            pass_em_mask=ct1.EmEF < 0.9,

        )

        cache = ak.concatenate([ak4_cache, ct1_cache], axis=1)

        stages = [("", None, 0.0)]

        if self.jet_systematics_storage != "nominal":

            stages += self._jes_variation_specs()

        result = {}

        for stage, source, sign in stages:

            sum_px, sum_py = self._type1_sum_from_cache(cache, isData=isData, source=source, sign=sign)

            px, py = met_px - sum_px, met_py - sum_py

            result[stage] = (np.hypot(px, py), np.arctan2(py, px))

        if return_inputs:

            return result, cache, ak.zip({"px": met_px, "py": met_py}, depth_limit=1)

        return result



    # ---------------- main process ---------------- #



    def process(self, events):

        self._ensure_veto_loaded()

        self._ensure_jerc_loaded()



        n_before = int(len(events))

        isData = not hasattr(events, "genWeight")

        if not isData and self.do_jer:

            self._ensure_jersmear_loaded()

        meta_systematics = self._systematics_metadata()

        meta_systematics["is_data"] = isData

        meta_systematics["mc_jet_jer_applied"] = self.do_jer and not isData

        print(f"\n[INFO] Starting events: {n_before} dataset='{self.dataset_name}' isData={isData} do_jer(jets)={self.do_jer}")



        # ---------- 1. Input normalization counters ----------

        # Count the original chunk, before ANY cut, for later MC normalization.

        # Zero genWeight contributes to nevents but neither sign counter.

        if not isData:

            gw = ak.to_numpy(events.genWeight)

            n_pos = int(np.count_nonzero(gw > 0))

            n_neg = int(np.count_nonzero(gw < 0))

        else:

            n_pos = 0

            n_neg = 0

        meta_counters = {"nevents": n_before, "nevents_pos": n_pos, "nevents_neg": n_neg}



        # ---------- 2. Base event requirements ----------

        # These masks are event-level arrays on the ORIGINAL chunk. Missing

        # configured flags/PV, or a missing data lumi mask, are explicit errors.

        if isData and (self._lumi_mask is not None):

            lumi_mask = ak.Array(self._lumi_mask(events.run, events.luminosityBlock))

        elif isData:

            raise RuntimeError("Data processing requires the 2024 golden JSON")

        else:

            lumi_mask = ak.ones_like(events.event, dtype=bool)

        _print_counts("After golden JSON", lumi_mask)



        if ("PV" in events.fields) and ("npvsGood" in ak.fields(events.PV)):

            pv_mask = events.PV.npvsGood >= 1

        else:

            raise RuntimeError("PV.npvsGood is required for the configured primary-vertex cut")

        _print_counts("After PV", lumi_mask & pv_mask)



        met_filter_mask = ak.ones_like(events.event, dtype=bool)

        if not hasattr(events, "Flag"):

            raise RuntimeError("Flag collection is required for the configured MET filters")

        for flag in self.met_filter_flags:

            f = flag.replace("Flag_", "")

            if not hasattr(events.Flag, f):

                raise RuntimeError(f"Configured MET filter Flag.{f} is missing")

            met_filter_mask = met_filter_mask & getattr(events.Flag, f)

        _print_counts("After MET filters (early)", lumi_mask & pv_mask & met_filter_mask)



        base_mask = ak.values_astype(lumi_mask & pv_mask & met_filter_mask, bool)

        n_base = int(ak.count_nonzero(base_mask))

        print(f"[INFO] Events after early base mask: {n_base} / {n_before}")

        if n_base == 0:

            return {"Events": {}, "MetaCounters": meta_counters, "MetaHists": {}, "MetaSystematics": meta_systematics}



        # Apply all base requirements together to the COMPLETE event record.

        # Every collection (Jet, GenJet, MET, HLT, etc.) follows the same slice.

        events_b = events[base_mask]

        assert len(events_b.event) == int(ak.count_nonzero(base_mask)), "[SANITY] base slicing mismatch!"



        # ---------- 3. Build correction inputs WITHOUT an output-jet cut ----------

        if hasattr(events_b, "Rho") and hasattr(events_b.Rho, "fixedGridRhoFastjetAll"):

            rho = events_b.Rho.fixedGridRhoFastjetAll

        else:

            raise RuntimeError("Rho.fixedGridRhoFastjetAll is required for JEC/JER")



        # Keep the FULL Jet collection for corrections, MET and the veto map.

        # The output selection is applied later, after any enabled MC JER.

        jets_all = events_b.Jet

        passJetIdTightLepVeto = self._jetid_tight_lepveto(jets_all)

        jets_all = ak.with_field(jets_all, passJetIdTightLepVeto, "passJetIdTightLepVeto")



        # ---------- 4. JEC from raw: L1 -> L2 -> L3 (+ data residual) ----------

        # jets_all still contains original NanoAOD pt/mass matching rawFactor.

        # _apply_jec_ak4 recovers raw values before evaluating the new payloads.

        pt_raw, mass_raw, pt_l1_jec, pt_full_jec, mass_full_jec = self._apply_jec_ak4(

            jets_all, rho=rho, run=events_b.run, isData=isData

        )

        # Store raw pT BEFORE replacing NanoAOD pT/mass. Keep all input jets.

        # The original rawFactor must never be used to undo this new JEC pT.

        jets_jec = ak.with_field(jets_all, pt_full_jec, "pt")

        jets_jec = ak.with_field(jets_jec, mass_full_jec, "mass")

        jets_jec = ak.with_field(jets_jec, pt_raw, "pt_raw")



        # ---------- 5. Jet-veto map: reject an ENTIRE event BEFORE MET ----------

        # Evaluate the map on all jets. Only jets passing jet_min can veto.

        # A jet with JEC pT >15 can veto even if it fails the final output cut.

        jets_for_veto = jets_jec

        jet_min = (

            (jets_for_veto.pt >= 15.0)

            & (abs(jets_for_veto.eta) < 5.191)

            & (abs(jets_for_veto.phi) <= np.pi)

            & ak.values_astype(jets_for_veto.passJetIdTightLepVeto, bool)

            & ((jets_for_veto.chEmEF + jets_for_veto.neEmEF) <= 0.9)

        )



        if self._jet_veto is not None:

            # Evaluate ONLY eligible jets, within the map's domain.

            counts = ak.num(jets_for_veto.pt[jet_min], axis=1)

            eta_flat = _as_np_flat(jets_for_veto.eta[jet_min]).astype(np.float64)

            phi_flat = _as_np_flat(jets_for_veto.phi[jet_min]).astype(np.float64)

            vflat = (self._jet_veto.evaluate(self._jet_veto_type, eta_flat, phi_flat)

                     if len(eta_flat) else np.empty(0, dtype=np.float64))

            bad_j = _unflatten_like(vflat, counts) > 0.0

        else:

            raise RuntimeError("The required jet-veto map was not loaded")



        # bad_j is a PER-JET mask; ak.any reduces it to one decision per EVENT.

        veto_event_mask = ~ak.any(bad_j, axis=1)

        _print_counts("After JetVetoMap (event veto)", veto_event_mask)

        assert len(veto_event_mask) == len(events_b.event), "[SANITY] veto_event_mask base mismatch!"



        n_veto = int(ak.count_nonzero(veto_event_mask))

        if n_veto == 0:

            return {"Events": {}, "MetaCounters": meta_counters, "MetaHists": {}, "MetaSystematics": meta_systematics}



        # This is an EVENT slice, not removal of individual vetoed jets.

        # Slice the complete record AND every derived array with the SAME mask.

        # GenJet/CorrT1METJet/HLT/MET follow the event record automatically;

        # local JEC/JetID/rho arrays must follow explicitly. Local genJetIdx

        # still indexes the GenJet collection inside its own event.

        events_b = events_b[veto_event_mask]

        rho = rho[veto_event_mask]

        jets_all = jets_all[veto_event_mask]

        jets_jec = jets_jec[veto_event_mask]

        pt_raw = pt_raw[veto_event_mask]

        mass_raw = mass_raw[veto_event_mask]

        pt_l1_jec = pt_l1_jec[veto_event_mask]

        pt_full_jec = pt_full_jec[veto_event_mask]

        mass_full_jec = mass_full_jec[veto_event_mask]

        jet_min = jet_min[veto_event_mask]

        assert len(events_b) == len(jets_jec) == len(pt_raw) == len(rho) == n_veto, "[SANITY] veto slicing mismatch!"



        # ---------- 6. Three definitions; nominal mode skips ALL variations ----------

        # All calculations are on full surviving Jet rows. No flavour selection

        # or output-pT cut has been applied. Standard JEC alone supplied the veto.

        jets_corrected = self._add_corrected_jet_fields(

            jets=jets_all, events=events_b, rho=rho, isData=isData,

            pt_raw=pt_raw, mass_raw=mass_raw,

            pt_jec=pt_full_jec, mass_jec=mass_full_jec,

        )

        pt_full_jecjer = jets_corrected.pt

        mass_full_jecjer = jets_corrected.mass



        # ---------- 7. Type-1 PUPPI MET on events passing the veto ----------

        # Start from RawPuppiMET; use FULL AK4 + CorrT1METJet inputs.

        # Even raw-pT < 15 GeV jets can contribute if corrected no-muon pT >15.

        # Apply |eta| < 5.2 and EM fraction <0.9, then subtract (full JEC - L1).

        # Do not use the selected or JER-smeared output jets in this calculation.

        if not hasattr(events_b, "RawPuppiMET"):

            raise RuntimeError("RawPuppiMET branch not found; cannot compute Type-1 PuppiMET.")



        met_raw_pt  = events_b.RawPuppiMET.pt

        met_raw_phi = events_b.RawPuppiMET.phi



        met_calculation = self._type1_met_from_ak4_and_corrt1(

            events=events_b,

            rho=rho,

            isData=isData,

            pt_raw_ak4=pt_raw,

            return_inputs=self.jet_systematics_storage == "nominal",

        )

        if self.jet_systematics_storage == "nominal":

            met_t1_variations, met_correction_jets, raw_met_vector = met_calculation

        else:

            met_t1_variations = met_calculation

        met_t1_jec_pt, met_t1_jec_phi = met_t1_variations[""]



        # ---------- 8. Define OUTPUT jets, THEN require >=2 per event ----------

        # Fixed candidates are invariant under this scalar JES/JER recipe.

        # Saved fixed candidates require original raw pT>15 GeV, central eta
        # and tight-lepton-veto ID. No corrected-pT cut is imposed here;
        # downstream JES/JER migrations are retained within this raw-pT skim.

        if self.jet_selection_mode == "fixed":

            output_jet_mask = (

                (jets_corrected.pt_raw > self.JET_RAW_PT_MIN)

                & (abs(jets_corrected.eta) < self.JET_ETA_MAX)

                & ak.values_astype(jets_corrected.passJetIdTightLepVeto, bool)

            )

        else:

            output_jet_mask = ak.values_astype(jets_corrected.pt > self.JET_PT_MIN, bool)

        if self.jet_selection_mode == "union":

            for suffix, _, _ in JET_PATHS:

                for stage in ("", *self._jet_variation_labels()):

                    ending = "_" + stage if stage else ""

                    output_jet_mask = output_jet_mask | (

                        jets_corrected["pt" + suffix + ending] > self.JET_PT_MIN

                    )

        selected_jets = jets_corrected[output_jet_mask]



        # Reduce the selected collection to one multiplicity decision per EVENT.

        # Count exactly the jets that will be saved, ensuring >=2 output jets.

        n_selected_jets = ak.num(selected_jets, axis=1)

        jet_multiplicity_mask = ak.values_astype(n_selected_jets >= self.MIN_SELECTED_JETS, bool)

        selection_label = (

            f">={self.MIN_SELECTED_JETS} tight-lepton-veto candidates, "
            f"|eta|<{self.JET_ETA_MAX:g}, raw pT>{self.JET_RAW_PT_MIN:g} GeV"

            if self.jet_selection_mode == "fixed" else

            f">={self.MIN_SELECTED_JETS} {self.jet_selection_mode} pT>{self.JET_PT_MIN:g} jets"

        )

        _print_counts("After event veto and " + selection_label, jet_multiplicity_mask)

        assert len(jet_multiplicity_mask) == len(events_b), "[SANITY] final jet multiplicity mask mismatch!"



        # ---------- 9. OR of ALL configured trigger groups ----------

        # All decisions use the same events_b as the jets and MET. HLT patterns

        # match available paths. trigger_type preserves the configured group bits.

        trigger_or = ak.zeros_like(events_b.event, dtype=bool)

        trigger_type = ak.zeros_like(events_b.event, dtype=np.int32)

        available_hlt = ak.fields(events_b.HLT)



        for bit, patterns in self.trigger_groups.items():

            group_fired = ak.zeros_like(events_b.event, dtype=bool)

            for pattern in patterns:

                patt = pattern.replace("HLT_", "")

                for trig in fnmatch.filter(available_hlt, patt):

                    group_fired = group_fired | events_b.HLT[trig]

            trigger_or = trigger_or | group_fired

            trigger_type = trigger_type | (ak.values_astype(group_fired, np.int32) << np.int32(bit))



        _print_counts("After " + selection_label + " and configured trigger OR", jet_multiplicity_mask & trigger_or)



        # Trigger selection is ON by default. Set require_trigger=False in the

        # constructor to retain events regardless of HLT while recording bits.

        final_mask_b = ak.values_astype(

            jet_multiplicity_mask & trigger_or if self.require_trigger else jet_multiplicity_mask, bool

        )

        _print_counts("FINAL (within veto-surviving events)", final_mask_b)



        # ---------- 10. Final event slice and selected output jets ----------

        # final_mask_b selects EVENT rows everywhere: event identities, triggers,

        # jet collections and MET. This guarantees saved jets belong to saved events.

        events_f = events_b[final_mask_b]

        trigger_type_f = trigger_type[final_mask_b]

        trigger_or_f = trigger_or[final_mask_b]



        # selected_jets already obeys the configured fixed/standard/union cut.

        # Apply the identical event mask used for identities, HLT and MET.

        jets_out = selected_jets[final_mask_b]

        met_t1_jec_pt_f = met_t1_jec_pt[final_mask_b]

        met_t1_jec_phi_f = met_t1_jec_phi[final_mask_b]



        # Check event-row alignment and the final jet multiplicity contract.

        assert len(jets_out) == len(events_f) == len(trigger_type_f) == len(trigger_or_f) == len(met_t1_jec_pt_f) == len(met_t1_jec_phi_f), "[SANITY] output event alignment mismatch!"

        assert bool(ak.all(ak.num(jets_out, axis=1) >= self.MIN_SELECTED_JETS)), "[SANITY] fewer than two selected output jets!"

        if self.jet_selection_mode == "standard":

            assert bool(ak.all(jets_out.pt > self.JET_PT_MIN)), "[SANITY] output jet below corrected-pT threshold!"

        elif self.jet_selection_mode == "fixed":

            assert bool(ak.all(jets_out.pt_raw > self.JET_RAW_PT_MIN)), "[SANITY] output candidate fails raw-pT requirement!"

            assert bool(ak.all(abs(jets_out.eta) < self.JET_ETA_MAX)), "[SANITY] noncentral output candidate!"

            assert bool(ak.all(jets_out.passJetIdTightLepVeto)), "[SANITY] output candidate fails ID!"



        # ---------- sanity prints ----------

        if self.debug and len(events_b) > 0:

            i = min(max(self.debug_event_index, 0), len(events_b) - 1)

            print(f"[DBG] event index (after event veto, before final cuts) = {i}")

            print(f"      RawPuppiMET pt/phi     : {float(met_raw_pt[i]):.3f} / {float(met_raw_phi[i]):.3f}")

            print(f"      T1(JEC)     pt/phi     : {float(met_t1_jec_pt[i]):.3f} / {float(met_t1_jec_phi[i]):.3f}")



        # ---------- 11. Correction-monitoring histograms ----------

        # Population: base + veto survivors, BEFORE final multiplicity/HLT cuts.

        # Jet histograms use standard veto-eligible jets, before the output cut.

        edges_jet = np.linspace(0, 500, 51, dtype=np.float64)

        edges_met = np.linspace(0, 500, 51, dtype=np.float64)



        jetpt_nano = _as_np_flat(jets_all.pt[jet_min])

        jetpt_jec = _as_np_flat(pt_full_jec[jet_min])

        jetpt_jecjer = _as_np_flat(pt_full_jecjer[jet_min]) if (not isData and self.do_jer) else jetpt_jec



        # Keep the existing histogram convention: hMet_raw contains the input

        # NanoAOD PuppiMET.pt, despite its historical "raw" histogram name.

        met_raw = ak.to_numpy(events_b.PuppiMET.pt)

        met_jec = ak.to_numpy(met_t1_jec_pt)



        meta_hists = {

            "edges_jet": edges_jet,

            "edges_met": edges_met,

            "hJetPt_nano": _hist_counts(jetpt_nano, edges_jet),

            "hJetPt_jec": _hist_counts(jetpt_jec, edges_jet),

            "hJetPt_jecjer": _hist_counts(jetpt_jecjer, edges_jet),

            "hMet_raw": _hist_counts(met_raw, edges_met),

            "hMet_t1_jec": _hist_counts(met_jec, edges_met),

        }



        # Keep input counters and pre-final-cut monitoring even if no event

        # passes the final selection. The driver can omit an empty Events tree.

        if len(events_f) == 0:

            return {"Events": {}, "MetaCounters": meta_counters, "MetaHists": meta_hists, "MetaSystematics": meta_systematics}



        # ---------- 12. Build saved branches on FINAL accepted events ----------

        out = {}

        out["run"] = events_f.run

        out["event"] = events_f.event

        out["luminosityBlock"] = events_f.luminosityBlock

        out["trigger_type"] = trigger_type_f

        if self.jet_systematics_storage == "nominal":

            # Separate from Jet: these entries NEVER enter its >=2 count.

            # Includes all statically eligible Jet+CorrT1METJet contributors,

            # including forward/failing-analysis-ID and pT-threshold migrants.

            out["METCorrectionJet"] = met_correction_jets[final_mask_b]

            out["RawPuppiMET"] = raw_met_vector[final_mask_b]

        # trigger_or_f is aligned with events_f; only trigger_type is saved,

        # preserving the existing output schema.



        # Optional ttbar top-pT weight: unit weight unless exactly two last-copy

        # tops exist. Build full-length arrays so mixed top multiplicities stay

        # aligned with all accepted events.

        if hasattr(events_f, "genWeight"):

            # Sign counters cannot replace the per-event generator weight.

            #out["genWeight"] = events_f.genWeight

            is_ttbar_sample = "TTto" in (self.dataset_name or "")

            if is_ttbar_sample and hasattr(events_f, "GenPart"):

                gen = events_f.GenPart

                LAST_COPY_BIT = 1 << 13

                is_lastcopy = (gen.statusFlags & LAST_COPY_BIT) != 0

                is_top = (abs(gen.pdgId) == 6) & is_lastcopy



                genTops = gen[is_top]

                n_tops = ak.num(genTops, axis=1)



                topptWeight = ak.ones_like(events_f.event, dtype=np.float32)

                has_two = (n_tops == 2)

                top_pts = ak.pad_none(genTops.pt, 2, axis=1, clip=True)

                pt1 = ak.fill_none(top_pts[:, 0], 0.0)

                pt2 = ak.fill_none(top_pts[:, 1], 0.0)

                w_event = ak.values_astype(toppt_run3(pt1) * toppt_run3(pt2), "float32")

                topptWeight = ak.where(has_two, w_event, topptWeight)



                out["topptWeight"] = ak.values_astype(topptWeight, "float32")



        # Save corrected MET (not RawPuppiMET), using the same final event mask.

        if hasattr(events_f, "PuppiMET"):

            puppi = events_f.PuppiMET

            puppi = ak.with_field(puppi, ak.values_astype(met_t1_jec_pt_f, "float32"), "pt")

            puppi = ak.with_field(puppi, ak.values_astype(met_t1_jec_phi_f, "float32"), "phi")

            for stage, _, _ in (() if self.jet_systematics_storage == "nominal" else self._jes_variation_specs()):

                stage_pt, stage_phi = met_t1_variations[stage]

                puppi = ak.with_field(puppi, ak.values_astype(stage_pt[final_mask_b], "float32"), "pt_" + stage)

                puppi = ak.with_field(puppi, ak.values_astype(stage_phi[final_mask_b], "float32"), "phi_" + stage)

            requested_met_fields = self.branches_to_keep.get("PuppiMET", [])

            missing = set(requested_met_fields) - set(ak.fields(puppi))

            if missing:

                raise RuntimeError(f"Requested PuppiMET fields are missing: {sorted(missing)}")

            out["PuppiMET"] = self.select_fields(puppi, self.branches_to_keep.get("PuppiMET", []))



        # Save requested fields. pt_raw is retained for MET and optional output;

        # include it in skim_config.py to save it. Other objects come from

        # events_f, so their event rows stay aligned with jets and MET.

        for obj in self.branches_to_keep:

            if obj in ("PuppiMET", "METCorrectionJet", "RawPuppiMET"):

                continue

            if obj == "Jet":

                requested_jet_fields = [f for f in self.branches_to_keep["Jet"] if "*" not in f]

                missing = set(requested_jet_fields) - set(ak.fields(jets_out))

                if isData:

                    missing.discard("hadronFlavour")  # MC-only NanoAOD field

                if missing:

                    raise RuntimeError(f"Requested Jet fields are missing: {sorted(missing)}")

                out["Jet"] = self.select_fields(jets_out, self.branches_to_keep["Jet"])

                continue

            if not hasattr(events_f, obj):

                continue

            if obj == "Muon":

                mu = events_f.Muon

                mu = mu[(mu.pt >= 10) & (abs(mu.eta) < 2.5)]

                out["Muon"] = self.select_fields(mu, self.branches_to_keep["Muon"])

            elif obj == "Electron":

                el = events_f.Electron

                el = el[(el.pt >= 10) & (abs(el.eta) < 2.5)]

                out["Electron"] = self.select_fields(el, self.branches_to_keep["Electron"])

            else:

                col = getattr(events_f, obj)

                out[obj] = self.select_fields(col, self.branches_to_keep[obj])



        return {"Events": out, "MetaCounters": meta_counters, "MetaHists": meta_hists, "MetaSystematics": meta_systematics}



    def postprocess(self, accumulator):

        return accumulator
