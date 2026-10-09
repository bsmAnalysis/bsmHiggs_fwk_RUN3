
# Event selection:
#   golden JSON (data) + PV + MET filters
#   -> JEC -> event-level jet-veto map
#   -> optional JER / Type-1 PUPPI MET using full input collections
#   -> define output jets with final corrected pT > 20 GeV -> require >= 2
#   -> configured trigger OR -> common final event slice

from coffea import processor
from coffea.lumi_tools import LumiMask
import correctionlib
import awkward as ak
import numpy as np
import fnmatch
import os
from typing import Optional, Tuple


# ------------------------------- helpers -------------------------------- #

def _unflatten_like(flat, counts):
    return ak.unflatten(ak.Array(flat), counts)


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
    raw_factor = ak.fill_none(_safe_get(jets, "rawFactor", ak.zeros_like(jets.pt)), 0.0)
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
    Packed uint64 key derived from (run,lumi,event) for tools that require an integer EventID.
    """
    run = ak.values_astype(events.run, np.uint64)
    lumi = ak.values_astype(events.luminosityBlock, np.uint64)
    evt = ak.values_astype(events.event, np.uint64)

    key = (run << np.uint64(44)) | (lumi << np.uint64(32)) | (evt & np.uint64(0xFFFFFFFF))
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
    AK4 skimmer with a common corrected-pT definition for counting and output.

    Selected output jets have final corrected pT > 20 GeV; require at least two.
    This definition and multiplicity cut are applied after corrections/MET.
    Golden JSON, PV and MET filters are applied first.
    The event veto uses the full JEC Jet collection with its own 15 GeV cut.
    Type-1 PUPPI MET uses full AK4 + CorrT1METJet inputs, with no JER.
    The configured trigger OR is required by default, as requested.

    Jet.pt and Jet.mass include nominal JER for MC when do_jer=True and the
    required inputs/payloads are available. Data use full JEC including the
    residual correction. Disabling/skipping JER retains JEC-only output.
    Normalization counters count input events before any selection.
    """

    JET_PT_MIN = 20.0  # final jet pT: JEC (+ nominal JER for MC), no muon subtraction
    MIN_SELECTED_JETS = 2

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
    ):
        self.branches_to_keep = branches_to_keep
        self.trigger_groups = trigger_groups
        self.met_filter_flags = met_filter_flags
        self.dataset_name = dataset_name

        self.do_jer = bool(do_jer)
        self.debug = bool(debug)
        self.debug_event_index = int(debug_event_index)
        # Set False only when trigger decisions should be recorded without a cut.
        # Existing drivers need no change: the requested trigger cut is on by default.
        self.require_trigger = bool(require_trigger)

        self.corrections_dir = corrections_dir or os.path.join(os.path.dirname(__file__), "corrections")
        self.golden_json_dir = golden_json_dir or os.path.join(os.path.dirname(__file__), "golden_json")

        # Load the certified run/lumisection list once; process() applies it to data.
        # Preserve the existing fallback: no golden JSON file means no lumi cut.
        self._lumi_mask = None
        golden_json_path = os.path.join(
            self.golden_json_dir,
            "Cert_Collisions2024_378981_386951_Golden.json",
        )
        if os.path.exists(golden_json_path):
            print(f"[INIT] Loading golden JSON from {golden_json_path}")
            self._lumi_mask = LumiMask(golden_json_path)
        else:
            print(f"[INIT] golden_json_path='{golden_json_path}' does not exist -> no golden JSON mask")

        # Jet veto map
        self._loaded_veto = False
        self._jet_veto = None
        self._jet_veto_key = "Summer24Prompt24_RunBCDEFGHI_V1"
        self._jet_veto_type = "jetvetomap"

        # JERC
        self._loaded_jerc = False
        self._cset_jerc = None

        # JEC keys (AK4PFPuppi)
        self._cL1_data = None
        self._cL2_data = None
        self._cL3_data = None
        self._cRes_data = None

        self._cL1_mc = None
        self._cL2_mc = None
        self._cL3_mc = None

        # JER resolution + SF from jet_jerc.json.gz (jets only)
        self._cReso = None
        self._cJerSF = None

        # JER smearing tool from jer_smear.json.gz (jets only)
        self._loaded_jersmear = False
        self._cJerSmear = None

    # ---------------- loaders ---------------- #

    def _ensure_veto_loaded(self):
        if self._loaded_veto:
            return
        path = os.path.join(self.corrections_dir, "jetvetomaps.json.gz")
        if not os.path.exists(path):
            print("[Skim:JetVeto] jetvetomaps.json.gz not found; veto disabled.")
            self._loaded_veto = True
            return
        try:
            cset = correctionlib.CorrectionSet.from_file(path)
            if self._jet_veto_key in cset:
                self._jet_veto = cset[self._jet_veto_key]
                print(f"[Skim:JetVeto] Loaded '{self._jet_veto_key}' from {path}")
            else:
                print(f"[Skim:JetVeto] Key '{self._jet_veto_key}' not found; veto disabled.")
        except Exception as e:
            print(f"[Skim:JetVeto] Failed to load veto map: {e}")
        self._loaded_veto = True

    def _ensure_jerc_loaded(self):
        if self._loaded_jerc:
            return
        path = os.path.join(self.corrections_dir, "jet_jerc.json.gz")
        if not os.path.exists(path):
            print(f"[Skim:JERC] Missing {path} -> JEC/JER disabled.")
            self._loaded_jerc = True
            return
        try:
            cset = correctionlib.CorrectionSet.from_file(path)
            self._cset_jerc = cset

            # ---- JEC 2024 AK4PFPuppi ----
            self._cL1_data = cset["Summer24Prompt24_V2_DATA_L1FastJet_AK4PFPuppi"]
            self._cL2_data = cset["Summer24Prompt24_V2_DATA_L2Relative_AK4PFPuppi"]
            self._cL3_data = cset["Summer24Prompt24_V2_DATA_L3Absolute_AK4PFPuppi"]
            self._cRes_data = cset["Summer24Prompt24_V2_DATA_L2L3Residual_AK4PFPuppi"]

            self._cL1_mc = cset["Summer24Prompt24_V2_MC_L1FastJet_AK4PFPuppi"]
            self._cL2_mc = cset["Summer24Prompt24_V2_MC_L2Relative_AK4PFPuppi"]
            self._cL3_mc = cset["Summer24Prompt24_V2_MC_L3Absolute_AK4PFPuppi"]

            # Preserve the JER payload keys from the supplied processor.
            self._cReso = cset["Summer23BPixPrompt23_RunD_JRV1_MC_PtResolution_AK4PFPuppi"]
            self._cJerSF = cset["Summer23BPixPrompt23_RunD_JRV1_MC_ScaleFactor_AK4PFPuppi"]

            self._loaded_jerc = True
            print(f"[Skim:JERC] Loaded jet_jerc.json.gz from {path}")
        except Exception as e:
            print(f"[Skim:JERC] Failed to load: {e}")
            self._loaded_jerc = True

    def _ensure_jersmear_loaded(self):
        if self._loaded_jersmear:
            return
        path = os.path.join(self.corrections_dir, "jer_smear.json.gz")
        if not os.path.exists(path):
            print("[Skim:JERSmear] jer_smear.json.gz not found -> JER smearing tool disabled.")
            self._loaded_jersmear = True
            return
        try:
            cset = correctionlib.CorrectionSet.from_file(path)
            if "JERSmear" in cset:
                self._cJerSmear = cset["JERSmear"]
                print(f"[Skim:JERSmear] Loaded JERSmear tool from {path}")
                print("[Skim:JERSmear] inputs:", [(i.name, i.type) for i in self._cJerSmear.inputs])
            else:
                print("[Skim:JERSmear] Key 'JERSmear' not found in jer_smear.json.gz")
        except Exception as e:
            print(f"[Skim:JERSmear] Failed to load: {e}")
        self._loaded_jersmear = True

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

    def _apply_jec_ak4(self, jets, rho, run, isData: bool) -> Tuple[ak.Array, ak.Array, ak.Array, ak.Array, ak.Array]:
        """
        JEC chain from RAW:
          raw -> L1 -> L2 -> L3 (+ residual for data)
        Returns jagged:
          pt_raw, mass_raw, pt_l1, pt_full, mass_full
        """
        pt_raw, mass_raw = _raw_ak4_kinematics(jets)

        if (not self._loaded_jerc) or (self._cset_jerc is None):
            return pt_raw, mass_raw, pt_raw, pt_raw, mass_raw

        if isData:
            cL1, cL2, cL3 = self._cL1_data, self._cL2_data, self._cL3_data
        else:
            cL1, cL2, cL3 = self._cL1_mc, self._cL2_mc, self._cL3_mc

        counts = ak.num(pt_raw, axis=1)
        rho_b, _ = ak.broadcast_arrays(rho, pt_raw)
        run_b, _ = ak.broadcast_arrays(run, pt_raw)

        JetA = _as_np_flat(jets.area)
        JetEta = _as_np_flat(jets.eta)
        JetPhi = _as_np_flat(jets.phi)
        Rho = _as_np_flat(rho_b)
        Run = _as_np_flat(run_b).astype(np.float64)

        pt0 = _as_np_flat(pt_raw)

        # Each JEC level receives the pT corrected by the previous level.
        l1 = cL1.evaluate(JetA, JetEta, pt0, Rho)
        pt1 = pt0 * l1

        l2 = cL2.evaluate(JetEta, JetPhi, pt1)
        pt2 = pt1 * l2

        l3 = cL3.evaluate(JetEta, pt2)
        pt3 = pt2 * l3

        if isData:
            res = self._cRes_data.evaluate(Run, JetEta, pt3)
            pt3 = pt3 * res

        pt0_safe = np.maximum(pt0, 1e-6)
        # Apply the same complete JEC factor to the jet mass.
        factor_full = pt3 / pt0_safe
        mass_full_flat = _as_np_flat(mass_raw) * factor_full

        pt_l1 = ak.unflatten(ak.Array(pt1), counts)
        pt_full = ak.unflatten(ak.Array(pt3), counts)
        mass_full = ak.unflatten(ak.Array(mass_full_flat), counts)

        return pt_raw, mass_raw, pt_l1, pt_full, mass_full

    # ---------------- JER smearing tool (jets only) ---------------- #

    def _jer_sf(self, JetEta_flat, JetPt_flat, cat: str):
        return self._cJerSF.evaluate(JetEta_flat, JetPt_flat, cat)

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
    ) -> ak.Array:
        """
        Apply jer_smear.json.gz tool to jets (NOT to MET).
        """
        if self._cJerSmear is None:
            return pt_in

        (pt_b, eta_b, phi_b, rho_b) = ak.broadcast_arrays(pt_in, eta, phi, rho)
        # Flatten for correctionlib and retain counts to recover event/jet rows.
        counts = ak.num(pt_b, axis=1)

        JetPt = _as_np_flat(pt_b).astype(np.float64)
        JetEta = _as_np_flat(eta_b).astype(np.float64)
        JetPhi = _as_np_flat(phi_b).astype(np.float64)
        Rho = _as_np_flat(rho_b).astype(np.float64)

        reso = self._cReso.evaluate(JetEta, JetPt, Rho).astype(np.float64)
        sf = self._jer_sf(JetEta, JetPt, var).astype(np.float64)

        idx = gen_idx_for_each_reco
        # Mask unmatched (-1) indices so they cannot select the last GenJet.
        idx_masked = ak.mask(idx, idx >= 0)

        gpt = ak.fill_none(gen_pt[idx_masked], -1.0)
        geta = ak.fill_none(gen_eta[idx_masked], 0.0)
        gphi = ak.fill_none(gen_phi[idx_masked], 0.0)

        dphi = _phi_mpi_pi(phi - gphi)
        dR = np.sqrt((eta - geta) ** 2 + dphi ** 2)

        reso_j = ak.unflatten(ak.Array(reso), counts)
        # Matched smearing requires both angular and resolution compatibility.
        # GenPt=-1 tells the tool to use stochastic smearing for unmatched jets.
        is_match = (gen_idx_for_each_reco >= 0) & (dR < mindr) & (abs(pt_in - gpt) < 3.0 * reso_j * pt_in)
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

        scale = ak.unflatten(ak.Array(scale_flat), counts)
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

    def _type1_sum_for_collection(
        self,
        *,
        events,
        rho: ak.Array,
        isData: bool,
        jet_eta: ak.Array,
        jet_phi: ak.Array,
        jet_area: ak.Array,
        jet_rawpt: ak.Array,               # raw pt (NO mu subtraction)
        jet_muSubFactor: ak.Array,         # muonSubtrFactor
        jet_muSubDeltaPhi: ak.Array,       # muonSubtrDeltaPhi
        pass_em_mask: ak.Array,            # per-jet EM requirement
    ) -> Tuple[ak.Array, ak.Array]:
        """
        Return (sum_px, sum_py) for Type-1:
          sum (pt_noMuFull - pt_noMuL1) * (cos(phi_noMuRaw), sin(phi_noMuRaw))

        - JEC factors are evaluated using jet_rawpt (no mu subtraction)
        - those factors are applied to pt_noMuRaw = rawpt*(1-muonSubtrFactor)
        - vector direction uses phi_noMuRaw = phi + muonSubtrDeltaPhi

        This is the prescription on slides 5-6 of Nurfikri's 2 February 2026
        JME presentation: retrieve factors with FULL raw jet pT first, then
        apply them to the muon-subtracted momentum. Do not evaluate the JEC
        lookup using pt_noMuRaw. Keep L1 from the payload rather than assume 1.
        """
        z = ak.zeros_like(events.event, dtype=np.float64)
        if len(jet_rawpt) == 0:
            return z, z
        if (not self._loaded_jerc) or (self._cset_jerc is None):
            return z, z

        if isData:
            cL1, cL2, cL3 = self._cL1_data, self._cL2_data, self._cL3_data
        else:
            cL1, cL2, cL3 = self._cL1_mc, self._cL2_mc, self._cL3_mc

        pt_noMuRaw  = jet_rawpt * (1.0 - jet_muSubFactor)
        phi_noMuRaw = jet_phi + jet_muSubDeltaPhi

        counts = ak.num(jet_rawpt, axis=1)
        rho_b, _ = ak.broadcast_arrays(rho, jet_rawpt)
        run_b, _ = ak.broadcast_arrays(events.run, jet_rawpt)

        JetA   = _as_np_flat(jet_area).astype(np.float64)
        JetEta = _as_np_flat(jet_eta).astype(np.float64)
        JetPhi = _as_np_flat(jet_phi).astype(np.float64)
        Rho    = _as_np_flat(rho_b).astype(np.float64)
        Run    = _as_np_flat(run_b).astype(np.float64)

        pt0 = _as_np_flat(jet_rawpt).astype(np.float64)   # <-- RAW pt for JEC lookup

        l1 = cL1.evaluate(JetA, JetEta, pt0, Rho)
        pt1 = pt0 * l1

        l2 = cL2.evaluate(JetEta, JetPhi, pt1)
        pt2 = pt1 * l2

        l3 = cL3.evaluate(JetEta, pt2)
        pt3 = pt2 * l3

        if isData:
            res = self._cRes_data.evaluate(Run, JetEta, pt3)
            pt3 = pt3 * res

        pt0_safe = np.maximum(pt0, 1e-6)
        fac_L1   = pt1 / pt0_safe
        fac_full = pt3 / pt0_safe

        pt_noMuRaw_flat = _as_np_flat(pt_noMuRaw).astype(np.float64)
        pt_noMuL1_flat   = pt_noMuRaw_flat * fac_L1
        pt_noMuFull_flat = pt_noMuRaw_flat * fac_full

        pt_noMuL1   = ak.unflatten(ak.Array(pt_noMuL1_flat), counts)
        pt_noMuFull = ak.unflatten(ak.Array(pt_noMuFull_flat), counts)

        # This MET-specific 15 GeV threshold is on corrected no-muon pT.
        # It is independent of the final corrected-pT >20 output jet selection.
        pass_pt  = pt_noMuFull > 15.0
        pass_eta = np.abs(jet_eta) < 5.2
        pass_all = pass_pt & pass_eta & pass_em_mask

        # Type-1 MET subtracts the vector sum of (full JEC - L1-only) jets.
        dpt = ak.where(pass_all, pt_noMuFull - pt_noMuL1, 0.0)
        sum_px = ak.sum(dpt * np.cos(phi_noMuRaw), axis=1)
        sum_py = ak.sum(dpt * np.sin(phi_noMuRaw), axis=1)
        return sum_px, sum_py

    def _type1_met_from_ak4_and_corrt1(
        self,
        *,
        events,
        rho: ak.Array,
        isData: bool,
        pt_raw_ak4: ak.Array,   # Jet_pt_raw = Jet_pt*(1-rawFactor)
    ) -> Tuple[ak.Array, ak.Array]:
        """
        Type-1 PuppiMET from RawPuppiMET using:
          - AK4 Jet
          - CorrT1METJet (if available)
        NO JER propagation to MET.
        """
        met_pt  = ak.Array(events.RawPuppiMET.pt)
        met_phi = ak.Array(events.RawPuppiMET.phi)
        met_px  = met_pt * np.cos(met_phi)
        met_py  = met_pt * np.sin(met_phi)

        # ---------- AK4 contribution ----------
        # Use every input AK4 jet here. Applying the output corrected-pT >20 cut
        # before this step would drop jets passing the MET-specific >15 cut.
        jets = events.Jet
        muSub  = _safe_get(jets, "muonSubtrFactor", ak.zeros_like(pt_raw_ak4))
        dphiMu = _safe_get(jets, "muonSubtrDeltaPhi", ak.zeros_like(pt_raw_ak4))
        chEm   = _safe_get(jets, "chEmEF", ak.zeros_like(pt_raw_ak4))
        neEm   = _safe_get(jets, "neEmEF", ak.zeros_like(pt_raw_ak4))
        pass_em_ak4 = (chEm + neEm) < 0.9

        sum_px_ak4, sum_py_ak4 = self._type1_sum_for_collection(
            events=events,
            rho=rho,
            isData=isData,
            jet_eta=jets.eta,
            jet_phi=jets.phi,
            jet_area=jets.area,
            jet_rawpt=pt_raw_ak4,
            jet_muSubFactor=muSub,
            jet_muSubDeltaPhi=dphiMu,
            pass_em_mask=pass_em_ak4,
        )

        # ---------- CorrT1METJet contribution ----------
        sum_px_ct1 = ak.zeros_like(sum_px_ak4, dtype=np.float64)
        sum_py_ct1 = ak.zeros_like(sum_py_ak4, dtype=np.float64)

        if hasattr(events, "CorrT1METJet"):
            ct1 = events.CorrT1METJet
            rawPt_ct1 = ct1.rawPt
            muSub_ct1  = _safe_get(ct1, "muonSubtrFactor", ak.zeros_like(rawPt_ct1))
            dphiMu_ct1 = _safe_get(ct1, "muonSubtrDeltaPhi", ak.zeros_like(rawPt_ct1))
            emef_ct1   = _safe_get(ct1, "EmEF", ak.zeros_like(rawPt_ct1))
            pass_em_ct1 = emef_ct1 < 0.9

            sum_px_ct1, sum_py_ct1 = self._type1_sum_for_collection(
                events=events,
                rho=rho,
                isData=isData,
                jet_eta=ct1.eta,
                jet_phi=ct1.phi,
                jet_area=ct1.area,
                jet_rawpt=rawPt_ct1,
                jet_muSubFactor=muSub_ct1,
                jet_muSubDeltaPhi=dphiMu_ct1,
                pass_em_mask=pass_em_ct1,
            )

        # Both collections contribute to the same event's MET correction.
        # No JER-smeared quantities enter this calculation.
        met_px_corr = met_px - (sum_px_ak4 + sum_px_ct1)
        met_py_corr = met_py - (sum_py_ak4 + sum_py_ct1)
        met_pt_corr  = np.hypot(met_px_corr, met_py_corr)
        met_phi_corr = np.arctan2(met_py_corr, met_px_corr)
        return met_pt_corr, met_phi_corr

    # ---------------- main process ---------------- #

    def process(self, events):
        self._ensure_veto_loaded()
        self._ensure_jerc_loaded()
        self._ensure_jersmear_loaded()

        n_before = int(len(events))
        isData = not hasattr(events, "genWeight")
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
        # optional flags/PV or an unloaded lumi mask keep the existing fallbacks.
        if isData and (self._lumi_mask is not None):
            lumi_mask = ak.Array(self._lumi_mask(events.run, events.luminosityBlock))
        else:
            lumi_mask = ak.ones_like(events.event, dtype=bool)
        _print_counts("After golden JSON", lumi_mask)

        if ("PV" in events.fields) and ("npvsGood" in ak.fields(events.PV)):
            pv_mask = events.PV.npvsGood >= 1
        else:
            pv_mask = ak.ones_like(events.event, dtype=bool)
        _print_counts("After PV", lumi_mask & pv_mask)

        met_filter_mask = ak.ones_like(events.event, dtype=bool)
        if hasattr(events, "Flag"):
            for flag in self.met_filter_flags:
                f = flag.replace("Flag_", "")
                if hasattr(events.Flag, f):
                    met_filter_mask = met_filter_mask & getattr(events.Flag, f)
        _print_counts("After MET filters (early)", lumi_mask & pv_mask & met_filter_mask)

        base_mask = ak.values_astype(lumi_mask & pv_mask & met_filter_mask, bool)
        n_base = int(ak.count_nonzero(base_mask))
        print(f"[INFO] Events after early base mask: {n_base} / {n_before}")
        if n_base == 0:
            return {"Events": {}, "MetaCounters": meta_counters, "MetaHists": {}}

        # Apply all base requirements together to the COMPLETE event record.
        # Every collection (Jet, GenJet, MET, HLT, etc.) follows the same slice.
        events_b = events[base_mask]
        assert len(events_b.event) == int(ak.count_nonzero(base_mask)), "[SANITY] base slicing mismatch!"

        # ---------- 3. Build correction inputs WITHOUT an output-jet cut ----------
        if hasattr(events_b, "Rho") and hasattr(events_b.Rho, "fixedGridRhoFastjetAll"):
            rho = events_b.Rho.fixedGridRhoFastjetAll
        else:
            rho = ak.zeros_like(events_b.event, dtype=np.float32)

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
            (jets_for_veto.pt > 15.0)
            & ak.values_astype(jets_for_veto.passJetIdTightLepVeto, bool)
            & ((_safe_get(jets_for_veto, "chEmEF", 0.0) + _safe_get(jets_for_veto, "neEmEF", 0.0)) < 0.9)
        )

        if self._jet_veto is not None:
            counts = ak.num(jets_for_veto.pt, axis=1)
            eta_flat = _as_np_flat(jets_for_veto.eta)
            phi_flat = _as_np_flat(jets_for_veto.phi)
            vflat = self._jet_veto.evaluate(self._jet_veto_type, eta_flat, phi_flat)
            veto_j = _unflatten_like(vflat, counts) > 0.0
        else:
            veto_j = ak.zeros_like(jets_for_veto.pt, dtype=bool)

        # bad_j is a PER-JET mask; ak.any reduces it to one decision per EVENT.
        bad_j = jet_min & veto_j
        veto_event_mask = ~ak.any(bad_j, axis=1)
        _print_counts("After JetVetoMap (event veto)", veto_event_mask)
        assert len(veto_event_mask) == len(events_b.event), "[SANITY] veto_event_mask base mismatch!"

        n_veto = int(ak.count_nonzero(veto_event_mask))
        if n_veto == 0:
            return {"Events": {}, "MetaCounters": meta_counters, "MetaHists": {}}

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

        # ---------- 6. Optional nominal MC JER for saved jets ----------
        # Smear full JEC pT, and scale mass by the same smearing factor.
        # The veto already used JEC-only jets; JER never enters Type-1 MET.
        # Veto failures have been rejected; output multiplicity is still pending.
        pt_full_jecjer = pt_full_jec
        mass_full_jecjer = mass_full_jec

        if (not isData) and self.do_jer and (self._cJerSmear is not None):
            if hasattr(events_b, "Jet") and hasattr(events_b.Jet, "genJetIdx") and hasattr(events_b, "GenJet"):
                ak4_gen_idx = ak.values_astype(events_b.Jet.genJetIdx, np.int32)
                pt_full_jecjer = self._jer_smear_tool(
                    pt_in=pt_full_jec,
                    eta=jets_all.eta,
                    phi=jets_all.phi,
                    rho=rho,
                    events=events_b,
                    gen_pt=events_b.GenJet.pt,
                    gen_eta=events_b.GenJet.eta,
                    gen_phi=events_b.GenJet.phi,
                    gen_idx_for_each_reco=ak4_gen_idx,
                    mindr=0.2,
                    var="nom",
                )
                sf_mass = pt_full_jecjer / ak.where(pt_full_jec > 0, pt_full_jec, 1.0)
                mass_full_jecjer = mass_full_jec * sf_mass
            else:
                print("[Skim:JER] Missing Jet.genJetIdx or GenJet: skipping AK4 jet smearing")

        # This is the FINAL jet momentum definition for both selection/output.
        # On data, or when JER is disabled/unavailable, these values equal JEC.
        # Keep true pt_raw and all other fields from the complete JEC collection.
        jets_corrected = ak.with_field(jets_jec, pt_full_jecjer, "pt")
        jets_corrected = ak.with_field(jets_corrected, mass_full_jecjer, "mass")

        # ---------- 7. Type-1 PUPPI MET on events passing the veto ----------
        # Start from RawPuppiMET; use FULL AK4 + CorrT1METJet inputs.
        # Even raw-pT < 15 GeV jets can contribute if corrected no-muon pT >15.
        # Apply |eta| < 5.2 and EM fraction <0.9, then subtract (full JEC - L1).
        # Do not use the selected or JER-smeared output jets in this calculation.
        if not hasattr(events_b, "RawPuppiMET"):
            raise RuntimeError("RawPuppiMET branch not found; cannot compute Type-1 PuppiMET.")

        met_raw_pt  = events_b.RawPuppiMET.pt
        met_raw_phi = events_b.RawPuppiMET.phi

        met_t1_jec_pt, met_t1_jec_phi = self._type1_met_from_ak4_and_corrt1(
            events=events_b,
            rho=rho,
            isData=isData,
            pt_raw_ak4=pt_raw,
        )

        # ---------- 8. Define OUTPUT jets, THEN require >=2 per event ----------
        
        output_jet_mask = ak.values_astype(jets_corrected.pt > self.JET_PT_MIN, bool)
        selected_jets = jets_corrected[output_jet_mask]

        # Reduce the selected collection to one multiplicity decision per EVENT.
        # Count exactly the jets that will be saved, ensuring >=2 output jets.
        n_selected_jets = ak.num(selected_jets, axis=1)
        jet_multiplicity_mask = ak.values_astype(n_selected_jets >= self.MIN_SELECTED_JETS, bool)
        _print_counts("After event veto and >=2 final corrected pt>20 jets", jet_multiplicity_mask)
        assert len(jet_multiplicity_mask) == len(events_b), "[SANITY] final jet multiplicity mask mismatch!"

        # ---------- 9. OR of ALL configured trigger groups ----------
        # All decisions use the same events_b as the jets and MET. HLT patterns
        # match available paths. trigger_type preserves the configured group bits.
        trigger_or = ak.zeros_like(events_b.event, dtype=bool)
        trigger_type = ak.zeros_like(events_b.event, dtype=np.int32)
        available_hlt = dir(events_b.HLT)

        for bit, patterns in self.trigger_groups.items():
            group_fired = ak.zeros_like(events_b.event, dtype=bool)
            for pattern in patterns:
                patt = pattern.replace("HLT_", "")
                for trig in fnmatch.filter(available_hlt, patt):
                    group_fired = group_fired | events_b.HLT[trig]
            trigger_or = trigger_or | group_fired
            trigger_type = trigger_type | (ak.values_astype(group_fired, np.int32) << np.int32(bit))

        _print_counts("After >=2 final corrected pt>20 jets and configured trigger OR", jet_multiplicity_mask & trigger_or)

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

        # selected_jets already contains only final corrected-pT >20 jets.
        # Apply the identical event mask used for identities, HLT and MET.
        jets_out = selected_jets[final_mask_b]
        met_t1_jec_pt_f = met_t1_jec_pt[final_mask_b]
        met_t1_jec_phi_f = met_t1_jec_phi[final_mask_b]

        # Check event-row alignment and the final jet multiplicity contract.
        assert len(jets_out) == len(events_f) == len(trigger_type_f) == len(trigger_or_f) == len(met_t1_jec_pt_f) == len(met_t1_jec_phi_f), "[SANITY] output event alignment mismatch!"
        assert bool(ak.all(ak.num(jets_out, axis=1) >= self.MIN_SELECTED_JETS)), "[SANITY] fewer than two selected output jets!"
        assert bool(ak.all(jets_out.pt > self.JET_PT_MIN)), "[SANITY] output jet below corrected-pT threshold!"

        # ---------- sanity prints ----------
        if self.debug and len(events_b) > 0:
            i = min(max(self.debug_event_index, 0), len(events_b) - 1)
            print(f"[DBG] event index (after event veto, before final cuts) = {i}")
            print(f"      RawPuppiMET pt/phi     : {float(met_raw_pt[i]):.3f} / {float(met_raw_phi[i]):.3f}")
            print(f"      T1(JEC)     pt/phi     : {float(met_t1_jec_pt[i]):.3f} / {float(met_t1_jec_phi[i]):.3f}")

        # ---------- 11. Correction-monitoring histograms ----------
        # Population: base + veto survivors, BEFORE final multiplicity/HLT cuts.
        # Jet histograms use veto-eligible jets; output uses final corrected pT >20.
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
            return {"Events": {}, "MetaCounters": meta_counters, "MetaHists": meta_hists}

        # ---------- 12. Build saved branches on FINAL accepted events ----------
        out = {}
        out["run"] = events_f.run
        out["event"] = events_f.event
        out["luminosityBlock"] = events_f.luminosityBlock
        out["trigger_type"] = trigger_type_f
        # trigger_or_f is aligned with events_f; only trigger_type is saved,
        # preserving the existing output schema.

        # Optional ttbar top-pT weight: unit weight unless exactly two last-copy
        # tops exist. Build full-length arrays so mixed top multiplicities stay
        # aligned with all accepted events.
        if hasattr(events_f, "genWeight"):
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
            out["PuppiMET"] = self.select_fields(puppi, self.branches_to_keep.get("PuppiMET", []))

        # Save requested fields. pt_raw is retained for MET and optional output;
        # include it in skim_config.py to save it. Other objects come from
        # events_f, so their event rows stay aligned with jets and MET.
        for obj in self.branches_to_keep:
            if obj in ("PuppiMET",):
                continue
            if obj == "Jet":
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

        return {"Events": out, "MetaCounters": meta_counters, "MetaHists": meta_hists}

    def postprocess(self, accumulator):
        return accumulator
