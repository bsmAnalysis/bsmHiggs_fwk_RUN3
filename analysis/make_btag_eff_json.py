#!/usr/bin/env python3
"""Build correctionlib b-tag efficiencies from Num/Denom ROOT histograms.

The stored ratio histograms are not used: efficiencies and sparse-bin fallbacks
are calculated from the raw counts. Fallbacks stay within one process/flavour.

The correction is named btag_eff and takes, in order:
    process, wp, flavour, nbjets, pt, abseta
Flavour indices are b=0, c=1, light=2. nbjets=4 means >=4 selected truth-b jets.

Requires NumPy and uproot; --validate also requires correctionlib. Example:
    python make_btag_eff_json.py ratios/*_BTag_ratio.root \
        --output corrections/btag_eff.json.gz \
        --summary corrections/btag_eff_summary.csv --validate

Convert each ttbar decay file separately: all three contain the same process
keys ttLF, ttCC and ttBB. Other inputs may share one JSON if their keys differ.
"""

import argparse
import csv
import gzip
import json
import os
import re
import sys

import numpy as np
import uproot


WPS = ("L", "M", "T")
FLAVORS = {"b": 0, "c": 1, "light": 2}
NBJETS = {"nb0": 0, "nb1": 1, "nb2": 2, "nb3": 3, "nb4p": 4}

HISTOGRAM_PATTERN = re.compile(
    r"^(?P<process>.+)_BTagEff_Denom_"
    r"(?P<flavour>b|c|light)_"
    r"(?P<nb>nb0|nb1|nb2|nb3|nb4p)_"
    r"WP(?P<wp>L|M|T)$"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input *_BTag_ratio.root files")
    parser.add_argument("--output", required=True, help="Output JSON or JSON.GZ")
    parser.add_argument(
        "--summary", default="btag_eff_json_summary.csv", help="CSV fallback summary"
    )
    parser.add_argument(
        "--min-denominator", type=float, default=10.0,
        help="Minimum count for an exact or pooled-bin estimate (default: 10)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Compare every JSON bin with the intended efficiency map",
    )
    args = parser.parse_args()
    if not np.isfinite(args.min_denominator) or args.min_denominator <= 0.0:
        parser.error("--min-denominator must be finite and greater than zero")
    if os.path.abspath(args.output) == os.path.abspath(args.summary):
        parser.error("--output and --summary must be different files")
    return args


def discover_processes(rootfile):
    processes = set()
    for name in rootfile.keys(cycle=False):
        match = HISTOGRAM_PATTERN.match(name)
        if match:
            processes.add(match.group("process"))
    return sorted(processes)


def histogram_name(process, kind, flavour, nbjet, wp):
    return f"{process}_BTagEff_{kind}_{flavour}_{nbjet}_WP{wp}"


def read_histogram(rootfile, name):
    """Read the visible pT/|eta| bins and reject malformed count maps."""
    if name not in rootfile:
        raise KeyError(f"Missing ROOT histogram: {name}")
    histogram = rootfile[name]
    if not hasattr(histogram, "to_numpy"):
        raise ValueError(f"{name}: expected a TH2 histogram")
    contents = histogram.to_numpy(flow=False)
    if len(contents) != 3:
        raise ValueError(f"{name}: expected a two-dimensional histogram")

    values, pt_edges, eta_edges = (
        np.asarray(array, dtype=np.float64) for array in contents
    )
    for label, edges in (("pT", pt_edges), ("|eta|", eta_edges)):
        if (edges.ndim != 1 or edges.size < 2
                or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0)):
            raise ValueError(f"{name}: {label} edges must be finite and increasing")
    expected_shape = (len(pt_edges) - 1, len(eta_edges) - 1)
    if values.shape != expected_shape:
        raise ValueError(f"{name}: shape {values.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: non-finite histogram counts")
    return values, pt_edges, eta_edges


def ratio(numerator, denominator):
    result = np.zeros_like(denominator, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def hierarchy_valid(efficiencies):
    """Require nonzero L-M, M-T and fail-L probabilities for multi-WP use."""
    loose = efficiencies["L"]
    medium = efficiencies["M"]
    tight = efficiencies["T"]
    tolerance = 1.0e-12
    return (
        np.isfinite(loose)
        & np.isfinite(medium)
        & np.isfinite(tight)
        & (loose >= 0.0)
        & (tight >= 0.0)
        & (loose <= 1.0)
        & (medium <= loose + tolerance)
        & (tight <= medium + tolerance)
        & ((loose - medium) > tolerance)
        & ((medium - tight) > tolerance)
        & ((1.0 - loose) > tolerance)
    )


def read_process(rootfile, process):
    """Load all categories with common binning and WP-independent denominators."""
    denominators = {}
    numerators = {wp: {} for wp in WPS}
    reference_pt_edges = None
    reference_eta_edges = None

    for flavour in FLAVORS:
        denominators[flavour] = {}
        for nbjet in NBJETS:
            wp_denominators = []
            for wp in WPS:
                den_name = histogram_name(process, "Denom", flavour, nbjet, wp)
                num_name = histogram_name(process, "Num", flavour, nbjet, wp)
                denominator, pt_edges, eta_edges = read_histogram(rootfile, den_name)
                numerator, num_pt_edges, num_eta_edges = read_histogram(rootfile, num_name)

                if not np.array_equal(pt_edges, num_pt_edges):
                    raise ValueError(f"pT edges disagree for {num_name}")
                if not np.array_equal(eta_edges, num_eta_edges):
                    raise ValueError(f"eta edges disagree for {num_name}")
                if reference_pt_edges is None:
                    reference_pt_edges = pt_edges
                    reference_eta_edges = eta_edges
                if not np.array_equal(reference_pt_edges, pt_edges):
                    raise ValueError(f"Inconsistent pT binning in {process}")
                if not np.array_equal(reference_eta_edges, eta_edges):
                    raise ValueError(f"Inconsistent eta binning in {process}")

                # Keep the original tolerance for round-off in stored counts.
                if np.any(numerator < -1.0e-9):
                    raise ValueError(f"Negative numerator in {num_name}")
                if np.any(denominator < -1.0e-9):
                    raise ValueError(f"Negative denominator in {den_name}")
                if np.any(numerator > denominator + 1.0e-9):
                    raise ValueError(f"Numerator exceeds denominator in {num_name}")
                wp_denominators.append(denominator)
                numerators[wp][(flavour, nbjet)] = numerator

            # Changing the tagging threshold must not change the selected jets.
            for other in wp_denominators[1:]:
                if not np.allclose(wp_denominators[0], other, rtol=1.0e-10, atol=1.0e-10):
                    raise ValueError(f"WP denominators disagree for {process}/{flavour}/{nbjet}")
            denominators[flavour][nbjet] = wp_denominators[0]

    return {
        "denominators": denominators,
        "numerators": numerators,
        "pt_edges": reference_pt_edges,
        "eta_edges": reference_eta_edges,
    }


def candidate_efficiencies(process_data, flavour, nbjets_to_sum):
    """Pool counts over the requested truth-b categories, then divide."""
    denominator = np.zeros_like(process_data["denominators"][flavour][next(iter(NBJETS))])
    numerators = {wp: np.zeros_like(denominator) for wp in WPS}
    for nbjet in nbjets_to_sum:
        denominator += process_data["denominators"][flavour][nbjet]
        for wp in WPS:
            numerators[wp] += process_data["numerators"][wp][(flavour, nbjet)]
    efficiencies = {wp: ratio(numerators[wp], denominator) for wp in WPS}
    return denominator, efficiencies


def global_efficiencies(process_data, flavour):
    """Last fallback: pool all pT, eta and truth-b bins for this flavour."""
    denominator = 0.0
    numerators = {wp: 0.0 for wp in WPS}
    for nbjet in NBJETS:
        denominator += float(np.sum(process_data["denominators"][flavour][nbjet]))
        for wp in WPS:
            numerators[wp] += float(np.sum(process_data["numerators"][wp][(flavour, nbjet)]))
    if denominator <= 0.0:
        raise RuntimeError(f"No denominator entries for flavour {flavour}")
    efficiencies = {wp: numerators[wp] / denominator for wp in WPS}
    as_arrays = {wp: np.asarray(efficiencies[wp]) for wp in WPS}
    if not bool(hierarchy_valid(as_arrays)):
        raise RuntimeError(
            f"Invalid global L/M/T efficiency hierarchy for flavour {flavour}: {efficiencies}"
        )
    return efficiencies


def build_final_maps(process, process_data, minimum_denominator):
    """Prefer exact bins, then high-nb, nb-inclusive and global estimates.

    The high-nb fallback applies only to nb3 and nb4p. All bin estimates must
    meet the count threshold and the L/M/T hierarchy. The global estimate
    must satisfy the hierarchy but has no minimum-count requirement.
    """
    final_maps = {}
    source_maps = {}
    summary_rows = []
    all_nbjets = list(NBJETS)

    for flavour in FLAVORS:
        global_eff = global_efficiencies(process_data, flavour)
        inclusive_den, inclusive_eff = candidate_efficiencies(process_data, flavour, all_nbjets)
        inclusive_valid = (inclusive_den >= minimum_denominator) & hierarchy_valid(inclusive_eff)
        high_den, high_eff = candidate_efficiencies(process_data, flavour, ("nb3", "nb4p"))
        high_valid = (high_den >= minimum_denominator) & hierarchy_valid(high_eff)

        for nbjet in NBJETS:
            exact_den, exact_eff = candidate_efficiencies(process_data, flavour, (nbjet,))
            exact_valid = (exact_den >= minimum_denominator) & hierarchy_valid(exact_eff)
            shape = exact_den.shape
            final = {wp: np.full(shape, global_eff[wp], dtype=np.float64) for wp in WPS}

            # Start with the broadest estimate; overwrite with better ones.
            # The CSV uses 0=exact, 1=nb3+nb4p, 2=nb-inclusive, 3=global.
            source = np.full(shape, 3, dtype=np.int32)
            for wp in WPS:
                final[wp][inclusive_valid] = inclusive_eff[wp][inclusive_valid]
            source[inclusive_valid] = 2

            if nbjet in {"nb3", "nb4p"}:
                for wp in WPS:
                    final[wp][high_valid] = high_eff[wp][high_valid]
                source[high_valid] = 1

            for wp in WPS:
                final[wp][exact_valid] = exact_eff[wp][exact_valid]
            source[exact_valid] = 0

            # A selected b jet cannot belong to nb0. Keep a global map for
            # schema completeness; the analysis should never request it.
            if flavour == "b" and nbjet == "nb0":
                source[...] = 3
                for wp in WPS:
                    final[wp][...] = global_eff[wp]

            if not np.all(hierarchy_valid(final)):
                raise RuntimeError(f"Final hierarchy failed for {process}/{flavour}/{nbjet}")
            for wp in WPS:
                final_maps[(FLAVORS[flavour], NBJETS[nbjet], wp)] = final[wp]
            source_maps[(FLAVORS[flavour], NBJETS[nbjet])] = source

            unique, counts = np.unique(source, return_counts=True)
            source_counts = {int(key): int(value) for key, value in zip(unique, counts)}
            summary_rows.append({
                "process": process,
                "flavour": flavour,
                "nbjet": nbjet,
                "total_bins": int(source.size),
                "exact_bins": source_counts.get(0, 0),
                "high_nb_bins": source_counts.get(1, 0),
                "nb_inclusive_bins": source_counts.get(2, 0),
                "global_bins": source_counts.get(3, 0),
            })
    return final_maps, source_maps, summary_rows


def multibinning_node(pt_edges, eta_edges, values):
    values = np.asarray(values, dtype=np.float64)
    expected_shape = (len(pt_edges) - 1, len(eta_edges) - 1)
    if values.shape != expected_shape:
        raise ValueError(f"Unexpected map shape {values.shape}; expected {expected_shape}")
    return {
        "nodetype": "multibinning",
        "inputs": ["pt", "abseta"],
        "edges": [pt_edges.tolist(), eta_edges.tolist()],
        # eta changes fastest for input order [pt, abseta].
        "content": values.flatten(order="C").tolist(),
        "flow": "clamp",
    }


def category_node(input_name, entries):
    return {
        "nodetype": "category",
        "input": input_name,
        "content": [{"key": key, "value": value} for key, value in entries],
    }


def process_correction_node(process_results):
    """Arrange categories as process -> WP -> flavour -> truth-b count."""
    process_entries = []
    for process in sorted(process_results):
        result = process_results[process]
        wp_entries = []
        for wp in WPS:
            flavour_entries = []
            for flavour_index in FLAVORS.values():
                nb_entries = []
                for nb_index in NBJETS.values():
                    values = result["maps"][(flavour_index, nb_index, wp)]
                    node = multibinning_node(result["pt_edges"], result["eta_edges"], values)
                    nb_entries.append((nb_index, node))
                flavour_entries.append((flavour_index, category_node("nbjets", nb_entries)))
            wp_entries.append((wp, category_node("flavour", flavour_entries)))
        process_entries.append((process, category_node("wp", wp_entries)))
    return category_node("process", process_entries)


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
    else:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)


def validate_json(path, process_results):
    """Check every bin through correctionlib, including its axis ordering."""
    import correctionlib

    correction_set = correctionlib.CorrectionSet.from_file(path)
    if "btag_eff" not in correction_set:
        raise RuntimeError("Generated JSON does not contain btag_eff")
    correction = correction_set["btag_eff"]
    expected_inputs = ("process", "wp", "flavour", "nbjets", "pt", "abseta")
    if tuple(item.name for item in correction.inputs) != expected_inputs:
        raise RuntimeError("Generated btag_eff has an unexpected input order")

    number_evaluated = 0
    for process in sorted(process_results):
        result = process_results[process]
        pt_edges = result["pt_edges"]
        eta_edges = result["eta_edges"]
        pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
        eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
        for flavour_index in FLAVORS.values():
            for nb_index in NBJETS.values():
                for ix, pt in enumerate(pt_centers):
                    for iy, eta in enumerate(eta_centers):
                        efficiencies = {}
                        for wp in WPS:
                            efficiency = correction.evaluate(
                                process, wp, flavour_index, nb_index, float(pt), float(eta)
                            )
                            expected = result["maps"][(flavour_index, nb_index, wp)][ix, iy]
                            if not np.isfinite(efficiency) or not 0.0 <= efficiency <= 1.0:
                                raise RuntimeError(f"Invalid efficiency: {efficiency}")
                            if not np.isclose(efficiency, expected, rtol=1.0e-12, atol=1.0e-12):
                                raise RuntimeError(
                                    f"JSON bin mismatch for {process}/{wp}/"
                                    f"{flavour_index}/{nb_index} at ({pt}, {eta}): "
                                    f"got {efficiency}, expected {expected}"
                                )
                            efficiencies[wp] = np.asarray(efficiency)
                            number_evaluated += 1
                        if not bool(hierarchy_valid(efficiencies)):
                            raise RuntimeError(
                                f"Invalid L/M/T hierarchy for {process}, "
                                f"flavour={flavour_index}, nb={nb_index}, pt={pt}, eta={eta}"
                            )
    print(f"[VALIDATE] Evaluated and matched {number_evaluated} JSON map points")


def main():
    args = parse_args()
    process_results = {}
    summary_rows = []
    for input_path in args.inputs:
        print(f"[INPUT] {input_path}")
        with uproot.open(input_path) as rootfile:
            processes = discover_processes(rootfile)
            if not processes:
                raise RuntimeError(f"No efficiency-map processes found in {input_path}")
            for process in processes:
                if process in process_results:
                    raise RuntimeError(f"Duplicate process key {process} in {input_path}")
                process_data = read_process(rootfile, process)
                maps, source_maps, rows = build_final_maps(
                    process, process_data, args.min_denominator
                )
                process_results[process] = {
                    "maps": maps, "sources": source_maps,
                    "pt_edges": process_data["pt_edges"],
                    "eta_edges": process_data["eta_edges"],
                }
                summary_rows.extend(rows)
                exact = sum(row["exact_bins"] for row in rows)
                total = sum(row["total_bins"] for row in rows)
                print(f"[MAP] {process}: exact={exact}/{total}, fallback={total - exact}")

    if not process_results:
        raise RuntimeError("No process maps were produced")
    correction_payload = {
        "schema_version": 2,
        "description": (
            "2024 UParTAK4 b-tag efficiencies with truth-b multiplicity "
            "dependence and documented sparse-bin fallbacks."
        ),
        "corrections": [{
            "name": "btag_eff",
            "description": (
                "B-tag efficiency as a function of process, WP, jet flavour, "
                "truth-b multiplicity, jet pT and |eta|."
            ),
            "version": 1,
            "inputs": [
                {"name": "process", "type": "string", "description": "Efficiency-map process key"},
                {"name": "wp", "type": "string", "description": "Working point: L, M or T"},
                {"name": "flavour", "type": "int", "description": "0=b, 1=c, 2=light"},
                {"name": "nbjets", "type": "int", "description": "0,1,2,3,4 where 4 means >=4"},
                {"name": "pt", "type": "real", "description": "Jet pT in GeV"},
                {"name": "abseta", "type": "real", "description": "Absolute jet eta"},
            ],
            "output": {
                "name": "efficiency", "type": "real", "description": "B-tagging efficiency"
            },
            "data": process_correction_node(process_results),
        }],
    }
    write_json(args.output, correction_payload)

    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "process", "flavour", "nbjet", "total_bins", "exact_bins",
            "high_nb_bins", "nb_inclusive_bins", "global_bins",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[DONE] Wrote {args.output} with {len(process_results)} process map(s)")
    print(f"[SUMMARY] Wrote {args.summary}")
    if args.validate:
        validate_json(args.output, process_results)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exception:
        print(f"[ERROR] {type(exception).__name__}: {exception}", file=sys.stderr)
        raise
