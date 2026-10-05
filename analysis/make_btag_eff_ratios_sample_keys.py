#!/usr/bin/env python3
"""Write ROOT efficiency maps from merged, unweighted jet counts.

Each output keeps Num and Denom alongside their binomial ratio. Process names
come from TARGETS_BY_INPUT_STEM; the two ratio scripts have different tables.
Nominal/extension counts are pooled before division. The three ttbar decay
files keep ttLF/ttCC/ttBB separate and must be converted to separate JSONs.

Requires PyROOT. Example:
    python make_btag_eff_ratios_sample_keys.py merged_counts/*_BTag.root --output-dir ratios --strict-names
"""

import argparse
import json
import math
import os
import sys

import ROOT


ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


WPS = ("L", "M", "T")
FLAVOURS = ("b", "c", "light")
NBJETS = ("nb0", "nb1", "nb2", "nb3", "nb4p")
TT_FLAVOURS = ("ttLF", "ttCC", "ttBB")


# Skip these only when the corresponding combined input is supplied.
# validate_input_paths checks this before any output files are created.
SKIP_COMPONENT_STEMS = {
    "GluGluH-Hto2B_2024",
    "GluGluH-Hto2B_ext_2024",
    "VBFH-Hto2B_2024",
    "VBFH-Hto2B_ext_2024",
}


# Expected internal prefixes in the two combined raw-count files.
COMBINED_SOURCES = {
    "GluGluH-Hto2B_combined_2024": ('GluGluH-Hto2B_2024', 'GluGluH-Hto2B_ext_2024'),
    "VBFH-Hto2B_combined_2024": ('VBFH-Hto2B_2024', 'VBFH-Hto2B_ext_2024'),
}


# Input filename stem -> process keys expected by the analysis.
# Keep spelling, hyphens, underscores and year suffixes exactly as listed.
TARGETS_BY_INPUT_STEM = {
    # Share the pooled map between the nominal and extension sample names.
    "GluGluH-Hto2B_combined_2024": ('GluGluH-Hto2B', 'GluGluH-Hto2B_ext'),
    "VBFH-Hto2B_combined_2024": ('VBFH-Hto2B', 'VBFH-Hto2B_ext'),

    # t-channel single top: sample names retain _2024.
    "TbarBQto2Q_2024": ("TbarBQto2Q_2024",),
    "TbarBQtoLNu_2024": ("TbarBQtoLNu_2024",),
    "TBbarQto2Q_2024": ("TBbarQto2Q_2024",),
    "TBbarQtoLNu_2024": ("TBbarQtoLNu_2024",),

    # s-channel single top: sample names omit _2024.
    "TbarBto2Q-s-channel_2024": ("TbarBto2Q-s-channel",),
    "TbarBtoLNu-s-channel_2024": ("TbarBtoLNu-s-channel",),
    "TBbarto2Q-s-channel_2024": ("TBbarto2Q-s-channel",),
    "TBbartoLNu-s-channel_2024": ("TBbartoLNu-s-channel",),

    # tW: sample names retain _2024.
    "TWminusto2L2Nu_2024": ("TWminusto2L2Nu_2024",),
    "TWminusto4Q_2024": ("TWminusto4Q_2024",),
    "TWminustoLNu2Q_2024": ("TWminustoLNu2Q_2024",),
    "TbarWplusto2L2Nu_2024": ("TbarWplusto2L2Nu_2024",),
    "TbarWplusto4Q_2024": ("TbarWplusto4Q_2024",),
    "TbarWplustoLNu2Q_2024": ("TbarWplustoLNu2Q_2024",),
    # QCD HT bins.
    "QCDB-4Jets_Bin-HT-100to200_2024": ("QCDB-4Jets_Bin-HT-100to200",),
    "QCDB-4Jets_Bin-HT-200to400_2024": ("QCDB-4Jets_Bin-HT-200to400",),
    "QCDB-4Jets_Bin-HT-400to600_2024": ("QCDB-4Jets_Bin-HT-400to600",),
    "QCDB-4Jets_Bin-HT-600to800_2024": ("QCDB-4Jets_Bin-HT-600to800",),
    "QCDB-4Jets_Bin-HT-800to1000_2024": ("QCDB-4Jets_Bin-HT-800to1000",),
    "QCDB-4Jets_Bin-HT-1000to1500_2024": ("QCDB-4Jets_Bin-HT-1000to1500",),
    "QCDB-4Jets_Bin-HT-1500to2000_2024": ("QCDB-4Jets_Bin-HT-1500to2000",),
    "QCDB-4Jets_Bin-HT-2000_2024": ("QCDB-4Jets_Bin-HT-2000",),
    # tt+X.
    "TTG-1Jets_Bin-PTG-200_2024": ("TTG-1Jets_Bin-PTG-200",),
    "TTG-1Jets_Bin-PTG-100_2024": ("TTG-1Jets-Bin-PTG-100",),
    "TTW-WtoQQ_2024": ("TTW-WtoQQ",),
    "TTZ-ZtoQQ_2024": ("TTZ-ZtoQQ",),
    "Zto2Q-4Jets_Bin-HT-100to400_2024": ("Zto2Q-4Jets_Bin-HT-100to400",),
    "Zto2Q-4Jets_Bin-HT-400to800_2024": ("Zto2Q-4Jets_Bin-HT-400to800",),
    "Zto2Q-4Jets_Bin-HT-800to1500_2024": ("Zto2Q-4Jets_Bin-HT-800to1500",),
    "Zto2Q-4Jets_Bin-HT-1500to2500_2024": ("Zto2Q-4Jets_Bin-HT-1500to2500",),
    "Zto2Q-4Jets_Bin-HT-2500_2024": ("Zto2Q-4Jets_Bin-HT-2500",),
    # SM Higgs.
    "TTH-Hto2B_2024": ("TTH-Hto2B",),
    "WminusH-Wto2Q-Hto2B_2024": ("WminusH-Wto2Q-Hto2B",),
    "WminusH-WtoLNu-Hto2B_2024": ("WminusH-WtoLNu-Hto2B",),
    "WplusH-Wto2Q-Hto2B_2024": ("WplusH-Wto2Q-Hto2B",),
    "WplusH-WtoLNu-Hto2B_2024": ("WplusH-WtoLNu-Hto2B",),
    "ZH-Zto2L-Hto2B_2024": ("ZH-Zto2L-Hto2B",),
    "ZH-Zto2Nu-Hto2B_2024": ("ZH-Zto2Nu-Hto2B",),
    "ZH-Zto2Q-Hto2B_2024": ("ZH-Zto2Q-Hto2B",),

    # Diboson and triboson.
    "WW_2024": ("WW",),
    "WZ_2024": ("WZ",),
    "ZZ_2024": ("ZZ",),
    "WWW_2024": ("WWW",),
    "WWZ_2024": ("WWZ",),
    "WZZ_2024": ("WZZ",),
    "ZZZ_2024": ("ZZZ",),

    # Z+jets HT bins kept as separate physical process maps.
    "Zto2Nu-4Jets-Bin-HT-100to200_2024": ('Zto2Nu-4Jets-Bin-HT-100to200',),
    "Zto2Nu-4Jets-Bin-HT-200to400_2024": ('Zto2Nu-4Jets-Bin-HT-200to400',),
    "Zto2Nu-4Jets-Bin-HT-400to800_2024": ('Zto2Nu-4Jets-Bin-HT-400to800',),
    "Zto2Nu-4Jets-Bin-HT-800to1500_2024": ('Zto2Nu-4Jets-Bin-HT-800to1500',),
    "Zto2Nu-4Jets-Bin-HT-1500to2500_2024": ('Zto2Nu-4Jets-Bin-HT-1500to2500',),
    "Zto2Nu-4Jets-Bin-HT-2500_2024": ('Zto2Nu-4Jets-Bin-HT-2500',),
    "WtoLNu-4Jets_Bin-1J_2024": ("WtoLNu-4Jets_Bin-1J",),
    "WtoLNu-4Jets_Bin-2J_2024": ("WtoLNu-4Jets_Bin-2J",),
    "WtoLNu-4Jets_Bin-3J_2024": ("WtoLNu-4Jets_Bin-3J",),
    "WtoLNu-4Jets_Bin-4J_2024": ("WtoLNu-4Jets_Bin-4J",),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input *_BTag.root files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--strict-names",
        action="store_true",
        help="Reject non-signal input stems absent from the explicit mapping",
    )
    return parser.parse_args()


def input_stem(path):
    stem = os.path.basename(path)
    if stem.endswith(".root"):
        stem = stem[:-5]
    if stem.endswith("_BTag"):
        stem = stem[:-5]
    return stem


def root_keys(root_file):
    return [key.GetName() for key in root_file.GetListOfKeys()]


def find_prefixes(root_file):
    prefixes = set()
    for name in root_keys(root_file):
        for flavour in FLAVOURS:
            for nb in NBJETS:
                for wp in WPS:
                    suffix = f"_BTagEff_Denom_{flavour}_{nb}_WP{wp}"
                    if name.endswith(suffix):
                        prefixes.add(name[: -len(suffix)])
                    elif name == f"BTagEff_Denom_{flavour}_{nb}_WP{wp}":
                        prefixes.add("")
    return sorted(prefixes)


def object_name(prefix, kind, flavour, nb, wp):
    head = f"{prefix}_" if prefix else ""
    return f"{head}BTagEff_{kind}_{flavour}_{nb}_WP{wp}"


def clone_detached(histogram, name):
    clone = histogram.Clone(name)
    clone.SetDirectory(0)
    return clone


def axis_edges(axis):
    edges = [axis.GetBinLowEdge(index) for index in range(1, axis.GetNbins() + 1)]
    return edges + [axis.GetBinUpEdge(axis.GetNbins())]


def assert_compatible(reference, candidate, label):
    if reference.GetDimension() != 2 or candidate.GetDimension() != 2:
        raise RuntimeError(f"{label}: expected TH2 histograms")
    if reference.GetNbinsX() != candidate.GetNbinsX():
        raise RuntimeError(f"{label}: different X bin counts")
    if reference.GetNbinsY() != candidate.GetNbinsY():
        raise RuntimeError(f"{label}: different Y bin counts")
    if axis_edges(reference.GetXaxis()) != axis_edges(candidate.GetXaxis()):
        raise RuntimeError(f"{label}: different X bin edges")
    if axis_edges(reference.GetYaxis()) != axis_edges(candidate.GetYaxis()):
        raise RuntimeError(f"{label}: different Y bin edges")


def summed_histogram(root_file, sources, kind, flavour, nb, wp, output_name):
    """Add raw counts; averaging precomputed efficiencies would be biased."""
    result = None
    for source in sources:
        name = object_name(source, kind, flavour, nb, wp)
        histogram = root_file.Get(name)
        if not histogram:
            raise KeyError(f"Missing ROOT object {name}")

        if result is None:
            result = clone_detached(histogram, output_name)
        else:
            assert_compatible(result, histogram, name)
            result.Add(histogram)

    if result is None:
        raise RuntimeError(f"No source histograms for {output_name}")
    result.SetName(output_name)
    return result


def process_maps(stem, prefixes, strict_names):
    prefix_set = set(prefixes)

    if stem in {"TTto2L2Nu_2024", "TTtoLNu2Q_2024", "TTto4Q_2024"}:
        missing = [key for key in TT_FLAVOURS if key not in prefix_set]
        if missing:
            raise RuntimeError(
                f"{stem}: missing TT prefixes {missing}; available={prefixes}"
            )
        unexpected = prefix_set - set(TT_FLAVOURS)
        if unexpected:
            raise RuntimeError(f"{stem}: unexpected TT prefixes {sorted(unexpected)}")
        return {key: (key,) for key in TT_FLAVOURS}

    if stem in COMBINED_SOURCES:
        expected = set(COMBINED_SOURCES[stem])
        missing = expected - prefix_set
        unexpected = prefix_set - expected
        if missing or unexpected:
            raise RuntimeError(
                f"{stem}: combined prefixes mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}, available={prefixes}"
            )
        sources = COMBINED_SOURCES[stem]
    else:
        if len(prefixes) != 1:
            raise RuntimeError(
                f"{stem}: expected exactly one source prefix, got {prefixes}"
            )
        sources = (prefixes[0],)

    targets = TARGETS_BY_INPUT_STEM.get(stem)
    if targets is None:
        # Keep the mass-specific signal process name.
        if stem.startswith("ZH-ZToAll-HToAATo4B_Par-M-"):
            targets = (stem,)
        elif strict_names:
            raise KeyError(
                f"No explicit metadata sample mapping for input stem {stem!r}"
            )
        else:
            targets = (stem,)
            print(f"[WARN] {stem}: no explicit mapping; using target={stem}")

    return {target: tuple(sources) for target in targets}


def validate_input_paths(paths):
    """Reject missing combined inputs before skipping their components."""
    stems = {input_stem(path) for path in paths}
    for combined, components in COMBINED_SOURCES.items():
        supplied = stems.intersection(components)
        if supplied and combined not in stems:
            raise ValueError(
                f"Found {sorted(supplied)} without {combined}_BTag.root. "
                "Include the combined file containing both internal prefixes."
            )
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)


def validate_counts(numerator, denominator, label):
    """Check the visible bins before using a binomial efficiency."""
    assert_compatible(denominator, numerator, label)
    for ix in range(1, denominator.GetNbinsX() + 1):
        for iy in range(1, denominator.GetNbinsY() + 1):
            num = numerator.GetBinContent(ix, iy)
            den = denominator.GetBinContent(ix, iy)
            if not (math.isfinite(num) and math.isfinite(den)):
                raise ValueError(f"{label}: non-finite counts in bin ({ix}, {iy})")
            if num < 0.0 or den < 0.0 or num > den:
                raise ValueError(
                    f"{label}: invalid counts in bin ({ix}, {iy}): "
                    f"Num={num}, Denom={den}"
                )


def count_zero_denominator_bins(histogram):
    count = 0
    for ix in range(1, histogram.GetNbinsX() + 1):
        for iy in range(1, histogram.GetNbinsY() + 1):
            if histogram.GetBinContent(ix, iy) <= 0.0:
                count += 1
    return count


def write_metadata(output_file, stem, maps):
    metadata = {
        "input_stem": stem,
        "process_maps": {target: list(sources) for target, sources in maps.items()},
        "working_points": list(WPS),
        "flavours": list(FLAVOURS),
        "nb_categories": list(NBJETS),
    }
    output_file.cd()
    ROOT.TObjString(json.dumps(metadata, sort_keys=True)).Write(
        "btag_eff_ratio_metadata"
    )


def process_file(path, output_dir, claimed_targets, strict_names):
    stem = input_stem(path)
    # main() has already checked that the combined file is present.
    if stem in SKIP_COMPONENT_STEMS:
        print(f"[SKIP COMPONENT] {path}")
        return None

    input_file = ROOT.TFile.Open(path, "READ")
    if not input_file or input_file.IsZombie():
        raise OSError(f"Cannot open input ROOT file {path}")

    try:
        prefixes = find_prefixes(input_file)
        if not prefixes:
            raise RuntimeError(f"{path}: no BTagEff denominator histograms found")

        maps = process_maps(stem, prefixes, strict_names)
        is_tt_decay_file = stem in {"TTto2L2Nu_2024", "TTtoLNu2Q_2024", "TTto4Q_2024"}
        for target in maps:
            # TT flavour keys may repeat across decay files. Other process
            # keys must be unique across the inputs for a common JSON.
            claim_key = f"{stem}:{target}" if is_tt_decay_file else target
            if claim_key in claimed_targets:
                raise RuntimeError(
                    f"Duplicate target process {target!r}: already produced from "
                    f"{claimed_targets[claim_key]}, encountered again in {path}"
                )
            claimed_targets[claim_key] = path

        output_path = os.path.join(output_dir, f"{stem}_BTag_ratio.root")
        output_file = ROOT.TFile.Open(output_path, "RECREATE")
        if not output_file or output_file.IsZombie():
            raise OSError(f"Cannot create output ROOT file {output_path}")

        maps_written = 0
        try:
            write_metadata(output_file, stem, maps)
            for target, sources in maps.items():
                print(f"[PROCESS MAP] target={target}, sources={list(sources)}")
                for flavour in FLAVOURS:
                    for nb in NBJETS:
                        for wp in WPS:
                            num_name = object_name(target, "Num", flavour, nb, wp)
                            den_name = object_name(target, "Denom", flavour, nb, wp)
                            eff_name = f"{target}_BTagEff_{flavour}_{nb}_WP{wp}"

                            numerator = summed_histogram(
                                input_file, sources, "Num", flavour, nb, wp,
                                num_name,
                            )
                            denominator = summed_histogram(
                                input_file, sources, "Denom", flavour, nb, wp,
                                den_name,
                            )
                            validate_counts(numerator, denominator, eff_name)

                            # Use binomial errors; empty denominator bins stay zero.
                            efficiency = clone_detached(numerator, eff_name)
                            efficiency.Reset("ICES")
                            efficiency.Divide(numerator, denominator, 1.0, 1.0, "B")
                            efficiency.SetTitle(
                                f"{target} {flavour}, {nb}, WP {wp};"
                                "p_{T} [GeV];|#eta|;Efficiency"
                            )
                            efficiency.SetMinimum(0.0)
                            efficiency.SetMaximum(1.0)

                            zero_bins = count_zero_denominator_bins(denominator)
                            if zero_bins:
                                print(
                                    f"[INFO] {eff_name}: {zero_bins} visible "
                                    "bins have zero denominator"
                                )

                            output_file.cd()
                            denominator.Write()
                            numerator.Write()
                            efficiency.Write()
                            maps_written += 1
        finally:
            output_file.Close()

        print(f"[OK] wrote {output_path} with {maps_written} 2D efficiency maps")
        return output_path
    finally:
        input_file.Close()


def main():
    args = parse_args()
    validate_input_paths(args.inputs)
    os.makedirs(args.output_dir, exist_ok=True)

    claimed_targets = {}
    outputs = []
    for path in args.inputs:
        print(f"\n[INPUT] {path}")
        output = process_file(path, args.output_dir, claimed_targets, args.strict_names)
        if output is not None:
            outputs.append(output)

    print(f"[DONE] Processed {len(outputs)} ROOT file(s)")
    print(f"[DONE] Produced {len(claimed_targets)} process map key(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
