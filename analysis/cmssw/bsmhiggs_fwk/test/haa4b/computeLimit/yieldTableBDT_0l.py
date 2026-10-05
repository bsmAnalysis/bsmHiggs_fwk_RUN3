#!/usr/bin/env python3

"""Create a grouped-BDT yield table for the 0-lepton veto channel.

Only the ``veto_A_SR_3b`` channel is read. BDT bins 1 and 2 are merged,
while bins 3 and 4 are printed separately. Data are shown only in the
unblinded [1,2] group; the data entries in bins 3 and 4 are left blank.
"""

import argparse
import os
import sys
from array import array

import ROOT as rt


rt.gROOT.SetBatch(True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Make the grouped-BDT yield table for veto_A_SR_3b."
    )
    parser.add_argument(
        "limit_dir",
        help=(
            "Main 0-lepton card directory, or one of its mass directories "
            "such as .../0060."
        ),
    )
    parser.add_argument(
        "--prefit",
        action="store_true",
        help="Use shapes_prefit instead of the default shapes_fit_b.",
    )
    parser.add_argument(
        "--fit-mass",
        type=int,
        default=60,
        help="Mass directory containing fitDiagnosticsTest.root (default: 60).",
    )
    parser.add_argument(
        "--signal-masses",
        type=int,
        nargs="+",
        default=[30, 60],
        help="Signal mass points to print (default: 30 60).",
    )
    parser.add_argument(
        "--regime",
        choices=["resolved", "boosted"],
        default=None,
        help="Analysis regime. If omitted, infer it from the directory name.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .tex path. By default it is written in the card directory.",
    )
    return parser.parse_args()


def open_root_file(path):
    root_file = rt.TFile.Open(path)
    if not root_file or root_file.IsZombie():
        raise RuntimeError("Cannot open ROOT file: {}".format(path))
    return root_file


def infer_cards_directory(input_directory):
    normalized = os.path.abspath(os.path.normpath(input_directory))
    if os.path.basename(normalized).isdigit():
        return os.path.dirname(normalized)
    return normalized


def infer_regime(cards_directory, requested_regime):
    if requested_regime:
        return requested_regime

    directory_name = cards_directory.lower()
    if "boosted" in directory_name or "boost" in directory_name:
        return "boosted"
    if "resolved" in directory_name or "resolv" in directory_name:
        return "resolved"
    return "0-lepton"


def get_first_existing(root_file, candidate_paths, required=True):
    for path in candidate_paths:
        obj = root_file.Get(path)
        if obj:
            return obj, path

    if required:
        raise RuntimeError(
            "None of these ROOT objects exists in '{}': {}".format(
                root_file.GetName(), ", ".join(candidate_paths)
            )
        )
    return None, None


def grouped_histogram_yield(histogram, first_bin, last_bin):
    if histogram.GetNbinsX() < last_bin:
        raise RuntimeError(
            "Histogram '{}' has {} bins, but bin {} was requested.".format(
                histogram.GetName(), histogram.GetNbinsX(), last_bin
            )
        )
    return sum(
        histogram.GetBinContent(bin_number)
        for bin_number in range(first_bin, last_bin + 1)
    )


def graph_point_y(graph, point_index):
    x_value = array("d", [0.0])
    y_value = array("d", [0.0])
    graph.GetPoint(point_index, x_value, y_value)
    return y_value[0]


def grouped_data_yield(data_object, first_bin, last_bin):
    if data_object.InheritsFrom("TH1"):
        return grouped_histogram_yield(data_object, first_bin, last_bin)

    if data_object.InheritsFrom("TGraph"):
        if data_object.GetN() < last_bin:
            raise RuntimeError(
                "Data graph '{}' has {} points, but point {} was requested.".format(
                    data_object.GetName(), data_object.GetN(), last_bin
                )
            )
        return sum(
            graph_point_y(data_object, bin_number - 1)
            for bin_number in range(first_bin, last_bin + 1)
        )

    raise RuntimeError(
        "Unsupported data object type '{}' for '{}'.".format(
            data_object.ClassName(), data_object.GetName()
        )
    )


def object_paths(shape_directory, channel, process_aliases):
    return [
        "{}/{}/{}".format(shape_directory, channel, process_name)
        for process_name in process_aliases
    ]


def main():
    args = parse_arguments()

    cards_directory = infer_cards_directory(args.limit_dir)
    regime = infer_regime(cards_directory, args.regime)
    fit_directory = os.path.join(cards_directory, "{:04d}".format(args.fit_mass))
    fit_path = os.path.join(fit_directory, "fitDiagnosticsTest.root")
    shape_directory = "shapes_prefit" if args.prefit else "shapes_fit_b"
    channel = "veto_A_SR_3b"

    output_path = args.output or os.path.join(
        cards_directory,
        "yield-table-in-bdt-bins-2024-zh-A-{}-0l-veto.tex".format(regime),
    )

    bin_groups = [
        {"first": 1, "last": 2, "label": "[1,2]", "blind": False},
        {"first": 3, "last": 3, "label": "[3,3]", "blind": True},
        {"first": 4, "last": 4, "label": "[4,4]", "blind": True},
    ]

    # Multiple aliases are accepted because some card versions use shortened
    # process names. A row is skipped, with a warning, if none of its aliases
    # is present in the FitDiagnostics file.
    background_rows = [
        (("otherbkg",), "Other Bkgs"),
        (("znunu", "zvv"), r"$Z\to\nu\bar{\nu}$"),
        (("wjets", "wjet"), r"$W\to\ell\nu+\mathrm{jets}$"),
        (("ttbarbba", "ttbb"), r"$t\bar{t}+b\bar{b}$"),
        (("ttbarcba", "ttcc"), r"$t\bar{t}+c\bar{c}$"),
        (("ttbarlig", "ttlight"), r"$t\bar{t}+\mathrm{light}$"),
        (("ddqcd", "qcd"), "DD-QCD"),
    ]

    fit_file = open_root_file(fit_path)

    table_rows = [
        r"\begin{tabular}{|l|ccc|}",
        r"\hline",
        (
            r"\multicolumn{4}{|c|}{\textbf{2024: $\PZ\PH$, zero-lepton "
            + regime
            + r" regime, signal region}} \\"
        ),
        r"\hline",
        r"Process & veto [1,2] & veto [3,3] & veto [4,4] \\",
        r"\hline",
    ]

    for process_aliases, process_label in background_rows:
        histogram, matched_path = get_first_existing(
            fit_file,
            object_paths(shape_directory, channel, process_aliases),
            required=False,
        )
        if not histogram:
            print(
                "WARNING: skipping '{}'; none of {} was found.".format(
                    process_label, ", ".join(process_aliases)
                ),
                file=sys.stderr,
            )
            continue

        row = [process_label]
        for group in bin_groups:
            value = grouped_histogram_yield(
                histogram, group["first"], group["last"]
            )
            row.append("{:.1f}".format(value))
        table_rows.append(" & ".join(row) + r" \\")

    table_rows.append(r"\hline")

    total_background, _ = get_first_existing(
        fit_file,
        object_paths(shape_directory, channel, ("total_background",)),
    )
    total_row = ["Total Bkg"]
    for group in bin_groups:
        value = grouped_histogram_yield(
            total_background, group["first"], group["last"]
        )
        total_row.append("{:.1f}".format(value))
    table_rows.append(" & ".join(total_row) + r" \\")

    table_rows.append(r"\hline")

    data_object, _ = get_first_existing(
        fit_file,
        object_paths(shape_directory, channel, ("data", "data_obs")),
    )
    data_row = ["Data"]
    for group in bin_groups:
        if group["blind"]:
            data_row.append("")
        else:
            value = grouped_data_yield(
                data_object, group["first"], group["last"]
            )
            data_row.append("{:.0f}".format(value))
    table_rows.append(" & ".join(data_row) + r" \\")

    table_rows.append(r"\hline")

    for signal_mass in args.signal_masses:
        signal_path = os.path.join(
            cards_directory,
            "{:04d}".format(signal_mass),
            "haa4b_{}_13p6TeV_zh.root".format(signal_mass),
        )
        signal_file = open_root_file(signal_path)
        signal_histogram, _ = get_first_existing(
            signal_file,
            ["{}/zh".format(channel)],
        )

        signal_row = [rf"$\PZ\PH$ ($m_{{\Pa}}={signal_mass}\GeV)"]
        for group in bin_groups:
            value = grouped_histogram_yield(
                signal_histogram, group["first"], group["last"]
            )
            signal_row.append("{:.1f}".format(value))
        table_rows.append(" & ".join(signal_row) + r" \\")
        signal_file.Close()

    table_rows.extend([r"\hline", r"\end{tabular}"])

    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as tex_file:
        tex_file.write("\n".join(table_rows) + "\n")

    fit_file.Close()
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
