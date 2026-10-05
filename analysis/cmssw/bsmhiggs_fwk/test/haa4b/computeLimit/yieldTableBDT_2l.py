#!/usr/bin/env python3

"""Create a combined ee/mumu yield table in grouped BDT bins.

The output columns are
  ee [1,2], ee [3,3], ee [4,4], mumu [1,2], mumu [3,3], mumu [4,4].

Bins 1 and 2 are merged. Data are printed only in the merged, unblinded
group [1,2]; the data cells for bins 3 and 4 are intentionally left empty.
The same script supports both the resolved and boosted regimes.
"""

import argparse
import os
import sys
from array import array

import ROOT as rt


rt.gROOT.SetBatch(True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Make a combined ee/mumu yield table in grouped BDT bins."
    )
    parser.add_argument(
        "limit_dir",
        help=(
            "Limit-card directory, or one of its mass subdirectories "
            "(for example .../resolved_3b_2l_T_1 or .../resolved_3b_2l_T_1/0060)."
        ),
    )
    parser.add_argument(
        "--prefit",
        action="store_true",
        help="Read backgrounds from shapes_prefit instead of shapes_fit_b.",
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
        "--channel-suffix",
        default="A_SR_3b",
        help=(
            "ROOT channel suffix appended to ee_ and mumu_ "
            "(default: A_SR_3b)."
        ),
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


def get_object(root_file, path):
    obj = root_file.Get(path)
    if not obj:
        raise RuntimeError(
            "Missing ROOT object '{}' in '{}'".format(path, root_file.GetName())
        )
    return obj


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
    """Sum data represented either by a TH1 or by a TGraphAsymmErrors."""
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


def infer_cards_directory(input_directory):
    normalized = os.path.abspath(os.path.normpath(input_directory))
    basename = os.path.basename(normalized)

    # Accept either the main card directory or a mass directory such as 0060.
    if basename.isdigit():
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

    raise RuntimeError(
        "Could not infer the regime from '{}'. Pass --regime resolved or "
        "--regime boosted.".format(cards_directory)
    )


def format_mc_yield(value):
    return "{:.1f}".format(value)


def format_data_yield(value):
    return "{:.0f}".format(value)


def main():
    args = parse_arguments()

    cards_directory = infer_cards_directory(args.limit_dir)
    regime = infer_regime(cards_directory, args.regime)
    fit_directory = os.path.join(cards_directory, "{:04d}".format(args.fit_mass))
    fit_path = os.path.join(fit_directory, "fitDiagnosticsTest.root")
    output_path = args.output or os.path.join(
        cards_directory,
        "yield-table-2024-zh-A-{}-2l.tex".format(regime),
    )

    shape_directory = "shapes_prefit" if args.prefit else "shapes_fit_b"
    channels = [
        ("ee", "ee_{}".format(args.channel_suffix)),
        ("mumu", "mumu_{}".format(args.channel_suffix)),
    ]
    bin_groups = [
        {"first": 1, "last": 2, "label": "[1,2]", "blind": False},
        {"first": 3, "last": 3, "label": "[3,3]", "blind": True},
        {"first": 4, "last": 4, "label": "[4,4]", "blind": True},
    ]

    # The order here is also the order used in the output table.
    background_rows = [
        ("otherbkg", "Other Bkgs"),
        ("zll", r"$Z\to\ell\ell$"),
        ("ttbarbba", r"$t\bar{t}+b\bar{b}$"),
        ("ttbarcba", r"$t\bar{t}+c\bar{c}$"),
        ("ttbarlig", r"$t\bar{t}+\mathrm{light}$"),
    ]

    fit_file = open_root_file(fit_path)

    table_rows = []
    table_rows.append(r"\begin{tabular}{|l|ccc|ccc|}")
    table_rows.append(r"\hline")
    table_rows.append(
        r"\multicolumn{7}{|c|}{\textbf{2024: $\PZ\PH$, two-lepton channel, "
        + regime
        + r" regime, signal region}} \\"
    )
    table_rows.append(r"\hline")

    header_cells = ["Process"]
    for flavour, _ in channels:
        display_flavour = r"$\mu\mu$" if flavour == "mumu" else r"$ee$"
        for group in bin_groups:
            header_cells.append("{} {}".format(display_flavour, group["label"]))
    table_rows.append(" & ".join(header_cells) + r" \\")
    table_rows.append(r"\hline")

    for process_name, process_label in background_rows:
        row = [process_label]
        for _, channel_name in channels:
            histogram = get_object(
                fit_file,
                "{}/{}/{}".format(shape_directory, channel_name, process_name),
            )
            for group in bin_groups:
                value = grouped_histogram_yield(
                    histogram, group["first"], group["last"]
                )
                row.append(format_mc_yield(value))
        table_rows.append(" & ".join(row) + r" \\")

    table_rows.append(r"\hline")

    total_row = ["Total Bkg"]
    for _, channel_name in channels:
        total_background = get_object(
            fit_file,
            "{}/{}/total_background".format(shape_directory, channel_name),
        )
        for group in bin_groups:
            value = grouped_histogram_yield(
                total_background, group["first"], group["last"]
            )
            total_row.append(format_mc_yield(value))
    table_rows.append(" & ".join(total_row) + r" \\")

    table_rows.append(r"\hline")

    data_row = ["Data"]
    for _, channel_name in channels:
        data_object = get_object(
            fit_file, "{}/{}/data".format(shape_directory, channel_name)
        )
        for group in bin_groups:
            if group["blind"]:
                data_row.append("")
            else:
                value = grouped_data_yield(
                    data_object, group["first"], group["last"]
                )
                data_row.append(format_data_yield(value))
    table_rows.append(" & ".join(data_row) + r" \\")

    table_rows.append(r"\hline")

    for signal_mass in args.signal_masses:
        signal_path = os.path.join(
            cards_directory,
            "{:04d}".format(signal_mass),
            "haa4b_{}_13p6TeV_zh.root".format(signal_mass),
        )
        signal_file = open_root_file(signal_path)
        signal_row = [rf"$\PZ\PH$ ($m_{{\Pa}}={signal_mass}\GeV)"]

        for _, channel_name in channels:
            signal_histogram = get_object(
                signal_file, "{}/zh".format(channel_name)
            )
            for group in bin_groups:
                value = grouped_histogram_yield(
                    signal_histogram, group["first"], group["last"]
                )
                signal_row.append(format_mc_yield(value))

        table_rows.append(" & ".join(signal_row) + r" \\")
        signal_file.Close()

    table_rows.append(r"\hline")
    table_rows.append(r"\end{tabular}")

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
