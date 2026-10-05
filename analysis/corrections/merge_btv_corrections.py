#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import gzip
import os
import pathlib
import tempfile
from typing import Generator, Literal

import correctionlib.schemav2


def read_correction_set(path: str | pathlib.Path) -> correctionlib.schemav2.CorrectionSet:
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if path.endswith(".gz"):
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            with gzip.open(path, "rt") as f:
                tmp.write(f.read().encode())
            tmp.flush()
            return correctionlib.schemav2.CorrectionSet.parse_file(tmp.name)

    return correctionlib.schemav2.CorrectionSet.parse_file(path)

def write_correction_set(
    cset: correctionlib.schemav2.CorrectionSet,
    path: str | pathlib.Path,
) -> None:
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))
    outdir = os.path.dirname(path)

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    json_text = cset.json(exclude_unset=True)

    if path.endswith(".gz"):
        with gzip.open(path, "wt") as f:
            f.write(json_text)
    else:
        with open(path, "w") as f:
            f.write(json_text)

    print(f"[OK] Wrote: {path}")
'''
def write_correction_set(
    cset: correctionlib.schemav2.CorrectionSet,
    path: str | pathlib.Path,
) -> None:
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))
    outdir = os.path.dirname(path)

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    if path.endswith(".gz"):
        with gzip.open(path, "wt") as f:
            f.write(cset.model_dump_json(exclude_unset=True))
    else:
        with open(path, "w") as f:
            f.write(cset.model_dump_json(exclude_unset=True))

    print(f"[OK] Wrote: {path}")

'''
def get_correction(
    cset: correctionlib.schemav2.CorrectionSet,
    name: str,
) -> correctionlib.schemav2.Correction:
    for corr in cset.corrections:
        if corr.name == name:
            return corr
    available = [corr.name for corr in cset.corrections]
    raise KeyError(f"Correction '{name}' not found. Available: {available}")


def iter_btv_correction(
    corr: correctionlib.schemav2.Correction,
    *,
    stop: Literal["systematic", "working_point", "flavor"] | None = None,
) -> Generator[tuple, None, None]:
    syst_items = list(corr.data.content)

    for i, syst_item in enumerate(syst_items):
        if syst_item.key == "central":
            syst_items.insert(0, syst_items.pop(i))
            break

    for syst_item in syst_items:
        if stop == "systematic":
            yield syst_item
            continue

        syst_cat = syst_item.value

        for wp_item in syst_cat.content:
            if stop == "working_point":
                yield syst_item, wp_item
                continue

            flavor_cat = wp_item.value

            for flavor_item in flavor_cat.content:
                if stop == "flavor":
                    yield syst_item, wp_item, flavor_item
                    continue

                eta_binning = flavor_item.value

                for i, pt_binning in enumerate(eta_binning.content):
                    eta_edges = (eta_binning.edges[i], eta_binning.edges[i + 1])
                    sf_values = pt_binning.content
                    yield (
                        syst_item,
                        wp_item,
                        flavor_item,
                        eta_binning,
                        eta_edges,
                        pt_binning,
                        sf_values,
                    )


def merge_2024_final_btv_corrections(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> None:
    cset = read_correction_set(input_path)

    # Final 2024 recommendation:
    # - UParTAK4_comb  for b/c jets, hadronFlavour = 5 or 4
    # - UParTAK4_light for light jets, hadronFlavour = 0
    bc_corr = copy.deepcopy(get_correction(cset, "UParTAK4_comb"))
    light_corr = copy.deepcopy(get_correction(cset, "UParTAK4_light"))

    bc_central = None
    light_central = None

    # Rename non-central b/c systematics with _bc suffix.
    for syst_item in iter_btv_correction(bc_corr, stop="systematic"):
        if syst_item.key == "central":
            bc_central = syst_item
        else:
            syst_item.key = f"{syst_item.key}_bc"

    # Rename non-central light systematics with _light suffix.
    for syst_item in iter_btv_correction(light_corr, stop="systematic"):
        if syst_item.key == "central":
            light_central = syst_item
        else:
            syst_item.key = f"{syst_item.key}_light"

    if bc_central is None:
        raise RuntimeError("Could not find central systematic in UParTAK4_comb")

    if light_central is None:
        raise RuntimeError("Could not find central systematic in UParTAK4_light")

    # For b/c-only variations, add central light response.
    for syst_item in list(iter_btv_correction(bc_corr, stop="systematic")):
        if syst_item.key.endswith("_bc"):
            cloned = copy.deepcopy(light_central)
            cloned.key = syst_item.key
            light_corr.data.content.append(cloned)

    # For light-only variations, add central b/c response.
    for syst_item in list(iter_btv_correction(light_corr, stop="systematic")):
        if syst_item.key.endswith("_light"):
            cloned = copy.deepcopy(bc_central)
            cloned.key = syst_item.key
            bc_corr.data.content.append(cloned)

    merged_corr = copy.deepcopy(bc_corr)
    merged_corr.name = "UParTAK4_merged"
    merged_corr.version = 1
    merged_corr.description = (
        "Scale factors for b, c and light jets. Created by merging "
        "'UParTAK4_comb' for b/c jets and 'UParTAK4_light' for light jets. "
        "Systematic variations ending in '_bc' affect only hadronFlavour 4/5 "
        "and return central values for hadronFlavour 0. Variations ending in "
        "'_light' affect only hadronFlavour 0 and return central values for "
        "hadronFlavour 4/5."
    )

    # Merge light flavor categories into every matching systematic/WP node.
    for syst_item, wp_item in iter_btv_correction(merged_corr, stop="working_point"):
        matched = False

        for light_syst_item, light_wp_item in iter_btv_correction(
            light_corr,
            stop="working_point",
        ):
            if (syst_item.key, wp_item.key) == (
                light_syst_item.key,
                light_wp_item.key,
            ):
                wp_item.value.content += copy.deepcopy(light_wp_item.value.content)
                matched = True
                break

        if not matched:
            raise RuntimeError(
                f"No matching light correction for systematic={syst_item.key}, "
                f"WP={wp_item.key}"
            )

    # Remove previous merged correction if present.
    cset.corrections = [
        corr for corr in cset.corrections if corr.name != "UParTAK4_merged"
    ]

    cset.corrections.append(merged_corr)

    write_correction_set(cset, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge final 2024 BTV AK4 UParT fixed-WP SFs into one correction: "
            "UParTAK4_merged = UParTAK4_comb(b/c) + UParTAK4_light(light)."
        )
    )
    parser.add_argument(
        "input_path",
        help="Input BTV btagging.json.gz file",
    )
    parser.add_argument(
        "output_path",
        help="Output merged JSON or JSON.GZ file",
    )

    args = parser.parse_args()

    merge_2024_final_btv_corrections(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
