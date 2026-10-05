#!/usr/bin/env python3

"""Minimal launcher for ZH analysis.
  4.0  build and run the 0-lepton limit jobs with computeLimit_0l
  4.1  build and run the 2-lepton limit jobs with computeLimit_2l
  6.0  merge and plot the 0-lepton limits
  6.1  merge and plot the 2-lepton limits
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Iterable, List, Optional, Sequence
PHASES = (4.0, 4.1, 6.0, 6.1)
DEFAULT_MASSES = {
    "boosted": [12, 15, 20, 25, 30],
    "resolved": [15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
}
DEFAULT_INPUTS = {
    "2024": "plotter_ZH_2024_2024_03_04_forLimits.root",
}
DEFAULT_JSONS = {
    "2024": "samples2024.json",
}
LUMINOSITY_PB = {
    "2016": 36330.0,
    "2017": 41307.99,
    "2018": 59740.565,
    "2024": 108960.0,
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Create 0l/2l ZH limit jobs or merge and plot their expected "
            "limits. Only phases 4.0, 4.1, 6.0 and 6.1 are supported."
        ),
    )
    parser.add_argument("year", choices=sorted(DEFAULT_INPUTS))
    parser.add_argument("phase", type=float, choices=PHASES)
    parser.add_argument(
        "--regime",
        choices=("boosted", "resolved", "all"),
        default="all",
        help="Run/plot the boosted regime, resolved regime, or both.",
    )
    parser.add_argument(
        "--masses",
        type=int,
        nargs="+",
        help=(
            "Override the default mass points. With --regime all, the same "
            "override is used for both regimes."
        ),
    )
    parser.add_argument("--bin", dest="analysis_bin", default="3b")
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        help="Override the default plotter ROOT file.",
    )
    parser.add_argument(
        "-j",
        "--json",
        dest="json_file",
        help="Override the default sample JSON file.",
    )
    parser.add_argument(
        "-o",
        "--workdir",
        default=os.getcwd(),
        help="Directory in which JOBS/, WORK/ and cards_* are created.",
    )
    parser.add_argument(
        "--sample-key",
        default="haa_mcbased",
        help="Sample-selection key passed to computeLimit.",
    )
    parser.add_argument(
        "--stat-mode",
        choices=("none", "correlated", "hybrid", "automcstats"),
        default="automcstats",
        help="Finite-template-statistics treatment.",
    )
    parser.add_argument(
        "--autoMCStats-threshold",
        type=int,
        default=10,
        help="Combine autoMCStats effective-event threshold.",
    )
    parser.add_argument(
        "--hybrid-stat-threshold",
        type=float,
        default=0.35,
        help="Threshold used for both custom hybrid-statistics tests.",
    )
    parser.add_argument(
        "--exe-0l",
        default="computeLimit_0l",
        help="Name or path of the executable built from computeLimit_0l.cc.",
    )
    parser.add_argument(
        "--exe-2l",
        default="computeLimit_2l",
        help="Name or path of the executable built from computeLimit_2l.cc.",
    )
    parser.add_argument(
        "--no-systematics",
        action="store_true",
        help="Do not pass --syst to computeLimit (intended only for tests).",
    )
    parser.add_argument(
        "--unblind",
        action="store_true",
        help="Use observed SR data. Do not use during the blinded analysis.",
    )
    parser.add_argument(
        "--r-min",
        type=float,
        help="Override the default lower POI boundary (-0.3 for 0l, -1 for 2l).",
    )
    parser.add_argument("--r-max", type=float, default=2.0)
    parser.add_argument(
        "--run-impacts",
        action="store_true",
        help="Also produce background-only Asimov impacts for every mass.",
    )
    parser.add_argument(
        "--run-nll",
        action="store_true",
        help="Also produce the total background-only Asimov NLL scan.",
    )
    parser.add_argument(
        "--run-gof",
        action="store_true",
        help="Also run the saturated background-only goodness-of-fit test.",
    )
    parser.add_argument("--gof-toys", type=int, default=500)
    parser.add_argument(
        "--noSubmit",
        dest="no_submit",
        action="store_true",
        help="Create the mass scripts and Condor description without submitting.",
    )
    parser.add_argument(
        "--farm-dir",
        default="FARM",
        help=(
            "LaunchOnCondor farm directory. It must remain relative; an "
            "absolute path is incompatible with LaunchOnCondor."
        ),
    )
    parser.add_argument("--queue", default="cmscaf1nd")
    parser.add_argument(
        "--cards-dir",
        help=(
            "For phase 6.x, plot this existing cards directory instead of "
            "the directory name constructed by this launcher. Requires one "
            "specific --regime."
        ),
    )
    parser.add_argument(
        "--plot-macro",
        default="plotLimit.C",
        help="ROOT limit-plot macro used by phases 6.0 and 6.1.",
    )
    parser.add_argument(
        "--lumi-pb",
        type=float,
        help="Luminosity passed to plotLimit.C and computeLimit, in pb^-1.",
    )
    parser.add_argument(
        "--linear-limits",
        action="store_true",
        help="Use a linear y axis in the limit plot.",
    )
    return parser


def quote_join(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def command(parts: Sequence[object], log: str) -> str:
    return quote_join(parts) + " > " + shlex.quote(log) + " 2>&1\n"


def selected_regimes(name: str) -> List[str]:
    return ["boosted", "resolved"] if name == "all" else [name]


def analysis_for_phase(phase: float) -> str:
    return "0l" if phase in (4.0, 6.0) else "2l"


def phase_is_plot(phase: float) -> bool:
    return phase in (6.0, 6.1)


def r_range(args: argparse.Namespace, analysis: str) -> List[str]:
    lower = args.r_min
    if lower is None:
        lower = -0.3 if analysis == "0l" else -1.0
    if lower >= args.r_max:
        raise ValueError("--r-min must be smaller than --r-max")
    return ["--setParameterRanges", "r={}:{}".format(lower, args.r_max)]


def statistics_args(args: argparse.Namespace) -> List[str]:
    if args.stat_mode == "none":
        return ["--statUncMode", "none"]
    if args.stat_mode == "correlated":
        return ["--statUncMode", "correlated"]
    if args.stat_mode == "hybrid":
        if args.hybrid_stat_threshold <= 0:
            raise ValueError("--hybrid-stat-threshold must be positive")
        threshold = str(args.hybrid_stat_threshold)
        return [
            "--statUncMode",
            "hybrid",
            "--statBinByBin",
            threshold,
            "--minErrOverSqrtNBGForBinByBin",
            threshold,
        ]
    if args.autoMCStats_threshold < 0:
        raise ValueError("--autoMCStats-threshold must be non-negative")
    return ["--statUncMode", "none", "--autoMCStats"]


def configuration_stem(
    args: argparse.Namespace, analysis: str, regime: str
) -> str:
    return (
        "SB13p6TeV_SM_Zh_{}_{}_{}_stat_{}_{}".format(
            args.year,
            regime,
            args.analysis_bin,
            args.stat_mode,
            analysis,
        )
    )


def cards_directory(
    args: argparse.Namespace, analysis: str, regime: str
) -> Path:
    if args.cards_dir:
        return Path(args.cards_dir).expanduser().resolve()
    return Path(args.workdir).resolve() / (
        "cards_" + configuration_stem(args, analysis, regime)
    )


def resolve_analysis_inputs(
    args: argparse.Namespace, cmssw_base: Path
) -> tuple[Path, Path]:
    base = cmssw_base / "src/UserCode/bsmhiggs_fwk/test/haa4b"
    input_file = (
        Path(args.input_file).expanduser().resolve()
        if args.input_file
        else base / DEFAULT_INPUTS[args.year]
    )
    json_file = (
        Path(args.json_file).expanduser().resolve()
        if args.json_file
        else base / DEFAULT_JSONS[args.year]
    )
    return input_file, json_file


def executable_help(executable: str) -> str:
    resolved = shutil.which(executable)
    if not resolved and Path(executable).is_file():
        resolved = str(Path(executable).resolve())
    if not resolved:
        raise RuntimeError(
            "Executable '{}' is not in PATH. Build it first or use the "
            "corresponding --exe-* option.".format(executable)
        )
    result = subprocess.run(
        [resolved, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return result.stdout


def check_executable_compatibility(
    args: argparse.Namespace, analysis: str
) -> None:
    executable = args.exe_0l if analysis == "0l" else args.exe_2l
    help_text = executable_help(executable)
    required = ["--statUncMode"]
    if args.stat_mode == "automcstats":
        required.append("--autoMCStats")
    missing = [option for option in required if option not in help_text]
    if missing:
        raise RuntimeError(
            "{} does not advertise the required option(s): {}. Rebuild the "
            "updated source before launching.".format(
                executable, ", ".join(missing)
            )
        )


def common_compute_args(
    args: argparse.Namespace,
    executable: str,
    regime: str,
    mass: int,
    input_file: Path,
    json_file: Path,
    luminosity_pb: float,
) -> List[str]:
    result = [
        executable,
        "--verbose",
        "--runZh",
        "--m",
        str(mass),
        "--lumi",
        str(luminosity_pb),
        "--BackExtrapol",
        "--in",
        str(input_file),
        "--simfit",
        "--index",
        "1",
        "--bins",
        args.analysis_bin,
        "--json",
        str(json_file),
        "--key",
        args.sample_key,
        "--shape",
        "--histo",
        "bdt_shapes_" + regime,
        "--systpostfix",
        "_13p6TeV",
    ]
    if not args.no_systematics:
        result.append("--syst")
    result.extend(statistics_args(args))
    return result


def shell_header(cmssw_base: Path, work_directory: Path) -> List[str]:
    scram_arch = os.environ.get("SCRAM_ARCH", "el9_amd64_gcc12")
    return [
        "#!/bin/bash\n",
        "set -euo pipefail\n",
        "cd {}\n".format(shlex.quote(str(cmssw_base))),
        "export SCRAM_ARCH={}\n".format(shlex.quote(scram_arch)),
        "eval \"$(scram runtime -sh)\"\n",
        "mkdir -p {}\n".format(shlex.quote(str(work_directory))),
        "RUN_DIRECTORY=$(mktemp -d {}/run.XXXXXX)\n".format(
            shlex.quote(str(work_directory))
        ),
        "cd \"${RUN_DIRECTORY}\"\n",
    ]


def auto_mc_stats_lines(args: argparse.Namespace, card_variable: str) -> List[str]:
    if args.stat_mode != "automcstats":
        return []
    return [
        "sed -i '/^[[:space:]]*\\*[[:space:]]\\+autoMCStats[[:space:]]/d' "
        + '"${' + card_variable + '}"\n',
        "printf '%s\\n' "
        + shlex.quote(
            "* autoMCStats {} 0 1".format(args.autoMCStats_threshold)
        )
        + " >> \"${" + card_variable + "}\"\n",
    ]

def card_builder_functions_0l() -> List[str]:
    return [
        "run_0l_card_builder() {\n",
        "  local script\n",
        "  for script in combineCards_0l_zh.sh _0l_zh.sh combineCards_zh.sh; do\n",
        "    if [ -s \"${script}\" ]; then sh \"${script}\"; return 0; fi\n",
        "  done\n",
        "  echo 'ERROR: computeLimit_0l did not create a recognized card-combination script' >&2\n",
        "  return 1\n",
        "}\n",
        "resolve_0l_card() {\n",
        "  local card\n",
        "  for card in card_0l_simfit_zh.dat card_combined_zh.dat; do\n",
        "    if [ -s \"${card}\" ]; then printf '%s\\n' \"${card}\"; return 0; fi\n",
        "  done\n",
        "  echo 'ERROR: the combined 0l datacard was not created' >&2\n",
        "  return 1\n",
        "}\n",
    ]


def fit_result_check_lines(filename: str) -> List[str]:
    return [
        "python3 - <<'PY'\n",
        "import ROOT\n",
        "f = ROOT.TFile.Open({!r})\n".format(filename),
        "if not f or f.IsZombie():\n",
        "    raise RuntimeError('Cannot open {}')\n".format(filename),
        "fit = f.Get('fit_b')\n",
        "if not fit:\n",
        "    raise RuntimeError('{} does not contain fit_b')\n".format(filename),
        "if fit.status() != 0 or fit.covQual() < 2:\n",
        "    raise RuntimeError('Invalid fit_b: status={} covQual={}'.format(fit.status(), fit.covQual()))\n",
        "print('fit_b status =', fit.status(), 'covQual =', fit.covQual())\n",
        "f.Close()\n",
        "PY\n",
    ]


def diagnostics_lines(
    args: argparse.Namespace,
    analysis: str,
    regime: str,
    mass: int,
    range_args: List[str],
    expected_mask_args: List[str],
) -> List[str]:
    lines: List[str] = []
    common = [
        "-d",
        "workspace.root",
        "-m",
        str(mass),
        "-t",
        "-1",
        "--expectSignal",
        "0",
        "--bypassFrequentistFit",
        "--robustFit",
        "1",
    ] + range_args + expected_mask_args

    if args.run_impacts:
        name = ".Impacts_bonly_m{}_{}_{}_{}".format(
            mass, analysis, regime, args.stat_mode
        )
        json_name = "impacts_bonly_m{}_{}_{}_{}.json".format(
            mass, analysis, regime, args.stat_mode
        )
        plot_name = json_name[:-5]
        lines.append(
            command(
                ["combineTool.py", "-M", "Impacts"]
                + common
                + ["--doInitialFit", "-n", name],
                "impacts_initial.log",
            )
        )
        lines.append(
            command(
                ["combineTool.py", "-M", "Impacts"]
                + common
                + ["--doFits", "-n", name],
                "impacts_fits.log",
            )
        )
        lines.append(
            command(
                [
                    "combineTool.py",
                    "-M",
                    "Impacts",
                    "-d",
                    "workspace.root",
                    "-m",
                    str(mass),
                    "-n",
                    name,
                    "-o",
                    json_name,
                ],
                "impacts_collect.log",
            )
        )
        lines.append(
            command(
                ["plotImpacts.py", "-i", json_name, "-o", plot_name],
                "impacts_plot.log",
            )
        )

    if args.run_nll:
        output_name = ".NLL_bonly_m{}_{}_{}".format(mass, analysis, regime)
        root_name = "higgsCombine{}.MultiDimFit.mH{}.root".format(
            output_name, mass
        )
        plot_name = "nll_scan_m{}_bonly_{}_{}".format(
            mass, analysis, regime
        )
        lines.append(
            command(
                ["combine", "-M", "MultiDimFit", "workspace.root"]
                + common[2:]
                + [
                    "--algo",
                    "grid",
                    "--points",
                    "300",
                    "--saveNLL",
                    "-n",
                    output_name,
                ],
                "NLL.log",
            )
        )
        lines.append(
            command(
                [
                    "plot1DScan.py",
                    root_name,
                    "--POI",
                    "r",
                    "--y-max",
                    "10",
                    "--y-cut",
                    "10",
                    "-o",
                    plot_name,
                ],
                "plotNLL.log",
            )
        )

    if args.run_gof:
        gof_common = [
            "-m",
            str(mass),
            "--algo",
            "saturated",
            "--fixedSignalStrength=0",
        ] + range_args + expected_mask_args
        obs_name = ".GOF_obs_m{}_{}_{}".format(mass, analysis, regime)
        toy_name = ".GOF_toys_m{}_{}_{}".format(mass, analysis, regime)
        lines.append(
            command(
                ["combine", "-M", "GoodnessOfFit", "workspace.root"]
                + gof_common
                + ["-n", obs_name],
                "GOF_obs.log",
            )
        )
        lines.append(
            command(
                ["combine", "-M", "GoodnessOfFit", "workspace.root"]
                + gof_common
                + [
                    "--toysFrequentist",
                    "-t",
                    str(args.gof_toys),
                    "-s",
                    "123456",
                    "-n",
                    toy_name,
                ],
                "GOF_toys.log",
            )
        )
        obs_file = "higgsCombine{}.GoodnessOfFit.mH{}.root".format(
            obs_name, mass
        )
        toy_file = "higgsCombine{}.GoodnessOfFit.mH{}.123456.root".format(
            toy_name, mass
        )
        json_name = "gof_m{}_{}_{}.json".format(mass, analysis, regime)
        plot_name = json_name[:-5]
        lines.append(
            command(
                [
                    "combineTool.py",
                    "-M",
                    "CollectGoodnessOfFit",
                    "--input",
                    obs_file,
                    toy_file,
                    "-m",
                    "{}.0".format(mass),
                    "-o",
                    json_name,
                ],
                "collectGOF.log",
            )
        )
        lines.append(
            command(
                [
                    "plotGof.py",
                    json_name,
                    "--statistic",
                    "saturated",
                    "--mass",
                    "{}.0".format(mass),
                    "-o",
                    plot_name,
                ],
                "plotGOF.log",
            )
        )
    return lines


def build_0l_script(
    args: argparse.Namespace,
    regime: str,
    mass: int,
    cmssw_base: Path,
    input_file: Path,
    json_file: Path,
    luminosity_pb: float,
    script_path: Path,
    output_directory: Path,
) -> None:
    stem = configuration_stem(args, "0l", regime)
    scratch = Path(args.workdir).resolve() / "WORK" / stem / "m{:04d}".format(mass)
    range_args = r_range(args, "0l")
    compute_args = common_compute_args(
        args,
        args.exe_0l,
        regime,
        mass,
        input_file,
        json_file,
        luminosity_pb,
    )
    compute_args += ["--modeDD", "--subFake"]
    if not args.unblind:
        compute_args.append("--replaceHighSensitivityBinsWithBG")

    lines = shell_header(cmssw_base, scratch)
    lines += card_builder_functions_0l()
    lines.append(command(compute_args, "cl-first.log"))
    lines += [
        "run_0l_card_builder\n",
        "DATACARD_FIRST=$(resolve_0l_card)\n",
        "test -s \"${DATACARD_FIRST}\"\n",
    ]
    lines += auto_mc_stats_lines(args, "DATACARD_FIRST")
    lines.append(
        command(
            [
                "text2workspace.py",
                "${DATACARD_FIRST}",
                "-o",
                "workspace-first.root",
                "--PO",
                "verbose",
                "--channel-masks",
                "--PO",
                "ishaa",
                "--PO",
                "m={}".format(mass),
            ],
            "t2w-first.log",
        ).replace("'${DATACARD_FIRST}'", '"${DATACARD_FIRST}"')
    )
    first_fit = [
        "combine",
        "-M",
        "FitDiagnostics",
        "workspace-first.root",
        "-m",
        str(mass),
        "-v",
        "3",
        "--setParameters",
        "r=0,mask_veto_A_SR_3b=1",
        "--freezeParameters",
        "r,mask_veto_A_SR_3b",
        "--skipSBFit",
        "--saveNormalizations",
        "--saveShapes",
        "--saveWithUncertainties",
        "--saveNLL",
        "--cminPreScan",
        "--cminDefaultMinimizerStrategy",
        "1",
        "--cminDefaultMinimizerTolerance",
        "0.1",
        "--cminFallbackAlgo",
        "Minuit2,Migrad,0:0.1",
    ]
    lines.append(command(first_fit, "log-first.txt"))
    lines += fit_result_check_lines("fitDiagnosticsTest.root")
    lines += [
        "mv fitDiagnosticsTest.root fitDiagnostics-first.root\n",
        "mkdir -p datacards-first-pass\n",
        "cp -a -- *.dat datacards-first-pass/\n",
    ]

    second_args = compute_args + [
        "--fitDiagnosticsInputFile",
        "fitDiagnostics-first.root",
    ]
    lines.append(command(second_args, "cl-second.log"))
    lines += [
        "run_0l_card_builder\n",
        "DATACARD=$(resolve_0l_card)\n",
        "test -s \"${DATACARD}\"\n",
    ]
    lines += auto_mc_stats_lines(args, "DATACARD")
    lines.append(
        command(
            [
                "text2workspace.py",
                "${DATACARD}",
                "-o",
                "workspace.root",
                "--PO",
                "verbose",
                "--channel-masks",
                "--PO",
                "ishaa",
                "--PO",
                "m={}".format(mass),
            ],
            "t2w.log",
        ).replace("'${DATACARD}'", '"${DATACARD}"')
    )
    final_fit = [
        "combine",
        "-M",
        "FitDiagnostics",
        "workspace.root",
        "-m",
        str(mass),
        "-v",
        "3",
        "--plots",
        "--saveWorkspace",
        "--saveNormalizations",
        "--saveShapes",
        "--saveOverallShapes",
        "--saveWithUncertainties",
        "--saveNLL",
        "--ignoreCovWarning",
        "--cminPreScan",
        "--cminDefaultMinimizerStrategy",
        "1",
        "--cminDefaultMinimizerTolerance",
        "0.01",
        "--stepSize=0.001",
        "--robustFit",
        "1",
    ] + range_args
    lines.append(command(final_fit, "log.txt"))
    lines += fit_result_check_lines("fitDiagnosticsTest.root")
    lines += standard_postfit_text_lines(cmssw_base, mass)
    lines += diagnostics_lines(args, "0l", regime, mass, range_args, [])
    lines += limit_lines(mass, range_args, [])
    lines += save_output_lines(output_directory)
    write_script(script_path, lines)


def build_2l_script(
    args: argparse.Namespace,
    regime: str,
    mass: int,
    cmssw_base: Path,
    input_file: Path,
    json_file: Path,
    luminosity_pb: float,
    script_path: Path,
    output_directory: Path,
) -> None:
    stem = configuration_stem(args, "2l", regime)
    scratch = Path(args.workdir).resolve() / "WORK" / stem / "m{:04d}".format(mass)
    range_args = r_range(args, "2l")
    compute_args = common_compute_args(
        args,
        args.exe_2l,
        regime,
        mass,
        input_file,
        json_file,
        luminosity_pb,
    )
    compute_args += [
        "--channels",
        "ee_A_SR,mumu_A_SR,emu_A_CR",
    ]
    if not args.unblind:
        compute_args.append("--replaceHighSensitivityBinsWithBG")

    lines = shell_header(cmssw_base, scratch)
    lines.append(command(compute_args, "cl.log"))
    lines += [
        "test -s combineCards_2l_zh.sh || { echo 'ERROR: missing combineCards_2l_zh.sh' >&2; exit 1; }\n",
        "sh combineCards_2l_zh.sh\n",
        "DATACARD=card_2l_simfit_zh.dat\n",
        "test -s \"${DATACARD}\" || { echo 'ERROR: missing 2l simultaneous-fit card' >&2; exit 1; }\n",
        "grep -q 'ee_A_SR_3b' \"${DATACARD}\" || { echo 'ERROR: ee SR missing from 2l card' >&2; exit 1; }\n",
        "grep -q 'mumu_A_SR_3b' \"${DATACARD}\" || { echo 'ERROR: mumu SR missing from 2l card' >&2; exit 1; }\n",
        "grep -q 'emu_A_CR_3b' \"${DATACARD}\" || { echo 'ERROR: e-mu CR missing from 2l card' >&2; exit 1; }\n",
        "! grep -qE '(ee|mumu)_A_(DYCR|CR)_3b' \"${DATACARD}\" || { echo 'ERROR: a split/dedicated same-flavour CR entered the nominal 2l card' >&2; exit 1; }\n",
        "grep -qE '^dy_norm_ee[[:space:]]+rateParam' \"${DATACARD}\" || { echo 'ERROR: dy_norm_ee rateParam missing' >&2; exit 1; }\n",
        "grep -qE '^dy_norm_mumu[[:space:]]+rateParam' \"${DATACARD}\" || { echo 'ERROR: dy_norm_mumu rateParam missing' >&2; exit 1; }\n",
    ]
    lines += auto_mc_stats_lines(args, "DATACARD")
    if args.stat_mode == "automcstats":
        lines += [
            "if grep -q 'CMS_haa4b_stat_' \"${DATACARD}\"; then\n",
            "  echo 'ERROR: custom MC-stat shapes remain together with autoMCStats' >&2\n",
            "  exit 1\n",
            "fi\n",
        ]
    lines.append(
        command(
            [
                "text2workspace.py",
                "${DATACARD}",
                "-o",
                "workspace.root",
                "--PO",
                "verbose",
                "--PO",
                "ishaa",
                "--PO",
                "m={}".format(mass),
            ],
            "t2w.log",
        ).replace("'${DATACARD}'", '"${DATACARD}"')
    )

    final_fit = [
        "combine",
        "-M",
        "FitDiagnostics",
        "workspace.root",
        "-m",
        str(mass),
        "-v",
        "3",
        "--plots",
        "--saveWorkspace",
        "--saveNormalizations",
        "--saveShapes",
        "--saveOverallShapes",
        "--saveWithUncertainties",
        "--saveNLL",
        "--ignoreCovWarning",
        "--cminPreScan",
        "--cminDefaultMinimizerStrategy",
        "1",
        "--cminDefaultMinimizerTolerance",
        "0.01",
        "--robustFit",
        "1",
    ] + range_args
    lines.append(command(final_fit, "log.txt"))
    lines += fit_result_check_lines("fitDiagnosticsTest.root")
    lines += standard_postfit_text_lines(cmssw_base, mass)

    lines += diagnostics_lines(args, "2l", regime, mass, range_args, [])
    lines += limit_lines(mass, range_args, [])
    lines += save_output_lines(output_directory)
    write_script(script_path, lines)


def standard_postfit_text_lines(cmssw_base: Path, mass: int) -> List[str]:
    print_script = (
        cmssw_base
        / "src/UserCode/bsmhiggs_fwk/test/haa4b/computeLimit/print.py"
    )
    diff_script = (
        cmssw_base
        / "src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py"
    )
    output = "fit_diagnostics_m{}.txt".format(mass)
    return [
        command(
            ["python3", str(print_script), "-u", "fitDiagnosticsTest.root"],
            output,
        ),
        quote_join(
            [
                "python3",
                str(diff_script),
                "-A",
                "-a",
                "fitDiagnosticsTest.root",
                "-g",
                "Nuisance_CrossCheck.root",
            ]
        )
        + " >> "
        + shlex.quote(output)
        + " 2>&1\n",
    ]


def limit_lines(
    mass: int, range_args: List[str], expected_mask_args: List[str]
) -> List[str]:
    output = "higgsCombineTest.AsymptoticLimits.mH{}.root".format(mass)
    renamed = "higgsCombineTest.AsymptoticLimits.mH{}-all.root".format(mass)
    return [
        command(
            [
                "combine",
                "-M",
                "AsymptoticLimits",
                "workspace.root",
                "-m",
                str(mass),
                "-v",
                "2",
                "-t",
                "-1",
                "--expectSignal",
                "0",
            ]
            + range_args
            + expected_mask_args,
            "COMB.log",
        ),
        "test -s {} || {{ echo 'ERROR: expected-limit ROOT file is missing' >&2; exit 1; }}\n".format(
            shlex.quote(output)
        ),
        "mv {} {}\n".format(shlex.quote(output), shlex.quote(renamed)),
    ]


def save_output_lines(output_directory: Path) -> List[str]:
    return [
        "mkdir -p {}\n".format(shlex.quote(str(output_directory))),
        "cp -a . {}/\n".format(shlex.quote(str(output_directory))),
        "echo 'Saved results in {}'\n".format(
            shlex.quote(str(output_directory))
        ),
    ]


def write_script(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    path.chmod(0o755)


def load_launch_on_condor(script_directory: Path):
    module_path = (script_directory / "../../../scripts").resolve()
    sys.path.insert(0, str(module_path))
    try:
        import LaunchOnCondor  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Cannot import LaunchOnCondor from {}".format(module_path)
        ) from error
    return LaunchOnCondor


def submit_jobs(args: argparse.Namespace, analysis: str) -> None:
    if Path(args.farm_dir).is_absolute():
        raise ValueError(
            "--farm-dir must be relative. Absolute paths are duplicated by "
            "LaunchOnCondor and cause missing-submit-file errors."
        )
    cmssw_text = os.environ.get("CMSSW_BASE")
    if not cmssw_text:
        raise RuntimeError("CMSSW_BASE is not set")
    cmssw_base = Path(cmssw_text).resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    input_file, json_file = resolve_analysis_inputs(args, cmssw_base)
    if not input_file.is_file():
        raise RuntimeError("Input ROOT file does not exist: {}".format(input_file))
    if not json_file.is_file():
        raise RuntimeError("Sample JSON file does not exist: {}".format(json_file))
    check_executable_compatibility(args, analysis)

    launch = load_launch_on_condor(Path(__file__).resolve().parent)
    launch.Jobs_Queue = args.queue
    launch.Jobs_RunHere = 0
    luminosity_pb = args.lumi_pb or LUMINOSITY_PB[args.year]

    for regime in selected_regimes(args.regime):
        masses = args.masses or DEFAULT_MASSES[regime]
        stem = configuration_stem(args, analysis, regime)
        job_directory = workdir / "JOBS" / stem
        job_directory.mkdir(parents=True, exist_ok=True)
        cards_dir = cards_directory(args, analysis, regime)
        cluster_name = "computeLimits__" + stem
        launch.SendCluster_Create(args.farm_dir, cluster_name)
        print("\nCreating {} {} jobs in {}".format(analysis, regime, job_directory))

        for mass in masses:
            script_path = job_directory / "script_mass_{}.sh".format(mass)
            output_directory = cards_dir / "{:04d}".format(mass)
            if analysis == "0l":
                build_0l_script(
                    args,
                    regime,
                    mass,
                    cmssw_base,
                    input_file,
                    json_file,
                    luminosity_pb,
                    script_path,
                    output_directory,
                )
            else:
                build_2l_script(
                    args,
                    regime,
                    mass,
                    cmssw_base,
                    input_file,
                    json_file,
                    luminosity_pb,
                    script_path,
                    output_directory,
                )
            launch.SendCluster_Push(["BASH", "sh " + str(script_path)])

        if args.no_submit:
            print("--noSubmit: generated scripts but did not submit {}".format(stem))
        else:
            launch.SendCluster_Submit()


def run_checked(parts: Sequence[object]) -> None:
    print("+ " + quote_join(parts))
    subprocess.run([str(part) for part in parts], check=True)


def plot_limits(args: argparse.Namespace, analysis: str) -> None:
    if args.cards_dir and args.regime == "all":
        raise ValueError("--cards-dir requires --regime boosted or resolved")
    workdir = Path(args.workdir).expanduser().resolve()
    macro = Path(args.plot_macro)
    if not macro.is_absolute():
        macro = workdir / macro
    if not macro.is_file():
        raise RuntimeError("Cannot find plot macro: {}".format(macro))
    luminosity_pb = args.lumi_pb or LUMINOSITY_PB[args.year]

    for regime in selected_regimes(args.regime):
        cards_dir = cards_directory(args, analysis, regime)
        if not cards_dir.is_dir():
            raise RuntimeError("Cards directory does not exist: {}".format(cards_dir))
        preferred = sorted(
            glob.glob(
                str(cards_dir / "*" / "higgsCombineTest.AsymptoticLimits.mH*-all.root")
            )
        )
        inputs = preferred or sorted(
            glob.glob(
                str(cards_dir / "*" / "higgsCombineTest.AsymptoticLimits.mH*.root")
            )
        )
        if not inputs:
            raise RuntimeError(
                "No AsymptoticLimits ROOT files were found below {}. "
                "Check which mass jobs completed before plotting.".format(cards_dir)
            )
        merged = cards_dir / "LimitTree.root"
        run_checked(["hadd", "-f", str(merged)] + inputs)
        output_prefix = cards_dir / "Strength_all_"
        label = "ZH {} channel, {}".format(analysis, regime)
        log_y = "false" if args.linear_limits else "true"
        macro_call = '{}+("{}","{}","",false,true,13.6,{},"{}",{})'.format(
            macro,
            output_prefix,
            merged,
            luminosity_pb,
            label,
            log_y,
        )
        run_checked(["root", "-l", "-b", "-q", macro_call])
        print("Limit plot written with prefix {}".format(output_prefix))


def validate_args(args: argparse.Namespace) -> None:
    if args.gof_toys <= 0:
        raise ValueError("--gof-toys must be positive")
    if args.cards_dir and not phase_is_plot(args.phase):
        raise ValueError("--cards-dir is only valid for phases 6.0 and 6.1")
    r_range(args, analysis_for_phase(args.phase))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        analysis = analysis_for_phase(args.phase)
        if phase_is_plot(args.phase):
            plot_limits(args, analysis)
        else:
            submit_jobs(args, analysis)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
