#!/usr/bin/env python3
"""Flat-output Condor workflow with modes and automatic folder tarballs."""
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import uuid

PROJECT = Path(__file__).resolve().parent
DATASET_DIR = "datasets"
IMAGE = (
    "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/"
    "coffea-base-almalinux8:0.7.21-fastjet-3.4.0.1"
)

# ONE configuration keeps the driver, processor and shell script synchronized
MODES = {
    "gen": {
        "driver": "run_gen_haa4b.py",
        "processor": "gen_haa4b_processor.py",
        "sh": "run_gen.sh",
    },
    "analysis-0lep": {
        "driver": "run_analysis.py",
        "processor": "ZH_0lep_processor_fixedWP.py",
        "sh": "run_analysis.sh",
    },
    "analysis-2lep": {
        "driver": "run_analysis.py",
        "processor": "ZH_2lep_processor_fixedWP.py",
        "sh": "run_analysis.sh",
    },
    "btag-eff": {
        "driver": "run_btag_efficiency.py",
        "processor": "btag_efficiency_processor.py",
        "sh": "run_btag_efficiency.sh",
    },
    "flavour-0lep": {
        "driver": "run_flavour_calibration_0lep.py",
        "processor": "flavour_calibration_processor_0l_TMMM.py",
        "sh": "run_analysis.sh",
    },
}

# All other visible top-level directories are archived automatically.
# Add any other folders of OUTPUTS here, or pass --exclude-dir NAME. datasets/ is never transferred as a whole directory.
EXCLUDE_DIRS = {
    "datasets", "out", "err", "log", "logs", "submissions", "cmssw","legacy",
    "results", "_analysis_return", "__pycache__", "venv", "env",
}


def token(value, label):
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", value) or value in {".", ".."}:
        raise ValueError(f"Unsupported {label}: {value!r}; use letters, digits, _, ., +, -")
    return value


def path_text(path):
    value = str(path)
    if not re.fullmatch(r"/[A-Za-z0-9_./+-]+", value):
        raise ValueError(f"Unsupported submit-side path: {value!r}; avoid whitespace and special characters")
    return value


def archive_filter(member):
    parts = Path(member.name).parts
    if any(part in {"__pycache__", ".git", ".pytest_cache", ".ipynb_checkpoints"} for part in parts):
        return None
    if member.name.endswith((".pyc", ".pyo")):
        return None
    source = PROJECT / member.name
    if source.is_symlink() and source.is_dir():
        raise ValueError(f"Materialize or exclude this directory symlink: {source}")
    return member


def synced_driver(source, selected_processor):
    """Update one ordinary processor import in the SUBMITTED COPY, never the source."""
    tree = ast.parse(source)
    configured = {Path(mode["processor"]).stem for mode in MODES.values()}
    candidates = set()
    for node in tree.body:
        names = [node.module] if isinstance(node, ast.ImportFrom) else (
            [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
        )
        for name in names:
            if name and "." not in name and (name in configured or "processor" in name.lower()):
                candidates.add(name)
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous processor imports {sorted(candidates)}; configure a dedicated driver")
    if not candidates:
        raise ValueError("No direct processor import found; configure a driver with an explicit processor import")
    old = candidates.pop()
    if old == selected_processor:
        return source, None
    # Replace module identifiers only; preserve imported class names and aliases.
    updated = re.sub(rf"(?m)^(\s*(?:from|import)\s+){re.escape(old)}\b", rf"\g<1>{selected_processor}", source)
    if updated == source:
        raise ValueError(f"Could not synchronize processor import {old}; configure a dedicated driver")
    compile(updated, "staged_driver", "exec")
    return updated, f"{old} -> {selected_processor}"


def synced_shell(source, selected_driver, folder_names):
    """Retain each existing shell's CLI; change only its direct Python driver name."""
    if "_analysis_return" in source or "RESULT_DIRECTORY" in source:
        raise ValueError("This is the newer output-folder wrapper; use the supplied run_analysis.sh")
    known = {mode["driver"] for mode in MODES.values()}
    known.add(selected_driver)
    for driver in known:
        # Typical commands: python run_analysis.py, python3 ./run_analysis.py.
        source = re.sub(rf"(\bpython(?:3(?:\.\d+)?)?\s+)(?:\./)?{re.escape(driver)}\b",
                        rf"\g<1>{selected_driver}", source)
    # The submission launcher already extracts every folder archive. Replace ordinary
    # legacy tar lines with a shell no-op (valid even inside an otherwise empty if).
    archives = {name + ".tar.gz" for name in folder_names}
    def replace_tar(match):
        if match.group(2) in archives:
            return match.group(1) + ": # archive already extracted by the submission launcher"
        return match.group(0)
    source = re.sub(r"(?m)^([ \t]*)tar[ \t]+-xzf[ \t]+([A-Za-z0-9_.+-]+\.tar\.gz)[ \t]*$",
                    replace_tar, source)
    # Legacy payloads assumed x509up was always present. The launcher now
    # configures it only when the user explicitly supplies --proxy.
    source = re.sub(
        r"(?m)^([ \t]*)export[ \t]+X509_USER_PROXY=[^\n]*x509up[^\n]*$",
        r"\1: # Optional proxy is configured by the launcher",
        source,
    )
    return source


def launcher_text(mode, config, archives):
    archive_words = " ".join(shlex.quote(name) for name in archives)
    return f'''#!/bin/bash
set -euo pipefail
[[ $# -eq 3 ]] || {{ echo "Expected JOB_INDEX DATASET_JSON DATASET_KEY" >&2; exit 64; }}
echo "Mode: {mode}"
echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"
if [[ -f x509up ]]; then
    export X509_USER_PROXY="$(pwd -P)/x509up"
fi
export PYTHONPATH="$(pwd -P)${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export ANALYSIS_DRIVER={shlex.quote(config['driver'])}
export ANALYSIS_PROCESSOR={shlex.quote(Path(config['processor']).stem)}
archives=({archive_words})
for archive in "${{archives[@]}}"; do
    echo "Extracting $archive"
    tar -xzf "$archive"
done
if bash -e payload_{mode}.sh "$@"; then
    status=0
else
    status=$?
fi
echo "Job exit code: $status"
exit "$status"
'''


def run_local_test(stage, launcher, snapshot, row, runtime):
    """Execute the same prepared payload locally, inside the configured image."""
    index, json_name, key = row.split()
    shutil.copy2(snapshot, stage / json_name)
    stdout_path = PROJECT / "out" / f"local_{stage.name}_{key}.out"
    stderr_path = PROJECT / "err" / f"local_{stage.name}_{key}.err"
    command = [runtime, "exec", "--bind", f"{stage}:/srv", "--bind", "/cvmfs:/cvmfs",
               "--pwd", "/srv", IMAGE, "bash", launcher.name, index, json_name, key]
    print("LOCAL TEST: one job; no Condor submission", flush=True)
    print(f"Job directory: {stage}", flush=True)
    print(f"stdout: {stdout_path}\nstderr: {stderr_path}", flush=True)
    print("Command:", shlex.join(command), flush=True)
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        result = subprocess.run(command, cwd=stage, stdout=out, stderr=err)
    print(f"Local job exit code: {result.returncode}")
    if result.returncode != 0:
        print("Read the .out/.err logs above. Partial outputs remain in the local job directory.")
        raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
    outputs = sorted(stage.glob("*.root"))
    for output in outputs:
        destination = PROJECT / output.name
        shutil.move(str(output), str(destination))
        print(f"ROOT output: {destination}")
    if not outputs:
        print("Local job returned zero but produced no top-level ROOT files; inspect its logs.")


def job_outputs_exist(dataset_key, dataset_info, job_idx, outdir,
                      layout="auto", templates=None):
    """Nonempty files indicate completion; this does not verify ROOT integrity."""
    outdir = Path(outdir)
    meta = dataset_info.get("metadata", {})
    sample_base = os.path.basename(str(meta.get("sample", dataset_key))).replace(".root", "").replace("/", "_")

    def present(name):
        path = outdir / name
        return path.is_file() and path.stat().st_size > 0

    if templates:
        return all(present(template.format(dataset_key=dataset_key, sample_base=sample_base,
                                           job_idx=job_idx)) for template in templates)
    single = f"{dataset_key}_{job_idx}.root"
    split = [f"{sample_base}_{flavour}_{job_idx}.root" for flavour in ("ttLF", "ttCC", "ttBB")]
    if layout == "single":
        return present(single)
    if layout == "tt-split":
        return all(present(name) for name in split)
    # A gen/btag job can produce a single TT file. TTH is not split ttbar.
    ttbar = any(re.match(r"^(?:TTto|TTTo|TTJets|TTbar|TT_|TT$)", name)
                for name in (sample_base, dataset_key))
    split_evidence = any((outdir / name).is_file() for name in split)
    return present(single) or ((ttbar or split_evidence) and all(present(name) for name in split))


def main(argv=None, *, only_missing=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", nargs="?", default="ZH-*.json")
    parser.add_argument("--mode", choices=MODES, default="gen")
    parser.add_argument("--list-modes", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--dry-run", action="store_true", help="Prepare files only; do not submit")
    execution.add_argument("--local-test", action="store_true",
                           help="Run exactly ONE prepared job locally inside Singularity/Apptainer; no Condor")
    parser.add_argument("--max-jobs", type=int, default=None, help="TOTAL job cap; use 1 for testing")
    parser.add_argument("--job-index", type=int, default=None,
                        help="Select this input-file index within each matching dataset")
    parser.add_argument("--filter-key", default=os.environ.get("FILTER_KEY"))
    parser.add_argument("--proxy", metavar="PATH", default=None,
                        help="Optional X.509 proxy to copy and transfer; none by default")
    parser.add_argument("--exclude-dir", action="append", default=[], help="Extra top-level folder to skip")
    parser.add_argument("--memory-mb", type=int, default=3000)
    parser.add_argument("--flavour", default="workday")
    if only_missing:
        parser.add_argument("--output-dir", default=".", help="Existing output folder; relative to project")
        parser.add_argument("--output-layout", choices=("auto", "single", "tt-split"), default="auto",
                            help="auto: nonempty single file OR all three nonempty ttbar split files")
        parser.add_argument("--output-template", action="append",
                            help="Custom relative filename with {dataset_key}, {sample_base}, {job_idx}; "
                                 "repeat to require ALL files; overrides --output-layout")
    args = parser.parse_args(argv)
    if args.job_index is not None and args.job_index < 0:
        parser.error("--job-index must be non-negative")
    if args.list_modes:
        for name, conf in MODES.items():
            print(f"{name}: {conf['driver']} | {conf['processor']} | {conf['sh']}")
        return
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max-jobs must be positive")
    if args.local_test:
        if args.max_jobs is not None and args.max_jobs != 1:
            parser.error("--local-test runs exactly one job; omit --max-jobs or set it to 1")
        args.max_jobs = 1
    if args.memory_mb <= 0:
        parser.error("--memory-mb must be positive")
    token(args.flavour, "job flavour")
    path_text(PROJECT)
    config = MODES[args.mode]
    exclusions = EXCLUDE_DIRS | {token(name, "excluded directory") for name in args.exclude_dir}
    if only_missing:
        output_dir = (PROJECT / args.output_dir).resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
        print(f"Checking existing outputs in: {output_dir}")
        if args.output_template:
            for template in args.output_template:
                example = template.format(dataset_key="sample", sample_base="sample", job_idx=0)
                if Path(example).is_absolute() or ".." in Path(example).parts:
                    parser.error("--output-template must stay inside --output-dir")
        # If outputs are below the project, do not package that output tree.
        if output_dir.is_relative_to(PROJECT) and output_dir != PROJECT:
            exclusions.add(output_dir.relative_to(PROJECT).parts[0])

    if Path(args.pattern).is_absolute() or ".." in Path(args.pattern).parts:
        parser.error("pattern must remain inside datasets/")
    json_paths = sorted((PROJECT / DATASET_DIR).glob(args.pattern))
    if not json_paths:
        raise FileNotFoundError(f"No datasets/{args.pattern}")
    key_regex = re.compile(args.filter_key) if args.filter_key else None
    batches, total = [], 0
    for json_path in json_paths:
        token(json_path.name, "dataset JSON filename")
        with json_path.open() as stream:
            data = json.load(stream)
        rows = []
        for key, info in data.items():
            if key_regex and not key_regex.search(key):
                continue
            token(key, "dataset key")
            if not isinstance(info["files"], list):
                raise ValueError(f"{key}: files must be a list")
            if args.job_index is None:
                indices = list(range(len(info["files"])))
            elif args.job_index < len(info["files"]):
                indices = [args.job_index]
            else:
                print(f"[SKIP] {key}: no file at index {args.job_index}")
                continue
            if only_missing:
                indices = [index for index in indices if not job_outputs_exist(
                    key, info, index, output_dir, args.output_layout, args.output_template)]
                print(f"{key}: {len(indices)} missing/incomplete of {len(info['files'])} jobs")
            for index in indices:
                if args.max_jobs is not None and total >= args.max_jobs:
                    break
                rows.append(f"{index} {json_path.name} {key}\n")
                total += 1
            if args.max_jobs is not None and total >= args.max_jobs:
                break
        if rows:
            batches.append((json_path, rows))
        if args.max_jobs is not None and total >= args.max_jobs:
            break
    if not batches:
        print("No matching jobs; nothing submitted.")
        return
    for name in (config["driver"], config["processor"], config["sh"]):
        token(name, "input filename")
        if not (PROJECT / name).is_file():
            raise FileNotFoundError(PROJECT / name)
    proxy_path = (PROJECT / args.proxy).resolve() if args.proxy else None
    if proxy_path is not None and not proxy_path.is_file():
        raise FileNotFoundError(f"Requested proxy does not exist: {proxy_path}")
    folders = []
    for entry in sorted(PROJECT.iterdir()):
        if entry.name.startswith((".", "results_", "submissions_")) or entry.name in exclusions:
            continue
        if entry.is_dir():
            token(entry.name, "resource directory")
            if entry.is_symlink():
                raise ValueError(f"Materialize or exclude this directory symlink: {entry}")
            folders.append(entry)
    print("Resource folders:", ", ".join(folder.name for folder in folders) or "(none)")
    driver_text, import_change = synced_driver((PROJECT / config["driver"]).read_text(), Path(config["processor"]).stem)
    shell_text = synced_shell((PROJECT / config["sh"]).read_text(), config["driver"], [p.name for p in folders])
    compile((PROJECT / config["processor"]).read_text(), config["processor"], "exec")
    if import_change:
        print("Processor import in submitted driver:", import_change)
    runtime = None
    if args.local_test:
        runtime = shutil.which("singularity") or shutil.which("apptainer")
        if runtime is None:
            raise RuntimeError("Neither singularity nor apptainer is available for --local-test")
    elif not args.dry_run and shutil.which("condor_submit") is None:
        raise RuntimeError("condor_submit is not available; use --dry-run off lxplus")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    stage = PROJECT / "submissions" / run_id
    stage.mkdir(parents=True)
    if proxy_path is not None:
        shutil.copy2(proxy_path, stage / "x509up")
        (stage / "x509up").chmod(0o600)
    for name in ("out", "err"):
        (PROJECT / name).mkdir(exist_ok=True)
    (stage / config["driver"]).write_text(driver_text)
    shutil.copy2(PROJECT / config["processor"], stage / config["processor"])
    payload = stage / f"payload_{args.mode}.sh"
    payload.write_text(shell_text)
    subprocess.run(["bash", "-n", str(payload)], check=True)
    launcher = stage / f"run_{args.mode}.sh"
    archive_names = [folder.name + ".tar.gz" for folder in folders]
    launcher.write_text(launcher_text(args.mode, config, archive_names))
    launcher.chmod(0o755)
    subprocess.run(["bash", "-n", str(launcher)], check=True)

    for folder, archive_name in zip(folders, archive_names):
        archive = stage / archive_name
        with tarfile.open(archive, "w:gz", dereference=True) as tar:
            tar.add(folder, arcname=folder.name, filter=archive_filter)
        print(f"Created {archive.name}: {archive.stat().st_size / 1024**2:.1f} MiB")
    common_inputs = [stage / config["driver"], stage / config["processor"], payload]
    common_inputs += [stage / name for name in archive_names]
    if proxy_path is not None:
        common_inputs.append(stage / "x509up")
    proxy_environment = 'environment = "X509_USER_PROXY=x509up"\n' if proxy_path else ""
    jdls = []
    for number, (json_path, rows) in enumerate(batches):
        json_stage = stage / f"dataset_{number}"
        json_stage.mkdir()
        snapshot = json_stage / json_path.name
        shutil.copy2(json_path, snapshot)
        joblist = json_stage / f"joblist_{json_path.name}.txt"
        joblist.write_text("".join(rows))
        jdl = json_stage / f"{'resubmit' if only_missing else 'submit'}_{json_path.name}.jdl"
        jdl.write_text(
            "universe = vanilla\n"
            f"initialdir = {path_text(PROJECT)}\n"
            f"executable = {path_text(launcher)}\n"
            'arguments = "$(jobindex) $(dataset_json) $(dataset_key)"\n'
            f"transfer_input_files = {', '.join(path_text(p) for p in common_inputs + [snapshot])}\n"
            "should_transfer_files = YES\n"
            "when_to_transfer_output = ON_EXIT\n"
            "output = out/job_$(Cluster)_$(Process)_$(dataset_key).out\n"
            "error = err/job_$(Cluster)_$(Process)_$(dataset_key).err\n"
            f'+SingularityImage = "{IMAGE}"\n'
            "+SingularityBindCVMFS = True\n"
            f'+JobFlavour = "{args.flavour}"\n'
            "request_cpus = 1\n"
            f"request_memory = {args.memory_mb}\n"
            + proxy_environment
            + f"queue jobindex, dataset_json, dataset_key from {path_text(joblist)}\n"
        )
        jdls.append(jdl)
        print(f"Prepared {len(rows)} {args.mode} jobs: {jdl}")
    print(f"Total jobs: {total}")
    if args.local_test:
        run_local_test(stage, launcher, snapshot, rows[0], runtime)
        return
    if args.dry_run:
        print("Dry run: no jobs submitted.")
        return
    for jdl in jdls:
        subprocess.run(["condor_submit", str(jdl)], cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
