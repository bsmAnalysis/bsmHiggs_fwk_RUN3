#!/usr/bin/env python3
"""Prepare, submit or locally test one-file NanoAOD skim jobs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
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
EOS_SERVER = "root://eosuser.cern.ch"
BASE_EOS_DIR = "/eos/user/a/ataxeidi/skim_MC_new"
INPUT_FILES = (
    "run_skim_ak4.py", "skim_processor_ak4.py", "skim_config.py",
    "run_skimming.sh",
)
REQUIRED_DIRS = {"corrections", "golden_json"}

# Every other visible top-level folder is archived once per invocation.
EXCLUDE_DIRS = {
    "datasets", "out", "err", "log", "logs", "submissions", "results",
    "cmssw", "cmsssw", "legacy", "__pycache__", "venv", "env",
}


def safe_name(value, label):
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", value) or value in {".", ".."}:
        raise ValueError(f"Invalid {label}: {value!r}; use letters, digits, _, ., +, -")
    return value


def submit_path(path):
    """Keep Condor paths unambiguous without mixing shell and submit quoting."""
    value = str(path)
    if not re.fullmatch(r"/[A-Za-z0-9_./+-]+", value):
        raise ValueError(f"Submit paths must not contain spaces or special characters: {value}")
    return value


def project_path(value):
    return (PROJECT / value).resolve()


def resolve_proxy(args):
    """Require the configured proxy unless staging is explicitly disabled."""
    if args.no_proxy:
        return None
    proxy = project_path(args.proxy)
    if not proxy.is_file():
        raise FileNotFoundError(
            f"Proxy file is missing: {proxy}. Create it, select --proxy PATH, "
            "or use --no-proxy if your inputs and outputs do not need it."
        )
    return proxy


def output_name(key, index):
    return f"{key}_{index}.root"


def eos_directory(base_dir, key):
    return f"{base_dir.rstrip('/')}/{key}"


def eos_outputs(server, directory, environment):
    """List one dataset directory; access errors are not missing outputs."""
    command = ["xrdfs", server, "ls", directory]
    result = subprocess.run(
        command, capture_output=True, text=True, env=environment, timeout=120,
    )
    if result.returncode:
        message = (result.stderr + "\n" + result.stdout).strip()
        lower = message.lower()
        access_error = any(word in lower for word in (
            "[3010]", "permission denied", "unauthorized", "auth failed",
        ))
        absent = "[3011]" in lower or "no such file or directory" in lower
        if absent and not access_error:
            print(f"[MISSING DIRECTORY] {directory}")
            return set()
        raise RuntimeError(
            f"Cannot check EOS outputs in {directory}. No jobs will be submitted.\n"
            f"{message or 'xrdfs failed without an error message'}"
        )
    return {PurePosixPath(line.strip()).name for line in result.stdout.splitlines()
            if line.strip()}


def local_output_exists(directory, key, index):
    name = output_name(key, index)
    # Local tests return flat files; an EOS-style local tree is also accepted.
    return any(path.is_file() and path.stat().st_size > 0
               for path in (directory / name, directory / key / name))


def select_jobs(args, environment, only_missing):
    paths = sorted((PROJECT / DATASET_DIR).glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No dataset JSONs match datasets/{args.pattern}")
    key_regex = re.compile(args.filter_key) if args.filter_key else None
    local_directory = project_path(args.output_dir) if args.output_dir else None
    if only_missing and local_directory is not None and not local_directory.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {local_directory}")
    batches, total = [], 0
    eos_cache = {}
    selected_outputs = {}
    for path in paths:
        safe_name(path.name, "dataset JSON filename")
        with path.open() as stream:
            datasets = json.load(stream)
        if not isinstance(datasets, dict):
            raise ValueError(f"{path}: expected a dictionary of datasets")
        rows = []
        for key, info in datasets.items():
            if key_regex and not key_regex.search(key):
                continue
            safe_name(key, "dataset key")
            files = info.get("files")
            if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
                raise ValueError(f"{path}: {key}: files must be a list of paths/URLs")
            indices = list(range(len(files)))
            if args.job_index is not None:
                if args.job_index >= len(files):
                    print(f"[SKIP] {key}: no file at index {args.job_index}")
                    continue
                indices = [args.job_index]
            if only_missing:
                if local_directory is not None:
                    indices = [i for i in indices
                               if not local_output_exists(local_directory, key, i)]
                else:
                    directory = eos_directory(args.base_eos_dir, key)
                    if directory not in eos_cache:
                        eos_cache[directory] = eos_outputs(args.eos_server, directory, environment)
                    indices = [i for i in indices
                               if output_name(key, i) not in eos_cache[directory]]
                scope = "selected index" if args.job_index is not None else "all input files"
                print(f"{key}: {len(indices)} missing jobs ({scope}; {len(files)} files total)")
            for index in indices:
                if args.max_jobs is not None and total >= args.max_jobs:
                    break
                identity = (key, index)
                # Two JSONs must not send different inputs to the same EOS filename.
                if identity in selected_outputs:
                    previous = selected_outputs[identity]
                    if previous != files[index]:
                        raise ValueError(f"Conflicting inputs for {output_name(key, index)}")
                    print(f"[SKIP] Duplicate job: {key}, index {index}")
                    continue
                selected_outputs[identity] = files[index]
                rows.append((index, path.name, key))
                total += 1
            if args.max_jobs is not None and total >= args.max_jobs:
                break
        if rows:
            batches.append((path, rows))
        if args.max_jobs is not None and total >= args.max_jobs:
            break
    return batches, total


def resource_folders(args):
    exclusions = EXCLUDE_DIRS | {
        safe_name(name, "excluded directory") for name in args.exclude_dir
    }
    if args.output_dir:
        destination = project_path(args.output_dir)
        try:
            relative = destination.relative_to(PROJECT)
        except ValueError:
            pass
        else:
            if relative.parts:
                exclusions.add(relative.parts[0])
    folders = []
    for entry in sorted(PROJECT.iterdir()):
        if (entry.name.startswith((".", "submissions_", "results_"))
                or entry.name in exclusions or not entry.is_dir()):
            continue
        safe_name(entry.name, "resource directory")
        if entry.is_symlink():
            raise ValueError(f"Copy or exclude directory symlink: {entry}")
        folders.append(entry)
    missing = REQUIRED_DIRS - {folder.name for folder in folders}
    if missing:
        raise FileNotFoundError(f"Required resource folders are missing or excluded: {sorted(missing)}")
    return folders


def archive_member(member):
    parts = PurePosixPath(member.name).parts
    if any(part in {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
           or part.startswith("x509up") for part in parts):
        return None
    if member.name.endswith((".pyc", ".pyo", "~")):
        return None
    source = PROJECT / member.name
    if source.is_symlink() and source.is_dir():
        raise ValueError(f"Copy or exclude directory symlink: {source}")
    return member


def settings_text(args, archives):
    driver_args = ["--debug-event", str(args.debug_event)]
    if args.do_jer:
        driver_args.append("--do-jer")
    if args.debug:
        driver_args.append("--debug")
    upload = not args.local_test or args.local_upload
    values = {
        "SKIM_EOS_SERVER": args.eos_server,
        "SKIM_BASE_EOS_DIR": args.base_eos_dir,
        "SKIM_COPY_TO_EOS": "1" if upload else "0",
        "SKIM_KEEP_LOCAL": "1" if args.local_test else "0",
    }
    lines = [f"{key}={shlex.quote(value)}" for key, value in values.items()]
    lines.append("SKIM_ARCHIVES=(" + " ".join(map(shlex.quote, archives)) + ")")
    lines.append("SKIM_DRIVER_ARGS=(" + " ".join(map(shlex.quote, driver_args)) + ")")
    return "\n".join(lines) + "\n"


def write_jdl(path, stage, snapshot, joblist, common_inputs, args):
    inputs = common_inputs + [snapshot]
    lines = [
        "universe = vanilla",
        f"initialdir = {submit_path(PROJECT)}",
        f"executable = {submit_path(stage / 'run_skimming.sh')}",
        'arguments = "$(jobindex) $(dataset_json) $(dataset_key)"',
        "transfer_input_files = " + ", ".join(submit_path(p) for p in inputs),
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        # ROOT files are uploaded by the wrapper, not copied back through Condor.
        'transfer_output_files = ""',
        "output = out/job_$(Cluster)_$(Process)_$(dataset_key).out",
        "error = err/job_$(Cluster)_$(Process)_$(dataset_key).err",
        f'+SingularityImage = "{args.image}"',
        "+SingularityBindCVMFS = True",
        f'+JobFlavour = "{args.flavour}"',
        "request_cpus = 1",
        f"request_memory = {args.memory_mb}",
        "on_exit_hold = (ExitBySignal == true) || (ExitCode != 0)",
    ]
    if args.proxy:
        lines.append('environment = "X509_USER_PROXY=x509up"')
    lines.append(f"queue jobindex, dataset_json, dataset_key from {submit_path(joblist)}")
    path.write_text("\n".join(lines) + "\n")


def prepare_jobs(args, batches, folders, proxy, only_missing):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    stage = PROJECT / "submissions" / run_id
    stage.mkdir(parents=True)
    for directory in ("out", "err"):
        (PROJECT / directory).mkdir(exist_ok=True)
    for name in INPUT_FILES:
        shutil.copy2(PROJECT / name, stage / name)
    (stage / "run_skimming.sh").chmod(0o755)
    if proxy is not None:
        shutil.copy2(proxy, stage / "x509up")
        (stage / "x509up").chmod(0o600)
    archives = []
    for folder in folders:
        name = folder.name + ".tar.gz"
        archive = stage / name
        with tarfile.open(archive, "w:gz", dereference=True) as tar:
            tar.add(folder, arcname=folder.name, filter=archive_member)
        archives.append(name)
        print(f"Created {name}: {archive.stat().st_size / 1024**2:.1f} MiB")
    settings = stage / "skim_job_settings.sh"
    settings.write_text(settings_text(args, archives))
    subprocess.run(["bash", "-n", str(settings)], check=True)
    common = [stage / name for name in INPUT_FILES if name != "run_skimming.sh"]
    common += [settings] + [stage / name for name in archives]
    if proxy is not None:
        common.append(stage / "x509up")
    jdls, local_job = [], None
    for number, (json_path, rows) in enumerate(batches):
        dataset_stage = stage / f"dataset_{number}"
        dataset_stage.mkdir()
        snapshot = dataset_stage / json_path.name
        shutil.copy2(json_path, snapshot)
        joblist = dataset_stage / f"joblist_{json_path.name}.txt"
        joblist.write_text("".join(f"{i} {name} {key}\n" for i, name, key in rows))
        prefix = "resubmit" if only_missing else "submit"
        jdl = dataset_stage / f"{prefix}_{json_path.name}.jdl"
        write_jdl(jdl, stage, snapshot, joblist, common, args)
        jdls.append(jdl)
        if local_job is None:
            local_job = (snapshot, rows[0])
        print(f"Prepared {len(rows)} skim jobs: {jdl}")
    (stage / "submission.json").write_text(json.dumps({
        "image": args.image, "eos_server": args.eos_server,
        "base_eos_dir": args.base_eos_dir, "resource_folders": [f.name for f in folders],
        "job_count": sum(len(rows) for _, rows in batches),
        "local_test": args.local_test, "only_missing": only_missing,
        "proxy_staged": proxy is not None,
        "do_jer": args.do_jer, "debug": args.debug, "debug_event": args.debug_event,
    }, indent=2) + "\n")
    return stage, jdls, local_job


def run_local(args, stage, job, runtime):
    snapshot, (index, json_name, key) = job
    shutil.copy2(snapshot, stage / json_name)
    log_stem = f"local_{stage.name}_{key}_{index}"
    stdout = PROJECT / "out" / f"{log_stem}.out"
    stderr = PROJECT / "err" / f"{log_stem}.err"
    command = [runtime, "exec", "--bind", f"{stage}:/srv", "--bind", "/cvmfs:/cvmfs"]
    for binding in args.bind:
        command += ["--bind", binding]
    command += ["--pwd", "/srv", args.image, "bash", "run_skimming.sh",
                str(index), json_name, key]
    print("LOCAL TEST: exactly one job; no Condor submission", flush=True)
    print(f"Job directory: {stage}\nstdout: {stdout}\nstderr: {stderr}", flush=True)
    print("Command:", shlex.join(command), flush=True)
    with stdout.open("wb") as out, stderr.open("wb") as err:
        result = subprocess.run(command, cwd=stage, stdout=out, stderr=err)
    print(f"Local job exit code: {result.returncode}")
    if result.returncode:
        print(f"Inspect the logs. Any partial ROOT output remains in {stage}")
        raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
    output = stage / output_name(key, index)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Local job returned zero without a nonempty output: {output}")
    destination = project_path(args.output_dir) if args.output_dir else PROJECT
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / output.name
    if output != target:
        shutil.move(str(output), str(target))
    print(f"ROOT output: {target}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", nargs="?", default="ZWQQ.json",
                        help="JSON filename/pattern under datasets/ (default: ZWQQ.json)")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--dry-run", action="store_true", help="Prepare files; do not run or submit")
    execution.add_argument("--local-test", action="store_true", help="Run exactly one job inside the image")
    parser.add_argument("--max-jobs", type=int, help="Total job cap across all matching datasets")
    parser.add_argument("--job-index", type=int, help="Select this zero-based input-file index")
    parser.add_argument("--filter-key", default=os.environ.get("FILTER_KEY"), help="Dataset-key regex")
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--proxy", metavar="PATH", default="x509up",
                             help="Stage this proxy as x509up (default: x509up beside this script)")
    credentials.add_argument("--no-proxy", action="store_true",
                             help="Do not stage a proxy, even if local x509up exists")
    parser.add_argument("--exclude-dir", action="append", default=[], help="Extra top-level folder to exclude")
    parser.add_argument("--image", default=IMAGE, help="Image used by both Condor and local tests")
    parser.add_argument("--eos-server", default=EOS_SERVER)
    parser.add_argument("--base-eos-dir", default=BASE_EOS_DIR)
    parser.add_argument("--output-dir", help="Local-test output directory; also check local outputs when resubmitting")
    parser.add_argument("--memory-mb", type=int, default=3000)
    parser.add_argument("--flavour", default="workday")
    jer = parser.add_mutually_exclusive_group()
    jer.add_argument("--do-jer", dest="do_jer", action="store_true", help="Enable JER diagnostics (default)")
    jer.add_argument("--no-jer", dest="do_jer", action="store_false", help="Disable JER diagnostics")
    parser.set_defaults(do_jer=True)
    parser.add_argument("--debug", action="store_true", help="Enable the driver's event-level debug output")
    parser.add_argument("--debug-event", type=int, default=10)
    parser.add_argument("--bind", action="append", default=[], help="Extra Singularity/Apptainer bind for local tests")
    parser.add_argument("--local-upload", action="store_true", help="Also upload a local-test output to EOS")
    args = parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max-jobs must be positive")
    if args.job_index is not None and args.job_index < 0:
        parser.error("--job-index must be nonnegative")
    if args.memory_mb <= 0 or args.debug_event < 0:
        parser.error("--memory-mb must be positive and --debug-event nonnegative")
    if args.local_test:
        if args.max_jobs is not None and args.max_jobs != 1:
            parser.error("--local-test runs one job; omit --max-jobs or use 1")
        args.max_jobs = 1
    if args.local_upload and not args.local_test:
        parser.error("--local-upload requires --local-test")
    if args.bind and not args.local_test:
        parser.error("--bind is only used with --local-test")
    pattern = Path(args.pattern)
    if pattern.is_absolute() or ".." in pattern.parts:
        parser.error("pattern must remain inside datasets/")
    if not re.fullmatch(r"root://[A-Za-z0-9_.:-]+/?", args.eos_server):
        parser.error("--eos-server must be a root://host[:port] endpoint")
    args.eos_server = args.eos_server.rstrip("/")
    base = PurePosixPath(args.base_eos_dir)
    if not str(base).startswith("/eos/") or ".." in base.parts or any(c.isspace() for c in str(base)):
        parser.error("--base-eos-dir must be an absolute /eos/... path")
    args.base_eos_dir = str(base).rstrip("/")
    if not args.image.startswith("/") or any(c in args.image for c in '\n\r"'):
        parser.error("--image must be an absolute container path without quotes/newlines")
    safe_name(args.flavour, "job flavour")
    if args.filter_key:
        try:
            re.compile(args.filter_key)
        except re.error as error:
            parser.error(f"Invalid dataset-key regex: {error}")
    return args


def main(argv=None, *, only_missing=False):
    args = parse_args(argv)
    submit_path(PROJECT)
    environment = os.environ.copy()
    proxy = resolve_proxy(args)
    # Normalize the effective choice so --no-proxy also suppresses the JDL setting.
    args.proxy = str(proxy) if proxy is not None else None
    if proxy is not None:
        environment["X509_USER_PROXY"] = str(proxy)
    print(f"Proxy to stage: {proxy}" if proxy else "Proxy to stage: none")
    if only_missing:
        location = str(project_path(args.output_dir)) if args.output_dir else (
            f"{args.eos_server}/{args.base_eos_dir}"
        )
        print(f"Checking existing skims in: {location}")
        if not args.output_dir and shutil.which("xrdfs") is None:
            raise RuntimeError("xrdfs is required to check EOS outputs")
    batches, total = select_jobs(args, environment, only_missing)
    if not total:
        print("No matching missing jobs." if only_missing else "No matching jobs.")
        return
    for name in INPUT_FILES:
        path = PROJECT / name
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
        if path.suffix == ".py":
            compile(path.read_text(), name, "exec")
    subprocess.run(["bash", "-n", str(PROJECT / "run_skimming.sh")], check=True)
    folders = resource_folders(args)
    runtime = None
    if args.local_test:
        runtime = shutil.which("singularity") or shutil.which("apptainer")
        if runtime is None:
            raise RuntimeError("--local-test requires singularity or apptainer")
        if not Path(args.image).exists():
            raise FileNotFoundError(f"Container image is not available: {args.image}")
    elif not args.dry_run and shutil.which("condor_submit") is None:
        raise RuntimeError("condor_submit is unavailable; use --dry-run")
    print("Resource folders:", ", ".join(folder.name for folder in folders))
    print(f"Image: {args.image}")
    stage, jdls, local_job = prepare_jobs(args, batches, folders, proxy, only_missing)
    print(f"Total jobs: {total}")
    if args.local_test:
        run_local(args, stage, local_job, runtime)
    elif args.dry_run:
        print("Dry run: JDLs and fresh archives written; no jobs submitted.")
    else:
        for jdl in jdls:
            subprocess.run(["condor_submit", str(jdl)], cwd=PROJECT, check=True, env=environment)


if __name__ == "__main__":
    main()
