#!/usr/bin/env python3
"""Run the time-budgeted SAST paper study: python run_paper_experiments.py --help."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def parser():
    p = argparse.ArgumentParser(description="SAST paper experiments on independent GPUs, with a 48-hour deadline.")
    commands = p.add_subparsers(dest="command", required=True)
    for name, help_text in (("doctor", "Check dependencies, GPUs, and actual CUDA backward execution."),
        ("plan", "Print the requested experiment matrix; no downloads or training."),
        ("prepare", "Download/verify datasets and build resumable frame caches."),
        ("run", "Benchmark, freeze a plan that fits the budget, and run it resumably."),
        ("status", "Show completed/failed jobs in an existing output directory."),
        ("report", "Regenerate CSV/JSON/PDF reports from completed jobs."),
        ("smoke", "Run every code path on small synthetic data, on CPU.")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--output", "-o", default="runs/paper_48h", help="Run directory (resume with the same path).")
        command.add_argument("--config", help="JSON overrides; init writes a complete example.")
        command.add_argument("--data", help="Shared raw/frame cache directory; default ./data.")
        command.add_argument("--microbatch", type=int, default=4, help="Per-GPU microbatch; effective batch remains fixed (default 4).")
        command.add_argument("--threads", type=int, default=2, help="CPU/BLAS threads per worker (default 2).")
        group = command.add_mutually_exclusive_group()
        group.add_argument("--gpus", default="0,1,2,3,4,5", help="Logical CUDA indices after CUDA_VISIBLE_DEVICES.")
        group.add_argument("--cpu", action="store_true", help="Explicit CPU execution (automatic for smoke).")
        command.add_argument("--budget-hours", type=float, help="Wall-time limit, at most 48 hours; includes automatic data preparation.")
        if name in ("run", "smoke"):
            command.add_argument("--stages", help="Comma-separated stages: toy,tune,confirm,causal,ablate,analysis,report.")
    init = commands.add_parser("init", help="Write the default 48-hour JSON configuration.")
    init.add_argument("path", nargs="?", default="paper_48h.json")
    return p


def gpu_indices(value):
    try:
        indices = [int(v) for v in value.split(",")]
    except ValueError as exc:
        raise ValueError("--gpus must contain comma-separated integer indices.") from exc
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise ValueError("GPU indices must be distinct and nonnegative.")
    return indices


def doctor(indices, cpu=False, data_root="."):
    import torch
    packages = ["torch", "numpy", "scipy", "pandas", "matplotlib", "tonic", "h5py", "aedat", "psutil"]
    versions, missing = {}, []
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    result = {"python": sys.version.split()[0], "packages": versions, "missing": missing,
              "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpus": []}
    if not cpu:
        if missing:
            raise RuntimeError("Install requirements/requirements-paper-linux.txt first; missing: "+", ".join(missing))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install the CUDA PyTorch wheel and a compatible NVIDIA driver; see docs/paper_experiments.md.")
        for index in indices:
            if index >= torch.cuda.device_count():
                raise RuntimeError(f"Requested GPU {index}, but only {torch.cuda.device_count()} GPUs are visible.")
            device = torch.device(f"cuda:{index}")
            torch.cuda.set_device(device)
            free, total = torch.cuda.mem_get_info(device)
            if free < 2*1024**3:
                raise RuntimeError(f"GPU {index} has under 2 GiB free. Free GPU memory before running.")
            # Executing kernels catches unsupported wheel architectures and driver problems.
            layer = torch.nn.Conv2d(2, 4, 3, padding=1).to(device)
            x = torch.randn(2, 2, 16, 16, device=device, requires_grad=True)
            (layer(x).square().mean()).backward()
            torch.cuda.synchronize(device)
            result["gpus"].append({"index": index, "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "free_gib": free/1024**3, "total_gib": total/1024**3, "backward": "passed"})
            del x, layer
            torch.cuda.empty_cache()
        result["wheel_architectures"] = torch.cuda.get_arch_list()
        try:
            result["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.free",
                "--format=csv"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            result["nvidia_smi"] = "unavailable; direct CUDA backward check passed"
    path = Path(data_root).resolve()
    while not path.exists():
        path = path.parent
    result["disk_free_gib"] = shutil.disk_usage(path).free/1024**3
    if not cpu and result["disk_free_gib"] < 75:
        result["disk_note"] = "Under 75 GiB free; full raw downloads/extraction/cache may not fit. Use a larger --data volume."
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    threads = getattr(args, "threads", 2)
    if threads < 1 or getattr(args, "microbatch", 4) < 1:
        raise ValueError("--threads and --microbatch must be positive.")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from src import paper_suite as suite
    from src.paper_suite_data import write_json
    if args.command == "init":
        path = Path(args.path)
        if path.exists():
            raise FileExistsError(f"Preserving existing configuration: {path}")
        write_json(path, suite.two_day_config())
        print(path.resolve())
        return 0
    root = Path(args.output).expanduser().resolve()
    c = suite.read_config(args.config, synthetic=args.command == "smoke", data_root=args.data)
    if args.command == "smoke":
        c["synthetic"] = True
        if not args.data:
            c["data_root"] = str(root/"synthetic_data")
    if args.budget_hours is not None:
        if not 0 < args.budget_hours <= 48:
            raise ValueError("--budget-hours must be positive and at most 48.")
        c["budget"]["hours"] = args.budget_hours
    suite.validate_config(c)
    if args.command in ("status", "report"):
        path = root/"protocol.json"
        if not path.exists():
            raise FileNotFoundError("No frozen run protocol in "+str(root))
        c = json.loads(path.read_text())["configuration"]
        if args.command == "status":
            print(json.dumps(suite.status(c, root), indent=2))
        else:
            suite.report(c, root)
        return 0
    if args.command == "plan":
        print(json.dumps(suite.plan(c), indent=2))
        return 0
    if args.command == "prepare":
        with suite.file_lock(root/"scheduler.lock", blocking=False):
            suite.prepare(c, root)
        return 0
    indices = gpu_indices(args.gpus)
    cpu = args.cpu or args.command == "smoke"
    if args.command == "doctor":
        print(json.dumps(doctor(indices, cpu, c["data_root"]), indent=2))
        return 0
    hardware = doctor(indices, cpu, c["data_root"])
    print(json.dumps(hardware, indent=2), flush=True)
    devices = ["cpu"] if cpu else [f"cuda:{index}" for index in indices]
    stages = [value.strip() for value in args.stages.split(",")] if args.stages else None
    suite.run(c, root, devices, args.microbatch, threads, stages, hardware=hardware)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume from completed epochs.", file=sys.stderr)
        sys.exit(130)
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(124)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
