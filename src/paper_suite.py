"""Resumable, independent-GPU execution of the six SAST paper experiment groups."""
import contextlib
import copy
import csv
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import torch
from torch.nn import functional as F

from src.paper_suite_data import (ROOT, LEGACY_SOURCES, atomic_torch_save, data_key,
    digest, file_hash, file_lock, load_data, prepare_case_data, stable_seed, SubsetSplit,
    stratified_ids, write_json)
from src.paper_suite_models import METHODS, SuiteSNN, train_step


CAUSAL_METHODS = ["erm", "erm2x", "sast", "margin", "sast_margin", "sam", "asam",
                  "asam_independent", "avg2", "random", "rho0", "hard_sg"]
STAGES = ["toy", "tune", "confirm", "causal", "ablate", "analysis", "report"]
_WORKER = {}


def default_config():
    return {
        "version": 1, "synthetic": False, "data_root": "./data",
        "datasets": ["nmnist", "dvsgesture", "cifar10dvs", "shd"],
        "architectures": {"nmnist": ["cnn2"], "dvsgesture": ["cnn2", "cnn4"],
                          "cifar10dvs": ["cnn2", "cnn4"], "shd": ["mlp2"]},
        "widths": {"cnn2": [16, 32], "cnn4": [16, 32, 64, 64], "mlp2": [256, 128]},
        "time_bins": {"nmnist": 16, "dvsgesture": 16, "cifar10dvs": 16, "shd": 64},
        "epochs": {"nmnist": 20, "dvsgesture": 50, "cifar10dvs": 100, "shd": 50},
        "batch_size": {"nmnist": 32, "dvsgesture": 8, "cifar10dvs": 16, "shd": 64},
        "leak": .5, "slope": 1., "threshold": 1., "margin_target": .25,
        "split_seeds": [2026], "pilot_seeds": [100, 101, 102],
        "seeds": list(range(1000, 1030)),
        "methods": ["erm2x", "sast", "sam", "asam", "hard_sg", "margin", "sast_margin", "input_gradient"],
        "tuning": {"candidates": 3, "lr": [.0003, .001, .003],
                   "weight_decay": [0., .0001, .001], "rho": [.05, .2, .4],
                   "beta": [.1, 1., 10.], "input_penalty": [.001, .01, .1], "sgd_lr": [.01, .03, .1]},
        "calibration": {"grid": [.25, .5, .75, 1., 1.5, 2.], "minimum": .015625,
                        "maximum": 8., "rounds": 2, "examples": None},
        "causal_datasets": ["nmnist", "dvsgesture"], "warmup_fraction": .6,
        "adaptation_fractions": [.02, .05, .1, .25],
        "ablation": {"datasets": ["dvsgesture", "shd"], "seeds": 5,
            "methods": ["erm2x", "sast", "hard_sg"], "slopes": [1., 10., 25.],
            "leaks": [.5, .75, .99], "time_bins": [8, 32, 64],
            "radii": [0., .005, .01, .1, .2], "batch_factors": [.5, 2., 4.],
            "train_fractions": [.1, .25, .5], "width_factors": [.5, 2.],
            "lr_factors": [.1, 10.], "betas": [.01, .1, 10.],
            "margin_targets": [.05, .1, .5], "detach_reset": True},
        "analysis": {"seeds": 5, "geometry_seeds": 3, "examples": 512,
                     "probe_train": 4000, "probe_validation": 1000,
                     "bounds_examples": 128, "geometry_examples": 32,
                     "planes": 2, "grid_size": 15, "radii": [.05, .2, .4],
                     "pgd_steps": 10, "pgd_starts": 3, "hessian_steps": 20,
                     "noise_batches": 32, "noise_batch_factors": [.5, 1., 2.],
                     "ridge": [.000001, .0001, .01, 1.],
                     "homotopy": [0., .25, .5, .75, 1.],
                     "corruption_levels": [.1, .25, .5, .75],
                     "attack_epsilons": [.01, .05, .1], "attack_steps": 10,
                     "attack_starts": 2, "timing_iterations": 50},
        "toy": {"slopes": [1., 5., 25.], "leaks": [.5, .9],
                "time_bins": [4, 16, 64], "radii": [.001, .02, .1]},
        "validation_speakers": 2, "limit_train_per_class": None,
        "limit_eval_per_class": None, "steps_per_epoch": None,
        "budget": {"hours": 48., "reserve_fraction": .2, "safety_factor": 1.5,
                   "minimum_seeds": 3, "minimum_epoch_fraction": .25},
    }


def extensive_config():
    c = default_config()
    c["methods"] += ["avg2", "random", "rho0", "sgd2x", "sast_sgd", "sam_sgd"]
    return c


def two_day_config():
    c = default_config()
    c["datasets"] = ["dvsgesture", "cifar10dvs", "shd"]
    c["architectures"]["cifar10dvs"] = ["cnn2"]
    c["seeds"], c["pilot_seeds"] = list(range(1000, 1005)), [100]
    c["epochs"].update(dvsgesture=30, cifar10dvs=25, shd=20)
    c["methods"] = ["erm2x", "sast", "sam", "asam", "hard_sg", "margin", "sast_margin"]
    c["causal_datasets"] = ["dvsgesture"]
    c["ablation"]["datasets"] = []
    c["adaptation_fractions"] = [.1]
    c["analysis"].update(seeds=1, geometry_seeds=1, examples=128, probe_train=512,
        probe_validation=256, bounds_examples=32,
        planes=1, grid_size=5, geometry_examples=32, radii=[.05, .4], pgd_steps=3, pgd_starts=1,
        hessian_steps=4, noise_batches=8, noise_batch_factors=[.5, 1., 2.],
        homotopy=[0., .5, 1.], corruption_levels=[.25, .5],
        attack_epsilons=[.05], attack_steps=3, attack_starts=1, timing_iterations=10)
    c["calibration"].update(rounds=1, examples=256)
    return c


def smoke_config():
    c = extensive_config()
    c.update(synthetic=True, seeds=[1000, 1001], pilot_seeds=[100], steps_per_epoch=1)
    c["widths"] = {"cnn2": [2, 3], "cnn4": [2, 3, 3, 3], "mlp2": [4, 3]}
    for field, value in (("time_bins", 2), ("epochs", 1), ("batch_size", 3)):
        c[field] = {name: value for name in c["datasets"]}
    c["tuning"]["candidates"] = 1
    c["calibration"].update(grid=[.5, 1., 2.], minimum=.5, maximum=2., rounds=1)
    c["analysis"].update(seeds=1, geometry_seeds=1, examples=6, probe_train=12,
        probe_validation=6, bounds_examples=3, geometry_examples=3, planes=1,
        grid_size=3, radii=[.02], pgd_steps=2, pgd_starts=1, hessian_steps=2,
        noise_batches=3, noise_batch_factors=[1.], ridge=[.001, 1.], homotopy=[0., 1.],
        corruption_levels=[.25], attack_epsilons=[.05], attack_steps=2,
        attack_starts=1, timing_iterations=2)
    c["adaptation_fractions"] = [.1]
    c["ablation"].update(seeds=1, slopes=[1.], leaks=[.5], time_bins=[3],
        radii=[0.], batch_factors=[2.], train_fractions=[.5], width_factors=[2.],
        lr_factors=[.1], betas=[.1], margin_targets=[.1])
    c["toy"] = {"slopes": [1., 5.], "leaks": [.5], "time_bins": [4, 16], "radii": [.02]}
    return c


def read_config(path=None, synthetic=False, data_root=None):
    c = smoke_config() if synthetic else two_day_config()
    if path:
        supplied = json.loads(Path(path).read_text())
        def merge(dst, src, prefix=""):
            if not isinstance(src, dict):
                raise ValueError(f"{prefix or 'configuration'} must be an object.")
            for key, value in src.items():
                if key not in dst:
                    raise ValueError(f"Unknown configuration key: {prefix+key}")
                if isinstance(dst[key], dict):
                    merge(dst[key], value, prefix+key+".")
                else:
                    dst[key] = value
        merge(c, supplied)
    if data_root is not None:
        c["data_root"] = data_root
    c["data_root"] = str(Path(c["data_root"]).expanduser().resolve())
    validate_config(c)
    return c


def validate_config(c):
    def numbers(values, name, low=0, integer=False, inclusive=False, distinct=True):
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty list.")
        for v in values:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"Invalid number in {name}.")
            if (v < low if inclusive else v <= low) or (integer and not isinstance(v, int)):
                raise ValueError(f"Out-of-range value in {name}: {v}")
        if distinct and len(set(values)) != len(values):
            raise ValueError(f"Duplicate values in {name}.")
    if c["version"] != 1 or type(c["synthetic"]) is not bool:
        raise ValueError("Unsupported configuration version or synthetic flag.")
    datasets = {"nmnist", "dvsgesture", "cifar10dvs", "shd"}
    if not c["datasets"] or not set(c["datasets"]) <= datasets or len(set(c["datasets"])) != len(c["datasets"]):
        raise ValueError("Choose distinct supported datasets.")
    if not set(c["causal_datasets"]) <= set(c["datasets"]):
        raise ValueError("causal_datasets must be included in datasets.")
    for field in ("time_bins", "epochs", "batch_size"):
        for dataset in c["datasets"]:
            numbers([c[field][dataset]], field+"."+dataset, integer=True)
    for dataset in c["datasets"]:
        archs = c["architectures"][dataset]
        allowed = {"mlp2"} if dataset == "shd" else {"cnn2", "cnn4"}
        if not archs or not set(archs) <= allowed or len(set(archs)) != len(archs):
            raise ValueError("Incompatible or duplicate architecture for "+dataset)
        for arch in archs:
            numbers(c["widths"][arch], "widths."+arch, integer=True, distinct=False)
            if len(c["widths"][arch]) != int(arch[-1]):
                raise ValueError("Architecture depth and width list disagree.")
    for field in ("seeds", "pilot_seeds", "split_seeds"):
        numbers(c[field], field, integer=True, inclusive=True)
    if set(c["seeds"]) & set(c["pilot_seeds"]):
        raise ValueError("Pilot and confirmation seeds must be disjoint.")
    if not c["methods"] or not set(c["methods"]) <= set(METHODS) or len(set(c["methods"])) != len(c["methods"]):
        raise ValueError("Choose distinct supported methods.")
    if c["causal_datasets"] and not {"erm2x", "sast", "margin"} <= set(c["methods"]):
        raise ValueError("Causal experiments require erm2x, sast and margin tuning.")
    for key, values in c["tuning"].items():
        numbers([values] if key == "candidates" else values, "tuning."+key,
                integer=key == "candidates", inclusive=key == "weight_decay")
    for key in ("slope", "threshold", "margin_target"):
        numbers([c[key]], key)
    if not 0 <= c["leak"] <= 1 or not 0 < c["warmup_fraction"] < 1:
        raise ValueError("leak must be in [0,1], warmup_fraction in (0,1).")
    cal, a = c["calibration"], c["analysis"]
    numbers(cal["grid"], "calibration.grid")
    numbers([cal["minimum"]], "calibration.minimum")
    if not cal["minimum"] <= min(cal["grid"]) <= max(cal["grid"]) <= cal["maximum"]:
        raise ValueError("Calibration grid must lie inside its positive bounds.")
    for key in ("rounds",):
        numbers([cal[key]], "calibration."+key, integer=True)
    for key in ("seeds", "geometry_seeds", "examples", "probe_train", "probe_validation",
                "bounds_examples", "geometry_examples", "planes", "grid_size", "pgd_steps",
                "pgd_starts", "hessian_steps", "noise_batches"):
        numbers([a[key]], "analysis."+key, integer=True)
    if a["grid_size"] < 3 or a["grid_size"] % 2 != 1:
        raise ValueError("Geometry grid_size must be odd and at least three.")
    for key in ("radii", "noise_batch_factors", "ridge"):
        numbers(a[key], "analysis."+key)
    numbers(a["homotopy"], "analysis.homotopy", inclusive=True)
    if max(a["homotopy"]) > 1 or not {0., 1.} <= set(a["homotopy"]):
        raise ValueError("Homotopy must include 0 and 1 and stay in [0,1].")
    for key in ("slopes", "leaks", "time_bins", "radii"):
        numbers(c["toy"][key], "toy."+key, integer=key == "time_bins", inclusive=key == "leaks")
    if max(c["toy"]["leaks"]) >= 1:
        raise ValueError("Toy interval bounds require leak < 1.")
    for key in ("limit_train_per_class", "limit_eval_per_class", "steps_per_epoch"):
        if c[key] is not None:
            numbers([c[key]], key, integer=True)
    if cal["examples"] is not None:
        numbers([cal["examples"]], "calibration.examples", integer=True)
    numbers([c["validation_speakers"]], "validation_speakers", integer=True)
    numbers(c["adaptation_fractions"], "adaptation_fractions")
    if max(c["adaptation_fractions"]) > 1:
        raise ValueError("Adaptation fractions must not exceed one.")
    ab = c["ablation"]
    if not set(ab["datasets"]) <= set(c["datasets"]) or not set(ab["methods"]) <= set(c["methods"]):
        raise ValueError("Ablation datasets/methods must be present in the confirmation protocol.")
    if type(ab["detach_reset"]) is not bool:
        raise ValueError("ablation.detach_reset must be boolean.")
    for key in ("seeds",):
        numbers([ab[key]], "ablation."+key, integer=True)
    for key in ("slopes", "leaks", "time_bins", "radii", "batch_factors", "train_fractions",
                "width_factors", "lr_factors", "betas", "margin_targets"):
        numbers(ab[key], "ablation."+key, integer=key == "time_bins", inclusive=key in ("radii", "leaks"))
    if max(ab["train_fractions"]) > 1 or max(ab["leaks"]) > 1:
        raise ValueError("Ablation fractions/leaks must not exceed one.")
    for key in ("attack_steps", "attack_starts", "timing_iterations"):
        numbers([a[key]], "analysis."+key, integer=True)
    for key in ("attack_epsilons", "corruption_levels"):
        numbers(a[key], "analysis."+key)
    if max(a["corruption_levels"]) >= 1 or max(a["attack_epsilons"]) > 1:
        raise ValueError("Corruption/attack levels must be within [0,1].")
    for key in ("hours", "safety_factor", "minimum_epoch_fraction", "reserve_fraction"):
        numbers([c["budget"][key]], "budget."+key)
    numbers([c["budget"]["minimum_seeds"]], "budget.minimum_seeds", integer=True)
    if not 0 < c["budget"]["reserve_fraction"] < 1 or not 0 < c["budget"]["minimum_epoch_fraction"] <= 1:
        raise ValueError("Invalid time-budget fractions.")
    if c["budget"]["hours"] > 48 or c["budget"]["safety_factor"] < 1:
        raise ValueError("The run must fit at most 48 hours and use a safety factor of at least one.")


def cases(c):
    return [{"dataset": d, "architecture": a, "split_seed": s, "time_bins": c["time_bins"][d],
             "id": f"{d}_{a}_t{c['time_bins'][d]}_s{s}"}
            for d in c["datasets"] for a in c["architectures"][d] for s in c["split_seeds"]]


def candidates(c, method):
    # Equal numbers of candidate configurations and pilot seeds for every method.
    fields = ["lr", "weight_decay"]
    canonical = {"sgd2x": "erm2x", "sast_sgd": "sast", "sam_sgd": "sam",
                 "asam_independent": "asam"}.get(method, method)
    if canonical in ("sast", "sam", "asam", "sast_margin", "random"):
        fields.append("rho")
    if method in ("margin", "sast_margin"):
        fields.append("beta")
    if method == "input_gradient":
        fields.append("input_penalty")
    count = c["tuning"]["candidates"]
    learning_rates = c["tuning"]["sgd_lr"] if method.endswith("sgd") or method == "sgd2x" else c["tuning"]["lr"]
    combinations = list(itertools.product(*(learning_rates if k == "lr" else c["tuning"][k] for k in fields)))
    if count > len(combinations):
        raise ValueError(f"{method} has only {len(combinations)} unique candidates, requested {count}.")
    # Balance all search dimensions within the small equal-budget candidate set.
    rng = np.random.default_rng(stable_seed("tuning", method))
    rng.shuffle(combinations)
    frequencies = [{} for _ in fields]
    ordered = []
    for _ in range(count):
        chosen = min(combinations, key=lambda item: sum(frequencies[i].get(value, 0) for i, value in enumerate(item)))
        ordered.append(chosen)
        combinations.remove(chosen)
        for i, value in enumerate(chosen):
            frequencies[i][value] = frequencies[i].get(value, 0)+1
    base = {"rho": .05, "beta": 1., "input_penalty": .01, "margin_target": c["margin_target"]}
    return [dict(base, **dict(zip(fields, item))) for item in ordered[:count]]


def job(kind, case=None, method=None, seed=None, candidate=None):
    value = {"kind": kind}
    for key, item in (("case", case), ("method", method), ("seed", seed), ("candidate", candidate)):
        if item is not None:
            value[key] = item
    parts = [kind, case["id"] if case else "all", method or "", f"c{candidate}" if candidate is not None else "",
             f"seed{seed}" if seed is not None else ""]
    value["id"] = "__".join(p for p in parts if p)
    return value


def stage_jobs(c, stage):
    all_cases = cases(c)
    if stage == "toy":
        return [job("toy")]
    if stage == "tune":
        return [job("pilot", case, method, seed, i) for case in all_cases for method in c["methods"]
                for i in range(len(candidates(c, method))) for seed in c["pilot_seeds"]]
    if stage == "confirm":
        return [job("confirm", case, method, seed) for case in all_cases
                for method in c["methods"] for seed in c["seeds"]]
    if stage == "causal":
        return [job("causal", case, seed=seed) for case in all_cases if case["architecture"] == "cnn2"
                and case["dataset"] in c["causal_datasets"] for seed in c["seeds"]]
    if stage == "analysis":
        return [job("analysis", case, method, seed) for case in all_cases for method in c["methods"]
                for seed in c["seeds"][:c["analysis"]["seeds"]]]
    if stage == "ablate":
        output = []
        for case in all_cases:
            if case["dataset"] not in c["ablation"]["datasets"] or case["architecture"] == "cnn4":
                continue
            for variant in ablation_variants(c, case):
                for method in variant["methods"]:
                    for seed in c["seeds"][:c["ablation"]["seeds"]]:
                        item = job("ablate", case, method, seed)
                        item["variant"] = variant
                        item["id"] += "__"+variant["name"]
                        output.append(item)
        return output
    return []


def ablation_variants(c, case):
    options = c["ablation"]
    ordinary = options["methods"]
    definitions = [("slope", options["slopes"], ordinary), ("leak", options["leaks"], ordinary),
        ("time_bins", options["time_bins"], ordinary),
        ("rho", options["radii"], [m for m in ordinary if m in ("sast", "sam", "asam", "sast_margin")]),
        ("batch_factor", options["batch_factors"], ordinary),
        ("train_fraction", options["train_fractions"], ordinary),
        ("width_factor", options["width_factors"], ordinary),
        ("lr_factor", options["lr_factors"], ordinary),
        ("beta", options["betas"], [m for m in ("margin", "sast_margin") if m in c["methods"]]),
        ("margin_target", options["margin_targets"], [m for m in ("margin", "sast_margin") if m in c["methods"]])]
    if options["detach_reset"]:
        definitions.append(("detach_reset", [True], ordinary))
    output = []
    for factor, values, methods in definitions:
        for value in values:
            base = case["time_bins"] if factor == "time_bins" else c.get(factor)
            if value == base or not methods:
                continue
            output.append({"factor": factor, "value": value, "methods": methods,
                           "name": factor+"_"+str(value).lower().replace(".", "p")})
    return output


def run_ablation(c, task, data, descriptor, settings, directory, device, microbatch):
    altered, case, settings = copy.deepcopy(c), dict(task["case"]), dict(settings)
    factor, value = task["variant"]["factor"], task["variant"]["value"]
    base_batch = c["batch_size"][case["dataset"]]
    base_steps = c["steps_per_epoch"] or math.ceil(len(data["train"])/base_batch)
    altered["steps_per_epoch"] = base_steps
    if factor in ("slope", "leak", "margin_target", "detach_reset"):
        altered[factor] = value
        if factor == "margin_target":
            settings[factor] = value
    elif factor == "time_bins":
        case["time_bins"] = value
    elif factor in ("rho", "beta"):
        settings[factor] = value
    elif factor == "lr_factor":
        settings["lr"] *= value
    elif factor == "width_factor":
        altered["widths"][case["architecture"]] = [max(2, round(w*value)) for w in c["widths"][case["architecture"]]]
    elif factor == "batch_factor":
        new_batch = max(1, round(base_batch*value))
        altered["batch_size"][case["dataset"]] = new_batch
        altered["steps_per_epoch"] = math.ceil(base_steps*base_batch/new_batch)
    elif factor == "train_fraction":
        data = dict(data)
        ids = stratified_ids(data["train"].labels, max(descriptor["classes"], math.ceil(len(data["train"])*value)),
                             stable_seed("data_fraction", case["id"], task["seed"]))
        data["train"] = SubsetSplit(data["train"], ids)
    else:
        raise ValueError("Unknown ablation factor: "+factor)
    model, calibration, costs = fit(altered, case, task["method"], task["seed"], settings, directory,
                                   data, descriptor, device, microbatch)
    return {"job": task, "synthetic": descriptor["synthetic"], "calibration": calibration, "costs": costs,
            "training_examples_available": len(data["train"]), "settings": settings,
            "test": primary_evaluation(model, data, descriptor, calibration, directory, device, microbatch, c, task["seed"]),
            "interpretation": "one-factor sensitivity using base validation-tuned settings; batch/data sweeps hold processed-example budget approximately fixed"}


def source_identity():
    names = ["run_paper_experiments.py"]+["src/"+name for name in LEGACY_SOURCES]+[
        "src/paper_suite.py", "src/paper_suite_models.py", "src/paper_suite_data.py", "src/paper_suite_analysis.py"]
    return {name: file_hash(ROOT/name) for name in names}


def freeze_protocol(c, root):
    root = Path(root)
    protocol = {"configuration": c, "source_sha256": source_identity()}
    protocol["sha256"] = digest(protocol)
    path = root/"protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError("Configuration or source changed. Use a new output directory; existing results are preserved.")
    else:
        write_json(path, protocol)
    return protocol["sha256"]


def completed(directory, expected=None):
    directory = Path(directory)
    marker = directory/"completed.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    if expected is not None and saved["identity"] != expected:
        raise RuntimeError(f"Completed-job identity changed: {directory}")
    for name, checksum in saved["files"].items():
        if not (directory/name).is_file() or file_hash(directory/name) != checksum:
            raise RuntimeError(f"Completed artifact is missing or changed: {directory/name}")
    return True


def mark_completed(directory, identity):
    files = {str(p.relative_to(directory)): file_hash(p) for p in Path(directory).rglob("*")
             if p.is_file() and p.suffix in (".json", ".pt", ".npz", ".csv", ".pdf", ".png")
             and p.name not in ("completed.json", "failure.json")}
    write_json(Path(directory)/"completed.json", {"identity": identity, "files": files})


def configure_device(device, threads=2):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def model_spec(c, case, descriptor):
    return {"architecture": case["architecture"], "widths": c["widths"][case["architecture"]],
            "classes": descriptor["classes"], "spatial_size": descriptor["spatial_size"],
            "input_channels": descriptor["input_channels"], "leak": c["leak"],
            "slope": c["slope"], "threshold": c["threshold"], "detach_reset": c.get("detach_reset", False)}


@torch.no_grad()
def evaluate(model, split, device, microbatch, mode="smooth", scales=None, ids=None,
             information=False, output_hardness=None, reset_hardness=None):
    model.eval()
    ids = np.arange(len(split)) if ids is None else np.asarray(ids)
    logits, features, statistics = [], None, {}
    for start in range(0, len(ids), microbatch):
        xx, _ = split.batch(ids[start:start+microbatch], device)
        result = model(xx, mode=mode, scales=scales, return_info=information,
                       output_hardness=output_hardness, reset_hardness=reset_hardness)
        if information:
            output, info = result
            if features is None:
                features = [[] for _ in info["features"]]
            for target, value in zip(features, info["features"]):
                target.append(value.cpu().numpy())
            for key in ("min_margin", "mean_margin", "near_threshold", "activity", "temporal_gain"):
                statistics.setdefault(key, []).append(info[key].cpu().numpy())
        else:
            output = result
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite evaluation logits.")
        logits.append(output.cpu().numpy())
    output = np.concatenate(logits)
    labels = np.asarray(split.labels[ids])
    summary = classification(output, labels)
    result = {"logits": output, "labels": labels, "ids": np.asarray(split.ids[ids]), "summary": summary}
    if information:
        result["features"] = [np.concatenate(values) for values in features]
        result["statistics"] = {key: np.concatenate(values) for key, values in statistics.items()}
    return result


def classification(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits-logits.max(1, keepdims=True)
    ce = -shifted[np.arange(len(labels)), labels]+np.log(np.exp(shifted).sum(1))
    return {"accuracy": float(np.mean(logits.argmax(1) == labels)), "loss": float(ce.mean()), "n": len(labels)}


def calibrate(model, validation, c, device, microbatch, seed):
    options = c["calibration"]
    ids = stratified_ids(validation.labels, options["examples"], stable_seed("calibration", seed))
    evaluated, trials = {}, []
    def score(scales):
        key = tuple(float(v) for v in scales)
        if key not in evaluated:
            metric = evaluate(model, validation, device, microbatch, "hard", key, ids)["summary"]
            evaluated[key] = (-metric["accuracy"], metric["loss"], sum(abs(math.log(v)) for v in key))
            trials.append({"scales": list(key), **metric})
        return evaluated[key]
    layers = len(model.blocks)
    initial_boundary, at_cap = False, False
    def search(current, coordinate=None):
        nonlocal initial_boundary, at_cap
        grid = sorted(set(options["grid"]+[current[coordinate] if coordinate is not None else current[0]]))
        while True:
            settings = [([v]*layers if coordinate is None else current[:coordinate]+[v]+current[coordinate+1:]) for v in grid]
            best = min(settings, key=score)
            value = best[0] if coordinate is None else best[coordinate]
            lower, upper = value == grid[0], value == grid[-1]
            initial_boundary |= lower or upper
            additions = []
            if lower and grid[0] > options["minimum"]:
                additions.append(max(options["minimum"], grid[0]/2))
            if upper and grid[-1] < options["maximum"]:
                additions.append(min(options["maximum"], grid[-1]*2))
            if not additions:
                at_cap |= (lower and value == options["minimum"]) or (upper and value == options["maximum"])
                return best
            grid = sorted(set(grid+additions))
    fixed_grid = min(([v]*layers for v in options["grid"]), key=score)
    best = search([1.]*layers)
    global_scales = best[:]
    for _ in range(options["rounds"]):
        previous = best[:]
        for i in range(layers):
            best = search(best, i)
        if previous == best:
            break
    metric = evaluate(model, validation, device, microbatch, "hard", best)["summary"]
    return {"scales": best, "global_scales": global_scales, "fixed_grid_scales": fixed_grid,
            "validation": metric, "selection_examples": len(ids),
            "initial_boundary_hit": initial_boundary,
            "boundary_at_cap": any(v in (options["minimum"], options["maximum"]) for v in best),
            "any_search_reached_cap": at_cap,
            "evaluations": len(trials)+1, "forward_examples": len(ids)*len(trials)+len(validation),
            "trials": trials}


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(c, case, method, seed, settings, directory, data, descriptor, device, microbatch,
        epochs=None, initial=None, continuation=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    specification = model_spec(c, case, descriptor)
    initialization_seed = stable_seed("initialization", case["id"], seed)
    seed_everything(initialization_seed, device)
    model = SuiteSNN(specification).to(device)
    if initial is not None:
        model.load_state_dict(initial)
    optimizer = (torch.optim.SGD(model.parameters(), lr=settings["lr"], momentum=.9, weight_decay=settings["weight_decay"])
                 if method in ("sgd2x", "sast_sgd", "sam_sgd") else
                 torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]))
    sampler_prefix = "continuation" if continuation else "from_scratch"
    descent = np.random.default_rng(stable_seed(sampler_prefix, "descent", case["id"], seed))
    ascent = np.random.default_rng(stable_seed(sampler_prefix, "ascent", case["id"], seed))
    generator = torch.Generator().manual_seed(stable_seed(sampler_prefix, "perturbations", case["id"], seed))
    base_epochs = c["epochs"][case["dataset"]] if epochs is None else epochs
    total_epochs = base_epochs*METHODS[method][1]
    batch_size = c["batch_size"][case["dataset"]]
    steps = c["steps_per_epoch"] or math.ceil(len(data["train"])/batch_size)
    identity = digest({"spec": specification, "method": method, "seed": seed, "settings": settings,
                       "epochs": total_epochs, "steps": steps, "batch": batch_size,
                       "data": descriptor["parameters"], "continuation": continuation})
    checkpoint = directory/"last.pt"
    history, start, best_loss, best_state, best_epoch = [], 0, float("inf"), None, None
    counters = {"optimizer_updates": 0, "pairs": 0, "micro_backward_calls": 0,
                "forward_backward_examples": 0, "input_vjps": 0, "training_seconds": 0.}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise RuntimeError("Incompatible checkpoint: "+str(checkpoint))
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, history, counters = saved["epoch"], saved["history"], saved["counters"]
        best_loss, best_state, best_epoch = saved["best_loss"], saved["best_model"], saved["best_epoch"]
        descent.bit_generator.state, ascent.bit_generator.state = saved["descent_rng"], saved["ascent_rng"]
        generator.set_state(saved["perturbation_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    for epoch in range(start, total_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before, objective = time.monotonic(), 0.
        for step in range(steps):
            update = epoch*steps+step
            learning_rate = settings["lr"]*.5*(1+math.cos(math.pi*update/(total_epochs*steps)))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            indices = descent.integers(0, len(data["train"]), size=batch_size)
            other = indices if method in ("sam", "sam_sgd", "asam") else ascent.integers(0, len(data["train"]), size=batch_size)
            b = data["train"].batch(indices, device)
            a = data["train"].batch(other, device) if METHODS[method][0] == 2 else b
            measured = train_step(model, optimizer, b, a, method, settings, microbatch, generator)
            for key in ("pairs", "micro_backward_calls", "forward_backward_examples", "input_vjps"):
                counters[key] += measured[key]
            counters["optimizer_updates"] += 1
            objective += measured["loss"]/steps
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        counters["training_seconds"] += time.monotonic()-before
        validation = evaluate(model, data["validation"], device, microbatch,
                              mode="hard" if method == "hard_sg" else "smooth")["summary"]
        if validation["loss"] < best_loss:
            best_loss, best_state, best_epoch = validation["loss"], _cpu_state(model), epoch+1
        history.append({"epoch": epoch+1, "training_objective": objective, "validation": validation,
                        "counters": dict(counters), "microbatch": microbatch})
        atomic_torch_save(checkpoint, {"identity": identity, "epoch": epoch+1, "model": _cpu_state(model),
            "optimizer": optimizer.state_dict(), "best_model": best_state, "best_loss": best_loss,
            "best_epoch": best_epoch, "history": history, "counters": counters,
            "descent_rng": descent.bit_generator.state, "ascent_rng": ascent.bit_generator.state,
            "perturbation_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None})
        print(f"{case['id']} {method} seed={seed}: epoch {epoch+1}/{total_epochs}, validation={validation}", flush=True)
    model.load_state_dict(best_state)
    atomic_torch_save(directory/"best.pt", {"spec": specification, "model": best_state, "epoch": best_epoch})
    calibration = calibrate(model, data["validation"], c, device, microbatch, case["split_seed"])
    write_json(directory/"training.json", {"method": method, "seed": seed, "settings": settings,
        "best_epoch": best_epoch, "epochs": total_epochs, "steps_per_epoch": steps,
        "effective_batch_size": batch_size, "counters": counters, "history": history,
        "calibration": calibration, "checkpoint_rule": "minimum validation CE (hard CE for hard_sg)",
        "sampling": "independent with replacement; same batch reused for standard SAM and ASAM"})
    return model, calibration, counters


def primary_evaluation(model, data, descriptor, calibration, directory, device, microbatch, c, seed):
    test = data["test"]
    results = {name: evaluate(model, test, device, microbatch, mode=mode, scales=scales)
               for name, mode, scales in (("smooth", "smooth", None), ("hard", "hard", None),
                                         ("calibrated", "hard", calibration["scales"]))}
    labels = results["smooth"]["labels"]
    soft = results["smooth"]["logits"].argmax(1)
    record = {name: result["summary"] for name, result in results.items()}
    for name in ("hard", "calibrated"):
        pred = results[name]["logits"].argmax(1)
        damage, rescue = (soft == labels)&(pred != labels), (soft != labels)&(pred == labels)
        record[name].update(damage=int(damage.sum()), rescue=int(rescue.sum()),
            damage_fraction=float(damage.mean()), rescue_fraction=float(rescue.mean()),
            conditional_damage=float(damage.sum()/(soft == labels).sum()) if (soft == labels).any() else None,
            disagreement=float(np.mean(soft != pred)))
        if test.speakers is not None:
            unseen = np.isin(test.speakers, descriptor.get("unseen_test_speakers", []))
            record[name]["unseen_speaker_accuracy"] = float(np.mean(pred[unseen] == labels[unseen])) if unseen.any() else None
            record[name]["unseen_speaker_n"] = int(unseen.sum())
            official_unseen = np.isin(test.speakers, descriptor.get("official_unseen_test_speakers", []))
            record[name]["official_unseen_speaker_accuracy"] = float(np.mean(pred[official_unseen] == labels[official_unseen])) if official_unseen.any() else None
            record[name]["official_unseen_speaker_n"] = int(official_unseen.sum())
    ids = stratified_ids(test.labels, c["analysis"]["examples"], stable_seed("statistics", seed))
    record["state_statistics"] = {}
    for name, mode, scales in (("smooth", "smooth", None), ("hard", "hard", None),
                               ("calibrated", "hard", calibration["scales"])):
        stats = evaluate(model, test, device, microbatch, mode, scales, ids, information=True)["statistics"]
        record["state_statistics"][name] = {key: float(value.mean()) for key, value in stats.items()}
    np.savez_compressed(Path(directory)/"predictions.npz", labels=labels, ids=results["smooth"]["ids"],
        **{name+"_logits": result["logits"] for name, result in results.items()})
    return record


def selections(c, root):
    path = Path(root)/"selection.json"
    if not path.exists():
        raise RuntimeError("Missing frozen tuning selection. Run --stages tune first.")
    value = json.loads(path.read_text())
    if value["protocol_sha256"] != json.loads((Path(root)/"protocol.json").read_text())["sha256"]:
        raise RuntimeError("Tuning selection belongs to a different protocol.")
    return value["cases"]


def freeze_selections(c, root):
    selected = {}
    for case in cases(c):
        selected[case["id"]] = {}
        for method in c["methods"]:
            scores = []
            options = candidates(c, method)
            for i, settings in enumerate(options):
                measurements = []
                for seed in c["pilot_seeds"]:
                    directory = Path(root)/"jobs"/job("pilot", case, method, seed, i)["id"]
                    if not completed(directory):
                        raise RuntimeError("Cannot select from incomplete pilots: "+str(directory))
                    result = json.loads((directory/"result.json").read_text())
                    measurements.append(result["calibration"]["validation"])
                scores.append({"candidate": i, "settings": settings,
                    "mean_accuracy": float(np.mean([r["accuracy"] for r in measurements])),
                    "mean_loss": float(np.mean([r["loss"] for r in measurements])), "pilot": measurements})
            winner = min(scores, key=lambda row: (-row["mean_accuracy"], row["mean_loss"], row["candidate"]))
            selected[case["id"]][method] = {"settings": winner["settings"], "candidate": winner["candidate"], "scores": scores}
    value = {"protocol_sha256": json.loads((Path(root)/"protocol.json").read_text())["sha256"],
             "rule": "mean calibrated validation accuracy, then CE, then candidate index; no test selection",
             "cases": selected}
    destination = Path(root)/"selection.json"
    if destination.exists() and json.loads(destination.read_text()) != value:
        raise RuntimeError("Frozen tuning selection changed; preserving the previous selection.")
    write_json(destination, value)


def execute_job(c, root, task, device, microbatch):
    from src.paper_suite_analysis import run_analysis, run_noise, run_toy
    directory = Path(root)/"jobs"/task["id"]
    identity = digest({"protocol": json.loads((Path(root)/"protocol.json").read_text())["sha256"], "job": task})
    with file_lock(directory/"job.lock", blocking=False):
        if completed(directory, identity):
            return "cached"
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        write_json(directory/"job.json", task)
        write_json(directory/"runtime.json", {"device": str(device), "microbatch": microbatch,
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "pid": os.getpid(), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if task["kind"] == "toy":
            result = run_toy(c, directory)
        else:
            case = task["case"]
            include_test = task["kind"] in ("confirm", "causal", "analysis", "ablate")
            bins = task["variant"]["value"] if task["kind"] == "ablate" and task["variant"]["factor"] == "time_bins" else case["time_bins"]
            data, descriptor = load_data(root, data_key(case["dataset"], bins, case["split_seed"]), include_test)
            if task["kind"] == "pilot":
                settings = candidates(c, task["method"])[task["candidate"]]
            else:
                chosen = selections(c, root)[case["id"]]
                settings = chosen[task["method"]]["settings"] if "method" in task else None
            if task["kind"] in ("pilot", "confirm"):
                model, calibration, costs = fit(c, case, task["method"], task["seed"], settings, directory,
                                               data, descriptor, device, microbatch)
                result = {"job": task, "synthetic": descriptor["synthetic"], "calibration": calibration, "costs": costs}
                if task["kind"] == "confirm":
                    result["test"] = primary_evaluation(model, data, descriptor, calibration, directory,
                                                        device, microbatch, c, task["seed"])
            elif task["kind"] == "causal":
                result = run_causal(c, root, task, data, descriptor, chosen, directory, device, microbatch)
            elif task["kind"] == "ablate":
                result = run_ablation(c, task, data, descriptor, settings, directory, device, microbatch)
            elif task["kind"] == "analysis":
                parent = Path(root)/"jobs"/job("confirm", case, task["method"], task["seed"])["id"]
                if not completed(parent):
                    raise RuntimeError("Missing completed confirmation checkpoint: "+str(parent))
                saved = torch.load(parent/"best.pt", map_location="cpu", weights_only=True)
                model = SuiteSNN(saved["spec"]).to(device)
                model.load_state_dict(saved["model"])
                calibration = json.loads((parent/"training.json").read_text())["calibration"]
                result = run_analysis(c, case, task["method"], task["seed"], model, data, descriptor,
                                      calibration, directory, device, microbatch)
            else:
                raise ValueError(task["kind"])
        result["wall_seconds_this_attempt"] = time.monotonic()-started
        result["peak_cuda_allocated_bytes"] = max(torch.cuda.max_memory_allocated(device),
            result.get("inference", {}).get("pre_timing_peak_allocated_bytes") or 0) if device.type == "cuda" else None
        write_json(directory/"result.json", result)
        mark_completed(directory, identity)
    return "done"


def run_causal(c, root, task, data, descriptor, chosen, directory, device, microbatch):
    from src.paper_suite_analysis import run_noise
    case, seed = task["case"], task["seed"]
    warmup = max(1, round(c["epochs"][case["dataset"]]*c["warmup_fraction"]))
    adaptation = max(1, c["epochs"][case["dataset"]]-warmup)
    common = dict(chosen["erm2x"]["settings"])
    common.update(rho=chosen["sast"]["settings"]["rho"], beta=chosen["margin"]["settings"]["beta"])
    pretrain = directory/"pretrain"
    pretrain_id = digest({"task": task, "settings": common, "epochs": warmup})
    if not completed(pretrain, pretrain_id):
        fit(c, case, "erm", seed, common, pretrain, data, descriptor, device, microbatch, epochs=warmup)
        latest = torch.load(pretrain/"last.pt", map_location="cpu", weights_only=True)
        model = SuiteSNN(model_spec(c, case, descriptor)).to(device)
        model.load_state_dict(latest["model"])
        calibration = calibrate(model, data["validation"], c, device, microbatch, case["split_seed"])
        zero_shot = primary_evaluation(model, data, descriptor, calibration, pretrain, device, microbatch, c, seed)
        write_json(pretrain/"result.json", {"test": zero_shot, "calibration": calibration})
        mark_completed(pretrain, pretrain_id)
    saved = torch.load(pretrain/"last.pt", map_location="cpu", weights_only=True)
    initial = saved["model"]
    initial_sha = file_hash(pretrain/"last.pt")
    def evaluate_arm(method, budget, arm):
        arm_id = digest({"parent": initial_sha, "method": method, "settings": common, "adaptation": budget})
        if not completed(arm, arm_id):
            model, calibration, costs = fit(c, case, method, seed, common, arm, data, descriptor, device,
                microbatch, epochs=budget, initial=initial, continuation=True)
            metrics = primary_evaluation(model, data, descriptor, calibration, arm, device, microbatch, c, seed)
            write_json(arm/"result.json", {"method": method, "test": metrics, "calibration": calibration,
                "costs": costs, "initial_checkpoint_sha256": initial_sha})
            mark_completed(arm, arm_id)
        return json.loads((arm/"result.json").read_text())
    arms = {method: evaluate_arm(method, adaptation, directory/method) for method in CAUSAL_METHODS}
    budget_curve = {}
    if seed in c["seeds"][:c["analysis"]["seeds"]]:
        for fraction in c["adaptation_fractions"]:
            budget = max(1, round(c["epochs"][case["dataset"]]*fraction))
            budget_curve[str(fraction)] = {method: evaluate_arm(method, budget,
                directory/("budget_"+str(fraction).replace(".", "p"))/method)
                for method in ("erm2x", "sast", "hard_sg")}
    noise = None
    if seed in c["seeds"][:c["analysis"]["seeds"]]:
        model = SuiteSNN(model_spec(c, case, descriptor)).to(device)
        model.load_state_dict(initial)
        noise = run_noise(c, case, seed, model, data["train"], common, device, microbatch, directory)
    return {"job": task, "synthetic": descriptor["synthetic"], "warmup_epochs": warmup,
            "continuation_base_epochs": adaptation, "initial_checkpoint_sha256": initial_sha,
            "arms": arms, "noise": noise, "pretrain_costs": saved["counters"],
            "zero_shot": json.loads((pretrain/"result.json").read_text()), "adaptation_curve": budget_curve}


def _initialize_worker(devices, c, root, microbatch, threads):
    device = torch.device(devices.get())
    configure_device(device, threads)
    _WORKER.update(device=device, c=c, root=root, microbatch=microbatch)


def _worker_job(task):
    directory = Path(_WORKER["root"])/"jobs"/task["id"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"worker.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                status = execute_job(_WORKER["c"], _WORKER["root"], task, _WORKER["device"], _WORKER["microbatch"])
                (directory/"failure.json").unlink(missing_ok=True)
                return {"id": task["id"], "status": status}
            except Exception as exc:
                traceback.print_exc()
                write_json(directory/"failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
                if _WORKER["device"].type == "cuda":
                    torch.cuda.empty_cache()
                return {"id": task["id"], "status": "failed", "error": str(exc)}


def _worker_benchmark(case):
    c, device, microbatch = _WORKER["c"], _WORKER["device"], _WORKER["microbatch"]
    data, descriptor = load_data(_WORKER["root"], data_key(case["dataset"], case["time_bins"], case["split_seed"]))
    seed_everything(stable_seed("timing", case["id"]), device)
    model = SuiteSNN(model_spec(c, case, descriptor)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    generator = torch.Generator().manual_seed(123)
    batch_size = c["batch_size"][case["dataset"]]
    b = data["train"].batch(np.arange(batch_size) % len(data["train"]), device)
    a = data["train"].batch((np.arange(batch_size)+batch_size) % len(data["train"]), device)
    settings = {"rho": .05, "margin_target": c["margin_target"]}
    for _ in range(2):
        train_step(model, optimizer, b, a, "sast", settings, microbatch, generator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    before = time.perf_counter()
    for _ in range(3):
        train_step(model, optimizer, b, a, "sast", settings, microbatch, generator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pair_time = (time.perf_counter()-before)/(6*batch_size)
    x = b[0][:microbatch]
    with torch.no_grad():
        for _ in range(2):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before = time.perf_counter()
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    result = {"case": case["id"], "device": str(device), "microbatch": microbatch,
        "pair_example_seconds": pair_time, "forward_example_seconds": (time.perf_counter()-before)/(5*len(x)),
        "counts": {k: descriptor["splits"][k]["count"] for k in ("train", "validation", "test")}}
    write_json(Path(_WORKER["root"])/"benchmarks"/(case["id"]+".json"), result)
    return result


def estimate_seconds(c, timings):
    total = 0.
    for case in cases(c):
        t = timings[case["id"]]
        counts = t["counts"]
        batch = c["batch_size"][case["dataset"]]
        steps = c["steps_per_epoch"] or math.ceil(counts["train"]/batch)
        epochs = c["epochs"][case["dataset"]]
        layers = len(c["widths"][case["architecture"]])
        cal = c["calibration"]
        grid_upper = len(cal["grid"])+math.ceil(math.log2(cal["maximum"]/cal["minimum"]))+2
        calibration_forwards = grid_upper*(1+layers*cal["rounds"])*min(counts["validation"], cal["examples"] or counts["validation"])+counts["validation"]
        def fit_cost(method, base_epochs, test=True):
            passes, multiplier = METHODS[method]
            e = base_epochs*multiplier
            extra = 2. if method == "input_gradient" else 1.
            return (steps*batch*e*passes*t["pair_example_seconds"]*extra+
                (e*counts["validation"]+calibration_forwards+
                 (3*counts["test"]+3*c["analysis"]["examples"] if test else 0))*t["forward_example_seconds"])
        for method in c["methods"]:
            total += fit_cost(method, epochs, False)*len(c["pilot_seeds"])*len(candidates(c, method))
            total += fit_cost(method, epochs)*len(c["seeds"])
            # Includes small probes/plots and conservative CPU time per analysis job.
            total += min(len(c["seeds"]), c["analysis"]["seeds"])*(20000*t["forward_example_seconds"]+
                3000*t["pair_example_seconds"]+120)
        if case["dataset"] in c["causal_datasets"] and case["architecture"] == "cnn2":
            warm = max(1, round(epochs*c["warmup_fraction"]))
            adapt = max(1, epochs-warm)
            total += len(c["seeds"])*(fit_cost("erm", warm)+sum(fit_cost(m, adapt) for m in CAUSAL_METHODS))
            for fraction in c["adaptation_fractions"]:
                total += min(len(c["seeds"]), c["analysis"]["seeds"])*sum(fit_cost(m, max(1, round(epochs*fraction)))
                    for m in ("erm2x", "sast", "hard_sg"))
            total += min(len(c["seeds"]), c["analysis"]["seeds"])*15000*t["pair_example_seconds"]
    return total


def budget_plan(c, pool, devices, root, deadline):
    timings = {}
    futures = [pool.submit(_worker_benchmark, case) for case in cases(c)]
    for future in futures:
        result = future.result()
        timings[result["case"]] = result
        print(f"Timing {result['case']}: {result['pair_example_seconds']:.6f} s/forward-backward example", flush=True)
    minimum = min(c["budget"]["minimum_seeds"], len(c["seeds"]))
    available = max(0., deadline-time.time())*(1-c["budget"]["reserve_fraction"])
    fractions = sorted(set([1., .8, .6, .4, c["budget"]["minimum_epoch_fraction"]]), reverse=True)
    for fraction in fractions:
        if fraction < c["budget"]["minimum_epoch_fraction"]:
            continue
        for number in range(len(c["seeds"]), minimum-1, -1):
            selected = copy.deepcopy(c)
            selected["seeds"] = c["seeds"][:number]
            selected["epochs"] = {k: max(1, round(v*fraction)) for k, v in c["epochs"].items()}
            seconds = estimate_seconds(selected, timings)
            hours = seconds*c["budget"]["safety_factor"]/len(devices)/3600
            if hours*3600 <= available:
                write_json(Path(root)/"budget_plan.json", {"requested_epochs": c["epochs"], "epochs": selected["epochs"],
                    "requested_seeds": c["seeds"], "seeds": selected["seeds"], "estimated_gpu_hours": seconds/3600,
                    "estimated_wall_hours_with_safety": hours, "available_wall_hours_after_reserve": available/3600,
                    "timings": timings, "rule": "retain longest training first, then most seeds; timing only, no accuracy-based decisions"})
                print(f"Frozen time plan: {number} seeds, epochs={selected['epochs']}, estimated {hours:.1f} hours plus reserve.", flush=True)
                return selected
    raise RuntimeError("Even the minimum protocol exceeds the measured time budget. No research training started. Reduce widths/time_bins in a new configuration/output directory.")


def run_jobs(pool, tasks, root):
    pending = [task for task in tasks if not completed(Path(root)/"jobs"/task["id"])]
    print(f"{len(tasks)-len(pending)}/{len(tasks)} cached; scheduling {len(pending)} jobs.", flush=True)
    active = {pool.submit(_worker_job, task): task for task in pending}
    failures, done = [], 0
    while active:
        finished, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
        if not finished:
            print(f"Running: {done}/{len(pending)} completed; logs in {Path(root)/'jobs'}", flush=True)
        for future in finished:
            task = active.pop(future)
            result = future.result()
            done += 1
            print(f"[{done}/{len(pending)}] {result['status']}: {task['id']}"+
                  (" — "+result["error"] if result.get("error") else ""), flush=True)
            if result["status"] == "failed":
                failures.append(result)
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; see worker.log/failure.json. Rerun the same command to resume.")


def prepare(c, root, ablations=True):
    for dataset in c["datasets"]:
        for split_seed in c["split_seeds"]:
            prepare_case_data(c, root, dataset, c["time_bins"][dataset], split_seed)
            if ablations and dataset in c["ablation"]["datasets"]:
                for bins in c["ablation"]["time_bins"]:
                    if bins != c["time_bins"][dataset]:
                        prepare_case_data(c, root, dataset, bins, split_seed)


def run(c, root, devices, microbatch=4, threads=2, stages=None, hardware=None):
    stages = STAGES if stages is None else stages
    unknown = set(stages)-set(STAGES)
    if unknown:
        raise ValueError("Unknown stages: "+", ".join(sorted(unknown)))
    root = Path(root).resolve()
    with file_lock(root/"scheduler.lock", blocking=False):
        write_json(root/"executions"/(str(time.time_ns())+".json"), {"devices": devices,
            "microbatch": microbatch, "threads": threads, "hardware": hardware,
            "torch": torch.__version__, "cuda": torch.version.cuda})
        requested = {"configuration": c, "source_sha256": source_identity()}
        request_path = root/"requested_protocol.json"
        if request_path.exists() and json.loads(request_path.read_text()) != requested:
            raise RuntimeError("Requested configuration/source changed. Use a new output directory.")
        write_json(request_path, requested)
        clock_path = root/"deadline.json"
        if not clock_path.exists():
            write_json(clock_path, {"started_unix": time.time(), "deadline_unix": time.time()+c["budget"]["hours"]*3600})
        deadline = json.loads(clock_path.read_text())["deadline_unix"]
        protocol_path = root/"protocol.json"
        if protocol_path.exists():
            c = json.loads(protocol_path.read_text())["configuration"]
        def timed_out(signum, frame):
            raise TimeoutError("Time budget reached. Workers stopped; completed epochs/results are preserved. Use report/status to inspect partial results.")
        previous_handler = signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.time()))
        context, pool = mp.get_context("spawn"), None
        def create_pool(active):
            queue = context.Queue()
            for device in devices:
                queue.put(device)
            return ProcessPoolExecutor(max_workers=len(devices), mp_context=context,
                initializer=_initialize_worker, initargs=(queue, active, str(root), microbatch, threads))
        try:
            needs_data = bool(set(stages) & {"tune", "confirm", "causal", "analysis", "ablate"})
            if needs_data:
                prepare(c, root, ablations="ablate" in stages)
            if not protocol_path.exists() and not c["synthetic"] and needs_data:
                pool = create_pool(c)
                c = budget_plan(c, pool, devices, root, deadline)
                pool.shutdown(wait=True)
                pool = None
            freeze_protocol(c, root)
            pool = create_pool(c)
            for stage in STAGES:
                if stage not in stages:
                    continue
                print(f"Stage {stage} ({'SYNTHETIC CHECK ONLY' if c['synthetic'] else 'research protocol'})", flush=True)
                if stage in ("confirm", "causal", "analysis", "ablate"):
                    selections(c, root)
                if stage == "report":
                    report(c, root)
                else:
                    run_jobs(pool, stage_jobs(c, stage), root)
                    if stage == "tune":
                        freeze_selections(c, root)
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if pool is not None:
                for process in (getattr(pool, "_processes", None) or {}).values():
                    process.terminate()
                pool.shutdown(wait=False, cancel_futures=True)
            if (root/"protocol.json").exists():
                try:
                    report(c, root)
                except Exception as error:
                    print(f"Partial report failed: {error}. Raw results remain in {root}.", file=sys.stderr)
            raise
        else:
            pool.shutdown(wait=True)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def plan(c):
    counts = {stage: len(stage_jobs(c, stage)) for stage in STAGES if stage != "report"}
    return {"synthetic": c["synthetic"], "cases": cases(c), "jobs": counts,
            "causal_fits_per_job": 1+len(CAUSAL_METHODS), "confirmation_seeds": len(c["seeds"]),
            "pilot_seeds": len(c["pilot_seeds"]), "candidates_per_method": c["tuning"]["candidates"],
            "time_budget_hours": c["budget"]["hours"],
            "note": "Maximum requested matrix, before hardware timing reduces epochs/seeds. One worker per GPU, no SLI/pooled VRAM. Long training sweeps are disabled in the 48-hour preset."}


def status(c, root):
    result = {}
    for stage in STAGES:
        if stage == "report":
            continue
        jobs = stage_jobs(c, stage)
        result[stage] = {"total": len(jobs), "completed": sum(completed(Path(root)/"jobs"/j["id"]) for j in jobs),
                         "failed": sum((Path(root)/"jobs"/j["id"]/"failure.json").exists() for j in jobs)}
    return result


def paired_interval(values, confidence=.95):
    from scipy.stats import t
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "low": None, "high": None}
    mean = float(values.mean())
    half = float(t.ppf((1+confidence)/2, len(values)-1)*values.std(ddof=1)/math.sqrt(len(values))) if len(values) > 1 else None
    return {"n": len(values), "mean": mean, "low": mean-half if half is not None else None,
            "high": mean+half if half is not None else None}


def report(c, root):
    from src.paper_suite_analysis import write_report
    return write_report(c, Path(root))
