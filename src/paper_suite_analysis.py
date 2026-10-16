"""Measured diagnostics; no landscape plot or floating-point bound is a global certificate."""
import copy
import csv
import itertools
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.paper_suite_data import stable_seed, stratified_ids, write_json
from src.paper_suite_models import (SuiteSNN, assign_vector, backward_mean,
    gradient_vector, parameter_vector, smooth, train_step, transfer_error_bound)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_toy(c, directory):
    rows = []
    for slope, leak, bins in itertools.product(c["toy"]["slopes"], c["toy"]["leaks"], c["toy"]["time_bins"]):
        x = torch.tensor([0., 1.], dtype=torch.float64)
        y = x.clone()
        a, b = .4*(1-leak), .2*(1-leak)
        def feature(parameters, hard=False):
            u = spike = None
            spikes = []
            for _ in range(bins):
                drive = parameters[0]*x+parameters[1]
                u = drive if u is None else leak*u+drive-spike
                spike = (u >= 1).double() if hard else smooth(u-1, slope)
                spikes.append(spike)
            return torch.stack(spikes).mean(0)
        initial = torch.tensor([a, b, 1000., 0.], dtype=torch.float64)
        initial[3] = -initial[2]*feature(initial).mean()
        def objective(p):
            return F.binary_cross_entropy_with_logits(p[2]*feature(p)+p[3], y)
        p = initial.clone().requires_grad_(True)
        loss = objective(p)
        gradient = torch.autograd.grad(loss, p)[0]
        hessian = torch.autograd.functional.hessian(objective, initial)
        soft_logits = initial[2]*feature(initial)+initial[3]
        hard_logits = initial[2]*feature(initial, True)+initial[3]
        for radius in c["toy"]["radii"]:
            lower, upper = [], []
            for input_value in (0., 1.):
                drive_lo = (a-radius)*input_value+b-radius
                drive_hi = (a+radius)*input_value+b+radius
                lo = hi = None
                spike_lows, spike_highs = [], []
                for _ in range(bins):
                    if lo is None:
                        lo, hi = drive_lo, drive_hi
                    else:
                        points = [lo, hi]
                        if leak > 0 and slope/math.pi > leak:
                            critical = math.sqrt(slope/(math.pi*leak)-1)/slope
                            points += [v for v in (1-critical, 1+critical) if lo <= v <= hi]
                        mapped = [leak*v-(.5+math.atan(slope*(v-1))/math.pi) for v in points]
                        lo, hi = min(mapped)+drive_lo, max(mapped)+drive_hi
                    spike_lows.append(.5+math.atan(slope*(lo-1))/math.pi)
                    spike_highs.append(.5+math.atan(slope*(hi-1))/math.pi)
                h_lo, h_hi = np.mean(spike_lows), np.mean(spike_highs)
                endpoints = [(initial[2].item()+dc)*h for dc in (-radius, radius) for h in (h_lo, h_hi)]
                lower.append(min(endpoints)+initial[3].item()-radius)
                upper.append(max(endpoints)+initial[3].item()+radius)
            numerical_upper = float((np.logaddexp(0, upper[0])+np.logaddexp(0, -lower[1]))/2)
            max_drive = a+b+2*radius
            constant_hard = max_drive*(1-leak**bins)/(1-leak) < 1 and initial[3].item()+radius < 0
            rows.append({"slope": slope, "leak": leak, "time_bins": bins, "rho": radius,
                "smooth_loss": float(loss.detach()), "gradient_norm": float(gradient.norm()),
                "hessian_operator_norm": float(torch.linalg.eigvalsh(hessian).abs().max()),
                "smooth_accuracy": float(((soft_logits >= 0) == y.bool()).double().mean()),
                "hard_accuracy": float(((hard_logits >= 0) == y.bool()).double().mean()),
                "box_loss_upper_numerical": numerical_upper,
                "hard_constant_throughout_box": constant_hard,
                "parameters": ";".join(format(v, ".17g") for v in initial.tolist())})
    write_csv(directory/"toy.csv", rows)
    return {"experiment": "two-input recurrent LIF counterexample", "rows": rows,
            "interpretation": "The componentwise box encloses the Euclidean rho-ball. Bounds are float64 numerical evaluations, without directed rounding. Some sweep settings need not be counterexamples."}


def cosine(a, b):
    denominator = float(a.norm()*b.norm())
    return float(torch.dot(a, b)/denominator) if denominator > 0 else None


def run_noise(c, case, seed, model, train, settings, device, microbatch, directory):
    options, rows = c["analysis"], []
    conditions = [(method, settings["rho"]) for method in ("erm", "avg2", "asam", "asam_independent", "random", "rho0")]
    conditions += [(method, radius) for method in ("sam", "sast") for radius in [0.]+options["radii"]]
    original = parameter_vector(model)
    try:
        for factor in options["noise_batch_factors"]:
            size = max(1, round(c["batch_size"][case["dataset"]]*factor))
            for method, radius in conditions:
                rng = np.random.default_rng(stable_seed("noise_batches", case["id"], seed, factor))
                generator = torch.Generator().manual_seed(stable_seed("noise_random", case["id"], seed, factor))
                optimizer = torch.optim.SGD(model.parameters(), lr=0.)
                mean, m2, cosines, independent_cosines, norms = None, 0., [], [], []
                for index in range(options["noise_batches"]):
                    a = train.batch(rng.integers(0, len(train), size=size), device)
                    b = train.batch(rng.integers(0, len(train), size=size), device)
                    model.zero_grad(set_to_none=True)
                    backward_mean(model, b, microbatch)
                    reference = gradient_vector(model).clone()
                    model.zero_grad(set_to_none=True)
                    backward_mean(model, a, microbatch)
                    independent = gradient_vector(model)
                    value = cosine(reference, independent)
                    if value is not None:
                        independent_cosines.append(value)
                    train_step(model, optimizer, b, a, method, dict(settings, rho=radius), microbatch, generator)
                    g = gradient_vector(model)
                    value = cosine(g, reference)
                    if value is not None:
                        cosines.append(value)
                    norms.append(float(g.norm()))
                    g = g.detach().cpu().double()
                    if mean is None:
                        mean = torch.zeros_like(g)
                    delta = g-mean
                    mean += delta/(index+1)
                    m2 += float(torch.dot(delta, g-mean))
                trace = m2/(options["noise_batches"]-1) if options["noise_batches"] > 1 else None
                mean_norm = float(mean.norm())
                rows.append({"method": method, "rho": radius, "batch_size": size,
                    "repetitions": options["noise_batches"], "gradient_covariance_trace": trace,
                    "mean_gradient_norm": mean_norm, "average_gradient_norm": float(np.mean(norms)),
                    "relative_noise": trace/mean_norm**2 if mean_norm and trace is not None else None,
                    "cosine_to_same_descent_gradient": float(np.mean(cosines)) if cosines else None,
                    "independent_batch_gradient_cosine": float(np.mean(independent_cosines)) if independent_cosines else None})
    finally:
        assign_vector(model, original)
    write_csv(directory/"gradient_noise.csv", rows)
    return {"rows": rows, "checkpoint": "common pretraining endpoint", "learning_rate": 0.,
            "note": "Welford covariance trace across independent batch draws; gradients are compared at identical weights. Diagnostic passes are extra to training budgets."}


def linear_cka(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x, y = x-x.mean(0), y-y.mean(0)
    k, l = x@x.T, y@y.T
    denominator = np.linalg.norm(k)*np.linalg.norm(l)
    return float(np.sum(k*l)/denominator) if denominator > 1e-30 else None


def ridge_probe(train_x, train_y, validation_x, validation_y, test_features, strengths, classes):
    from scipy.linalg import cho_factor, cho_solve
    from src.paper_suite import classification
    x = np.asarray(train_x, dtype=np.float64)
    center, scale = x.mean(0), x.std(0)
    scale = np.maximum(scale, 1e-6)*math.sqrt(x.shape[1])
    x = (x-center)/scale
    y = np.eye(classes)[train_y]
    intercept = y.mean(0)
    y -= intercept
    primal = x.shape[1] <= x.shape[0]
    gram = x.T@x if primal else x@x.T
    right = x.T@y if primal else y
    validation = (np.asarray(validation_x, dtype=np.float64)-center)/scale
    records, heads = [], []
    for strength in strengths:
        matrix = gram.copy()
        matrix.flat[::len(matrix)+1] += strength*len(x)
        jitter = 0.
        for attempt in range(6):
            try:
                factor = cho_factor(matrix, lower=True, check_finite=True)
                solved = cho_solve(factor, right)
                break
            except np.linalg.LinAlgError:
                addition = max(float(np.trace(gram)/len(gram)), 1.)*10**(-10+attempt)
                matrix.flat[::len(matrix)+1] += addition
                jitter += addition
        else:
            raise FloatingPointError("Ridge Cholesky failed despite positive regularization.")
        weights = solved if primal else x.T@solved
        metric = classification(validation@weights+intercept, validation_y)
        heads.append(weights)
        records.append({"lambda": strength, "jitter": jitter, "validation": metric})
    index = min(range(len(records)), key=lambda i: (-records[i]["validation"]["accuracy"], records[i]["validation"]["loss"], i))
    weights = heads[index]
    predictions = {name: (np.asarray(values, dtype=np.float64)-center)/scale@weights+intercept
                   for name, values in test_features.items()}
    return {"selected_lambda": strengths[index], "candidates": records,
            "form": "primal" if primal else "dual", "head_norm": float(np.linalg.norm(weights))}, predictions, {
                "center": center, "scale": scale, "weights": weights, "intercept": intercept}


@torch.no_grad()
def conventional_bound(model, x, scales):
    soft_logits = model(x)
    hard_logits, info = model(x, mode="hard", scales=scales, trace=True)
    incoming = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)
    for block, saved in zip(model.blocks, info["traces"]):
        gain = block.weight.abs().flatten(1).sum(1).max()
        errors, previous_u, previous_z = [], None, None
        for t in range(x.shape[1]):
            error = gain*incoming[:, t]
            if previous_u is not None:
                hard_max = saved["s"][:, t-1].flatten(1).amax(1)
                error += model.leak*previous_u+model.threshold*previous_z+(saved["threshold"]-model.threshold).abs()*hard_max
            residual = (smooth(saved["u"][:, t]-model.threshold, model.slope)-saved["s"][:, t]).abs().flatten(1).amax(1)
            spike_error = torch.clamp(model.slope/math.pi*error+residual, max=1.)
            errors.append(spike_error)
            previous_u, previous_z = error, spike_error
        incoming = torch.stack(errors, 1)
    feature_error = incoming.mean(1)
    maximum = model.readout.weight.abs().sum(1).max()*feature_error
    winner = soft_logits.argmax(1)
    rows = model.readout.weight[winner, None, :]-model.readout.weight[None, :, :]
    pair_bounds = rows.abs().sum(2)*feature_error[:, None]
    margins = soft_logits.gather(1, winner[:, None])-soft_logits
    mask = F.one_hot(winner, soft_logits.shape[1]).bool()
    return {"agreement_condition": ((margins > pair_bounds)|mask).all(1), "logit_bound": maximum,
            "observed_logit_error": (soft_logits-hard_logits).abs().amax(1)}


def bound_diagnostics(c, model, test, ids, scales, device, microbatch, directory):
    model64 = copy.deepcopy(model).double()
    records, arrays = {}, {}
    for name, threshold_scales in (("raw", None), ("calibrated", scales)):
        collected = {}
        for start in range(0, len(ids), min(microbatch, 2)):
            x, _ = test.batch(ids[start:start+min(microbatch, 2)], device)
            x = x.double()
            refined = transfer_error_bound(model64, x, threshold_scales)
            loose = conventional_bound(model64, x, threshold_scales)
            quantities = {"condition": refined["agreement_condition"], "agreement": refined["observed_agreement"],
                "bound": refined["logit_bound"], "error": refined["observed_logit_error"],
                "loose_bound": loose["logit_bound"], "loose_condition": loose["agreement_condition"],
                "feature_violation": (refined["observed_feature_error"]-refined["feature_error_bound"]).amax(1)}
            for key, value in quantities.items():
                collected.setdefault(key, []).append(value.cpu().numpy())
        a = {key: np.concatenate(value) for key, value in collected.items()}
        arrays.update({name+"_"+key: value for key, value in a.items()})
        positive = a["error"] > 1e-12
        ratios = a["bound"][positive]/a["error"][positive]
        tolerance = 1e-9+1e-7*a["bound"]
        records[name] = {"n": len(ids), "agreement_coverage": float(a["condition"].mean()),
            "loose_agreement_coverage": float(a["loose_condition"].mean()),
            "observed_agreement": float(a["agreement"].mean()),
            "false_agreement_conditions": int(np.sum(a["condition"]&~a["agreement"])),
            "logit_bound_violations": int(np.sum(a["error"] > a["bound"]+tolerance)),
            "feature_bound_violations": int(np.sum(a["feature_violation"] > 1e-9)),
            "max_logit_excess": float(np.max(a["error"]-a["bound"])),
            "mean_bound": float(a["bound"].mean()), "mean_loose_bound": float(a["loose_bound"].mean()),
            "mean_observed_error": float(a["error"].mean()),
            "median_bound_to_error": float(np.median(ratios)) if len(ratios) else None}
    np.savez_compressed(directory/"transfer_bounds.npz", ids=test.ids[ids], **arrays)
    return {"profiles": records, "arithmetic": "float64 without directed rounding",
            "scope": "a posteriori prediction-agreement analysis using hard trajectories; numerical coverage is not a new theorem or formal certificate"}


def normalized_direction(model, generator):
    values = []
    for name, p in model.named_parameters():
        value = torch.randn(p.shape, generator=generator, dtype=p.dtype).to(p.device)
        if name.endswith("weight") and p.ndim >= 2:
            flat = value.flatten(1)
            sizes = p.detach().flatten(1).norm(dim=1)/(flat.norm(dim=1)+1e-12)
            value *= sizes.reshape((-1,)+(1,)*(p.ndim-1))
        else:
            value *= p.detach().norm()/(value.norm()+1e-12)
        values.append(value.flatten())
    vector = torch.cat(values)
    return vector/(vector.norm()+1e-12)


def hessian_ritz(model, split, ids, device, microbatch, iterations, generator):
    parameters = tuple(model.parameters())
    dimension = sum(p.numel() for p in parameters)
    q = torch.randn(dimension, generator=generator).to(device)
    q /= q.norm()
    previous, beta, diagonals, off_diagonals = torch.zeros_like(q), 0., [], []
    for iteration in range(iterations):
        product = torch.zeros_like(q)
        for start in range(0, len(ids), microbatch):
            x, y = split.batch(ids[start:start+microbatch], device)
            loss = F.cross_entropy(model(x), y)
            g = torch.autograd.grad(loss, parameters, create_graph=True)
            vector = torch.cat([value.flatten() for value in g])
            second = torch.autograd.grad(torch.dot(vector, q), parameters)
            product += torch.cat([value.detach().flatten() for value in second])*(len(y)/len(ids))
        if not torch.isfinite(product).all():
            raise FloatingPointError("Non-finite Hessian-vector product.")
        alpha = float(torch.dot(q, product))
        residual = product-alpha*q-beta*previous
        next_beta = float(residual.norm())
        diagonals.append(alpha)
        if iteration:
            off_diagonals.append(beta)
        if next_beta < 1e-10:
            break
        previous, q, beta = q, residual/next_beta, next_beta
    matrix = np.diag(diagonals)
    if off_diagonals:
        matrix += np.diag(off_diagonals, 1)+np.diag(off_diagonals, -1)
    values = np.linalg.eigvalsh(matrix)
    return {"largest_ritz_value": float(values.max()), "smallest_ritz_value": float(values.min()),
            "operator_norm_lower_estimate": float(np.abs(values).max()), "iterations": len(diagonals),
            "note": "finite-step Lanczos estimate on a fixed validation subset; not the exact spectrum"}


def geometry(c, case, seed, model, validation, scales, device, microbatch, directory, label=""):
    from src.paper_suite import evaluate
    options = c["analysis"]
    ids = stratified_ids(validation.labels, options["geometry_examples"], stable_seed("geometry_ids", case["id"]))
    generator = torch.Generator().manual_seed(stable_seed("geometry_planes", case["id"], seed))
    base = parameter_vector(model)
    curves, arrays = [], {"ids": validation.ids[ids]}
    def metric():
        return {"smooth_loss": evaluate(model, validation, device, microbatch, ids=ids)["summary"]["loss"],
            "hard_accuracy": evaluate(model, validation, device, microbatch, "hard", ids=ids)["summary"]["accuracy"],
            "calibrated_accuracy": evaluate(model, validation, device, microbatch, "hard", scales, ids)["summary"]["accuracy"]}
    def gradient():
        model.zero_grad(set_to_none=True)
        for start in range(0, len(ids), microbatch):
            x, y = validation.batch(ids[start:start+microbatch], device)
            (F.cross_entropy(model(x), y)*(len(y)/len(ids))).backward()
        return gradient_vector(model)
    try:
        center = metric()
        ritz = hessian_ritz(model, validation, ids, device, microbatch, options["hessian_steps"], generator)
        for radius in options["radii"]:
            best = dict(center, norm=0., start=-1)
            for start in range(options["pgd_starts"]):
                perturbation = torch.randn(base.shape, generator=generator).to(device)
                perturbation *= radius/(perturbation.norm()+1e-12)
                for _ in range(options["pgd_steps"]):
                    assign_vector(model, base+perturbation)
                    g = gradient()
                    perturbation += (2*radius/options["pgd_steps"])*g/(g.norm()+1e-12)
                    perturbation *= min(1., radius/(float(perturbation.norm())+1e-12))
                assign_vector(model, base+perturbation)
                measured = metric()
                if measured["smooth_loss"] > best["smooth_loss"]:
                    best = dict(measured, norm=float(perturbation.norm()), start=start)
            curves.append({"rho": radius, "sampled_loss_increase": best["smooth_loss"]-center["smooth_loss"], **best})
        for plane in range(options["planes"]):
            assign_vector(model, base)
            d1, d2 = normalized_direction(model, generator), normalized_direction(model, generator)
            coordinates = np.linspace(-max(options["radii"]), max(options["radii"]), options["grid_size"])
            grids = {key: np.empty((len(coordinates), len(coordinates))) for key in center}
            for i, a in enumerate(coordinates):
                for j, b in enumerate(coordinates):
                    assign_vector(model, base+float(a)*d1+float(b)*d2)
                    for key, value in metric().items():
                        grids[key][i, j] = value
            arrays.update({f"plane{plane}_"+key: value for key, value in grids.items()})
            arrays[f"plane{plane}_coordinates"] = coordinates
            arrays[f"plane{plane}_direction1"] = d1.cpu().numpy()
            arrays[f"plane{plane}_direction2"] = d2.cpu().numpy()
        np.savez_compressed(directory/"landscapes.npz", **arrays)
        plot_landscape(arrays, directory/"landscape", ("SYNTHETIC CHECK: " if c["synthetic"] else "")+case["id"]+" / "+label)
    finally:
        assign_vector(model, base)
        model.zero_grad(set_to_none=True)
    return {"center": center, "neighborhood_searches": curves, "hessian": ritz,
            "plane_normalization": "per-output-filter weight norms, then unit global L2; independent saved directions",
            "scope": "fixed validation subset; grid corners can exceed the single-axis radius; numerical searches are lower estimates of maximum loss"}


def plot_landscape(arrays, destination, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams["pdf.fonttype"] = 42
    coordinates = arrays["plane0_coordinates"]
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    fig = plt.figure(figsize=(7, 3.3), constrained_layout=True)
    for i, (key, label) in enumerate((("smooth_loss", "Smooth CE"), ("calibrated_accuracy", "Calibrated hard accuracy"))):
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        ax.plot_surface(x, y, arrays["plane0_"+key], cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Direction 1", fontsize=8, labelpad=0)
        ax.set_ylabel("Direction 2", fontsize=8, labelpad=0)
        ax.set_xticks([coordinates[0], 0, coordinates[-1]])
        ax.set_yticks([coordinates[0], 0, coordinates[-1]])
        ax.zaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=7, pad=1)
        if i:
            ax.set_zlim(0, 1)
    fig.suptitle(title, fontsize=9)
    fig.savefig(str(destination)+".pdf", bbox_inches="tight")
    fig.savefig(str(destination)+".png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def confidence_metrics(logits, labels, temperature=1.):
    from src.paper_suite import classification
    z = np.asarray(logits, dtype=np.float64)/temperature
    p = np.exp(z-z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    confidence, correct = p.max(1), p.argmax(1) == labels
    ece = 0.
    for index in range(10):
        mask = (confidence >= index/10)&((confidence < (index+1)/10) if index < 9 else (confidence <= 1))
        if mask.any():
            ece += mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    return {**classification(z, labels), "ece_10_fixed_bins": float(ece),
            "brier": float(np.square(p-np.eye(p.shape[1])[labels]).sum(1).mean()), "temperature": temperature}


def input_attack(model, split, ids, mode, scales, epsilon, options, device, microbatch, seed):
    total_correct, total_loss = 0, 0.
    generator = torch.Generator().manual_seed(seed)
    target_mode = "smooth" if mode == "smooth" else "ste"
    evaluation_mode = "smooth" if mode == "smooth" else "hard"
    for start in range(0, len(ids), microbatch):
        x, y = split.batch(ids[start:start+microbatch], device)
        lo, hi = (x-epsilon).clamp(0, 1), (x+epsilon).clamp(0, 1)
        with torch.no_grad():
            best_logits = model(x, mode=evaluation_mode, scales=scales)
            best_loss = F.cross_entropy(best_logits, y, reduction="none")
        for _ in range(options["attack_starts"]):
            delta = (2*torch.rand(x.shape, generator=generator)-1).to(device)*epsilon
            adversarial = torch.maximum(lo, torch.minimum(hi, x+delta))
            for _ in range(options["attack_steps"]):
                adversarial = adversarial.detach().requires_grad_(True)
                objective = F.cross_entropy(model(adversarial, mode=target_mode, scales=scales), y)
                g = torch.autograd.grad(objective, adversarial)[0]
                with torch.no_grad():
                    adversarial = torch.maximum(lo, torch.minimum(hi, adversarial+2*epsilon/options["attack_steps"]*g.sign()))
                    logits = model(adversarial, mode=evaluation_mode, scales=scales)
                    losses = F.cross_entropy(logits, y, reduction="none")
                    take = losses > best_loss
                    best_loss = torch.where(take, losses, best_loss)
                    best_logits = torch.where(take[:, None], logits, best_logits)
        total_correct += int((best_logits.argmax(1) == y).sum())
        total_loss += float(best_loss.sum())
    return {"accuracy": total_correct/len(ids), "loss": total_loss/len(ids), "n": len(ids),
            "epsilon_linf": epsilon, "steps": options["attack_steps"], "starts": options["attack_starts"],
            "gradient": "exact smooth" if mode == "smooth" else "hard forward, arctan surrogate backward"}


def corrupt(x, kind, level, generator):
    if kind == "cell_dropout":
        return x*(torch.rand(x.shape, generator=generator).to(x.device) >= level)
    if kind == "bin_dropout":
        shape = x.shape[:2]+(1,)*(x.ndim-2)
        return x*(torch.rand(shape, generator=generator).to(x.device) >= level)
    if kind == "background_occupancy":
        return torch.maximum(x, (torch.rand(x.shape, generator=generator).to(x.device) < level*.1).to(x.dtype))
    if kind == "time_reverse":
        return x.flip(1)
    if kind == "time_shift":
        return x.roll(max(1, x.shape[1]//4), 1)
    if kind == "time_shuffle":
        indices = torch.rand(x.shape[:2], generator=generator).argsort(1).to(x.device)
        return x[torch.arange(len(x), device=x.device)[:, None], indices]
    if kind == "spatial_shift":
        amount = int(level)
        out = x.roll((amount, amount), (-2, -1))
        out[..., :amount, :] = 0
        out[..., :, :amount] = 0
        return out
    raise ValueError(kind)


@torch.no_grad()
def corruption_tests(c, case, seed, model, test, ids, scales, device, microbatch):
    from src.paper_suite import classification
    rows = []
    conditions = [(kind, level) for kind in ("cell_dropout", "bin_dropout", "background_occupancy")
                  for level in c["analysis"]["corruption_levels"]]
    conditions += [(kind, 0.) for kind in ("time_reverse", "time_shift", "time_shuffle")]
    if model.vision:
        conditions += [("spatial_shift", 2), ("spatial_shift", 4)]
    for kind, level in conditions:
        for name, mode, thresholds in (("smooth", "smooth", None), ("hard", "hard", None), ("calibrated", "hard", scales)):
            generator = torch.Generator().manual_seed(stable_seed("corruption", case["id"], seed, kind, level))
            predictions, labels = [], []
            for start in range(0, len(ids), microbatch):
                x, y = test.batch(ids[start:start+microbatch], device)
                changed = corrupt(x, kind, level, generator)
                predictions.append(model(changed, mode=mode, scales=thresholds).cpu().numpy())
                labels.append(y.cpu().numpy())
            rows.append({"corruption": kind, "level": level, "mode": name,
                         **classification(np.concatenate(predictions), np.concatenate(labels))})
    return rows


@torch.no_grad()
def inference_timing(c, model, test, device, microbatch, scales):
    x, _ = test.batch(np.arange(min(microbatch, len(test))), device)
    previous_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    rows = {}
    for name, mode, thresholds in (("smooth", "smooth", None), ("hard", "hard", None), ("calibrated", "hard", scales)):
        for _ in range(3):
            model(x, mode=mode, scales=thresholds)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(c["analysis"]["timing_iterations"]):
            before = time.perf_counter()
            model(x, mode=mode, scales=thresholds)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter()-before)
        rows[name] = {"batch_size": len(x), "median_ms": float(np.median(durations)*1000),
            "p95_ms": float(np.quantile(durations, .95)*1000),
            "examples_per_second": len(x)/float(np.median(durations)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    return {"parameters": sum(p.numel() for p in model.parameters()), "modes": rows,
            "pre_timing_peak_allocated_bytes": previous_peak,
            "scope": "dense PyTorch inference, no transfer/preprocessing time; activity is a proxy, not measured neuromorphic energy"}


def run_analysis(c, case, method, seed, model, data, descriptor, calibration, directory, device, microbatch):
    from scipy.optimize import minimize_scalar
    from src.paper_suite import classification, evaluate
    options = c["analysis"]
    ids = {name: stratified_ids(data[name].labels, count, stable_seed("analysis", case["id"], name))
           for name, count in (("train", options["probe_train"]), ("validation", options["probe_validation"]),
                               ("test", options["examples"]))}
    modes = (("smooth", "smooth", None), ("hard", "hard", None), ("calibrated", "hard", calibration["scales"]))
    representations = {name: {split: evaluate(model, data[split], device, microbatch, mode, scales,
                                              ids[split], information=True) for split in ids}
                       for name, mode, scales in modes}
    heads, probes, cka, arrays, confidence = {}, {}, {}, {}, {}
    test_features = {name: values["test"]["features"][-1] for name, values in representations.items()}
    labels = representations["smooth"]["test"]["labels"]
    for name, values in representations.items():
        metadata, logits, head = ridge_probe(values["train"]["features"][-1], values["train"]["labels"],
            values["validation"]["features"][-1], values["validation"]["labels"], test_features,
            options["ridge"], descriptor["classes"])
        probes[name] = {**metadata, "test": {key: classification(value, labels) for key, value in logits.items()}}
        heads.update({name+"_"+key: value for key, value in head.items()})
        val_logits, val_labels = values["validation"]["logits"], values["validation"]["labels"]
        chosen = minimize_scalar(lambda value: classification(val_logits/math.exp(value), val_labels)["loss"],
                                 bounds=(-4., 4.), method="bounded")
        temperature = math.exp(float(chosen.x))
        confidence[name] = {"raw": confidence_metrics(values["test"]["logits"], labels),
                            "temperature_scaled": confidence_metrics(values["test"]["logits"], labels, temperature),
                            "temperature_selected_on": "fixed validation subset"}
        arrays[name+"_logits"] = values["test"]["logits"]
        arrays.update({name+"_"+key: value for key, value in values["test"]["statistics"].items()})
    for a, b in (("smooth", "hard"), ("smooth", "calibrated"), ("hard", "calibrated")):
        cka[a+"_"+b] = [linear_cka(x, y) for x, y in zip(representations[a]["test"]["features"],
                                                         representations[b]["test"]["features"])]
    collapse = {}
    for name, x in test_features.items():
        centered = x.astype(np.float64)
        centered -= centered.mean(0)
        gram_values = np.linalg.eigvalsh(centered@centered.T)
        maximum = float(gram_values.max())
        collapse[name] = {"mean_feature_variance": float(np.var(x, axis=0).mean()),
            "numerical_rank": int(np.sum(gram_values > max(maximum*1e-8, 1e-16))),
            "zero_feature_fraction": float(np.mean(np.all(x == 0, axis=0))),
            "undefined_cka_is_null": True}
    np.savez_compressed(directory/"ridge_heads.npz", **heads)
    np.savez_compressed(directory/"analysis_samples.npz", **arrays, labels=labels,
        **{split+"_ids": data[split].ids[index] for split, index in ids.items()})
    bounds_ids = stratified_ids(data["test"].labels, options["bounds_examples"], stable_seed("bounds", case["id"]))
    bounds = bound_diagnostics(c, model, data["test"], bounds_ids, calibration["scales"], device, microbatch, directory)
    homotopy = []
    for output_tau, reset_tau in itertools.product(options["homotopy"], repeat=2):
        measurement = evaluate(model, data["test"], device, microbatch, ids=ids["test"],
            information=True, output_hardness=output_tau, reset_hardness=reset_tau)
        homotopy.append({"output_hardness": output_tau, "reset_hardness": reset_tau,
            **measurement["summary"], "mean_state_margin": float(measurement["statistics"]["mean_margin"].mean())})
    calibration_ablation = {}
    for key in ("fixed_grid_scales", "global_scales", "scales"):
        calibration_ablation[key] = {"scales": calibration[key], "test": evaluate(model, data["test"],
            device, microbatch, "hard", calibration[key], ids["test"])["summary"]}
    corruptions = corruption_tests(c, case, seed, model, data["test"], ids["test"], calibration["scales"], device, microbatch)
    attacks = [{"mode": name, **input_attack(model, data["test"], ids["test"], mode, scales, epsilon,
        options, device, microbatch, stable_seed("attack", case["id"], seed, epsilon))}
        for name, mode, scales in modes for epsilon in options["attack_epsilons"]]
    landscape = None
    if seed in c["seeds"][:options["geometry_seeds"]]:
        landscape = geometry(c, case, seed, model, data["validation"], calibration["scales"], device, microbatch, directory, label=method)
    write_csv(directory/"corruptions.csv", corruptions)
    write_csv(directory/"input_attacks.csv", attacks)
    write_csv(directory/"homotopy.csv", homotopy)
    return {"case": case, "method": method, "seed": seed, "synthetic": descriptor["synthetic"],
        "probes": probes, "cka": cka, "collapse": collapse, "confidence": confidence,
        "bounds": bounds, "homotopy": homotopy, "calibration_ablation": calibration_ablation,
        "corruptions": corruptions, "input_attacks": attacks, "geometry": landscape,
        "inference": inference_timing(c, model, data["test"], device, microbatch, calibration["scales"]),
        "interpretation": "Analysis subsets and conditions were fixed before testing. Corruptions act on cached normalized frames, not raw events. Attacks use continuous frame perturbations and do not certify robustness."}


def detectable_n(sd, effect, maximum=1000):
    from scipy.stats import nct, t
    if sd == 0:
        return 2
    def power(n):
        cutoff = t.ppf(.975, n-1)
        nc = effect*math.sqrt(n)/sd
        return nct.sf(cutoff, n-1, nc)+nct.cdf(-cutoff, n-1, nc)
    if power(maximum) < .8:
        return None
    low, high = 2, maximum
    while low < high:
        middle = (low+high)//2
        if power(middle) >= .8:
            high = middle
        else:
            low = middle+1
    return low


def write_report(c, root):
    import json
    from scipy.stats import ttest_1samp
    from src.paper_suite import cases, completed, job, paired_interval, stage_jobs, status
    output = root/"reports"
    output.mkdir(parents=True, exist_ok=True)
    primary, comparisons, factorial, analyses, ablations, causal_rows, audit = [], [], [], [], [], [], []
    index, causal_index = {}, {}
    for task in stage_jobs(c, "confirm"):
        directory = root/"jobs"/task["id"]
        if not completed(directory):
            continue
        result = json.loads((directory/"result.json").read_text())
        test = result["test"]
        row = {"case": task["case"]["id"], "method": task["method"], "seed": task["seed"],
            "synthetic": result["synthetic"], **result["costs"],
            "calibration_boundary_at_cap": result["calibration"]["boundary_at_cap"],
            "calibration_forward_examples": result["calibration"]["forward_examples"]}
        for mode in ("smooth", "hard", "calibrated"):
            row.update({mode+"_"+key: value for key, value in test[mode].items()})
        primary.append(row)
        index[(row["case"], row["method"], row["seed"])] = row
        audit.append({"phase": "confirmation", "case": row["case"], "method": row["method"],
                      "seed": row["seed"], **result["costs"]})
    for task in stage_jobs(c, "tune"):
        directory = root/"jobs"/task["id"]
        if completed(directory):
            result = json.loads((directory/"result.json").read_text())
            audit.append({"phase": "tuning", "case": task["case"]["id"], "method": task["method"],
                "seed": task["seed"], "candidate": task["candidate"], **result["costs"]})
    primary_p = []
    for case in cases(c):
        for method in c["methods"]:
            if method == "sast":
                continue
            paired = [(index[(case["id"], "sast", s)]["calibrated_accuracy"]-
                       index[(case["id"], method, s)]["calibrated_accuracy"])*100
                for s in c["seeds"] if (case["id"], "sast", s) in index and (case["id"], method, s) in index]
            interval = paired_interval(paired)
            row = {"case": case["id"], "comparison": "sast_minus_"+method, "units": "percentage_points", **interval}
            if len(paired) > 1:
                sd = float(np.std(paired, ddof=1))
                row.update(paired_sd=sd, seeds_for_1pp_80pct=detectable_n(sd, 1.),
                           seeds_for_2pp_80pct=detectable_n(sd, 2.))
                p = float(ttest_1samp(paired, 0).pvalue) if sd > 0 else (1. if np.mean(paired) == 0 else 0.)
                row["p_unadjusted"] = p
                if method == "erm2x":
                    primary_p.append((row, p))
            comparisons.append(row)
    previous = 0.
    for order, (row, p) in enumerate(sorted(primary_p, key=lambda item: item[1])):
        adjusted = max(previous, min(1., p*(len(cases(c))-order)))
        row["p_holm_primary_family"] = adjusted
        previous = adjusted
    for task in stage_jobs(c, "causal"):
        directory = root/"jobs"/task["id"]
        if completed(directory):
            result = json.loads((directory/"result.json").read_text())
        else:
            # A deadline can fall between arms; retain every completed comparison.
            from src.paper_suite import CAUSAL_METHODS
            arms = {m: json.loads((directory/m/"result.json").read_text())
                    for m in CAUSAL_METHODS if completed(directory/m)}
            if not arms:
                continue
            training = directory/"pretrain"/"training.json"
            result = {"arms": arms, "adaptation_curve": {},
                "pretrain_costs": json.loads(training.read_text())["counters"] if training.exists() else {},
                "initial_checkpoint_sha256": next(iter(arms.values()))["initial_checkpoint_sha256"]}
        audit.append({"phase": "pretraining", "case": task["case"]["id"], "method": "erm",
                      "seed": task["seed"], **result["pretrain_costs"]})
        values = {m: r["test"]["calibrated"]["accuracy"] for m, r in result["arms"].items()}
        if {"erm2x", "sast", "margin", "sast_margin"} <= set(values):
            causal_index.setdefault(task["case"]["id"], []).append(values)
        for method, arm in result["arms"].items():
            causal_rows.append({"case": task["case"]["id"], "seed": task["seed"], "method": method,
                "initial_checkpoint_sha256": result["initial_checkpoint_sha256"], **arm["costs"],
                **{mode+"_accuracy": arm["test"][mode]["accuracy"] for mode in ("smooth", "hard", "calibrated")}})
            audit.append({"phase": "causal", "case": task["case"]["id"], "method": method,
                          "seed": task["seed"], **arm["costs"]})
        for fraction, arms in result["adaptation_curve"].items():
            for method, arm in arms.items():
                audit.append({"phase": "adaptation_curve", "case": task["case"]["id"], "method": method,
                              "seed": task["seed"], "budget_fraction": fraction, **arm["costs"]})
    for case_id, rows in causal_index.items():
        contrasts = {"sast_effect_without_margin": lambda x: x["sast"]-x["erm2x"],
            "sast_effect_with_margin": lambda x: x["sast_margin"]-x["margin"],
            "margin_effect_without_sast": lambda x: x["margin"]-x["erm2x"],
            "factorial_interaction": lambda x: x["sast_margin"]-x["sast"]-x["margin"]+x["erm2x"]}
        for name, contrast in contrasts.items():
            factorial.append({"case": case_id, "contrast": name, "units": "percentage_points",
                              **paired_interval([100*contrast(r) for r in rows])})
    for task in stage_jobs(c, "analysis"):
        directory = root/"jobs"/task["id"]
        if completed(directory):
            analyses.append({"job": task["id"], "result": json.loads((directory/"result.json").read_text())})
    for task in stage_jobs(c, "ablate"):
        directory = root/"jobs"/task["id"]
        if completed(directory):
            result = json.loads((directory/"result.json").read_text())
            ablations.append({"case": task["case"]["id"], "method": task["method"], "seed": task["seed"],
                "factor": task["variant"]["factor"], "value": task["variant"]["value"],
                "calibrated_accuracy": result["test"]["calibrated"]["accuracy"], **result["costs"]})
    current_status = status(c, root)
    complete = all(v["completed"] == v["total"] for v in current_status.values())
    violations = sum(profile["false_agreement_conditions"]+profile["logit_bound_violations"]+profile["feature_bound_violations"]
        for a in analyses for profile in a["result"]["bounds"]["profiles"].values())
    summary = {"synthetic": c["synthetic"], "complete": complete, "status": current_status,
        "primary_rows": len(primary), "paired_comparisons": comparisons, "factorial": factorial,
        "numerical_bound_violations": violations, "analysis": analyses,
        "interpretation": "CIs use independent training seeds as paired units. Holm correction covers SAST versus ERM2x across cases. Other contrasts and post-run power calculations are exploratory. Five seeds can leave wide intervals. Numerical bounds are not formal certificates; no result guarantees acceptance."}
    write_csv(output/"primary.csv", primary)
    write_csv(output/"paired_comparisons.csv", comparisons)
    write_csv(output/"causal.csv", causal_rows)
    write_csv(output/"factorial.csv", factorial)
    write_csv(output/"ablations.csv", ablations)
    write_csv(output/"compute_audit.csv", audit)
    write_json(output/"summary.json", summary)
    if primary:
        plot_accuracy(primary, output)
    for case in cases(c):
        plot_competing_landscapes(c, root, case, output)
    report_text = ("SYNTHETIC PIPELINE CHECK — NOT RESEARCH EVIDENCE\n" if c["synthetic"] else "SAST experiment report\n")
    report_text += f"Complete: {complete}\nConfirmation runs: {len(primary)}\nNumerical bound violations: {violations}\n"
    report_text += "\nPaired calibrated accuracy differences (SAST minus comparator, percentage points):\n"
    for row in comparisons:
        if row["mean"] is not None:
            report_text += f"{row['case']} {row['comparison']}: {row['mean']:.3f}, 95% CI [{row['low']}, {row['high']}], n={row['n']}\n"
    report_text += "\n"+summary["interpretation"]+"\nNo manuscript was edited automatically. Inspect raw results and failures before using these artifacts.\n"
    (output/"report.txt").write_text(report_text)
    print(f"Report: {output/'report.txt'} (complete={complete}, synthetic={c['synthetic']})", flush=True)
    return summary


def plot_competing_landscapes(c, root, case, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from src.paper_suite import completed, job
    saved = []
    for method in c["methods"]:
        path = root/"jobs"/job("analysis", case, method, c["seeds"][0])["id"]/"landscapes.npz"
        if path.exists() and completed(path.parent):
            with np.load(path, allow_pickle=False) as a:
                saved.append((method, {key: a[key] for key in ("plane0_coordinates", "plane0_smooth_loss", "plane0_calibrated_accuracy")}))
    if not saved:
        return
    plt.rcParams["pdf.fonttype"] = 42
    fig = plt.figure(figsize=(7, 2*len(saved)), constrained_layout=True)
    for row, (method, a) in enumerate(saved):
        x, y = np.meshgrid(a["plane0_coordinates"], a["plane0_coordinates"], indexing="ij")
        for column, (key, title) in enumerate((("smooth_loss", "Smooth CE"), ("calibrated_accuracy", "Calibrated hard accuracy"))):
            ax = fig.add_subplot(len(saved), 2, row*2+column+1, projection="3d")
            ax.plot_surface(x, y, a["plane0_"+key], cmap="viridis", linewidth=0)
            ax.set_title(method+": "+title, fontsize=9)
            ax.set_xlabel("Direction 1", fontsize=8, labelpad=0)
            ax.set_ylabel("Direction 2", fontsize=8, labelpad=0)
            coordinates = a["plane0_coordinates"]
            ax.set_xticks([coordinates[0], 0, coordinates[-1]])
            ax.set_yticks([coordinates[0], 0, coordinates[-1]])
            ax.zaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=7, pad=1)
            if column:
                ax.set_zlim(0, 1)
    fig.suptitle(("SYNTHETIC PIPELINE CHECK — NOT RESEARCH\n" if c["synthetic"] else "")+
                 case["id"]+" — each method at its own trained checkpoint", fontsize=10)
    for suffix in ("pdf", "png"):
        fig.savefig(output/(case["id"]+"_landscapes."+suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.paper_suite import paired_interval
    plt.rcParams["pdf.fonttype"] = 42
    case_ids = list(dict.fromkeys(r["case"] for r in rows))
    fig, axes = plt.subplots(len(case_ids), 1, figsize=(7, max(2.2, len(case_ids)*2.2)), squeeze=False, constrained_layout=True)
    if any(row["synthetic"] for row in rows):
        fig.suptitle("SYNTHETIC PIPELINE CHECK — NOT RESEARCH", fontsize=11)
    for case_id, ax in zip(case_ids, axes[:, 0]):
        methods = list(dict.fromkeys(r["method"] for r in rows if r["case"] == case_id))
        for i, method in enumerate(methods):
            values = [100*r["calibrated_accuracy"] for r in rows if r["case"] == case_id and r["method"] == method]
            summary = paired_interval(values)
            ax.errorbar(i, summary["mean"], yerr=summary["high"]-summary["mean"] if summary["high"] is not None else 0,
                        fmt="o", color="#16856a" if method == "sast" else "#4c5970", capsize=3)
            ax.scatter([i]*len(values), values, s=8, alpha=.4, color="#4c5970")
        ax.set(xticks=range(len(methods)), xticklabels=methods, ylabel="Hard accuracy (%)", title=case_id)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.15)
    fig.savefig(output/"accuracy.pdf", bbox_inches="tight")
    fig.savefig(output/"accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
