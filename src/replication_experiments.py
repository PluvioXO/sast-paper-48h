# %% Full replication configuration and experiment inventory
"""Embedded after sast_core.py and colab_support.py. No import of companion files at runtime."""


@dataclass
class ReplicationConfig(Config):
    radius_grid: tuple = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
    # Appendix B mentions testing up to 1 without listing points. This grid is our choice.
    extra_radius_grid: tuple = (0.60, 0.70, 0.80, 0.90, 1.00)
    slope_grid: tuple = (0.5, 1., 2.)
    calibration_grid: tuple = tuple(round(0.5 + 0.1*i, 1) for i in range(11))
    calibration_rounds: int = 2
    diagnostic_samples: int = 128
    jacobian_samples: int = 16
    secant_samples: int = 64
    secant_radii: tuple = (0.0001, 0.001, 0.01)
    secant_directions: int = 4
    asam_eta: float = 0.01
    input_penalty: float = 0.01
    surrogate: str = "arctan"
    pooling: str = "avg"
    batch_norm: bool = False
    algorithm: str = "baseline"
    variant_id: str = "baseline"
    variant_group: str = "main"
    trial_rho: float = 0.

    def validate(self):
        super().validate()
        assert self.surrogate in ("arctan", "fast_sigmoid")
        assert self.pooling in ("avg", "max")
        assert self.algorithm in ("baseline", "compute_matched", "sast", "reused",
                                  "asam", "hard_sg", "input_gradient")
        assert all(r > 0 for r in self.radius_grid + self.extra_radius_grid + self.secant_radii)
        assert len(set(self.radius_grid + self.extra_radius_grid)) == len(self.radius_grid + self.extra_radius_grid)
        assert all(k > 0 for k in self.slope_grid)
        assert 1. in self.calibration_grid and min(self.calibration_grid) > 0
        assert self.calibration_rounds > 0 and self.secant_directions > 0
        assert 0 < self.jacobian_samples <= self.diagnostic_samples
        assert self.secant_samples > 0 and self.asam_eta > 0 and self.input_penalty >= 0
        return self


def replication_smoke(cfg):
    return replace(
        cfg, profile="smoke", synthetic=True, seeds=(0,), time_bins=3, channels=(2, 3),
        epochs={"nmnist": 1, "dvsgesture": 1},
        batch_sizes={"nmnist": 2, "dvsgesture": 2}, steps_per_epoch=1,
        radius_grid=(0.1,), extra_radius_grid=(0.8,), slope_grid=(0.5, 1.),
        calibration_grid=(0.5, 1., 1.5), calibration_rounds=1,
        diagnostic_samples=2, jacobian_samples=1, secant_samples=2,
        secant_radii=(0.001,), secant_directions=1,
        benchmark_steps=1, log_every=0,
    )


def variant_plan(cfg, name):
    variants = []

    def add(identifier, algorithm, group, rho=0., **extra):
        variants.append(dict(id=identifier, algorithm=algorithm, group=group, rho=rho,
                             surrogate=extra.get("surrogate", "arctan"),
                             slope=extra.get("slope", 1.), pooling=extra.get("pooling", "avg"),
                             batch_norm=extra.get("batch_norm", False)))

    add("baseline", "baseline", "main")
    for rho in cfg.radius_grid:
        add(f"sast_r{rho:.2f}", "sast", "main", rho)
    add("compute_matched", "compute_matched", "controls")
    for rho in cfg.radius_grid:
        add(f"reused_r{rho:.2f}", "reused", "controls", rho)
        add(f"asam_r{rho:.2f}", "asam", "controls", rho)
    for family in ("arctan", "fast_sigmoid"):
        for slope in cfg.slope_grid:
            if family == "arctan" and slope == 1.:
                continue
            for algorithm in ("baseline", "sast"):
                add(f"{family}_k{slope:g}_{algorithm}", algorithm, "surrogates",
                    cfg.radii[name] if algorithm == "sast" else 0.,
                    surrogate=family, slope=slope)
    for algorithm in ("baseline", "sast"):
        add(f"maxpool_batchnorm_{algorithm}", algorithm, "practical",
            cfg.radii[name] if algorithm == "sast" else 0., pooling="max", batch_norm=True)
    add("hard_forward_surrogate_backward", "hard_sg", "practical")
    add("input_gradient_regularization", "input_gradient", "controls")
    for rho in cfg.extra_radius_grid:
        add(f"extended_sast_r{rho:.2f}", "sast", "extended", rho)
    assert len({v["id"] for v in variants}) == len(variants)
    return variants


def trial_config(cfg, name, variant):
    radii = dict(cfg.radii)
    radii[name] = variant["rho"] or cfg.radii[name]
    return replace(cfg, radii=radii, algorithm=variant["algorithm"],
                   variant_id=variant["id"], variant_group=variant["group"],
                   trial_rho=variant["rho"], surrogate=variant["surrogate"],
                   slope=variant["slope"], pooling=variant["pooling"],
                   batch_norm=variant["batch_norm"])


def budget_method(cfg):
    if cfg.algorithm in ("sast", "reused", "asam"):
        return "sast"
    return "compute_matched" if cfg.algorithm == "compute_matched" else "baseline"


def experiment_manifest(cfg):
    return pd.DataFrame([
        dict(dataset=name, seed=seed, variant=v["id"], algorithm=v["algorithm"],
             group=v["group"], rho=v["rho"], surrogate=v["surrogate"], slope=v["slope"],
             pooling=v["pooling"], batch_norm=v["batch_norm"],
             epochs=cfg.epochs[name]*(2 if v["algorithm"] == "compute_matched" else 1),
             batch_size=cfg.batch_sizes[name])
        for name in cfg.datasets for seed in cfg.seeds for v in variant_plan(cfg, name)
    ])


def paper_accuracy_reference():
    rows = {
        "nmnist": [
            (0., .9606, .0033, .6572, .0974, .3034),
            (.02, .9789, .0033, .7453, .0251, .2336),
            (.05, .9831, .0007, .7812, .0847, .2019),
            (.10, .9786, .0009, .8335, .0894, .1451),
            (.20, .9753, .0013, .9018, .0496, .0735),
            (.30, .9721, .0012, .9473, .0462, .0248),
            (.40, .9697, .0009, .9424, .0332, .0273),
            (.50, .9655, .0013, .9318, .0323, .0337)],
        "dvsgesture": [
            (0., .7502, .0142, .3182, .0732, .4320),
            (.02, .7670, .0134, .4489, .0716, .3181),
            (.05, .7519, .0187, .4754, .0623, .2765),
            (.10, .7727, .0107, .5341, .0357, .2386),
            (.20, .8087, .0043, .5957, .0116, .2130),
            (.30, .7778, .0076, .5926, .0151, .1852),
            (.40, .7685, .0151, .6327, .0116, .1358),
            (.50, .7747, .0430, .6022, .0191, .1725)],
    }
    return pd.DataFrame([
        dict(dataset=name, variant="baseline" if r == 0 else f"sast_r{r:.2f}", rho=r,
             published_smooth_mean=sm, published_smooth_std=ss,
             published_hard_mean=hm, published_hard_std=hs, published_gap=gap)
        for name, values in rows.items() for r, sm, ss, hm, hs, gap in values
    ]).assign(source=PAPER_URL + "#S5.T2", provenance="PUBLISHED_ONLY")


def paper_cost_reference():
    return pd.DataFrame([
        ("nmnist", "baseline", 80.48, .25, 1., .40, .01, 1.),
        ("nmnist", "sast", 167.72, 1.81, 2.08, .40, .01, 1.),
        ("dvsgesture", "baseline", 2.87, .07, 1., 2.48, .01, 1.),
        ("dvsgesture", "sast", 5.09, .09, 1.77, 2.48, .01, 1.),
    ], columns=["dataset", "algorithm", "published_epoch_seconds_mean", "published_epoch_seconds_std",
                "published_time_factor", "published_memory_GB_mean", "published_memory_GB_std",
                "published_memory_factor"]).assign(source=PAPER_URL + "#S5.T4", provenance="PUBLISHED_ONLY")


def coverage_inventory():
    rows = [
        ("Figure 1 / Algorithm 1", "Architecture and SAST", "schematic + executable",
         "architecture.png; tested forward/reset/SAM implementation"),
        ("Table I", "Theory scope by architecture/training mode", "qualitative",
         "theory_scope.csv; C2 and parameterization caveats"),
        ("Table II", "Both datasets, baseline and seven SAST radii", "16 numerical rows",
         "table_II_measured.csv; published_comparison.csv; per_seed.csv"),
        ("Figure 2 / V-C", "Voxel drop; temporal drop on both datasets", "published figure, no raw series",
         "figure_2_measured.png/pdf; robustness.csv; robustness_summary.csv"),
        ("V-D / Appendix B", "Sharpness, gradients, Gram, secants, margins, firing rates", "protocol only",
         "per_example.csv; sharpness.csv; secants.csv; diagnostic_summary.csv"),
        ("Table III / Appendix B", "Compute matched, reused batches and ASAM", "protocol only",
         "control_comparisons.csv; budget_audit.csv"),
        ("Table III / Appendix B", "Global and per-layer threshold calibration", "protocol only",
         "calibration_validation.csv; calibration_test.csv; calibrated_robustness.csv"),
        ("Table III", "Surrogate/slope sweep and MaxPool + BatchNorm", "protocol only",
         "control_results.csv; theory_scope.csv"),
        ("Table I / I-A", "Hard-forward baseline and input-gradient penalty", "mentioned, no numerical rows",
         "control_results.csv; differentiated compute accounting"),
        ("Table IV", "Epoch time and memory ratios", "4 numerical rows; hardware unspecified",
         "table_IV_measured.csv; hardware-specific ratios; published cost separate"),
        ("Appendix B", "Per-seed hard accuracy, gaps, median/IQR", "protocol only",
         "per_seed.csv; seed_summary.csv; seed_sensitivity.png/pdf"),
        ("Appendix B", "Radius range through 1.0", "range stated, exact grid absent",
         "extended_radii.csv; our additional 0.6:0.1:1.0 grid"),
        ("IV / Appendix A.1–A.2", "State/Jacobian/Hessian bounds and constants", "analytical claims",
         "theory_checks.csv; explicit constants audit and reproducible counterexample"),
        ("IV.10 / IV.12 / Appendix A-C", "SAM approximation and stochastic convergence", "analytical claims",
         "theory_checks.csv; convergence_checks.csv; assumptions and proof derivations"),
        ("IV.13 / Appendix A.3–A.5", "Loss stability and local conditioning mechanism", "analytical claims",
         "theory_checks.csv; per-example true Jacobians and local inequalities"),
        ("V-A", "CIFAR10-DVS", "optional future extension; no reported result or protocol",
         "not a replication target in this version"),
        ("Original exact recipe / seeds / Figure 2 series", "Author artifacts", "not supplied",
         "assumptions.json; exact numerical replication is not certified"),
    ]
    return pd.DataFrame(rows, columns=["paper_item", "target", "paper_status", "notebook_output"])


def replication_assumptions(cfg):
    return {
        "paper_version": "2603.18039v1",
        "original_code_or_logs_available": False,
        "purpose": "Coverage-complete independent implementation; exact reported numbers not guaranteed.",
        "choices": {
            "architecture": "Two 3x3 Conv 16/32 -> 2x2 AvgPool -> LIF blocks; flattened temporal mean + linear.",
            "training": "AdamW, lr .001 -> .00001 cosine, wd .0001, 20/50 epochs, batches32/8.",
            "seeds": list(cfg.seeds), "preprocessing": "16 equal-duration bins, per-recording/polarity maximum.",
            "thresholds": "Fixed 1.0, leak .5; no learned threshold statistics implied.",
            "validation": "Fixed stratified recording-level 10%; smooth accuracy then loss selects checkpoints.",
            "asam": "Weight tensors scaled by abs(w)+eta; bias unscaled; eta .01; independent second batch.",
            "fast_sigmoid": ".5+.5*k*x/(1+abs(k*x)); C1 but not C2 at zero; practical extension.",
            "slope_grid": list(cfg.slope_grid), "extra_radius_grid": list(cfg.extra_radius_grid),
            "calibration": "Global exhaustive grid; per-layer coordinate search, 2 rounds; validation only.",
            "hard_gradients": "True hard derivatives almost everywhere; STE diagnostics separately labeled.",
            "sharpness": "One gradient-ascent direction on a fixed validation subset, shared radii; a proxy, not exact maximum.",
            "secants": "Shared random directions on a fixed validation subset; empirical lower estimates of local smoothness.",
            "diagnostic_samples": cfg.diagnostic_samples, "jacobian_samples": cfg.jacobian_samples,
            "input_penalty": cfg.input_penalty,
            "practical_batchnorm": "Conv -> BN over merged batch/time -> MaxPool -> LIF; ascent BN stats restored.",
            "timing": "Per-epoch training wall time includes loading/instrumentation; separate validation time.",
            "memory": "Sampled RSS and CUDA allocator peaks; paper hardware/memory instrumentation unknown.",
        },
    }


# %% Network variants and optimization controls
def surrogate_value(x, family, slope):
    if family == "arctan":
        return .5 + torch.atan(slope*x)/math.pi
    return .5 + .5*slope*x/(1 + (slope*x).abs())


def surrogate_derivative(x, family, slope):
    if family == "arctan":
        return slope/(math.pi*(1 + (slope*x).square()))
    return .5*slope/(1 + (slope*x).abs()).square()


class _HardSurrogateBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, family, slope):
        ctx.save_for_backward(x)
        ctx.family, ctx.slope = family, slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return gradient*surrogate_derivative(x, ctx.family, ctx.slope), None, None


class ReplicationSNN(nn.Module):
    def __init__(self, cfg, name):
        super().__init__()
        size, classes = dataset_shape(name)
        self.cfg = cfg
        self.leak, self.slope = cfg.leak, cfg.slope
        self.surrogate, self.pooling = cfg.surrogate, cfg.pooling
        self.register_buffer("threshold", torch.tensor(cfg.threshold, dtype=torch.float32))
        a, b = cfg.channels
        self.convs = nn.ModuleList([nn.Conv2d(2, a, 3, padding=1), nn.Conv2d(a, b, 3, padding=1)])
        self.norms = nn.ModuleList([nn.BatchNorm2d(c) if cfg.batch_norm else nn.Identity() for c in (a, b)])
        self.readout = nn.Linear(b*(size//4)**2, classes)

    def forward(self, x, hard=False, mode=None, threshold_scales=None, return_trace=False):
        mode = mode or ("hard" if hard else "smooth")
        scales = threshold_scales if threshold_scales is not None else (1., 1.)
        batch, bins = x.shape[:2]
        z, traces = x, []
        pool = F.avg_pool2d if self.pooling == "avg" else F.max_pool2d
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            drives = pool(norm(conv(z.flatten(0, 1))), 2)
            drives = drives.reshape(batch, bins, *drives.shape[1:])
            threshold = self.threshold*scales[layer]
            state, spike, sequence, states = None, None, [], []
            for t in range(bins):
                u = drives[:, t] if state is None else self.leak*state + drives[:, t] - threshold*spike
                distance = u-threshold
                if mode == "hard":
                    spike = (distance >= 0).to(x.dtype)
                elif mode == "ste":
                    spike = _HardSurrogateBackward.apply(distance, self.surrogate, self.slope)
                elif mode == "smooth":
                    spike = surrogate_value(distance, self.surrogate, self.slope)
                else:
                    raise ValueError(mode)
                state = u
                sequence.append(spike)
                if return_trace:
                    states.append(u)
            z = torch.stack(sequence, 1)
            if return_trace:
                traces.append({"u": torch.stack(states, 1), "s": z, "threshold": threshold})
        total = z[:, 0]
        for t in range(1, bins):
            total = total + z[:, t]
        logits = self.readout((total/bins).flatten(1))
        return (logits, traces) if return_trace else logits


SurrogateSNN = ReplicationSNN
_replication_base_budget = run_budget


def run_budget(cfg, name, n_train, method):
    steps, epochs, passes = _replication_base_budget(cfg, name, n_train, method)
    if getattr(cfg, "algorithm", "") == "input_gradient":
        # One extra input VJP per update, distinct from a complete extra forward/backward pair.
        passes = 2*steps*epochs
    return steps, epochs, passes


def _restore_buffers(model, original):
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(original[name])


def training_step(model, optimizer, descent_batch, ascent_batch=None, rho=0., delta=1e-12,
                  memory_callback=None):
    cfg = model.cfg
    model.train()
    optimizer.zero_grad(set_to_none=True)
    params = list(model.named_parameters())
    originals = [p.detach().clone() for _, p in params]
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    two_pass = cfg.algorithm in ("sast", "reused", "asam")
    try:
        if two_pass:
            ascent_batch = descent_batch if cfg.algorithm == "reused" else ascent_batch
            if ascent_batch is None:
                raise ValueError("Independent ascent batch required.")
            first = F.cross_entropy(model(ascent_batch[0]), ascent_batch[1])
            if not torch.isfinite(first):
                raise FloatingPointError("Non-finite ascent loss.")
            first.backward()
            gradient_norm(model)
            scales = [p.detach().abs()+cfg.asam_eta if cfg.algorithm == "asam" and "weight" in n
                      else torch.ones_like(p) for n, p in params]
            norm = torch.linalg.vector_norm(torch.stack([
                (p.grad*s).norm() for (_, p), s in zip(params, scales) if p.grad is not None]))
            with torch.no_grad():
                for (_, p), scale in zip(params, scales):
                    if p.grad is not None:
                        p.add_(rho*scale.square()*p.grad/(norm+delta))
            # First-pass BN running statistics must not contaminate the second pass.
            _restore_buffers(model, buffers)
            optimizer.zero_grad(set_to_none=True)
        x, y = descent_batch
        if cfg.algorithm == "input_gradient":
            x = x.detach().requires_grad_(True)
        logits = model(x, mode="ste" if cfg.algorithm == "hard_sg" else "smooth")
        loss = F.cross_entropy(logits, y)
        if cfg.algorithm == "input_gradient":
            per_example_gradient = torch.autograd.grad(loss, x, create_graph=True)[0]*len(y)
            loss = loss + cfg.input_penalty*per_example_gradient.flatten(1).square().sum(1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite descent loss.")
        loss.backward()
        gradient_norm(model)
        if memory_callback:
            memory_callback()
    except BaseException:
        _restore_buffers(model, buffers)
        raise
    finally:
        with torch.no_grad():
            for (_, parameter), old in zip(params, originals):
                parameter.copy_(old)
    optimizer.step()
    return float(loss.detach()), 2 if two_pass or cfg.algorithm == "input_gradient" else 1


class CalibratedModel(nn.Module):
    def __init__(self, model, scales):
        super().__init__()
        self.model, self.scales = model, tuple(scales)

    def forward(self, x, hard=False):
        return self.model(x, hard=hard, threshold_scales=self.scales)


def calibrate_model(model, data, cfg, name, directory, device):
    """Freeze all weights and choose thresholds using only validation hard accuracy/loss."""
    directory = Path(directory)
    before = state_digest(cpu_state(model))
    progress = directory / "calibration_validation.json"
    tested = json.loads(progress.read_text()) if progress.exists() else []
    cache = {tuple(row["scales"]): row for row in tested}

    def score(scales):
        key = tuple(round(float(s), 8) for s in scales)
        if key not in cache:
            metrics = evaluate(CalibratedModel(model, key), data["validation"],
                               cfg.batch_sizes[name], device, name, modes=("hard",))
            row = dict(scales=list(key), accuracy=metrics["hard_accuracy"], loss=metrics["hard_loss"])
            cache[key] = row
            tested.append(row)
            atomic_json(progress, tested)
        row = cache[key]
        # Deterministic tie: accuracy, lower loss, then closest to unchanged thresholds.
        return row["accuracy"], -row["loss"], -sum(abs(s-1.) for s in key)

    global_scales = max(((s, s) for s in cfg.calibration_grid), key=score)
    scales = list(global_scales)
    for _ in range(cfg.calibration_rounds):
        for layer in range(2):
            candidates = [tuple(s if j == layer else scales[j] for j in range(2))
                          for s in cfg.calibration_grid]
            scales = list(max(candidates, key=score))
    choices = {"global": global_scales, "per_layer": tuple(scales)}
    best_family = max(choices, key=lambda family: score(choices[family]))
    test_path = directory / "calibration_test.json"
    results = json.loads(test_path.read_text()) if test_path.exists() else []
    existing = {r["calibration"] for r in results}
    for family, selected in choices.items():
        if family in existing:
            continue
        metrics = evaluate(CalibratedModel(model, selected), data["test"],
                           cfg.batch_sizes[name], device, name, modes=("hard",))
        results.append(dict(calibration=family, scales=list(selected),
                            validation_hard_accuracy=score(selected)[0],
                            selected_best_family=(family == best_family),
                            hard_accuracy=metrics["hard_accuracy"], hard_loss=metrics["hard_loss"]))
        atomic_json(test_path, results)
    pd.DataFrame(tested).to_csv(directory / "calibration_validation.csv", index=False)
    pd.DataFrame(results).to_csv(directory / "calibration_test.csv", index=False)
    # Robustness for the family selected on validation; no test-based family selection.
    path = directory / "calibrated_robustness.json"
    robustness = json.loads(path.read_text()) if path.exists() else []
    done = {(r["family"], r["severity"]) for r in robustness}
    calibrated = CalibratedModel(model, choices[best_family])
    for family in ("voxel_drop", "time_bin_drop"):
        for severity in cfg.severities:
            if (family, severity) not in done:
                metrics = evaluate(calibrated, data["test"], cfg.batch_sizes[name], device, name,
                                   family, severity, cfg.corruption_seed, modes=("hard",))
                robustness.append(dict(family=family, severity=severity, calibration=best_family,
                                       hard_accuracy=metrics["hard_accuracy"]))
                atomic_json(path, robustness)
    pd.DataFrame(robustness).to_csv(directory / "calibrated_robustness.csv", index=False)
    assert state_digest(cpu_state(model)) == before, "Calibration mutated the trained model."
    return results
