# %% CIFAR10-DVS configuration and fixed study protocol
"""New dataset adapter, embedded after the unchanged replication/follow-up engines."""

CIFAR_DATASET = "cifar10dvs"
CIFAR_VERSION = "1.0"
CIFAR_DIGEST = globals().get("CIFAR_DIGEST", "development")
CIFAR_METHODS = ("baseline", "compute_matched", "sast_selected", "hard_forward_hard_selected")
CIFAR_LABELS = ("Vanilla", "2x updates", "SAST", "Direct hard")
_cifar_legacy_shape = dataset_shape
_cifar_legacy_environment = environment_info


@dataclass
class Cifar10DVSConfig(ReplicationConfig):
    datasets: tuple = (CIFAR_DATASET,)
    seeds: tuple = tuple(range(10))
    pilot_seeds: tuple = (10, 11, 12)
    epochs: dict = field(default_factory=lambda: {CIFAR_DATASET: 100})
    batch_sizes: dict = field(default_factory=lambda: {CIFAR_DATASET: 16})
    radii: dict = field(default_factory=lambda: {CIFAR_DATASET: .1})
    # The radii field is only a placeholder until the validation-only pilot freezes rho.
    extra_radius_grid: tuple = ()
    slope_grid: tuple = (1.,)
    test_fraction: float = .1
    validation_fraction: float = .1  # Fraction of ALL recordings, not of the remaining 90%.

    def validate(self):
        assert self.datasets == (CIFAR_DATASET,)
        assert 0 < self.test_fraction < 1 and 0 < self.validation_fraction < 1
        assert self.test_fraction + self.validation_fraction < 1
        assert self.pilot_seeds and len(set(self.pilot_seeds)) == len(self.pilot_seeds)
        assert not set(self.pilot_seeds) & set(self.seeds)
        assert all(isinstance(s, int) and s >= 0 for s in self.seeds + self.pilot_seeds)
        assert self.radius_grid and tuple(sorted(set(self.radius_grid))) == self.radius_grid
        assert len({f"{r:.2f}" for r in self.radius_grid}) == len(self.radius_grid)
        assert not self.extra_radius_grid and self.slope_grid == (1.,)
        assert self.pooling == "avg" and not self.batch_norm and self.surrogate == "arctan"
        # Exercise every frozen engine constraint without changing its dataset allowlist.
        # This temporary validator view is never used for training or provenance.
        view = replace(self, datasets=("dvsgesture",),
                       epochs={"dvsgesture": self.epochs[CIFAR_DATASET]},
                       batch_sizes={"dvsgesture": self.batch_sizes[CIFAR_DATASET]},
                       radii={"dvsgesture": self.radii[CIFAR_DATASET]})
        ReplicationConfig.validate(view)
        if self.profile == "evidence":
            assert self.seeds == tuple(range(10)), "The evidence protocol requires all ten seeds."
            assert self.pilot_seeds == (10, 11, 12)
            assert .1 in self.radius_grid, "Diagnostics require the fixed 0.1 sharpness probe."
        return self


def cifar_smoke_config(cfg):
    return replace(cfg, profile="smoke", synthetic=True, device="cpu", seeds=(0, 1),
                   pilot_seeds=(10,), time_bins=2, channels=(2, 3),
                   epochs={CIFAR_DATASET: 1}, batch_sizes={CIFAR_DATASET: 2},
                   steps_per_epoch=1, radius_grid=(.1, .3),
                   calibration_grid=(.5, 1., 1.5), calibration_rounds=1,
                   diagnostic_samples=2, jacobian_samples=1, secant_samples=2,
                   secant_radii=(.001,), secant_directions=1, benchmark_steps=1, log_every=0)


def dataset_shape(name):
    return (64, 10) if name == CIFAR_DATASET else _cifar_legacy_shape(name)


def environment_info(device):
    result = _cifar_legacy_environment(device)
    try:
        result["packages"]["aedat"] = importlib.metadata.version("aedat")
    except importlib.metadata.PackageNotFoundError:
        result["packages"]["aedat"] = "not installed (synthetic smoke only)"
    result["cifar_engine_digest"] = CIFAR_DIGEST
    return result


def cifar_manifest(cfg):
    cfg.validate()
    rows = []
    for seed in cfg.pilot_seeds:
        for rho in cfg.radius_grid:
            rows.append(dict(phase="pilot", dataset=CIFAR_DATASET, seed=seed,
                             variant=f"sast_r{rho:.2f}", architecture="original",
                             checkpoint_selection="smooth", transfer_focus=False))
    for seed in cfg.seeds:
        for method in CIFAR_METHODS:
            rows.append(dict(phase="main", dataset=CIFAR_DATASET, seed=seed, variant=method,
                             architecture="original", transfer_focus=True,
                             checkpoint_selection="hard" if method == "hard_forward_hard_selected" else "smooth"))
    return pd.DataFrame(rows)


def cifar_protocol(cfg):
    config = asdict(cfg)
    for key in ("output_dir", "data_dir"):
        config.pop(key)
    return dict(version=CIFAR_VERSION, config=config, cifar_engine_sha256=CIFAR_DIGEST,
                inherited_engine_sha256=ENGINE_DIGEST, followup_engine_sha256=FOLLOWUP_DIGEST,
                split="Fixed class-stratified 80/10/10 by sorted relative filename; same split for every seed/method.",
                pilot="Seeds 10,11,12, all radius candidates; validation only, no test inference. Never pooled with main seeds.",
                radius_selection="Maximize mean calibrated validation hard accuracy across pilot seeds; tie: lower mean loss, then smaller rho.",
                primary="Calibrated test hard accuracy: SAST minus compute-matched vanilla across paired main seeds 0-9.",
                checkpoint_selection="Smooth validation accuracy then loss; corrected direct-hard control uses hard validation.",
                calibration="Global/per-layer threshold family selected on validation only, with frozen weights.",
                statistics="Mean, sample SD and unadjusted paired Student-t 95% intervals across seeds; other outcomes exploratory.",
                scope="New dataset-specific protocol, not a claim of optimal hyperparameters or a reproduction of published CIFAR10-DVS accuracy.",
                diagnostics="All main seeds: full validation/test predictions, sampled gradients/Jacobians, sharpness, secants and activity.",
                sources={"dataset": "https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full",
                         "loader": "https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.CIFAR10DVS.html"})


def cifar_directory(cfg):
    digest = hashlib.sha256(json.dumps(cifar_protocol(cfg), sort_keys=True).encode()).hexdigest()[:12]
    return Path(cfg.output_dir) / ("cifar10dvs_smoke" if cfg.synthetic else "cifar10dvs") / digest


def cifar_job_directory(root, job):
    if job["phase"] == "pilot":
        return Path(root) / "pilot" / f"seed_{job['seed']}" / job["variant"]
    return followup_job_directory(root, job)


def cifar_trial(cfg, job, selected_radius=None):
    variant = job["variant"]
    algorithm = "sast" if variant.startswith("sast_") else (
        "hard_sg" if variant == "hard_forward_hard_selected" else variant)
    rho = float(variant.removeprefix("sast_r")) if job["phase"] == "pilot" else selected_radius
    if algorithm == "sast" and (rho is None or rho not in cfg.radius_grid):
        raise ValueError("Freeze the pilot-selected SAST radius before main training.")
    return replace(cfg, algorithm=algorithm, variant_id=variant, variant_group=job["phase"],
                   trial_rho=rho if algorithm == "sast" else 0.,
                   radii={CIFAR_DATASET: rho or cfg.radii[CIFAR_DATASET]})


# %% CIFAR10-DVS verified download, stable splits, and resumable local caches
class _UnopenedTestSet:
    """Training can record test size, but cannot read test events during the pilot."""
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def batch(self, *args, **kwargs):
        raise RuntimeError("Test data is locked until the pilot radius is frozen.")


def cifar_split(labels, cfg):
    labels = np.asarray(labels)
    rng = np.random.default_rng(cfg.split_seed)
    parts = {k: [] for k in ("train", "validation", "test")}
    for label in sorted(np.unique(labels)):
        ids = rng.permutation(np.flatnonzero(labels == label))
        ntest = max(1, int(round(len(ids)*cfg.test_fraction)))
        nval = max(1, int(round(len(ids)*cfg.validation_fraction)))
        if ntest+nval >= len(ids):
            raise ValueError("Each class needs examples in all three partitions.")
        parts["test"].extend(ids[:ntest])
        parts["validation"].extend(ids[ntest:ntest+nval])
        parts["train"].extend(ids[ntest+nval:])
    parts = {k: np.sort(np.asarray(v, dtype=np.int64)) for k,v in parts.items()}
    assert sorted(np.concatenate(list(parts.values())).tolist()) == list(range(len(labels)))
    return parts


def cifar_raw_dataset(cfg):
    import tonic
    import aedat  # Fail before downloading if the extra decoder dependency is absent.
    from tonic.download_utils import extract_archive
    cls = tonic.datasets.CIFAR10DVS
    local = Path(cfg.data_dir) / "raw" / cls.__name__
    local.mkdir(parents=True, exist_ok=True)
    archive = local / cls.filename
    backup = Path(ARCHIVE_DIR) / cls.__name__ / cls.filename if ARCHIVE_DIR else None
    parts = (backup.parent if backup is not None else local) / (cls.filename + ".chunks")
    urls = (cls.url.replace("https://figshare.com/ndownloader/files/",
                           "https://ndownloader.figshare.com/files/"),
            "https://api.figshare.com/v2/file/download/38023437")
    print("Preparing CIFAR10-DVS with 64 MiB resumable chunks and 1 MiB buffers.", flush=True)
    print("Download checkpoints:", parts, flush=True)
    cifar_download_archive(archive, parts, urls, cls.file_md5, legacy_backup=backup,
                           progress=lambda stage, **details: cifar_data_progress(cfg, stage, **details))
    cifar_data_progress(cfg, "extracting_recordings")
    raw = cls(save_to=str(Path(cfg.data_dir) / "raw"))
    if len(raw) != 10000:
        # Tonic accepts only 1,000 extracted files; repair an interrupted 10,000-file extraction.
        print("Repairing incomplete CIFAR10-DVS extraction.", flush=True)
        extract_archive(str(archive))
        for filename in cls.data_filename:
            extract_archive(str(local / filename))
        raw = cls(save_to=str(Path(cfg.data_dir) / "raw"))
    order = sorted(range(len(raw)), key=lambda i: Path(raw.data[i]).relative_to(local).as_posix())
    raw.data = [raw.data[i] for i in order]
    raw.targets = [int(raw.targets[i]) for i in order]
    if len(raw) != 10000 or pd.Series(raw.targets).value_counts().sort_index().to_dict() != dict.fromkeys(range(10), 1000):
        raise RuntimeError("CIFAR10-DVS must contain exactly 1,000 recordings in each of 10 classes.")
    if len(set(raw.data)) != 10000 or any(Path(p).stat().st_size == 0 for p in raw.data):
        raise RuntimeError("Duplicate or empty CIFAR10-DVS recordings.")
    return raw, dict(filename=cls.filename, md5=cls.file_md5, url=cls.url, classes=cls.classes)


def prepare_cifar_data(cfg, include_test=False):
    cfg.validate()
    if cfg.synthetic:
        data, offset = {}, 0
        for split,n in (("train", 20), ("validation", 10), ("test", 10)):
            rng = np.random.default_rng(stable_seed("cifar-smoke", split))
            frames = rng.binomial(1, .03, (n, cfg.time_bins, 2, 64, 64)).astype(np.float32)
            data[split] = FrameSet(frames, np.arange(n)%10, np.arange(offset,offset+n))
            offset += n
        if not include_test:
            data["test"] = _UnopenedTestSet(10)
        return data
    raw, archive = cifar_raw_dataset(cfg)
    cifar_data_progress(cfg, "recordings_ready")
    labels = np.asarray(raw.targets, dtype=np.int64)
    parts = cifar_split(labels, cfg)
    raw_root = Path(cfg.data_dir) / "raw" / "CIFAR10DVS"
    spec = dict(dataset=CIFAR_DATASET, archive=archive,
                tonic=importlib.metadata.version("tonic"), aedat=importlib.metadata.version("aedat"),
                recordings=[dict(path=Path(p).relative_to(raw_root).as_posix(), label=int(y),
                                 bytes=Path(p).stat().st_size) for p,y in zip(raw.data,labels)],
                partitions={k:v.tolist() for k,v in parts.items()}, split_seed=cfg.split_seed,
                time_bins=cfg.time_bins, sensor_size=128, spatial_size=64,
                preprocessing="equal_duration_per_channel_max_v1; integer spatial binning 128->64; no augmentation")
    digest = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cfg.data_dir) / "frames" / CIFAR_DATASET / digest
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory / "manifest.json",spec)
    data = {"manifest":str(directory / "manifest.json")}
    for split,ids in parts.items():
        if split != "test" or include_test:
            cifar_data_progress(cfg, "caching_frames", split=split)
        data[split] = cache_frames(raw,ids,labels,cfg,CIFAR_DATASET,split,directory) if (
            split != "test" or include_test) else _UnopenedTestSet(len(ids))
    cifar_data_progress(cfg, "frames_ready", include_test=include_test)
    return data


# %% Validation-only calibration and frozen pilot radius selection
def cifar_validation_calibration(model, validation, cfg, directory, device):
    """Receives validation data only; there is no path to test data in this function."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    digest = state_digest(cpu_state(model))
    identity = dict(checkpoint=digest, grid=list(cfg.calibration_grid), rounds=cfg.calibration_rounds,
                    example_ids=validation.ids.tolist(), engine=CIFAR_DIGEST)
    path = directory / "calibration_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise RuntimeError("Calibration cache does not match this checkpoint/protocol.")
    atomic_json(path,identity)
    progress = directory / "calibration_validation.json"
    rows = json.loads(progress.read_text()) if progress.exists() else []
    cache = {tuple(row["scales"]):row for row in rows}

    def score(scales):
        scales = tuple(round(float(s),8) for s in scales)
        if scales not in cache:
            metrics = evaluate(CalibratedModel(model,scales),validation,cfg.batch_sizes[CIFAR_DATASET],
                               device,CIFAR_DATASET,modes=("hard",))
            row = dict(scales=list(scales),accuracy=metrics["hard_accuracy"],loss=metrics["hard_loss"])
            rows.append(row)
            cache[scales] = row
            atomic_json(progress,rows)
        row = cache[scales]
        return row["accuracy"],-row["loss"],-sum(abs(s-1) for s in scales)

    global_scales = max(((s,s) for s in cfg.calibration_grid),key=score)
    scales = list(global_scales)
    for _ in range(cfg.calibration_rounds):
        for layer in range(2):
            scales = list(max((tuple(s if j == layer else scales[j] for j in range(2))
                               for s in cfg.calibration_grid),key=score))
    choices = {"global":global_scales,"per_layer":tuple(scales)}
    family = max(choices,key=lambda k:score(choices[k]))
    chosen = dict(calibration=family,scales=list(choices[family]),
                  validation_hard_accuracy=score(choices[family])[0],
                  validation_hard_loss=-score(choices[family])[1],checkpoint_digest=digest)
    _atomic_csv(directory / "calibration_validation.csv",pd.DataFrame(rows))
    atomic_json(directory / "validation_choice.json",chosen)
    assert state_digest(cpu_state(model)) == digest
    return chosen


def freeze_cifar_radius(cfg,root):
    records=[]
    for job in cifar_manifest(cfg).query("phase == 'pilot'").to_dict("records"):
        directory = cifar_job_directory(root,job)
        if not (directory / "trial_complete.json").exists():
            raise RuntimeError("Every pilot radius/seed must finish before main training.")
        record=json.loads((directory / "pilot_result.json").read_text())
        assert record["seed"] == job["seed"] and record["variant"] == job["variant"]
        records.append(record)
    frame=pd.DataFrame(records)
    assert not frame.duplicated(["seed","rho"]).any()
    ranking=frame.groupby("rho").agg(validation_accuracy=("validation_hard_accuracy","mean"),
                                    validation_loss=("validation_hard_loss","mean"),n_seeds=("seed","count")).reset_index()
    assert set(ranking.rho) == set(cfg.radius_grid) and (ranking.n_seeds == len(cfg.pilot_seeds)).all()
    ranking=ranking.sort_values(["validation_accuracy","validation_loss","rho"],ascending=[False,True,True])
    selected=dict(rho=float(ranking.iloc[0].rho),pilot_seeds=list(cfg.pilot_seeds),
                  criterion="mean calibrated validation hard accuracy; tie: lower mean loss, then smaller rho",
                  pilot_records_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
                  test_used=False,main_training_seeds=list(cfg.seeds))
    path=Path(root) / "selected_radius.json"
    if path.exists() and json.loads(path.read_text()) != selected:
        raise RuntimeError("Frozen pilot radius or its source evidence changed.")
    atomic_json(path,selected)
    _atomic_csv(Path(root) / "pilot_results.csv",frame)
    _atomic_csv(Path(root) / "pilot_radius_summary.csv",ranking)
    return selected["rho"]
