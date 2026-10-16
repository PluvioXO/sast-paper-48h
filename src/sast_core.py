# %% Imports and configuration
"""Executable source embedded, section by section, in sast_evidence.ipynb."""
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import stats
from torch import nn
from torch.nn import functional as F

ENGINE_VERSION = "1.0"
ENGINE_DIGEST = globals().get("ENGINE_DIGEST", "embedded")
if "__file__" in globals():
    ENGINE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
METHODS = ("baseline", "compute_matched", "sast")
PAPER_URL = "https://arxiv.org/html/2603.18039v1"


@dataclass
class Config:
    profile: str = "evidence"
    synthetic: bool = False
    datasets: tuple = ("nmnist", "dvsgesture")
    seeds: tuple = (0, 1, 2)
    data_dir: str = "data"
    output_dir: str = "outputs"
    device: str = "auto"
    time_bins: int = 16
    channels: tuple = (16, 32)
    leak: float = 0.5
    threshold: float = 1.0
    slope: float = 1.0
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    sam_delta: float = 1e-12
    epochs: dict = field(default_factory=lambda: {"nmnist": 20, "dvsgesture": 50})
    batch_sizes: dict = field(default_factory=lambda: {"nmnist": 32, "dvsgesture": 8})
    radii: dict = field(default_factory=lambda: {"nmnist": 0.30, "dvsgesture": 0.40})
    validation_fraction: float = 0.1
    split_seed: int = 2026
    corruption_seed: int = 314159
    severities: tuple = (0., 0.1, 0.2, 0.3, 0.4)
    smooth_tolerance: float = 0.02
    # Only the smoke profile limits samples or updates.
    steps_per_epoch: int | None = None
    train_per_class: int | None = None
    eval_per_class: int | None = None
    cache_chunk: int = 256
    benchmark_steps: int = 5
    log_every: int = 50

    def validate(self):
        assert self.profile in ("evidence", "smoke")
        assert set(self.datasets) <= {"nmnist", "dvsgesture"} and self.datasets
        assert len(set(self.seeds)) == len(self.seeds) and self.seeds
        assert 0 < self.validation_fraction < 1
        assert self.time_bins > 0 and len(self.channels) == 2
        assert all(c > 0 for c in self.channels)
        assert 0 < self.leak < 1 and self.threshold > 0 and self.slope > 0
        assert 0 < self.min_learning_rate <= self.learning_rate
        assert self.sam_delta > 0 and self.smooth_tolerance >= 0
        assert self.cache_chunk > 0 and self.benchmark_steps > 0
        assert tuple(sorted(set(self.severities))) == self.severities
        assert self.severities[0] == 0 and self.severities[-1] <= 1
        for name in self.datasets:
            assert self.epochs[name] > 0 and self.batch_sizes[name] > 0
            assert self.radii[name] > 0
        if self.profile == "evidence":
            assert not self.synthetic, "Synthetic data cannot produce scientific evidence."
            assert self.steps_per_epoch is None
            assert self.train_per_class is None and self.eval_per_class is None
        return self


def smoke_config(cfg=None, synthetic=True):
    return replace(
        cfg or Config(), profile="smoke", synthetic=synthetic, seeds=(0,),
        time_bins=4, channels=(4, 8), epochs={"nmnist": 1, "dvsgesture": 1},
        batch_sizes={"nmnist": 4, "dvsgesture": 4}, steps_per_epoch=2,
        train_per_class=4, eval_per_class=2, benchmark_steps=2, log_every=1,
    )


def stable_seed(*parts):
    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def choose_device(requested="auto"):
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use device='cpu'.")
    return torch.device(requested)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def rng_state(device):
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state()}
    if device.type == "mps":
        result["device"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        result["device"] = torch.cuda.get_rng_state()
    return result


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "mps":
        torch.mps.set_rng_state(state["device"].cpu())
    elif device.type == "cuda":
        torch.cuda.set_rng_state(state["device"].cpu())


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def environment_info(device):
    packages = {}
    for name in ("torch", "tonic", "numpy", "scipy", "pandas", "matplotlib"):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": sys.version, "platform": platform.platform(),
        "device": str(device), "packages": packages,
        "memory_bytes": psutil.virtual_memory().total,
        "mps_available": torch.backends.mps.is_available(),
        "engine_digest": ENGINE_DIGEST,
        "reproducibility": "Seeds and RNG states saved; MPS bitwise determinism is not guaranteed.",
    }


def study_directory(cfg):
    spec = asdict(cfg)
    spec.pop("output_dir")
    spec.pop("data_dir")
    digest = hashlib.sha256(json.dumps(
        {"config": spec, "engine": ENGINE_DIGEST}, sort_keys=True).encode()).hexdigest()[:12]
    return Path(cfg.output_dir) / cfg.profile / digest


# %% Event data and resumable frame caches
def dataset_shape(name):
    return (34, 10) if name == "nmnist" else (64, 11)


def events_to_frames(events, name, bins):
    """Equal-duration bins spanning each recording; the final event stays in the last bin."""
    size, _ = dataset_shape(name)
    source_size = 34 if name == "nmnist" else 128
    frames = np.zeros((bins, 2, size, size), dtype=np.float32)
    if len(events) == 0:
        return frames
    x = np.asarray(events["x"], dtype=np.int64) * size // source_size
    y = np.asarray(events["y"], dtype=np.int64) * size // source_size
    p = np.asarray(events["p"], dtype=np.int64)
    t = np.asarray(events["t"], dtype=np.float64)
    if not (np.isfinite(t).all() and ((p == 0) | (p == 1)).all()):
        raise ValueError("Invalid timestamps or polarities.")
    if not (((x >= 0) & (x < size) & (y >= 0) & (y < size)).all()):
        raise ValueError("Event coordinates outside the sensor.")
    duration = t.max() - t.min()
    ti = np.zeros(len(t), dtype=np.int64) if duration == 0 else np.minimum(
        ((t - t.min()) * (bins / duration)).astype(np.int64), bins - 1)
    np.add.at(frames, (ti, p, y, x), 1)
    maxima = frames.max(axis=(0, 2, 3), keepdims=True)
    frames /= np.maximum(maxima, 1)
    return frames


def stratified_split(labels, fraction, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    labels = np.asarray(labels)
    for label in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == label))
        if len(ids) < 2:
            raise ValueError("Each class needs at least two training examples.")
        n = min(len(ids) - 1, max(1, int(round(len(ids) * fraction))))
        validation.extend(ids[:n])
        train.extend(ids[n:])
    return np.sort(train), np.sort(validation)


def select_per_class(ids, labels, limit):
    if limit is None:
        return np.asarray(ids, dtype=np.int64)
    return np.sort(np.concatenate([
        np.asarray(ids)[np.asarray(labels)[ids] == label][:limit]
        for label in np.unique(np.asarray(labels)[ids])
    ]))


@dataclass
class FrameSet:
    frames: np.ndarray
    labels: np.ndarray
    ids: np.ndarray

    def __len__(self):
        return len(self.labels)

    def batch(self, indices, device):
        # Advanced indexing copies only this batch; the full cache stays on disk.
        return (torch.from_numpy(np.asarray(self.frames[indices], dtype=np.float32)).to(device),
                torch.as_tensor(np.asarray(self.labels[indices]), dtype=torch.long, device=device))


def cache_frames(raw, ids, labels, cfg, name, split, cache_dir):
    size, _ = dataset_shape(name)
    path = cache_dir / f"{split}.npy"
    progress = cache_dir / f"{split}_progress.json"
    shape = (len(ids), cfg.time_bins, 2, size, size)
    done = json.loads(progress.read_text())["completed"] if progress.exists() else 0
    if done and not path.exists():
        raise RuntimeError(f"Incomplete cache metadata: {path}")
    if path.exists():
        frames = np.lib.format.open_memmap(path, mode="r+")
        if frames.shape != shape or frames.dtype != np.float32:
            raise RuntimeError(f"Cache shape/dtype mismatch: {path}")
    else:
        required = math.prod(shape) * 4
        if psutil.disk_usage(str(cache_dir)).free < required + 2 * 1024**3:
            raise RuntimeError(f"Insufficient space for {required / 1024**3:.1f} GiB frame cache.")
        frames = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(done, len(ids), cfg.cache_chunk):
        end = min(start + cfg.cache_chunk, len(ids))
        for j in range(start, end):
            events, label = raw[int(ids[j])]
            if int(label) != int(labels[ids[j]]):
                raise RuntimeError("Dataset label/order changed during caching.")
            frames[j] = events_to_frames(events, name, cfg.time_bins)
        frames.flush()
        atomic_json(progress, {"completed": end, "total": len(ids)})
        if start == done or end == len(ids) or end % (cfg.cache_chunk * 10) == 0:
            print(f"Cache {name}/{split}: {end}/{len(ids)}", flush=True)
    del frames
    return FrameSet(np.load(path, mmap_mode="r"), np.asarray(labels)[ids], ids)


def prepare_dataset(cfg, name):
    """Downloads official partitions through Tonic; no augmentation or test-derived normalization."""
    cfg.validate()
    if cfg.synthetic:
        result = {}
        size, classes = dataset_shape(name)
        for split, per_class in (("train", 4), ("validation", 2), ("test", 2)):
            rng = np.random.default_rng(stable_seed("synthetic", name, split))
            labels = np.repeat(np.arange(classes), per_class)
            frames = rng.binomial(1, 0.03, (len(labels), cfg.time_bins, 2, size, size)).astype(np.float32)
            for i, label in enumerate(labels):
                frames[i, :, label % 2, 2 + label:4 + label, 2:6] = 1
            result[split] = FrameSet(frames, labels, np.arange(len(labels)))
        return result

    import tonic
    cls = tonic.datasets.NMNIST if name == "nmnist" else tonic.datasets.DVSGesture
    # Figshare documents this public API download route. Its HTML endpoint may
    # return an empty 202 response to automated clients. Preserve Tonic's MD5 checks.
    original_urls = cls.train_url, cls.test_url
    try:
        if name == "dvsgesture":
            cls.train_url = cls.train_url.replace(
                "https://figshare.com/ndownloader/files/", "https://api.figshare.com/v2/file/download/")
            cls.test_url = cls.test_url.replace(
                "https://figshare.com/ndownloader/files/", "https://api.figshare.com/v2/file/download/")
        expected_sizes = (60000, 10000) if name == "nmnist" else (1077, 264)
        partitions = []
        for train, expected in zip((True, False), expected_sizes):
            raw = cls(save_to=str(Path(cfg.data_dir) / "raw"), train=train)
            # Tonic's existence check is permissive. Repair an interrupted extraction
            # from the checksum-verified archive before accepting the partition.
            if len(raw) != expected:
                raw.download()
                raw = cls(save_to=str(Path(cfg.data_dir) / "raw"), train=train)
            if len(raw) != expected:
                raise ValueError(f"Incomplete {name} partition: expected {expected}, found {len(raw)}.")
            partitions.append(raw)
        raw_train, raw_test = partitions
    finally:
        cls.train_url, cls.test_url = original_urls
    # Some Tonic releases leave DVS Gesture directory enumeration unsorted.
    for raw in (raw_train, raw_test):
        order = sorted(range(len(raw.data)), key=lambda i: str(raw.data[i]))
        for attr in ("data", "targets", "users", "lighting"):
            values = getattr(raw, attr, None)
            if values is not None and len(values) == len(order):
                setattr(raw, attr, [values[i] for i in order])
    train_labels = np.asarray(raw_train.targets, dtype=np.int64)
    test_labels = np.asarray(raw_test.targets, dtype=np.int64)
    _, classes = dataset_shape(name)
    if set(np.unique(train_labels)) != set(range(classes)) or set(np.unique(test_labels)) != set(range(classes)):
        raise ValueError(f"Unexpected class labels or incomplete dataset: {name}")
    train_ids, val_ids = stratified_split(train_labels, cfg.validation_fraction, cfg.split_seed)
    selected = {
        "train": select_per_class(train_ids, train_labels, cfg.train_per_class),
        "validation": select_per_class(val_ids, train_labels, cfg.eval_per_class),
        "test": select_per_class(np.arange(len(test_labels)), test_labels, cfg.eval_per_class),
    }
    # Include raw ordering, dataset version, and preprocessing in the cache identity.
    raw_fingerprint = hashlib.sha256(json.dumps({
        "train": [str(p) for p in raw_train.data], "test": [str(p) for p in raw_test.data],
        "train_labels": train_labels.tolist(), "test_labels": test_labels.tolist(),
    }, sort_keys=True).encode()).hexdigest()
    spec = {"dataset": name, "tonic": tonic.__version__, "time_bins": cfg.time_bins,
            "preprocessing": "equal_duration_per_channel_max_v1", "raw": raw_fingerprint,
            "partitions": {k: v.tolist() for k, v in selected.items()}}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = Path(cfg.data_dir) / "frames" / name / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(cache_dir / "manifest.json", spec)
    assert not np.intersect1d(selected["train"], selected["validation"]).size
    result = {}
    for split, ids in selected.items():
        raw = raw_test if split == "test" else raw_train
        labels = test_labels if split == "test" else train_labels
        result[split] = cache_frames(raw, ids, labels, cfg, name, split, cache_dir)
    result["manifest"] = str(cache_dir / "manifest.json")
    return result


# %% Smooth-forward LIF network
class SurrogateSNN(nn.Module):
    def __init__(self, cfg, name):
        super().__init__()
        size, classes = dataset_shape(name)
        self.leak = cfg.leak
        self.slope = cfg.slope
        self.register_buffer("threshold", torch.tensor(cfg.threshold, dtype=torch.float32))
        c1, c2 = cfg.channels
        self.convs = nn.ModuleList([
            nn.Conv2d(2, c1, 3, padding=1), nn.Conv2d(c1, c2, 3, padding=1)])
        self.readout = nn.Linear(c2 * (size // 4)**2, classes)

    def forward(self, x, hard=False):
        # x: [batch, time, polarity, height, width].
        # All states are local variables: every sequence/SAM pass starts at zero.
        states = [None, None]
        spikes = [None, None]
        total = None
        for t in range(x.shape[1]):
            z = x[:, t]
            for layer, conv in enumerate(self.convs):
                drive = F.avg_pool2d(conv(z), 2)
                if states[layer] is None:
                    u = drive
                else:
                    u = self.leak * states[layer] + drive - self.threshold * spikes[layer]
                distance = u - self.threshold
                z = (distance >= 0).to(x.dtype) if hard else (
                    0.5 + torch.atan(self.slope * distance) / math.pi)
                states[layer], spikes[layer] = u, z
            total = z if total is None else total + z
        return self.readout((total / x.shape[1]).flatten(1))


def make_model(cfg, name, seed, device):
    seed_everything(seed)
    # Initialize on CPU for identical weights independent of device.
    return SurrogateSNN(cfg, name).to(device)


class IndependentBatches:
    """IID index draws with replacement from separate ascent and descent RNG streams.

    Independent batches can overlap by chance; forcing disjointness would couple them.
    """
    def __init__(self, n, batch_size, seed, name):
        self.n, self.batch_size = n, batch_size
        self.ascent = torch.Generator().manual_seed(stable_seed(seed, name, "ascent"))
        self.descent = torch.Generator().manual_seed(stable_seed(seed, name, "descent"))

    def next(self, sam):
        a = torch.randint(self.n, (self.batch_size,), generator=self.ascent).numpy() if sam else None
        b = torch.randint(self.n, (self.batch_size,), generator=self.descent).numpy()
        return a, b

    def state_dict(self):
        return {"ascent": self.ascent.get_state(), "descent": self.descent.get_state()}

    def load_state_dict(self, state):
        self.ascent.set_state(state["ascent"].cpu())
        self.descent.set_state(state["descent"].cpu())


# %% Optimization and clean evaluation
def gradient_norm(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("Non-finite gradient; no optimizer update was applied.")
    return norm


def training_step(model, optimizer, descent_batch, ascent_batch=None, rho=0., delta=1e-12,
                  memory_callback=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if rho > 0:
        if ascent_batch is None:
            raise ValueError("SAST needs an independent ascent minibatch.")
        loss_first = F.cross_entropy(model(ascent_batch[0]), ascent_batch[1])
        if not bool(torch.isfinite(loss_first)):
            raise FloatingPointError("Non-finite ascent loss.")
        loss_first.backward()
        norm = gradient_norm(model)
        originals = [(p, p.detach().clone()) for p in model.parameters()]
        try:
            with torch.no_grad():
                for p, _ in originals:
                    if p.grad is not None:
                        p.add_(p.grad * (rho / (norm + delta)))
            optimizer.zero_grad(set_to_none=True)
            logits = model(descent_batch[0])
            loss = F.cross_entropy(logits, descent_batch[1])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite perturbed loss.")
            loss.backward()
            gradient_norm(model)
            if memory_callback:
                memory_callback()
        finally:
            # Restore even if a forward/backward pass is interrupted.
            with torch.no_grad():
                for p, old in originals:
                    p.copy_(old)
        evaluations = 2
    else:
        logits = model(descent_batch[0])
        loss = F.cross_entropy(logits, descent_batch[1])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite baseline loss.")
        loss.backward()
        gradient_norm(model)
        if memory_callback:
            memory_callback()
        evaluations = 1
    optimizer.step()
    return float(loss.detach()), evaluations


def corruption(x, ids, name, family, severity, seed):
    if family == "clean" or severity == 0:
        return x
    masks = []
    for index in ids:
        generator = torch.Generator().manual_seed(
            stable_seed(seed, name, family, int(index)))
        shape = x.shape[1:] if family == "voxel_drop" else (x.shape[1], 1, 1, 1)
        if family not in ("voxel_drop", "time_bin_drop"):
            raise ValueError(f"Unknown corruption: {family}")
        masks.append(torch.rand(shape, generator=generator) >= severity)
    return x * torch.stack(masks).to(device=x.device, dtype=x.dtype)


@torch.no_grad()
def evaluate(model, data, batch_size, device, name, family="clean", severity=0., seed=0,
             modes=("smooth", "hard")):
    model.eval()
    totals = {mode: {"correct": 0, "loss_sum": 0.} for mode in modes}
    predictions = {mode: [] for mode in modes}
    for start in range(0, len(data), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data)))
        x, y = data.batch(indices, device)
        x = corruption(x, data.ids[indices], name, family, severity, seed)
        for mode in modes:
            logits = model(x, hard=(mode == "hard"))
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Non-finite evaluation logits.")
            pred = logits.argmax(1)
            totals[mode]["correct"] += int((pred == y).sum())
            totals[mode]["loss_sum"] += float(F.cross_entropy(logits, y, reduction="sum"))
            predictions[mode].extend(pred.cpu().tolist())
    result = {"n": len(data)}
    for mode in modes:
        result[f"{mode}_accuracy"] = totals[mode]["correct"] / len(data)
        result[f"{mode}_loss"] = totals[mode]["loss_sum"] / len(data)
        result[f"{mode}_predictions"] = predictions[mode]
    if len(modes) == 2:
        result["transfer_gap"] = result["smooth_accuracy"] - result["hard_accuracy"]
    return result


class MemoryMeter:
    """Sampled process RSS and MPS allocator sizes; these are not CUDA peak-memory statistics."""
    def __init__(self, device):
        self.device = device
        self.values = {"sampled_peak_rss_bytes": 0, "sampled_peak_mps_allocated_bytes": 0,
                       "sampled_peak_mps_driver_bytes": 0}

    def sample(self):
        self.values["sampled_peak_rss_bytes"] = max(
            self.values["sampled_peak_rss_bytes"], psutil.Process().memory_info().rss)
        if self.device.type == "mps":
            for key, value in (
                ("sampled_peak_mps_allocated_bytes", torch.mps.current_allocated_memory()),
                ("sampled_peak_mps_driver_bytes", torch.mps.driver_allocated_memory()),
            ):
                self.values[key] = max(self.values[key], value)


def run_budget(cfg, name, n_train, method):
    steps = cfg.steps_per_epoch or math.ceil(n_train / cfg.batch_sizes[name])
    epochs = cfg.epochs[name] * (2 if method == "compute_matched" else 1)
    return steps, epochs, steps * epochs * (2 if method == "sast" else 1)


def checkpoint_spec(cfg, name, seed, method, data, device):
    return {
        "config": asdict(cfg), "dataset": name, "seed": seed, "method": method,
        "device": str(device), "engine_digest": ENGINE_DIGEST,
        "data_manifest": data.get("manifest", "synthetic"),
        "partition_sizes": {k: len(data[k]) for k in ("train", "validation", "test")},
    }


def train_run(cfg, name, seed, method, data, directory, device, max_epochs_this_call=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = checkpoint_spec(cfg, name, seed, method, data, device)
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    last_path = directory / "last.pt"
    model = make_model(cfg, name, seed, device)
    initial_digest = state_digest(cpu_state(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay, foreach=False)
    steps, epochs, expected_passes = run_budget(cfg, name, len(data["train"]), method)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps, eta_min=cfg.min_learning_rate)
    batches = IndependentBatches(len(data["train"]), cfg.batch_sizes[name], seed, name)
    history, start_epoch, passes, train_seconds = [], 0, 0, 0.
    best_score, best_state, best_epoch = (-math.inf, -math.inf), None, None
    meter = MemoryMeter(device)
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise RuntimeError(f"Configuration changed; refusing incompatible resume: {directory}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        batches.load_state_dict(saved["batches"])
        restore_rng(saved["rng"], device)
        history, start_epoch = saved["history"], saved["epoch"]
        passes, train_seconds = saved["passes"], saved["train_seconds"]
        best_score, best_state, best_epoch = saved["best_score"], saved["best_state"], saved["best_epoch"]
        meter.values = saved["memory"]
        print(f"Resume {name}/{method}/seed{seed} at epoch {start_epoch}/{epochs}", flush=True)
    else:
        atomic_json(directory / "config.json", spec)
    stop = epochs if max_epochs_this_call is None else min(epochs, start_epoch + max_epochs_this_call)
    for epoch in range(start_epoch, stop):
        synchronize(device)
        started = time.perf_counter()
        losses = []
        for step in range(steps):
            a, b = batches.next(sam=(method == "sast"))
            ascent = data["train"].batch(a, device) if a is not None else None
            descent = data["train"].batch(b, device)
            loss, used = training_step(
                model, optimizer, descent, ascent,
                rho=cfg.radii[name] if method == "sast" else 0.,
                delta=cfg.sam_delta, memory_callback=meter.sample)
            scheduler.step()
            losses.append(loss)
            passes += used
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                print(f"{name}/{method}/s{seed} epoch {epoch+1}/{epochs} "
                      f"step {step+1}/{steps} loss={np.mean(losses):.4f}", flush=True)
        synchronize(device)
        elapsed = time.perf_counter() - started
        train_seconds += elapsed
        validation_started = time.perf_counter()
        validation = evaluate(model, data["validation"], cfg.batch_sizes[name], device, name,
                              modes=("smooth",))
        synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
        score = (validation["smooth_accuracy"], -validation["smooth_loss"])
        if score > best_score:
            best_score, best_state, best_epoch = score, cpu_state(model), epoch + 1
        history.append({
            "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
            "validation_smooth_accuracy": validation["smooth_accuracy"],
            "validation_smooth_loss": validation["smooth_loss"],
            "training_seconds": elapsed, "validation_seconds": validation_seconds,
            "gradient_evaluations": passes, "learning_rate": optimizer.param_groups[0]["lr"],
        })
        saved = {
            "signature": signature, "model": cpu_state(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "batches": batches.state_dict(),
            "rng": rng_state(device), "epoch": epoch + 1, "history": history,
            "passes": passes, "train_seconds": train_seconds, "best_score": best_score,
            "best_state": best_state, "best_epoch": best_epoch, "memory": meter.values,
            "initial_digest": initial_digest,
        }
        atomic_save(last_path, saved)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        atomic_json(directory / "progress.json", {
            "epoch": epoch + 1, "epochs": epochs, "passes": passes,
            "expected_passes": expected_passes, "training_seconds": train_seconds,
            "best_validation_smooth_accuracy": best_score[0],
        })
        print(f"{name}/{method}/s{seed}: epoch {epoch+1}/{epochs}; "
              f"validation={score[0]:.3f}; train={elapsed:.1f}s", flush=True)
    if len(history) < epochs:
        return {"completed": False, "epoch": len(history), "passes": passes}
    assert passes == expected_passes
    model.load_state_dict(best_state)
    atomic_save(directory / "best.pt", {"model": best_state, "epoch": best_epoch, "spec": spec})
    return {
        "completed": True, "model": model, "best_epoch": best_epoch, "passes": passes,
        "training_seconds": train_seconds, "mean_epoch_seconds": train_seconds / epochs,
        "validation_seconds": sum(r["validation_seconds"] for r in history),
        "initial_digest": initial_digest, **meter.values,
    }


# %% Benchmarking and resumable study execution
def benchmark(cfg, name, data, device):
    records = []
    for method in ("baseline", "sast"):
        model = make_model(cfg, name, 987654, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                      weight_decay=cfg.weight_decay, foreach=False)
        batches = IndependentBatches(len(data["train"]), cfg.batch_sizes[name], 987654, name)
        meter = MemoryMeter(device)
        durations = []
        for step in range(cfg.benchmark_steps + 2):
            a, b = batches.next(method == "sast")
            ascent = data["train"].batch(a, device) if a is not None else None
            descent = data["train"].batch(b, device)
            synchronize(device)
            started = time.perf_counter()
            training_step(model, optimizer, descent, ascent,
                          cfg.radii[name] if method == "sast" else 0.,
                          cfg.sam_delta, meter.sample)
            synchronize(device)
            if step >= 2:
                durations.append(time.perf_counter() - started)
        seconds = float(np.mean(durations))
        steps, epochs, _ = run_budget(cfg, name, len(data["train"]), method)
        records.append({"dataset": name, "method": method, "seconds_per_update": seconds,
                        "estimated_training_hours_per_run": seconds * steps * epochs / 3600,
                        **meter.values})
        print(f"Benchmark {name}/{method}: {seconds:.3f}s/update; "
              f"estimated training {seconds*steps*epochs/3600:.2f}h/run "
              "(excludes validation, downloads, preprocessing, and test evaluation)", flush=True)
        del model, optimizer, ascent, descent
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    return records


def evaluate_run(cfg, name, seed, method, training, data, directory, device):
    directory = Path(directory)
    model = training.pop("model")
    weights_before = state_digest(cpu_state(model))
    clean = evaluate(model, data["test"], cfg.batch_sizes[name], device, name)
    np.savez_compressed(
        directory / "test_predictions.npz", ids=data["test"].ids, labels=data["test"].labels,
        smooth=np.asarray(clean.pop("smooth_predictions")),
        hard=np.asarray(clean.pop("hard_predictions")))
    atomic_json(directory / "clean.json", clean)
    progress_path = directory / "robustness_progress.json"
    rows = json.loads(progress_path.read_text()) if progress_path.exists() else []
    completed = {(row["family"], row["severity"]) for row in rows}
    for family in ("voxel_drop", "time_bin_drop"):
        for severity in cfg.severities:
            if (family, severity) in completed:
                continue
            metrics = clean if severity == 0 else evaluate(
                model, data["test"], cfg.batch_sizes[name], device, name,
                family, severity, cfg.corruption_seed)
            rows.append({
                "dataset": name, "seed": seed, "method": method,
                "family": family, "severity": severity,
                "smooth_accuracy": metrics["smooth_accuracy"],
                "hard_accuracy": metrics["hard_accuracy"], "transfer_gap": metrics["transfer_gap"],
            })
            atomic_json(progress_path, rows)
            print(f"Evaluate {name}/{method}/s{seed}: {family} p={severity:.1f}", flush=True)
    if state_digest(cpu_state(model)) != weights_before:
        raise AssertionError("Evaluation changed model parameters or buffers.")
    pd.DataFrame(rows).to_csv(directory / "robustness.csv", index=False)
    result = {
        "dataset": name, "seed": seed, "method": method, "profile": cfg.profile,
        "synthetic": cfg.synthetic, "data_manifest": data.get("manifest", "synthetic"),
        "rho": cfg.radii[name] if method == "sast" else 0.,
        "device": str(device), "selected_state_digest": weights_before,
        **clean, **training,
    }
    result.pop("completed", None)
    atomic_json(directory / "result.json", result)
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


@contextmanager
def study_lock(root):
    """Prevent two processes from writing the same checkpoints."""
    path = Path(root) / ".running.json"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            prior = json.loads(path.read_text())
        except (ValueError, OSError):
            raise RuntimeError(f"Unreadable run lock: {path}. Inspect it before restarting.")
        if psutil.pid_exists(prior["pid"]):
            raise RuntimeError(f"Study is already running in PID {prior['pid']}: {root}")
        path.unlink()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump({"pid": os.getpid(), "started": time.time()}, handle)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def run_study(cfg, benchmark_first=True):
    cfg.validate()
    device = choose_device(cfg.device)
    root = study_directory(cfg)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Study directory: {root.resolve()}\nDevice: {device}; "
          f"planned runs: {len(cfg.datasets)*len(cfg.seeds)*len(METHODS)}", flush=True)
    with study_lock(root):
        atomic_json(root / "config.json", asdict(cfg))
        env_path = root / "environment.json"
        env = environment_info(device)
        if env_path.exists() and json.loads(env_path.read_text()) != env:
            raise RuntimeError("Execution environment changed; use a new output directory.")
        atomic_json(env_path, env)
        if not (root / "report.md").exists():
            (root / "report.md").write_text(
                "# SAST evidence report\n\nStudy is preparing data. Completed runs: 0.\n\n"
                "**No scientific conclusion is available yet.**\n")
        # Prepare both datasets before spending the long training budget.
        prepared = {}
        for name in cfg.datasets:
            atomic_json(root / "status.json", {"phase": "preparing", "dataset": name})
            prepared[name] = prepare_dataset(cfg, name)
        for name in cfg.datasets:
            data = prepared.pop(name)
            bench_path = root / f"benchmark_{name}.json"
            if benchmark_first and not bench_path.exists():
                atomic_json(bench_path, benchmark(cfg, name, data, device))
            for seed in cfg.seeds:
                for method in METHODS:
                    atomic_json(root / "status.json", {
                        "phase": "training_or_evaluation", "dataset": name, "seed": seed, "method": method})
                    directory = root / name / f"seed_{seed}" / method
                    result_path = directory / "result.json"
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        if result["data_manifest"] != data.get("manifest", "synthetic"):
                            raise RuntimeError("Dataset cache changed: use a new output directory.")
                        print(f"Already complete: {name}/{method}/seed{seed}", flush=True)
                    else:
                        training = train_run(cfg, name, seed, method, data, directory, device)
                        result = evaluate_run(cfg, name, seed, method, training, data, directory, device)
                        del training
                    write_analysis(cfg, root)
            del data
            gc.collect()
        atomic_json(root / "completed.json", {"runs": len(cfg.datasets)*len(cfg.seeds)*3,
                                            "completed_at": time.time()})
        atomic_json(root / "status.json", {"phase": "complete", "runs": len(cfg.datasets)*len(cfg.seeds)*3})
    return root


# %% Paired comparisons and evidence reports
def paired_comparisons(results, cfg):
    rows = []
    if results.empty:
        return pd.DataFrame()
    for name in cfg.datasets:
        subset = results[results.dataset == name]
        for control in ("baseline", "compute_matched"):
            sast = subset[subset.method == "sast"].set_index("seed")
            base = subset[subset.method == control].set_index("seed")
            paired = sorted(set(sast.index) & set(base.index) & set(cfg.seeds))
            if not paired:
                continue
            for metric, sign, boundary in (
                ("hard_accuracy", 1, 0.),
                ("transfer_gap", -1, 0.),
                ("smooth_accuracy", 1, -cfg.smooth_tolerance),
            ):
                diffs = sign * (sast.loc[paired, metric] - base.loc[paired, metric]).to_numpy()
                n = len(diffs)
                mean = float(diffs.mean())
                std = float(diffs.std(ddof=1)) if n > 1 else float("nan")
                half_width = float(stats.t.ppf(.975, n - 1) * std / math.sqrt(n)) if n > 1 else float("nan")
                lower, upper = mean - half_width, mean + half_width
                if cfg.profile != "evidence" or cfg.synthetic:
                    verdict = "smoke only"
                elif set(paired) != set(cfg.seeds):
                    verdict = "pending"
                elif n < 3:
                    verdict = "insufficient seeds"
                elif lower > boundary:
                    verdict = "supports"
                elif upper < boundary:
                    verdict = "contradicts"
                else:
                    verdict = "inconclusive"
                rows.append({
                    "dataset": name, "control": control, "metric": metric,
                    "effect": "baseline gap minus SAST gap" if metric == "transfer_gap" else "SAST minus baseline",
                    "n_seeds": n, "mean_difference": mean, "std_difference": std,
                    "ci95_lower": lower, "ci95_upper": upper, "decision_boundary": boundary,
                    "verdict": verdict,
                })
    return pd.DataFrame(rows)


def paper_reference():
    # These values are reference data only and are never used by the training/evidence pipeline.
    return pd.DataFrame([
        ("nmnist", "baseline", 0., .9606, .0033, .6572, .0974),
        ("nmnist", "sast", .30, .9721, .0012, .9473, .0462),
        ("dvsgesture", "baseline", 0., .7502, .0142, .3182, .0732),
        ("dvsgesture", "sast", .40, .7685, .0151, .6327, .0116),
    ], columns=["dataset", "method", "rho", "published_smooth_mean", "published_smooth_std",
                "published_hard_mean", "published_hard_std"]).assign(source=PAPER_URL + "#S5.T2")


def plot_results(results, robustness, histories, root, cfg):
    plot_dir = Path(root) / "figures"
    plot_dir.mkdir(exist_ok=True)
    colors = {"baseline": "#667085", "compute_matched": "#3171C6", "sast": "#16836B"}
    labels = {"baseline": "Baseline", "compute_matched": "2× baseline", "sast": "SAST"}
    suffix = " — SMOKE CHECK ONLY" if cfg.profile == "smoke" else ""

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(len(cfg.datasets), 3, figsize=(12, 3.5*len(cfg.datasets)), squeeze=False)
    for row, name in enumerate(cfg.datasets):
        for col, metric in enumerate(("smooth_accuracy", "hard_accuracy", "transfer_gap")):
            ax = axes[row, col]
            subset = results[results.dataset == name]
            for j, method in enumerate(METHODS):
                values = subset[subset.method == method][metric].to_numpy() * 100
                if not len(values):
                    continue
                sd = values.std(ddof=1) if len(values) > 1 else 0
                ax.errorbar(j, values.mean(), yerr=sd, color=colors[method], fmt="s", capsize=5,
                            markersize=8, linewidth=2)
                ax.scatter(np.full(len(values), j) + np.linspace(-.1, .1, len(values)),
                           values, color=colors[method], s=22, alpha=.65)
            ax.set(xticks=range(3), xticklabels=[labels[m] for m in METHODS],
                   title=f"{name}: {metric.replace('_', ' ')}",
                   ylabel="Percentage points" if metric == "transfer_gap" else "Accuracy (%)")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Measured results: means ± seed SD; dots are individual runs" + suffix, y=1.02)
    save(fig, "clean_results")

    if not robustness.empty:
        fig, axes = plt.subplots(len(cfg.datasets), 2, figsize=(11, 3.8*len(cfg.datasets)), squeeze=False)
        for row, name in enumerate(cfg.datasets):
            for col, family in enumerate(("voxel_drop", "time_bin_drop")):
                ax = axes[row, col]
                for method in METHODS:
                    group = robustness[(robustness.dataset == name) &
                                       (robustness.family == family) & (robustness.method == method)]
                    if group.empty:
                        continue
                    for mode, style in (("smooth", "-"), ("hard", "--")):
                        aggregate = group.groupby("severity")[f"{mode}_accuracy"].agg(["mean", "std"])
                        x = aggregate.index.to_numpy()
                        y = aggregate["mean"].to_numpy() * 100
                        sd = aggregate["std"].fillna(0).to_numpy() * 100
                        ax.plot(x, y, style, color=colors[method], label=f"{labels[method]} ({mode})")
                        ax.fill_between(x, y-sd, y+sd, color=colors[method], alpha=.08)
                ax.set(title=f"{name}: {family.replace('_', ' ')}",
                       xlabel="Drop probability", ylabel="Accuracy (%)", ylim=(0, 100))
                ax.legend(fontsize=7, ncol=2)
                ax.grid(alpha=.2)
        fig.suptitle("Secondary robustness measurements" + suffix, y=1.02)
        save(fig, "robustness")

    if not histories.empty:
        fig, axes = plt.subplots(len(cfg.datasets), 2, figsize=(11, 3.6*len(cfg.datasets)), squeeze=False)
        for row, name in enumerate(cfg.datasets):
            for col, metric in enumerate(("train_loss", "validation_smooth_accuracy")):
                for (method, seed), group in histories[histories.dataset == name].groupby(["method", "seed"]):
                    axes[row, col].plot(group.epoch, group[metric], color=colors[method], alpha=.7,
                                        label=f"{labels[method]} s{seed}")
                axes[row, col].set(title=f"{name}: {metric.replace('_', ' ')}", xlabel="Epoch")
                axes[row, col].legend(fontsize=7)
                axes[row, col].grid(alpha=.2)
        fig.suptitle("Training and checkpoint-selection histories" + suffix, y=1.02)
        save(fig, "learning_curves")


def write_analysis(cfg, root=None):
    root = Path(root or study_directory(cfg))
    results = pd.DataFrame([json.loads(p.read_text()) for p in sorted(root.glob("*/*/*/result.json"))])
    if results.empty:
        print("No completed runs yet. Check per-run progress.json files.", flush=True)
        return {}
    results.to_csv(root / "results.csv", index=False)
    paired = paired_comparisons(results, cfg)
    paired.to_csv(root / "paired_comparisons.csv", index=False)
    summary = results.groupby(["dataset", "method"])[
        ["smooth_accuracy", "hard_accuracy", "transfer_gap", "training_seconds",
         "mean_epoch_seconds", "sampled_peak_rss_bytes",
         "sampled_peak_mps_allocated_bytes", "sampled_peak_mps_driver_bytes"]
    ].agg(["count", "mean", "std"])
    summary.to_csv(root / "summary.csv")
    robustness_parts = [pd.read_csv(p) for p in sorted(root.glob("*/*/*/robustness.csv"))]
    robustness = pd.concat(robustness_parts, ignore_index=True) if robustness_parts else pd.DataFrame()
    robustness.to_csv(root / "robustness.csv", index=False)
    robustness_summary = []
    if not robustness.empty:
        for keys, group in robustness.groupby(["dataset", "method", "seed", "family"]):
            group = group.sort_values("severity")
            for mode in ("smooth", "hard"):
                x = group.severity.to_numpy()
                y = group[f"{mode}_accuracy"].to_numpy()
                robustness_summary.append(dict(
                    zip(("dataset", "method", "seed", "family"), keys),
                    mode=mode, mean_accuracy=float(y.mean()),
                    area_under_curve=float(np.trapz(y, x)),
                    normalized_area=float(np.trapz(y, x) / (x[-1]-x[0])) if x[-1] > x[0] else float(y[0])))
    pd.DataFrame(robustness_summary).to_csv(root / "robustness_summary.csv", index=False)
    histories = []
    for p in sorted(root.glob("*/*/*/history.csv")):
        history = pd.read_csv(p)
        history["dataset"] = p.parent.parent.parent.name
        history["seed"] = int(p.parent.parent.name.removeprefix("seed_"))
        history["method"] = p.parent.name
        histories.append(history)
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    history_df.to_csv(root / "learning_curves.csv", index=False)
    paper_reference().to_csv(root / "published_reference_ONLY.csv", index=False)
    plot_results(results, robustness, history_df, root, cfg)
    expected = len(cfg.datasets) * len(cfg.seeds) * 3
    complete = len(results) == expected
    lines = [
        "# SAST evidence report", "",
        f"Completed runs: **{len(results)}/{expected}**. Profile: **{cfg.profile}**. "
        f"Synthetic inputs: **{cfg.synthetic}**.", "",
        "## Interpretation", "",
        "Effects are paired by training seed. Positive effects favor SAST; the transfer-gap "
        "effect is baseline gap minus SAST gap. Smooth-accuracy preservation uses a "
        f"{100*cfg.smooth_tolerance:g} percentage-point noninferiority margin.",
        "",
        "Intervals are two-sided 95% Student-t intervals across seed differences, with n−1 "
        "degrees of freedom. Three seeds give imprecise intervals and cannot establish a universal "
        "claim. Normality of seed effects is an approximation. These are separate, unadjusted "
        "comparisons; they do not constitute a familywise significance claim.",
        "",
        "Supports: the entire interval exceeds the declared boundary. Contradicts: the entire "
        "interval is below it. Otherwise: inconclusive. Incomplete studies are pending. Smoke "
        "runs produce no scientific verdict.",
        "",
    ]
    if not complete:
        lines += ["**Study incomplete. Final conclusions are pending.**", ""]
    if not paired.empty:
        for row in paired.to_dict("records"):
            lines.append(
                f"- {row['dataset']} / vs {row['control']} / {row['metric']}: "
                f"**{row['verdict']}**; effect {100*row['mean_difference']:+.2f} pp; "
                f"95% CI [{100*row['ci95_lower']:+.2f}, {100*row['ci95_upper']:+.2f}] pp "
                f"(n={row['n_seeds']}).")
    lines += [
        "", "## Limits and provenance", "",
        "This tests the mechanism under a documented implementation. Architecture, temporal "
        "window, preprocessing, fixed thresholds, optimizer and epoch budgets are our choices "
        "because the paper does not fully specify its setup. Failure here does not by itself "
        "invalidate the original experiments. Improvements here do not verify their exact numbers.",
        "",
        "Training compute is matched by forward/backward evaluation counts at equal batch size. "
        "The compute-matched control has twice the optimizer updates and a cosine schedule stretched "
        "over that budget. Validation selects the checkpoint; test accuracy never selects settings. "
        "The selected checkpoint can occur before the final budget, which is still spent in full.",
        "",
        "Memory columns are sampled process RSS and MPS allocator/driver maxima, not hardware "
        "peak-memory measurements. They overlap in unified memory and must not be added together. "
        "Training time includes batch loading and memory instrumentation, but excludes validation, "
        "checkpoint serialization, downloads and test evaluation. Benchmark estimates exclude loading.",
        "",
        "DVS Gesture uses Tonic's presegmented, millisecond-timestamp dataset. Validation is a "
        "stratified recording split within official training data, not a subject-held-out split. "
        "Temporal binning covers each recording's full duration. Corruptions mask normalized "
        "voxels or entire bins without renormalization.",
        "",
        f"[Paper and method]({PAPER_URL}) · "
        "[Tonic DVS Gesture provenance](https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.DVSGesture.html)",
        "",
        "## Artifacts", "",
        "- results.csv: measured per-run metrics.",
        "- paired_comparisons.csv: effects, uncertainty and claim-specific verdicts.",
        "- summary.csv: aggregate metrics and costs.",
        "- robustness.csv / robustness_summary.csv: secondary corruption measurements.",
        "- figures/: PNG and PDF plots.",
        "- published_reference_ONLY.csv: published values, excluded from all verdicts.",
        "- Per-run config.json, last.pt, best.pt, history.csv and test_predictions.npz: audit trail.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    return {"results": results, "paired": paired, "robustness": robustness,
            "summary": summary, "root": root}


# %% Built-in correctness checks
def run_correctness_checks(device="cpu"):
    """Small deterministic checks, also callable directly from the notebook."""
    target = choose_device(device)
    cfg = smoke_config()
    model = make_model(cfg, "nmnist", 123, target)
    x = torch.rand((2, cfg.time_bins, 2, 34, 34), device=target)
    y = torch.tensor([1, 3], device=target)
    before = cpu_state(model)
    out1 = model(x)
    model(x.flip(1))
    torch.testing.assert_close(model(x), out1, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        model(x, hard=True)
    assert state_digest(before) == state_digest(cpu_state(model))
    F.cross_entropy(model(x), y).backward()
    assert gradient_norm(model) > 0
    sampler_a = IndependentBatches(100, 8, 7, "nmnist")
    sampler_b = IndependentBatches(100, 8, 7, "nmnist")
    first, second = sampler_a.next(True)
    _, baseline = sampler_b.next(False)
    assert np.array_equal(second, baseline)
    assert not np.array_equal(first, second)
    ids = np.arange(2)
    corrupted = corruption(x, ids, "nmnist", "voxel_drop", .2, 5)
    torch.testing.assert_close(corrupted, corruption(x, ids, "nmnist", "voxel_drop", .2, 5))
    for name in cfg.datasets:
        b = run_budget(cfg, name, 40, "compute_matched")[2]
        s = run_budget(cfg, name, 40, "sast")[2]
        assert b == s
    # Check the analytic smooth surrogate against finite differences in float64 on CPU.
    v = torch.linspace(-2, 2, 9, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: .5 + torch.atan(a)/math.pi, (v,))
    print(f"Correctness checks passed on {target}: state resets, fixed evaluation weights, "
          "smooth gradients, independent RNG streams, shared corruptions and matched budgets.")
    return True
