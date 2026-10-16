"""Versioned frame caches and the existing verified SAST dataset adapters."""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SOURCES = ("sast_core.py", "colab_support.py", "replication_experiments.py",
                  "cifar10dvs_download.py", "cifar10dvs_study.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_seed(*values):
    return int(digest(values)[:15], 16)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


@contextmanager
def file_lock(path, blocking=True):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}.") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@lru_cache(maxsize=1)
def legacy_engine():
    # The repository's audited notebook modules intentionally share one namespace.
    module = types.ModuleType("paper_suite_legacy")
    sys.modules[module.__name__] = module
    for name in LEGACY_SOURCES:
        path = ROOT/"src"/name
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def stratified_ids(labels, count, seed):
    labels = np.asarray(labels)
    if count is None or count >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    classes = np.unique(labels)
    if count < len(classes):
        raise ValueError("A diagnostic subset must contain at least one example per class.")
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(labels == c)).tolist() for c in classes]
    selected = []
    for offset in itertools.count():
        for pool in pools:
            if offset < len(pool):
                selected.append(pool[offset])
                if len(selected) == count:
                    return np.sort(np.asarray(selected, dtype=np.int64))


def speaker_split(labels, speakers, seed, count=2):
    labels, speakers = np.asarray(labels), np.asarray(speakers)
    groups = np.unique(speakers)
    candidates = list(itertools.combinations(groups.tolist(), count))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    classes = set(labels.tolist())
    for choice in candidates:
        validation = np.flatnonzero(np.isin(speakers, choice))
        train = np.flatnonzero(~np.isin(speakers, choice))
        if set(labels[train]) == classes and set(labels[validation]) == classes:
            return {"train": train, "validation": validation}
    raise ValueError("No speaker-disjoint validation split covers every class; change validation_speakers.")


def shd_frames(events, bins):
    frames = np.zeros((bins, 700), dtype=np.float32)
    if not len(events):
        return frames
    units = np.asarray(events["x"], dtype=np.int64)
    times = np.asarray(events["t"], dtype=np.float64)
    if not np.isfinite(times).all() or np.any((units < 0)|(units >= 700)):
        raise ValueError("Invalid SHD event times or input channels.")
    duration = times.max()-times.min()
    ids = np.zeros(len(events), dtype=np.int64) if duration == 0 else np.minimum(
        ((times-times.min())*bins/duration).astype(np.int64), bins-1)
    np.add.at(frames, (ids, units), 1)
    frames /= max(float(frames.max()), 1.)
    return frames


def data_key(dataset, bins, split_seed):
    return f"{dataset}_t{bins}_split{split_seed}"


class CachedSplit:
    def __init__(self, info):
        self.info = info
        self.frames = np.load(info["frames"], mmap_mode="r", allow_pickle=False)
        self.labels = np.load(info["labels"], allow_pickle=False)
        self.ids = np.load(info["ids"], allow_pickle=False)
        self.speakers = np.load(info["speakers"], allow_pickle=False) if info.get("speakers") else None
        if self.frames.dtype != np.float32 or len(self.frames) != len(self.labels) or len(self.ids) != len(self.labels):
            raise ValueError("Dataset cache shape/dtype mismatch.")

    def __len__(self):
        return len(self.labels)

    def batch(self, ids, device):
        x = torch.from_numpy(np.array(self.frames[ids], dtype=np.float32, copy=True))
        y = torch.as_tensor(self.labels[ids], dtype=torch.long)
        if device.type == "cuda":
            x, y = x.pin_memory(), y.pin_memory()
        return x.to(device, non_blocking=device.type == "cuda"), y.to(device, non_blocking=device.type == "cuda")


class SubsetSplit:
    def __init__(self, original, indices):
        self.original, self.indices = original, np.asarray(indices)
        self.labels, self.ids = original.labels[self.indices], original.ids[self.indices]
        self.speakers = original.speakers[self.indices] if original.speakers is not None else None

    def __len__(self):
        return len(self.indices)

    def batch(self, ids, device):
        return self.original.batch(self.indices[ids], device)


def load_data(root, key, include_test=False):
    descriptor = json.loads((Path(root)/"datasets"/(key+".json")).read_text())
    names = ("train", "validation", "test") if include_test else ("train", "validation")
    data = {name: CachedSplit(descriptor["splits"][name]) for name in names}
    return data, descriptor


def _cache_split(directory, name, ids, labels, shape, get_frame, speakers=None):
    directory.mkdir(parents=True, exist_ok=True)
    path, progress = directory/(name+".npy"), directory/(name+"_progress.json")
    expected = (len(ids),)+tuple(shape)
    done = json.loads(progress.read_text())["completed"] if progress.exists() else 0
    if done and not path.exists():
        raise RuntimeError("Cache progress exists without its frame file.")
    if not path.exists() and shutil.disk_usage(directory).free < math.prod(expected)*4+2*1024**3:
        raise RuntimeError(f"Insufficient disk space for frame cache {expected}.")
    frames = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
                                     dtype=np.float32, shape=expected)
    if frames.shape != expected or frames.dtype != np.float32 or not 0 <= done <= len(ids):
        raise ValueError("Incompatible partial frame cache.")
    for start in range(done, len(ids), 128):
        end = min(start+128, len(ids))
        for j in range(start, end):
            frames[j] = get_frame(int(ids[j]))
        frames.flush()
        write_json(progress, {"completed": end, "total": len(ids)})
        if start == done or end == len(ids) or end % 4096 == 0:
            print(f"Cache {directory.name}/{name}: {end}/{len(ids)}", flush=True)
    del frames
    paths = {"frames": str(path.resolve()), "labels": str((directory/(name+"_labels.npy")).resolve()),
             "ids": str((directory/(name+"_ids.npy")).resolve())}
    np.save(paths["labels"], np.asarray(labels)[ids], allow_pickle=False)
    np.save(paths["ids"], np.asarray(ids, dtype=np.int64), allow_pickle=False)
    if speakers is not None:
        paths["speakers"] = str((directory/(name+"_speakers.npy")).resolve())
        np.save(paths["speakers"], np.asarray(speakers)[ids], allow_pickle=False)
    paths["sha256"] = {key: file_hash(value) for key, value in paths.items()}
    paths["count"] = len(ids)
    return paths


def prepare_case_data(config, root, dataset, bins, split_seed):
    root, data_root = Path(root), Path(config["data_root"])
    key = data_key(dataset, bins, split_seed)
    destination = root/"datasets"/(key+".json")
    parameters = {k: config[k] for k in ("synthetic", "limit_train_per_class", "limit_eval_per_class", "validation_speakers")}
    parameters.update(dataset=dataset, time_bins=bins, split_seed=split_seed,
                      preprocessing="equal_duration_per_recording_max_v1", version=1)
    parameters["adapter_sha256"] = digest({name: file_hash(ROOT/"src"/name)
        for name in ("paper_suite_data.py",)+LEGACY_SOURCES})
    identity = digest(parameters)
    directory = data_root/"frames"/"paper_suite"/identity[:16]
    with file_lock(data_root/"locks"/(dataset+".lock")):
        shared = directory/"manifest.json"
        if shared.exists():
            descriptor = json.loads(shared.read_text())
            if descriptor["parameters"] != parameters:
                raise ValueError("Dataset cache identity mismatch.")
            for info in descriptor["splits"].values():
                for kind, expected in info["sha256"].items():
                    if file_hash(info[kind]) != expected:
                        raise RuntimeError(f"Dataset cache changed: {info[kind]}")
            write_json(destination, descriptor)
            return descriptor
        if config["synthetic"]:
            descriptor = _synthetic_data(parameters, directory)
        else:
            descriptor = _real_data(config, parameters, directory)
        write_json(shared, descriptor)
        write_json(destination, descriptor)
        return descriptor


def _synthetic_data(parameters, directory):
    dataset, bins = parameters["dataset"], parameters["time_bins"]
    audio, classes = dataset == "shd", 3
    shape = (bins, 8) if audio else (bins, 2, 16, 16)
    result = {"parameters": parameters, "synthetic": True, "classes": classes,
              "spatial_size": None if audio else 16, "input_channels": 8 if audio else 2,
              "splits": {}, "split_scope": "synthetic validation only"}
    for offset, (split, per_class) in enumerate((("train", 4), ("validation", 2), ("test", 2))):
        labels = np.repeat(np.arange(classes), per_class)
        rng = np.random.default_rng(stable_seed("synthetic", dataset, split, parameters["split_seed"]))
        frames = rng.binomial(1, .1, (len(labels),)+shape).astype(np.float32)
        for i, label in enumerate(labels):
            if audio:
                frames[i, :, label] = 1
            else:
                frames[i, :, label % 2, 2+label:4+label, 2:6] = 1
        result["splits"][split] = _cache_split(directory, split, np.arange(len(labels)), labels, shape,
                                                      lambda i, f=frames: f[i])
        if split == "test":
            result["splits"][split]["synthetic_partition_offset"] = offset
    return result


def _real_data(config, parameters, directory):
    import importlib.metadata
    import tonic
    engine = legacy_engine()
    name, bins, seed = parameters["dataset"], parameters["time_bins"], parameters["split_seed"]
    data_root = Path(config["data_root"])
    result = {"parameters": parameters, "synthetic": False, "splits": {},
              "tonic": importlib.metadata.version("tonic")}
    if name == "cifar10dvs":
        cfg = engine.Cifar10DVSConfig(data_dir=str(data_root), output_dir=str(directory), split_seed=seed,
                                     time_bins=bins)
        raw, archive = engine.cifar_raw_dataset(cfg)
        labels = np.asarray(raw.targets, dtype=np.int64)
        parts = engine.cifar_split(labels, cfg)
        sets = {split: raw for split in parts}
        result.update(classes=10, spatial_size=64, input_channels=2, archive=archive,
                      split_scope="declared stratified 80/10/10; not an official partition")
        speakers = None
    elif name in ("nmnist", "dvsgesture"):
        from tonic.download_utils import download_url
        cls = tonic.datasets.NMNIST if name == "nmnist" else tonic.datasets.DVSGesture
        urls = cls.train_url, cls.test_url
        try:
            for split in ("train", "test"):
                setattr(cls, split+"_url", getattr(cls, split+"_url").replace(
                    "https://figshare.com/ndownloader/files/", "https://api.figshare.com/v2/file/download/"))
            partitions = []
            expected = (60000, 10000) if name == "nmnist" else (1077, 264)
            for is_train, count in zip((True, False), expected):
                split = "train" if is_train else "test"
                download_url(getattr(cls, split+"_url"), str(data_root/"raw"/cls.__name__),
                             getattr(cls, split+"_filename"), getattr(cls, split+"_md5"))
                raw = cls(save_to=str(data_root/"raw"), train=is_train)
                if len(raw) != count:
                    raw.download()  # Repair partial extraction from the verified archive.
                    raw = cls(save_to=str(data_root/"raw"), train=is_train)
                partitions.append(raw)
            train, test = partitions
        finally:
            cls.train_url, cls.test_url = urls
        expected = (60000, 10000) if name == "nmnist" else (1077, 264)
        for raw, count in zip((train, test), expected):
            if len(raw) != count:
                raise RuntimeError(f"Incomplete {name} partition: {len(raw)} != {count}.")
            order = sorted(range(len(raw.data)), key=lambda i: str(raw.data[i]))
            for field in ("data", "targets", "users", "lighting"):
                values = getattr(raw, field, None)
                if values is not None and len(values) == len(order):
                    setattr(raw, field, [values[i] for i in order])
        labels = np.asarray(train.targets, dtype=np.int64)
        a, b = engine.stratified_split(labels, .1, seed)
        parts = {"train": a, "validation": b, "test": np.arange(len(test))}
        sets = {"train": train, "validation": train, "test": test}
        result.update(classes=10 if name == "nmnist" else 11, spatial_size=34 if name == "nmnist" else 64,
                      input_channels=2, split_scope="official train/test; stratified validation within official training")
        speakers = None
    elif name == "shd":
        import h5py
        from tonic.download_utils import download_url, extract_archive
        cls = tonic.datasets.SHD
        local = data_root/"raw"/cls.__name__
        for split, expected in (("train", 8156), ("test", 2264)):
            filename = getattr(cls, split+"_zip")
            download_url(cls.base_url+filename, str(local), filename, getattr(cls, split+"_md5"))
            path, valid = local/filename[:-4], False
            try:
                with h5py.File(path, "r") as f:
                    valid = all(len(f[key]) == expected for key in ("labels", "spikes/times", "spikes/units", "extra/speaker")) and len(f["extra/keys"]) == 20
            except (OSError, KeyError):
                pass
            if not valid:
                extract_archive(str(local/filename))
        train, test = tonic.datasets.SHD(save_to=str(data_root/"raw"), train=True), tonic.datasets.SHD(save_to=str(data_root/"raw"), train=False)
        if (len(train), len(test)) != (8156, 2264):
            raise RuntimeError("Incomplete SHD dataset; expected 8156 train and 2264 test examples.")
        with h5py.File(Path(train.location_on_system)/train.data_filename, "r") as f:
            labels = f["labels"][()].astype(np.int64)
        parts = speaker_split(labels, train.speaker, seed, config["validation_speakers"])
        parts["test"] = np.arange(len(test))
        sets = {"train": train, "validation": train, "test": test}
        speakers = {"train": train.speaker, "validation": train.speaker, "test": test.speaker}
        result.update(classes=20, spatial_size=None, input_channels=700,
                      split_scope="official test; speaker-disjoint validation within training",
                      train_speakers=np.unique(train.speaker[parts["train"]]).tolist(),
                      validation_speakers=np.unique(train.speaker[parts["validation"]]).tolist(),
                      unseen_test_speakers=np.setdiff1d(np.unique(test.speaker), np.unique(train.speaker[parts["train"]])).tolist(),
                      official_unseen_test_speakers=np.setdiff1d(np.unique(test.speaker), np.unique(train.speaker)).tolist())
    else:
        raise ValueError(name)
    if result["spatial_size"] is not None:
        shape = (bins, 2, result["spatial_size"], result["spatial_size"])
    else:
        shape = (bins, result["input_channels"])
    for split, ids in parts.items():
        raw = sets[split]
        if name == "shd":
            import h5py
            with h5py.File(Path(raw.location_on_system)/raw.data_filename, "r") as f:
                current_labels = f["labels"][()].astype(np.int64)
        else:
            current_labels = np.asarray(raw.targets, dtype=np.int64)
        if set(current_labels.tolist()) != set(range(result["classes"])):
            raise ValueError("Unexpected dataset label set.")
        limit = config["limit_train_per_class"] if split == "train" else config["limit_eval_per_class"]
        if limit is not None:
            sub = stratified_ids(current_labels[ids], min(len(ids), limit*result["classes"]), stable_seed(seed, split))
            ids = ids[sub]
        def get_frame(i, raw=raw, split=split, labels=current_labels):
            events, label = raw[i]
            if int(label) != int(labels[i]):
                raise ValueError("Dataset label/order changed during preparation.")
            return shd_frames(events, bins) if name == "shd" else engine.events_to_frames(events, name, bins)
        result["splits"][split] = _cache_split(directory, split, ids, current_labels, shape, get_frame,
                                              speakers[split] if speakers else None)
    if np.intersect1d(parts["train"], parts["validation"]).size:
        raise AssertionError("Training and validation partitions overlap.")
    raw_files = {}
    for split, raw in sets.items():
        if name == "shd":
            path = Path(raw.location_on_system)/raw.data_filename
            raw_files[split] = {"filename": path.name, "sha256": file_hash(path)}
        else:
            paths = [Path(p) for p in raw.data]
            base = Path(os.path.commonpath([str(p.parent) for p in paths]))
            raw_files[split] = {"count": len(paths), "sorted_names_and_sizes_sha256": digest(
                sorted((str(p.relative_to(base)), p.stat().st_size) for p in paths))}
    result["raw_files"] = raw_files
    result["raw_identity"] = digest(raw_files)
    return result
