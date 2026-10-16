# %% Colab GPU execution and persistent recovery
"""Embedded after sast_core.py; keeps the original Mac implementation unchanged."""
import shutil
import threading
import uuid

_TimeLoopSNN = SurrogateSNN
_OriginalFrameSet = FrameSet
_OriginalMemoryMeter = MemoryMeter
_original_prepare_dataset = prepare_dataset
_original_train_run = train_run
_original_write_analysis = write_analysis
_original_run_study = run_study
COLAB_SESSION_ID = uuid.uuid4().hex
ARCHIVE_DIR = None  # Set in the notebook's configuration cell.
LOCK_TIMEOUT_SECONDS = 180
LOCK_REFRESH_SECONDS = 30


def choose_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No NVIDIA GPU detected. Select Runtime > Change runtime type > GPU, "
                           "then reconnect and run the notebook again.")
    return device


def configure_cuda():
    # Preserve float32 arithmetic; do not silently change the scientific setup to AMP/TF32.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class SurrogateSNN(_TimeLoopSNN):
    """Batch time-independent convolutions; retain full recurrent gradients and reset feedback."""
    def forward(self, x, hard=False):
        batch, bins = x.shape[:2]
        z = x
        for conv in self.convs:
            drives = F.avg_pool2d(conv(z.flatten(0, 1)), 2)
            drives = drives.reshape(batch, bins, *drives.shape[1:])
            state, spike, sequence = None, None, []
            for t in range(bins):
                u = drives[:, t] if state is None else (
                    self.leak * state + drives[:, t] - self.threshold * spike)
                distance = u - self.threshold
                spike = (distance >= 0).to(x.dtype) if hard else (
                    0.5 + torch.atan(self.slope * distance) / math.pi)
                state = u
                sequence.append(spike)
            z = torch.stack(sequence, dim=1)
        # Keep the original summation order for temporal averaging.
        total = z[:, 0]
        for t in range(1, bins):
            total = total + z[:, t]
        return self.readout((total / bins).flatten(1))


class FrameSet(_OriginalFrameSet):
    def batch(self, indices, device):
        if device.type != "cuda":
            return super().batch(indices, device)
        x = torch.from_numpy(np.asarray(self.frames[indices], dtype=np.float32)).pin_memory()
        y = torch.as_tensor(np.asarray(self.labels[indices]), dtype=torch.long).pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class MemoryMeter(_OriginalMemoryMeter):
    def __init__(self, device):
        super().__init__(device)
        self.values.update(cuda_peak_allocated_bytes=0, cuda_peak_reserved_bytes=0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def sample(self):
        super().sample()
        if self.device.type == "cuda":
            for key, value in (
                ("cuda_peak_allocated_bytes", torch.cuda.max_memory_allocated(self.device)),
                ("cuda_peak_reserved_bytes", torch.cuda.max_memory_reserved(self.device)),
            ):
                self.values[key] = max(self.values.get(key, 0), value)


def environment_info(device):
    """Resume compatibility: software and arithmetic, independent of a VM's hostname/RAM."""
    return {
        "python": list(platform.python_version_tuple()[:2]),
        "device": str(device),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "tonic", "numpy", "scipy", "pandas", "matplotlib")},
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "precision": "float32; AMP off; TF32 off; deterministic cuDNN; batched-time convolutions",
        "engine_digest": ENGINE_DIGEST,
    }


def session_info(device):
    info = {
        "session_id": COLAB_SESSION_ID, "recorded_at": time.time(),
        "python": sys.version, "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
        "environment": environment_info(device),
        "packages": dict(sorted((dist.metadata["Name"], dist.version)
                                for dist in importlib.metadata.distributions()
                                if dist.metadata.get("Name"))),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(gpu_name=props.name, gpu_total_bytes=props.total_memory,
                    gpu_capability=list(torch.cuda.get_device_capability(device)))
    return info


def file_md5(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "md5").hexdigest()


def copy_verified_archive(source, destination, expected_md5):
    """Only publish a complete, verified archive; never delete the source."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        shutil.copyfile(source, temporary)
        if file_md5(temporary) != expected_md5:
            raise ValueError(f"Archive checksum mismatch: {source}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_dataset(cfg, name):
    if not cfg.synthetic:
        import tonic
        from tonic.download_utils import download_url
        cls = tonic.datasets.NMNIST if name == "nmnist" else tonic.datasets.DVSGesture
        local = Path(cfg.data_dir) / "raw" / cls.__name__
        local.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            filename = getattr(cls, f"{split}_filename")
            md5 = getattr(cls, f"{split}_md5")
            archive = local / filename
            backup = Path(ARCHIVE_DIR) / cls.__name__ / filename if ARCHIVE_DIR else None
            valid_local = archive.exists() and file_md5(archive) == md5
            valid_backup = backup is not None and backup.exists() and file_md5(backup) == md5
            if not valid_local:
                if valid_backup:
                    print(f"Restore archive from Drive: {filename}", flush=True)
                    copy_verified_archive(backup, archive, md5)
                else:
                    url = getattr(cls, f"{split}_url").replace(
                        "https://figshare.com/ndownloader/files/",
                        "https://api.figshare.com/v2/file/download/")
                    download_url(url, str(local), filename, md5)
            if backup is not None and not valid_backup:
                print(f"Save verified archive to Drive: {filename}", flush=True)
                copy_verified_archive(archive, backup, md5)
    return _original_prepare_dataset(cfg, name)


def _read_lock(path):
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"Cannot read study lock {path}; ensure the previous runtime has "
                           "stopped before removing that file.") from exc


@contextmanager
def study_lock(root):
    """Advisory renewable lease across Colab VMs. Use only one active notebook per study."""
    path = Path(root) / ".running.json"
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    host = boot_path.read_text().strip() if boot_path.exists() else platform.node()
    if path.exists():
        prior = _read_lock(path)
        same_host = prior.get("host") == host
        active = (psutil.pid_exists(prior["pid"]) if same_host else
                  time.time() - prior.get("heartbeat", time.time()) < LOCK_TIMEOUT_SECONDS)
        if active:
            raise RuntimeError("This study has an active notebook lease. Stop its other runtime. "
                               "After a lost VM, wait three minutes and rerun this cell.")
        path.unlink()
    owner = uuid.uuid4().hex
    lease = {"owner": owner, "host": host, "pid": os.getpid(), "heartbeat": time.time()}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(lease, handle)
    stopped = threading.Event()
    failures = []

    def refresh():
        while not stopped.wait(LOCK_REFRESH_SECONDS):
            try:
                if _read_lock(path).get("owner") != owner:
                    raise RuntimeError("Study lease ownership changed.")
                lease["heartbeat"] = time.time()
                atomic_json(path, lease)
            except Exception as exc:
                failures.append(exc)
                # Stop writes at the next checkpoint/result boundary if Drive fails.
                break

    thread = threading.Thread(target=refresh, daemon=True)
    thread.start()
    global _ACTIVE_LEASE
    _ACTIVE_LEASE = (path, owner, failures)
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=5)
        _ACTIVE_LEASE = None
        if path.exists() and _read_lock(path).get("owner") == owner:
            path.unlink()


_ACTIVE_LEASE = None
_original_atomic_save = atomic_save


def atomic_save(path, value):
    if _ACTIVE_LEASE is not None:
        lock_path, owner, failures = _ACTIVE_LEASE
        if failures or _read_lock(lock_path).get("owner") != owner:
            raise RuntimeError("Study lease was lost; stopped before writing another checkpoint.")
    _original_atomic_save(path, value)


def train_run(cfg, name, seed, method, data, directory, device, max_epochs_this_call=None):
    directory = Path(directory)
    session_path = directory / "training_sessions.json"
    sessions = json.loads(session_path.read_text()) if session_path.exists() else []
    if COLAB_SESSION_ID not in sessions:
        sessions.append(COLAB_SESSION_ID)
        atomic_json(session_path, sessions)
    result = _original_train_run(cfg, name, seed, method, data, directory, device, max_epochs_this_call)
    result["training_session_ids"] = sessions
    return result


def write_analysis(cfg, root=None):
    root = Path(root or study_directory(cfg))
    tables = _original_write_analysis(cfg, root)
    if tables:
        columns = ["sampled_peak_rss_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"]
        tables["results"].groupby(["dataset", "method"])[columns].agg(
            ["count", "mean", "std"]).to_csv(root / "colab_memory_summary.csv")
        report = root / "report.md"
        text = report.read_text().replace(
            "Memory columns are sampled process RSS and MPS allocator/driver maxima, not hardware "
            "peak-memory measurements. They overlap in unified memory and must not be added together. ",
            "Host memory is sampled process RSS. CUDA columns are PyTorch allocator peak allocated "
            "and reserved bytes during training/validation since each run or resumed segment began "
            "(maxima retained across segments). Reserved includes allocated memory; do not add them. "
            "These exclude non-PyTorch GPU allocations. MPS columns are unused zeros in Colab. ")
        text += ("\n## Colab execution\n\nFloat32 with AMP and TF32 disabled. Convolutions batch the "
                 "time dimension; recurrent dynamics and gradients are unchanged. GPU models and full "
                 "package versions are recorded in sessions/*.json; per-run training_sessions.json "
                 "links resumed segments. Hardware changes may affect timing and numerical results. "
                 "Bitwise equivalence across GPUs is not promised. colab_memory_summary.csv contains "
                 "the CUDA memory aggregates. Mac results are stored separately.\n")
        report.write_text(text)
    return tables


def run_study(cfg, benchmark_first=True):
    device = choose_device(cfg.device)
    if device.type == "cuda":
        configure_cuda()
        torch.cuda.empty_cache()
    root = study_directory(cfg)
    atomic_json(root / "sessions" / f"{COLAB_SESSION_ID}.json", session_info(device))
    return _original_run_study(cfg, benchmark_first)


def run_colab_checks(device):
    """Compare the optimized forward, input gradients, and parameter gradients to the time loop."""
    cfg = smoke_config()
    target = choose_device(device)
    dtype = torch.float64 if target.type == "cpu" else torch.float32
    for name in cfg.datasets:
        seed_everything(42)
        reference = _TimeLoopSNN(cfg, name).to(device=target, dtype=dtype)
        optimized = SurrogateSNN(cfg, name).to(device=target, dtype=dtype)
        optimized.load_state_dict(reference.state_dict())
        size, _ = dataset_shape(name)
        x = torch.rand(2, 4, 2, size, size, device=target, dtype=dtype, requires_grad=True)
        a, b = reference(x), optimized(x)
        tolerance = dict(rtol=2e-4, atol=2e-6) if target.type == "cuda" else dict(rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(a, b, **tolerance)
        ga = torch.autograd.grad(a.square().sum(), (x,) + tuple(reference.parameters()))
        gb = torch.autograd.grad(b.square().sum(), (x,) + tuple(optimized.parameters()))
        for actual, expected in zip(gb, ga):
            torch.testing.assert_close(actual, expected, **tolerance)
        before = state_digest(cpu_state(optimized))
        with torch.no_grad():
            torch.testing.assert_close(reference(x, hard=True), optimized(x, hard=True), **tolerance)
            torch.testing.assert_close(optimized(x), optimized(x), rtol=0, atol=0)
        assert state_digest(cpu_state(optimized)) == before
    print(f"Colab convolution batching checks passed on {target}.")
