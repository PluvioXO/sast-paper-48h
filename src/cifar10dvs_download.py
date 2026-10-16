# %% Bounded-memory CIFAR10-DVS downloads with persistent chunk recovery
"""I/O-only recovery layer; never changes training or event preprocessing."""
import urllib.error
import urllib.request
import http.client

CIFAR_ARCHIVE_BYTES = 11168490039
CIFAR_DOWNLOAD_CHUNK = 64 * 1024**2
CIFAR_DOWNLOAD_BUFFER = 1024**2


def _cifar_flush_file(handle):
    handle.flush()
    os.fsync(handle.fileno())
    # Release this file's clean page-cache pages where the OS supports it.
    # This is advisory, not a system-wide cache drop or a RAM guarantee.
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _cifar_file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CIFAR_DOWNLOAD_BUFFER), b""):
            digest.update(block)
    return digest.hexdigest()


def _cifar_save_download_manifest(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        _cifar_flush_file(handle)
    temporary.replace(path)


def cifar_data_progress(cfg, stage, **details):
    """Append small snapshots to Drive, so a lost VM does not erase all evidence."""
    directory = Path(cfg.output_dir).parent / "data_preparation"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = dict(time=time.time(), session=COLAB_SESSION_ID, stage=stage,
                    rss_bytes=psutil.Process().memory_info().rss,
                    available_ram_bytes=psutil.virtual_memory().available,
                    local_free_bytes=psutil.disk_usage(cfg.data_dir).free, **details)
    for field_name, filename in (("cgroup_memory_bytes", "memory.current"),
                                 ("cgroup_memory_limit", "memory.max"),
                                 ("cgroup_memory_events", "memory.events")):
        path = Path("/sys/fs/cgroup") / filename
        if path.exists():
            try:
                snapshot[field_name] = path.read_text().strip()
            except OSError:
                pass
    with (directory / f"{COLAB_SESSION_ID}.jsonl").open("a") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        _cifar_flush_file(handle)
    progress = (f" {details['completed_bytes']/1024**3:.2f}/{details['total_bytes']/1024**3:.2f} GiB"
                if "completed_bytes" in details else "")
    print(f"[data] {stage}{progress}; RAM available "
          f"{snapshot['available_ram_bytes']/1024**3:.1f} GiB; local disk free "
          f"{snapshot['local_free_bytes']/1024**3:.1f} GiB", flush=True)


def _cifar_open_range(url, start, end):
    # Request a fresh redirect for EVERY range/retry, never reuse a signed S3 URL.
    # A unique query avoids stale redirects / cached responses for another Range.
    fresh_url = url + ("&" if "?" in url else "?") + "sast_download=" + uuid.uuid4().hex
    request = urllib.request.Request(fresh_url, headers={
        "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
        "Cache-Control": "no-cache", "User-Agent": "sast-cifar10dvs/1.1"})
    return urllib.request.urlopen(request, timeout=30)


def _cifar_receive_range(url, path, start, end, total, progress):
    expected = end - start + 1
    digest = hashlib.sha256()
    with _cifar_open_range(url, start, end) as response:
        wanted = f"bytes {start}-{end}/{total}"
        if response.status != 206 or response.headers.get("Content-Range") != wanted:
            raise RuntimeError("Server did not return the exact requested byte range.")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != expected:
            raise RuntimeError("Server returned an incorrect range length.")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise RuntimeError("Server encoded the range; refusing ambiguous archive bytes.")
        with Path(path).open("wb") as handle:
            remaining, last_log = expected, time.monotonic()
            while remaining:
                block = response.read(min(CIFAR_DOWNLOAD_BUFFER, remaining))
                if not block:
                    raise RuntimeError("Truncated range download; completed chunks are retained.")
                handle.write(block)
                digest.update(block)
                remaining -= len(block)
                if time.monotonic() - last_log >= 10:
                    _cifar_flush_file(handle)
                    progress("download_active", chunk_start=start,
                             received_in_chunk=expected-remaining)
                    last_log = time.monotonic()
            if response.read(1):
                raise RuntimeError("Server returned bytes beyond the requested range.")
            _cifar_flush_file(handle)
    return digest.hexdigest()


def cifar_download_archive(archive, parts_dir, urls, expected_md5,
                           total=CIFAR_ARCHIVE_BYTES, chunk_size=CIFAR_DOWNLOAD_CHUNK,
                           progress=None, attempts=4, retry_delay=2., legacy_backup=None):
    """Persist independently checked ranges, then assemble/MD5-check only locally.

    Part files remain the backup: no second 11-GB monolithic Drive copy is made.
    A dropped VM loses at most the unfinished part, provided Drive has synced it.
    The study's single-writer lease must already be held by the caller.
    """
    progress = progress or (lambda stage, **details: None)
    archive, parts_dir = Path(archive), Path(parts_dir)
    archive.parent.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    if total <= 0 or chunk_size <= 0 or attempts < 1 or not urls:
        raise ValueError("Invalid download dimensions or retry settings.")
    identity = dict(format=1, total_bytes=total, md5=expected_md5,
                    chunk_bytes=chunk_size, urls=list(urls))
    manifest_path = parts_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        identity=identity, chunks={})
    if manifest.get("identity") != identity:
        raise RuntimeError("Download manifest changed; preserve the old cache and use a different folder.")
    _cifar_save_download_manifest(manifest_path, manifest)
    progress("checking_archive", total_bytes=total)
    local_valid = (archive.is_file() and archive.stat().st_size == total
                   and _cifar_file_hash(archive, "md5") == expected_md5)
    source = archive if local_valid else None
    if legacy_backup is not None:
        legacy_backup = Path(legacy_backup)
        if (legacy_backup.is_file() and legacy_backup.stat().st_size == total
                and _cifar_file_hash(legacy_backup, "md5") == expected_md5):
            if local_valid:
                progress("archive_verified", completed_bytes=total, total_bytes=total)
                return archive
            source = legacy_backup
    # Reuse an existing complete monolithic backup without duplicating it in Drive.
    if source is not None and source != archive:
        if psutil.disk_usage(str(archive.parent)).free < total + 2*1024**3:
            raise RuntimeError("Insufficient local disk to restore the dataset archive.")
        progress("restoring_legacy_backup", total_bytes=total)
        temporary = archive.with_name(archive.name + ".assembling")
        with source.open("rb") as reader, temporary.open("wb") as writer:
            copied = 0
            for block in iter(lambda: reader.read(CIFAR_DOWNLOAD_BUFFER), b""):
                writer.write(block)
                copied += len(block)
                if copied % chunk_size == 0:
                    _cifar_flush_file(writer)
                    progress("restoring_legacy_backup", completed_bytes=copied, total_bytes=total)
            _cifar_flush_file(writer)
        if _cifar_file_hash(temporary, "md5") != expected_md5:
            raise RuntimeError("Restored archive checksum mismatch; source backup was preserved.")
        temporary.replace(archive)
        progress("archive_verified", completed_bytes=total, total_bytes=total)
        return archive

    try:
        completed = 0
        for index, start in enumerate(range(0, total, chunk_size)):
            end = min(total, start + chunk_size) - 1
            size = end - start + 1
            part = parts_dir / f"part-{index:05d}.bin"
            saved = manifest["chunks"].get(str(index), {})
            valid = (part.is_file() and part.stat().st_size == size
                     and saved.get("bytes") == size
                     and _cifar_file_hash(part) == saved.get("sha256"))
            if not valid:
                # Account for assembly on this same filesystem when backups are local.
                reserve = 2*1024**3 + (total if parts_dir.stat().st_dev == archive.parent.stat().st_dev
                                      and not local_valid else 0)
                if psutil.disk_usage(str(parts_dir)).free < size + reserve:
                    raise RuntimeError("Insufficient space for download chunks and local assembly.")
                progress("download_chunk_start", chunk=index, completed_bytes=completed, total_bytes=total)
                partial = part.with_suffix(".partial")
                if source is not None:
                    with source.open("rb") as reader, partial.open("wb") as writer:
                        reader.seek(start)
                        remaining = size
                        while remaining:
                            block = reader.read(min(CIFAR_DOWNLOAD_BUFFER, remaining))
                            if not block:
                                raise RuntimeError("Verified source was truncated during copying.")
                            writer.write(block)
                            remaining -= len(block)
                        _cifar_flush_file(writer)
                    digest = _cifar_file_hash(partial)
                else:
                    for attempt in range(attempts):
                        try:
                            digest = _cifar_receive_range(urls[attempt % len(urls)], partial,
                                                         start, end, total, progress)
                            break
                        except (OSError, RuntimeError, ValueError, http.client.HTTPException) as error:
                            # Do not log temporary signed URLs or their credentials.
                            progress("download_retry", chunk=index, attempt=attempt+1,
                                     error_type=type(error).__name__,
                                     http_status=getattr(error, "code", None))
                            if attempt + 1 == attempts:
                                raise RuntimeError(f"Download chunk {index} failed after {attempts} attempts "
                                                   f"({type(error).__name__}); rerun to resume.") from None
                            time.sleep(min(retry_delay * 2**attempt, 8.))
                if partial.stat().st_size != size or _cifar_file_hash(partial) != digest:
                    raise RuntimeError("Stored chunk failed read-back verification; rerun to retry.")
                partial.replace(part)
                manifest["chunks"][str(index)] = dict(bytes=size, sha256=digest)
                _cifar_save_download_manifest(manifest_path, manifest)
            completed += size
            progress("chunk_reused" if valid else "chunk_saved", completed_bytes=completed, total_bytes=total)

        if not local_valid:
            if psutil.disk_usage(str(archive.parent)).free < total + 2*1024**3:
                raise RuntimeError("Insufficient local disk to assemble archive; Drive chunks are retained.")
            temporary = archive.with_name(archive.name + ".assembling")
            digest, assembled = hashlib.md5(), 0
            progress("assembling_archive", completed_bytes=0, total_bytes=total)
            with temporary.open("wb") as writer:
                for index in range(math.ceil(total/chunk_size)):
                    with (parts_dir / f"part-{index:05d}.bin").open("rb") as reader:
                        for block in iter(lambda: reader.read(CIFAR_DOWNLOAD_BUFFER), b""):
                            writer.write(block)
                            digest.update(block)
                            assembled += len(block)
                    _cifar_flush_file(writer)
                    progress("assembling_archive", completed_bytes=assembled, total_bytes=total)
            if assembled != total or digest.hexdigest() != expected_md5:
                # Locally computed chunk hashes detect later corruption, not an
                # incorrect response body. The pinned whole-archive MD5 is final.
                manifest["chunks"] = {}
                _cifar_save_download_manifest(manifest_path, manifest)
                raise RuntimeError("CIFAR10-DVS archive checksum mismatch; chunks will be refetched on rerun. "
                                   "No extraction or training started.")
            if _cifar_file_hash(temporary, "md5") != expected_md5:
                raise RuntimeError("Assembled archive failed read-back checksum verification.")
            temporary.replace(archive)
        progress("archive_verified", completed_bytes=total, total_bytes=total)
        return archive
    except BaseException as error:
        progress("download_interrupted", error_type=type(error).__name__)
        raise
