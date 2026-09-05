"""Bounded, asynchronous, lossless shot storage. No video encoding on intake."""

from __future__ import annotations

import os
import queue
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import numpy as np

from dropwatch._capture import _CapturedSequence
from dropwatch.models import ApolloLifecycleError


def save_npy(sequence: np.ndarray, path: str | Path) -> Path:
    """Atomically save a pixel-exact sequence, without a full-array copy."""
    output = Path(path)
    if output.suffix.lower() != ".npy":
        raise ValueError("raw sequence path must end in .npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.part")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, sequence, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


class _SequenceSpool:
    """One FIFO writer; publish read-only memmaps only after atomic finalization."""

    def __init__(self) -> None:
        self.directory: Path | None = None
        self.error: Exception | None = None
        self._worker: threading.Thread | None = None
        self._tasks: queue.Queue[_CapturedSequence] = queue.Queue()
        self._done = threading.Event()
        self._publish: Callable[[_CapturedSequence], None] | None = None

    @property
    def is_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(
        self,
        directory: str | Path,
        capacity: int,
        required_disk_bytes: int,
        publish: Callable[[_CapturedSequence], None],
    ) -> None:
        if self.is_alive:
            raise ApolloLifecycleError("previous raw recording writer is still running")
        self.directory = None
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < required_disk_bytes + 1024**2:
            raise OSError(f"not enough disk space: recording needs at least {required_disk_bytes} bytes")
        self.directory = root / f"acquisition_{uuid4().hex}"
        self.directory.mkdir()
        self.error = None
        self._tasks = queue.Queue(maxsize=capacity)
        self._done.clear()
        self._publish = publish
        self._worker = threading.Thread(target=self._run, name="dropwatch-apollo-storage", daemon=True)
        self._worker.start()

    def submit(self, sequence: _CapturedSequence) -> None:
        self._tasks.put_nowait(sequence)

    def finish(self, timeout_s: float = 30.0) -> None:
        self._done.set()
        if self._worker is not None:
            self._worker.join(timeout_s)
            if self._worker.is_alive():
                raise ApolloLifecycleError("raw recording writer is still finishing; do not restart this instance")
        self._publish = None

    def detach(self) -> None:
        """Let a blocked OS write finish, but never publish into a closed recorder."""
        self._publish = None
        self._done.set()

    def release(self) -> None:
        """Release Python-owned results/errors after shutdown; keep files intact."""
        if not self.is_alive:
            self._tasks = queue.Queue()
            self._worker = None
            self._publish = None
            self.error = None

    def _run(self) -> None:
        assert self.directory is not None and self._publish is not None
        shot = 0
        while True:
            try:
                sequence = self._tasks.get(timeout=0.01)
            except queue.Empty:
                if self._done.is_set():
                    return
                continue
            try:
                if self.error is None:
                    path = save_npy(sequence.materialize(), self.directory / f"shot_{shot:03d}.npy")
                    sequence.frames = np.load(path, mmap_mode="r", allow_pickle=False)
                    sequence.lease = None
            except Exception as exc:
                self.error = exc.with_traceback(None)
            # On a disk failure keep the validated in-memory shot recoverable.
            # Its block must never return to the acquisition pool afterwards.
            if self.error is not None and sequence.lease is not None:
                sequence.lease.release = None
            publish = self._publish
            if publish is not None:
                publish(sequence)
            del publish
            del sequence
            shot += 1
