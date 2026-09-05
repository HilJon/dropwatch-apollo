"""Long trigger windows using a fixed buffer pool and one sequential disk writer."""

from __future__ import annotations

import os
import queue
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any
from typing import BinaryIO
from uuid import uuid4

import numpy as np

from dropwatch_apollo._capture import _CapturedSequence
from dropwatch_apollo._capture import _foreground_pixels
from dropwatch_apollo._capture import _validate_roi
from dropwatch_apollo.models import ApolloLifecycleError
from dropwatch_apollo.models import ApolloSettings


@dataclass
class _Chunk:
    buffer: np.ndarray
    count: int
    shot: int
    final: bool
    total_shape: tuple[int, int, int]
    release: Callable[[np.ndarray], None]


class _ChunkedCapture:
    """No full-window allocation, waiting on disk, or silent buffer eviction."""

    def __init__(self, settings: ApolloSettings, submit: Callable[[_Chunk], None]) -> None:
        self._settings = settings
        self._submit = submit
        self._max_sequences = 1
        self.reset()

    def reset(self, *, max_sequences: int | None = None) -> None:
        if max_sequences is not None:
            self._max_sequences = max_sequences
        self._pool: queue.Queue[np.ndarray] = queue.Queue()
        self._block: np.ndarray | None = None
        self._history: np.ndarray | None = None
        self._history_pos = self._history_count = self._pos = self._chunk_pos = 0
        self._shot = self._clear = self.trigger_count = 0
        self._armed = False
        self._frame_shape: tuple[int, int] | None = None
        self._dtype: np.dtype[Any] | None = None

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def is_done(self) -> bool:
        return self._shot >= self._max_sequences

    @property
    def is_capturing(self) -> bool:
        return self._pos > 0

    @property
    def frames_remaining(self) -> int:
        return self._settings.max_number_frames - self._pos if self.is_capturing else 0

    def prepare(
        self, frame_shape: tuple[int, int], frame_dtype: np.dtype[Any] | type[Any], *, additional_buffer_bytes: int = 0
    ) -> None:
        if len(frame_shape) != 2 or any(d < 1 for d in frame_shape):
            raise ValueError("expected a positive 2D frame shape")
        _validate_roi(self._settings, frame_shape)
        self._frame_shape, self._dtype = frame_shape, np.dtype(frame_dtype)
        assert self._settings.spool_chunk_frames is not None
        chunk_frames = min(self._settings.spool_chunk_frames, self._settings.max_number_frames)
        count = self._settings.spool_buffer_count
        required = (count * chunk_frames + self._settings.pre_trigger) * prod(frame_shape) * self._dtype.itemsize
        if required + additional_buffer_bytes > self._settings.max_buffer_bytes:
            raise MemoryError("chunk pool, lookback and camera exceed max_buffer_bytes")
        for _ in range(count):
            block = np.empty((chunk_frames, *frame_shape), dtype=self._dtype)
            block.fill(255)  # Fault in the pages before arming, not on first live use.
            self._pool.put_nowait(block)
        self._history = np.empty((self._settings.pre_trigger, *frame_shape), dtype=self._dtype)
        self._history.fill(255)

    def push(self, frame: np.ndarray) -> bool | None:
        if frame.shape != self._frame_shape or frame.dtype != self._dtype:
            raise ValueError("frame shape and dtype must remain constant during acquisition")
        if self.is_done:
            raise RuntimeError("capture already completed")
        pixels = _foreground_pixels(self._settings, frame)
        start = not self.is_capturing and self._armed and pixels > self._settings.trigger_on_pixels
        if start:
            self.trigger_count += 1
            assert self._history is not None
            for i in range(self._history_count):
                self._append(self._history[(self._history_pos + i) % len(self._history)])
        completed = self._append(frame) if start or self.is_capturing else False
        if self._settings.pre_trigger:
            assert self._history is not None
            self._history[self._history_pos] = frame
            self._history_pos = (self._history_pos + 1) % len(self._history)
            self._history_count = min(self._history_count + 1, len(self._history))
        # Always require a clear ROI at startup, including legacy level mode.
        if not (self._settings.trigger_policy == "level" and self._armed):
            if pixels <= self._settings.trigger_off_pixels:
                self._clear += 1
                if (
                    self._clear >= self._settings.rearm_clear_frames
                    and self._history_count == self._settings.pre_trigger
                ):
                    self._armed = True
            else:
                self._clear = 0
                if pixels > self._settings.trigger_on_pixels:
                    self._armed = False
        return True if completed else None

    def _append(self, frame: np.ndarray) -> bool:
        if self._block is None:
            try:
                self._block = self._pool.get_nowait()
            except queue.Empty as exc:
                raise ApolloLifecycleError(
                    "disk writer cannot keep up: chunk buffers exhausted; acquisition stopped"
                ) from exc
        self._block[self._chunk_pos] = frame
        self._chunk_pos += 1
        self._pos += 1
        final = self._pos == self._settings.max_number_frames
        if self._chunk_pos == len(self._block) or final:
            assert self._frame_shape is not None
            task = _Chunk(
                self._block,
                self._chunk_pos,
                self._shot,
                final,
                (self._settings.max_number_frames, *self._frame_shape),
                self._pool.put_nowait,
            )
            self._submit(task)
            self._block = None
            self._chunk_pos = 0
        if final:
            self._shot += 1
            self._pos = 0
        return final


class _ChunkedSpool:
    """Publish only complete, fsynced NPY files; incomplete files are removed."""

    def __init__(self, settings: ApolloSettings) -> None:
        self._settings = settings
        self.directory: Path | None = None
        self.error: Exception | None = None
        self._worker: threading.Thread | None = None
        self._tasks: queue.Queue[_Chunk] = queue.Queue()
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
            raise ApolloLifecycleError("previous chunk writer is still running")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        minimum = required_disk_bytes + 256
        if minimum > self._settings.max_spool_bytes:
            raise OSError("one window exceeds max_spool_bytes; increase the quota before recording")
        if shutil.disk_usage(root).free < minimum + 1024**2:
            raise OSError("not enough disk space for one complete trigger window")
        # Check each window's space when it starts, not a hypothetical session
        # containing triggers in every frame. The session also has a hard quota.
        self.directory = root / f"acquisition_{uuid4().hex}"
        self.directory.mkdir()
        self.error = None
        self._tasks = queue.Queue(maxsize=capacity)
        self._publish = publish
        self._done.clear()
        self._worker = threading.Thread(target=self._run, name="dropwatch-apollo-storage", daemon=True)
        self._worker.start()

    def submit(self, chunk: _Chunk) -> None:
        if self.error is not None:
            raise self.error
        self._tasks.put_nowait(chunk)

    def finish(self, timeout_s: float = 30.0) -> None:
        self._done.set()
        if self._worker is not None:
            self._worker.join(timeout_s)
            if self._worker.is_alive():
                raise ApolloLifecycleError("chunk writer is still finishing; do not restart this instance")
        self._publish = None

    def detach(self) -> None:
        self._publish = None
        self._done.set()

    def release(self) -> None:
        if not self.is_alive:
            self._tasks = queue.Queue()
            self._worker = None
            self._publish = None
            self.error = None

    def _run(self) -> None:
        assert self.directory is not None
        handle: BinaryIO | None = None
        temporary: Path | None = None
        written = used_bytes = 0
        try:
            while True:
                try:
                    chunk = self._tasks.get(timeout=0.01)
                except queue.Empty:
                    if self._done.is_set():
                        return
                    continue
                try:
                    if self.error is not None:
                        continue
                    if handle is None:
                        size = prod(chunk.total_shape) * chunk.buffer.dtype.itemsize + 256
                        if used_bytes + size > self._settings.max_spool_bytes:
                            raise OSError("recording exceeds max_spool_bytes; completed windows are preserved")
                        if shutil.disk_usage(self.directory).free < size + 1024**2:
                            raise OSError("not enough disk space for the next trigger window")
                        used_bytes += size
                        temporary = self.directory / f".shot_{chunk.shot:03d}.part"
                        handle = temporary.open("xb")
                        np.lib.format.write_array_header_2_0(
                            handle,
                            {
                                "descr": np.lib.format.dtype_to_descr(chunk.buffer.dtype),
                                "fortran_order": False,
                                "shape": chunk.total_shape,
                            },
                        )
                        written = 0
                    data = memoryview(chunk.buffer[: chunk.count]).cast("B")
                    if handle.write(data) != data.nbytes:
                        raise OSError("short disk write")
                    del data
                    written += chunk.count
                    if chunk.final:
                        if written != chunk.total_shape[0]:
                            raise OSError("incomplete chunked window")
                        handle.flush()
                        os.fsync(handle.fileno())
                        handle.close()
                        handle = None
                        path = self.directory / f"shot_{chunk.shot:03d}.npy"
                        assert temporary is not None
                        os.replace(temporary, path)
                        temporary = None
                        if self._publish is not None:
                            self._publish(_CapturedSequence(np.load(path, mmap_mode="r", allow_pickle=False)))
                except Exception as exc:
                    self.error = exc.with_traceback(None)
                finally:
                    chunk.release(chunk.buffer)
                    del chunk
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:
                    self.error = self.error or exc.with_traceback(None)
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except Exception as exc:
                    self.error = self.error or exc.with_traceback(None)
