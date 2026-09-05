"""Public lifecycle orchestration for Dropwatch Apollo."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from enum import Enum
from math import prod
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import Literal

import numpy as np
from typing_extensions import Self

from dropwatch._capture import _CapturedSequence
from dropwatch._capture import _find_bottom_object_position
from dropwatch._capture import _SequenceCapture
from dropwatch._evaluation import _EvaluationRunner
from dropwatch._source import _FastEyeApolloSource
from dropwatch._storage import _SequenceSpool
from dropwatch._storage import save_npy
from dropwatch._video import save_png
from dropwatch._video import save_sequence_videos as _save_sequence_videos
from dropwatch._video import save_video as _save_video
from dropwatch.models import ApolloEvaluationError
from dropwatch.models import ApolloFrameLossError
from dropwatch.models import ApolloFrameSource
from dropwatch.models import ApolloIncompleteSequenceError
from dropwatch.models import ApolloLifecycleError
from dropwatch.models import ApolloSequenceEvaluator
from dropwatch.models import ApolloSettings
from dropwatch.models import ApolloStats
from dropwatch.models import ApolloVideoSettings
from dropwatch.models import require_finite
from dropwatch.models import require_integer

logger = logging.getLogger(__name__)


class _ApolloState(Enum):
    CLOSED = "closed"
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DropwatchApollo:
    """Minimal reusable single-view Dropwatch recorder."""

    _POLL_INTERVAL_S = 0.001
    _RESULT_WAIT_INTERVAL_S = 0.05
    _STOP_TIMEOUT_S = 5.0

    def __init__(
        self,
        settings: ApolloSettings,
        *,
        frame_source: ApolloFrameSource | None = None,
        evaluator: ApolloSequenceEvaluator | None = None,
    ) -> None:
        self._settings = settings
        self._source = frame_source if frame_source is not None else _FastEyeApolloSource(settings)
        self._evaluation = _EvaluationRunner(evaluator)
        self._spool = _SequenceSpool()
        self._spooling = False
        self._state = _ApolloState.CLOSED
        self._state_lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._drain_event = threading.Event()
        self._armed_event = threading.Event()
        self._worker_done = threading.Event()
        self._worker: threading.Thread | None = None
        self._worker_error: Exception | None = None
        self._max_sequences = 1
        self._capture = _SequenceCapture(settings)
        self._results: queue.Queue[_CapturedSequence] = queue.Queue(maxsize=1)
        self._frames_received = 0
        self._sequences_captured = 0
        self._frame_gaps = 0
        self._incomplete_sequences = 0
        self._drain_not_before = 0.0
        self._recording_deadline: float | None = None
        self._exporting = False

    @property
    def settings(self) -> ApolloSettings:
        """Immutable acquisition configuration (auto-trigger placement may replace it)."""
        return self._settings

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._state is _ApolloState.RUNNING and self._worker is not None and self._worker.is_alive()

    @property
    def recording_directory(self) -> Path | None:
        """Unique durable raw-shot directory for the latest spooled acquisition."""
        return self._spool.directory if self._spooling else None

    @property
    def stats(self) -> ApolloStats:
        source_stats = getattr(self._source, "diagnostics", None)
        with self._stats_lock:
            return ApolloStats(
                frames_received=self._frames_received,
                sequences_captured=self._sequences_captured,
                frame_gaps=self._frame_gaps,
                daq_reads=int(getattr(source_stats, "daq_reads", 0)),
                zero_byte_reads=int(getattr(source_stats, "zero_byte_reads", 0)),
                recovered_zero_byte_reads=int(getattr(source_stats, "recovered_zero_byte_reads", 0)),
                transport_failures=int(getattr(source_stats, "transport_failures", 0)),
                last_daq_read_ms=float(getattr(source_stats, "last_daq_read_ms", 0.0)),
                max_daq_read_ms=float(getattr(source_stats, "max_daq_read_ms", 0.0)),
                incomplete_sequences=self._incomplete_sequences,
                last_vendor_error=getattr(source_stats, "last_vendor_error", None),
                last_frame_counter=getattr(source_stats, "last_frame_counter", None),
            )

    def open(self) -> None:
        with self._state_lock:
            if self._state is not _ApolloState.CLOSED:
                return
            self._source.open()
            self._state = _ApolloState.IDLE

    def start(self, timeout_s: float = 5.0, *, max_sequences: int = 1, max_duration_s: float | None = None) -> None:
        """Start a bounded acquisition and return only when it is safe to dispense."""
        require_finite("timeout_s", timeout_s)
        require_integer("max_sequences", max_sequences)
        if max_duration_s is not None:
            require_finite("max_duration_s", max_duration_s)
            if max_duration_s <= 0:
                raise ValueError("max_duration_s must be > 0")
        if timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        if max_sequences < 1:
            raise ValueError("max_sequences must be >= 1")

        with self._state_lock:
            if self._exporting or self._spool.is_alive:
                raise ApolloLifecycleError("cannot start while export or raw recording storage is running")
            if self._state is _ApolloState.RUNNING:
                raise ApolloLifecycleError("Apollo acquisition is already running")
            if self._state is _ApolloState.COMPLETED:
                if not self._results.empty():
                    raise ApolloLifecycleError(
                        "collect completed sequences with get_sequences() or stop() before restart"
                    )
                self._worker = None
                self._capture.reset()
                self._state = _ApolloState.IDLE
            if self._state is _ApolloState.FAILED:
                if self._worker is not None and self._worker.is_alive():
                    raise ApolloLifecycleError("the failed Apollo acquisition is still stopping")
                if self._evaluation.is_alive:
                    raise ApolloLifecycleError("evaluations from the failed acquisition are still stopping")
                if not self._results.empty():
                    self._raise_worker_error()
                self._worker = None
                self._worker_error = None
                self._capture.reset()
                self._clear_results()
                self._evaluation.reset()
                self._state = _ApolloState.IDLE
            if self._state is _ApolloState.CLOSED:
                self._source.open()
                self._state = _ApolloState.IDLE

            evaluation_prepared = False
            try:
                self._evaluation.prepare(max_sequences)
                evaluation_prepared = True
                self._clear_results()
                self._max_sequences = max_sequences
                self._results = queue.Queue(maxsize=max_sequences)
                self._capture.reset(max_sequences=max_sequences)
                self._worker_error = None
                self._stop_event.clear()
                self._drain_event.clear()
                self._armed_event.clear()
                self._worker_done.clear()
                with self._stats_lock:
                    self._frames_received = 0
                    self._sequences_captured = 0
                    self._frame_gaps = 0
                    self._incomplete_sequences = 0

                frame_shape = getattr(self._source, "frame_shape", None)
                frame_dtype = getattr(self._source, "frame_dtype", None)
                if frame_shape is None or frame_dtype is None:
                    raise TypeError("Apollo frame sources must declare frame_shape and frame_dtype")
                additional_buffer_bytes = int(getattr(self._source, "reserved_buffer_bytes", 0))
                self._capture.prepare(
                    frame_shape,
                    frame_dtype,
                    additional_buffer_bytes=additional_buffer_bytes,
                )
                self._spooling = self.settings.spool_directory is not None
                if self.settings.spool_directory is not None:
                    self._spool.start(
                        self.settings.spool_directory,
                        self.settings.spool_buffer_count,
                        max_sequences
                        * self.settings.max_number_frames
                        * prod(frame_shape)
                        * np.dtype(frame_dtype).itemsize,
                        self._publish_sequence,
                    )
                self._source.start()
                self._evaluation.start(self._worker_done)
                self._recording_deadline = None if max_duration_s is None else time.monotonic() + max_duration_s
                worker = threading.Thread(
                    target=self._acquisition_loop,
                    name="dropwatch-apollo-acquisition",
                    daemon=False,
                )
                self._worker = worker
                self._state = _ApolloState.RUNNING
                worker.start()
            except Exception:
                self._worker_done.set()
                self._state = _ApolloState.IDLE
                self._worker = None
                self._capture.reset()
                if evaluation_prepared:
                    self._evaluation.finish()
                self._spool.finish()
                try:
                    self._source.stop()
                except Exception:
                    logger.exception("failed to stop Apollo source after an incomplete start")
                raise

        deadline = time.monotonic() + timeout_s
        while not self._armed_event.is_set():
            if self._worker_done.is_set():
                self.stop()
                raise ApolloLifecycleError("Apollo acquisition stopped before it became armed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise TimeoutError(f"Apollo did not become armed after {timeout_s}s")
            self._armed_event.wait(min(self._RESULT_WAIT_INTERVAL_S, remaining))
        self._raise_worker_error()

    def set_trigger_size(self, width: int = 100, timeout_s: float = 5.0) -> int:
        """Place a trigger band directly above the object entering from below.

        Apollo keeps the unrotated first camera view. Its physical bottom is
        therefore the trailing (right) image edge. The highest foreground
        connected to that edge sets the upper boundary of the plate.

        Returns the detected ``trigger_position_px`` measured from the bottom.
        """
        require_integer("width", width)
        require_finite("timeout_s", timeout_s)
        if width < 1:
            raise ValueError("width must be >= 1")
        if timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")

        with self._state_lock:
            if self._exporting or self._spool.is_alive or self._state is _ApolloState.RUNNING:
                raise ApolloLifecycleError("cannot set trigger size while Apollo acquisition is running")
            if self._state is _ApolloState.COMPLETED:
                if not self._results.empty():
                    raise ApolloLifecycleError(
                        "collect completed sequences with get_sequences() or stop() before taking a snapshot"
                    )
                self._worker = None
                self._capture.reset()
                self._state = _ApolloState.IDLE
            if self._state is _ApolloState.FAILED:
                self._raise_worker_error()
            if self._state is _ApolloState.CLOSED:
                self._source.open()
                self._state = _ApolloState.IDLE

            snapshot = self._read_snapshot(timeout_s)
            trigger_position_px = _find_bottom_object_position(snapshot)
            if trigger_position_px + width > snapshot.shape[1]:
                raise ValueError(
                    f"trigger width {width} does not fit above the detected object "
                    f"at x={snapshot.shape[1] - trigger_position_px}"
                )

            self._settings = replace(
                self.settings,
                trigger_position_px=trigger_position_px,
                trigger_from_top=False,
                trigger_width_px=width,
            )
            self._capture = _SequenceCapture(self.settings)
            return trigger_position_px

    def save_avi(self, sequence: np.ndarray, path: str | Path, fps: float = 25.0) -> Path:
        """Save one unannotated sequence with the legacy Apollo AVI polarity."""
        with self._exclusive_io():
            return _save_video(
                sequence,
                path,
                frame_period_ms=self.settings.frame_period_ms,
                pre_trigger=self.settings.pre_trigger,
                options=ApolloVideoSettings(playback_fps=fps, codec="MJPG", annotate=False, invert=True),
            )

    def save_video(
        self,
        sequences: np.ndarray | Iterable[np.ndarray],
        path: str | Path,
        *,
        options: ApolloVideoSettings | None = None,
    ) -> Path:
        """Atomically export one or multiple completed shots to AVI or MP4.

        Encoding runs only after camera intake has stopped, so OpenCV and disk
        I/O cannot delay an RLE read.
        """
        with self._exclusive_io():
            return _save_video(
                sequences,
                path,
                frame_period_ms=self.settings.frame_period_ms,
                pre_trigger=self.settings.pre_trigger,
                options=options or ApolloVideoSettings(),
            )

    def save_videos(
        self,
        sequences: np.ndarray | Iterable[np.ndarray],
        directory: str | Path,
        *,
        prefix: str = "shot",
        suffix: str = ".avi",
        options: ApolloVideoSettings | None = None,
    ) -> list[Path]:
        """Atomically export one independently recoverable video per shot."""
        with self._exclusive_io():
            return _save_sequence_videos(
                sequences,
                directory,
                prefix=prefix,
                suffix=suffix,
                frame_period_ms=self.settings.frame_period_ms,
                pre_trigger=self.settings.pre_trigger,
                options=options or ApolloVideoSettings(),
            )

    def snapshot(self, path: str | Path | None = None, *, timeout_s: float = 5.0) -> np.ndarray:
        """Return the raw left view; optionally save a physical-orientation PNG."""
        require_finite("timeout_s", timeout_s)
        if timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        with self._exclusive_io():
            self.open()
            image = self._read_snapshot(timeout_s)
            if path is not None:
                save_png(image, path)
            return image

    def save_raw(self, sequences: np.ndarray | Iterable[np.ndarray], directory: str | Path) -> list[Path]:
        """Save pixel-exact, unrotated NPY files, one per shot."""
        with self._exclusive_io():
            shots = (sequences,) if isinstance(sequences, np.ndarray) else sequences
            paths = []
            for shot, sequence in enumerate(shots):
                if sequence.ndim != 3 or not len(sequence):
                    raise ValueError("raw sequences must be non-empty 3D arrays")
                paths.append(save_npy(sequence, Path(directory) / f"shot_{shot:03d}.npy"))
            return paths

    def save_frames(self, sequences: np.ndarray | Iterable[np.ndarray], directory: str | Path) -> list[Path]:
        """Export lossless physical-orientation PNG frames."""
        with self._exclusive_io():
            shots = (sequences,) if isinstance(sequences, np.ndarray) else sequences
            paths = []
            for shot, sequence in enumerate(shots):
                for index, frame in enumerate(sequence):
                    paths.append(save_png(frame, Path(directory) / f"shot_{shot:03d}" / f"frame_{index:06d}.png"))
            return paths

    def preview(self, duration_s: float = 10.0, *, display_fps: float = 10.0) -> None:
        """Idle-only live preview; press Escape to close. Requires GUI OpenCV."""
        import cv2

        require_finite("duration_s", duration_s)
        require_finite("display_fps", display_fps)
        if duration_s <= 0 or display_fps <= 0:
            raise ValueError("duration_s and display_fps must be > 0")
        with self._exclusive_io():
            self.open()
            deadline, next_display = time.monotonic() + duration_s, 0.0
            try:
                self._source.start()
                while time.monotonic() < deadline:
                    batch = self._source.read()
                    if batch is None or not len(batch):
                        if getattr(self._source, "exhausted", False):
                            return
                        time.sleep(self._POLL_INTERVAL_S)
                        continue
                    if time.monotonic() >= next_display:
                        image = np.ascontiguousarray((batch[-1] != 0).T, dtype=np.uint8) * 255
                        cv2.imshow("Dropwatch Apollo", image)
                        if cv2.waitKey(1) & 0xFF == 27:
                            return
                        next_display = time.monotonic() + 1 / display_fps
            finally:
                try:
                    self._source.stop()
                finally:
                    cv2.destroyAllWindows()

    def get_sequence(self, timeout_s: float | None = None) -> np.ndarray:
        """Return the next completed sequence without stopping a multi-trigger session."""
        if timeout_s is not None:
            require_finite("timeout_s", timeout_s)
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s

        while True:
            self._raise_worker_error()
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                pass
            else:
                return self._materialize_result(result)

            with self._state_lock:
                if self._state is not _ApolloState.RUNNING:
                    raise ApolloLifecycleError("Apollo acquisition is not running")
            if self._worker_done.is_set():
                self._raise_worker_error()
                raise ApolloLifecycleError("Apollo acquisition stopped before a sequence was available")

            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"no triggered sequence available after {timeout_s}s")
            wait_time = (
                self._RESULT_WAIT_INTERVAL_S if remaining is None else min(self._RESULT_WAIT_INTERVAL_S, remaining)
            )
            try:
                result = self._results.get(timeout=wait_time)
            except queue.Empty:
                continue
            return self._materialize_result(result)

    def get_sequences(self, timeout_s: float | None = None) -> list[np.ndarray]:
        """Wait for the configured limit and return all not-yet-fetched sequences."""
        if timeout_s is not None:
            require_finite("timeout_s", timeout_s)
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s

        while not self._worker_done.is_set():
            self._raise_worker_error()
            with self._state_lock:
                if self._state not in {_ApolloState.RUNNING, _ApolloState.COMPLETED}:
                    raise ApolloLifecycleError("Apollo acquisition is not running")
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(
                    f"only {self.stats.sequences_captured} of {self._max_sequences} "
                    f"sequences were captured after {timeout_s}s"
                )
            wait_time = (
                self._RESULT_WAIT_INTERVAL_S if remaining is None else min(self._RESULT_WAIT_INTERVAL_S, remaining)
            )
            self._worker_done.wait(wait_time)

        return self.stop()

    def get_evaluations(self, timeout_s: float | None = None) -> Any:
        """Return one DataFrame containing all evaluations from the current session."""
        self._evaluation.wait(self._worker_done, timeout_s)
        self._raise_worker_error()
        return self._evaluation.collect()

    def stop(self, *, drain: bool = True, timeout_s: float | None = None) -> list[np.ndarray]:
        """Drain queued/partially flushed batches, then finish any active window.

        The conservative stop boundary includes one RLE flush interval after
        the request and a caught-up source poll. Triggers inside that interval
        are retained. A timeout is explicit even when no window is active.
        Use :meth:`abort` to discard unread/in-progress data intentionally.
        """
        if timeout_s is not None:
            require_finite("timeout_s", timeout_s)
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        drain_timed_out = False
        wait_timeout_s = self._STOP_TIMEOUT_S if timeout_s is None else timeout_s
        with self._state_lock:
            if self._state not in {_ApolloState.RUNNING, _ApolloState.COMPLETED, _ApolloState.FAILED}:
                return self._take_results()
            should_drain = self._state is _ApolloState.RUNNING and drain
            if should_drain:
                self._request_drain()
                if timeout_s is None:
                    remaining_frames = self._capture.frames_remaining or self.settings.max_number_frames
                    remaining_recording_s = remaining_frames * self.settings.frame_period_ms / 1000
                    transfer_margin_s = self.settings.effective_read_timeout_ms / 1000 + 1
                    wait_timeout_s = max(self._STOP_TIMEOUT_S, remaining_recording_s + transfer_margin_s)
            else:
                self._stop_event.set()
            worker = self._worker

        if worker is not None:
            worker.join(timeout=wait_timeout_s)
            if worker.is_alive():
                if should_drain:
                    drain_timed_out = True
                    self._stop_event.set()
                    worker.join(timeout=self._STOP_TIMEOUT_S)
                if worker.is_alive():
                    raise ApolloLifecycleError(f"Apollo acquisition worker did not stop within {self._STOP_TIMEOUT_S}s")

        with self._state_lock:
            self._worker = None
            self._state = _ApolloState.FAILED if self._worker_error is not None else _ApolloState.IDLE
            self._capture.reset()
            self._drain_event.clear()
        if self._worker_error is not None:
            self._raise_worker_error()
        results = self._take_results()
        if drain_timed_out:
            raise ApolloIncompleteSequenceError(
                f"recording could not drain queued frames or finish its triggered sequence within {wait_timeout_s}s",
                results,
            )
        return results

    def abort(self) -> list[np.ndarray]:
        """Request an immediate cutoff; wait for outstanding read/cleanup to finish."""
        return self.stop(drain=False)

    def close(self) -> None:
        stop_error: Exception | None = None
        evaluation_error: Exception | None = None
        try:
            self.stop()
        except Exception as exc:  # noqa: BLE001 - cleanup must continue for every acquisition failure
            stop_error = exc
        with self._state_lock:
            if self._worker is not None and self._worker.is_alive():
                if stop_error is not None:
                    raise stop_error
                raise ApolloLifecycleError("cannot close Apollo while its acquisition worker is still running")
        if self._spool.is_alive:
            try:
                self._spool.finish(timeout_s=self._STOP_TIMEOUT_S)
            except ApolloLifecycleError as exc:
                self._spool.detach()
                stop_error = stop_error or exc
        evaluation_error = self._evaluation.finish()
        self._spool.release()
        with self._state_lock:
            if self._state is not _ApolloState.CLOSED:
                # Retain the source and lifecycle state if vendor close fails,
                # so a second close can retry the same handle.
                self._source.close()
                self._state = _ApolloState.CLOSED
                self._worker = None
                self._worker_error = None
                self._capture.reset()
                self._clear_results()
        if stop_error is not None:
            if evaluation_error is not None:
                logger.error("sequence evaluation also failed during close", exc_info=evaluation_error)
            raise stop_error
        if evaluation_error is not None:
            if isinstance(evaluation_error, ApolloLifecycleError):
                raise evaluation_error
            raise ApolloEvaluationError("sequence evaluation failed") from evaluation_error

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if exc_type is not None:
            try:
                self.close()
            except Exception:
                logger.exception("failed to close Apollo while handling another exception")
            return False
        self.close()
        return False

    def _acquisition_loop(self) -> None:
        finish_active_only = False
        try:
            while not self._stop_event.is_set():
                if self._spooling and self._spool.error is not None:
                    raise self._spool.error
                if self._recording_deadline is not None and time.monotonic() >= self._recording_deadline:
                    self._request_drain()
                batch = self._source.read()
                if batch is None or len(batch) == 0:
                    if getattr(self._source, "exhausted", False):
                        if self._capture.is_capturing:
                            raise ApolloFrameLossError("replay ended inside a triggered sequence")
                        return
                    if self._drain_event.is_set() and time.monotonic() >= self._drain_not_before:
                        if not self._capture.is_capturing:
                            return
                        finish_active_only = True
                    self._stop_event.wait(self._POLL_INTERVAL_S)
                    continue
                if batch.ndim != 3:
                    raise ValueError(f"Apollo frame source returned invalid batch shape {batch.shape}")
                with self._stats_lock:
                    self._frames_received += len(batch)

                for frame in batch:
                    if self._stop_event.is_set():
                        break
                    sequence = self._capture.push(frame)
                    if self._capture.is_armed and not self._armed_event.is_set():
                        self._armed_event.set()
                    if sequence is not None:
                        if self._spooling:
                            self._spool.submit(sequence)
                        else:
                            self._publish_sequence(sequence)
                        with self._stats_lock:
                            self._sequences_captured += 1
                        del sequence
                        if self._capture.is_done or finish_active_only:
                            return
        except Exception as exc:  # noqa: BLE001 - worker failures are propagated to the calling thread
            with self._state_lock:
                self._worker_error = exc.with_traceback(None)
            if isinstance(exc, ApolloFrameLossError):
                with self._stats_lock:
                    self._frame_gaps += 1
        finally:
            try:
                self._source.stop()
            except Exception as exc:
                with self._state_lock:
                    if self._worker_error is None:
                        self._worker_error = exc
                    else:
                        logger.exception("failed to stop Apollo frame source")
            if self._spooling:
                try:
                    self._spool.finish()
                except Exception as exc:
                    self._worker_error = self._worker_error or exc.with_traceback(None)
                self._worker_error = self._worker_error or self._spool.error
            # Release incomplete/unused blocks before signalling completion.
            if self._capture.is_capturing:
                with self._stats_lock:
                    self._incomplete_sequences += 1
            self._capture.reset()
            with self._state_lock:
                if self._state is _ApolloState.RUNNING:
                    self._state = _ApolloState.FAILED if self._worker_error is not None else _ApolloState.COMPLETED
            self._worker_done.set()

    def _read_snapshot(self, timeout_s: float) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        try:
            self._source.start()
            while True:
                batch = self._source.read()
                if batch is not None and len(batch) > 0:
                    if batch.ndim != 3:
                        raise ValueError(f"Apollo frame source returned invalid batch shape {batch.shape}")
                    return batch[-1].copy()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"no Apollo snapshot available after {timeout_s}s")
                time.sleep(self._POLL_INTERVAL_S)
        finally:
            self._source.stop()

    def _clear_results(self) -> None:
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                return

    def _take_results(self) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        while True:
            try:
                results.append(self._results.get_nowait().materialize())
            except queue.Empty:
                return results

    def _raise_worker_error(self) -> None:
        with self._state_lock:
            error = self._worker_error
        if error is not None:
            previous = getattr(error, "completed_sequences", [])
            previous.extend(self._take_results())
            error.completed_sequences = previous  # type: ignore[attr-defined]
            raise error

    def _publish_sequence(self, sequence: _CapturedSequence) -> None:
        self._results.put_nowait(sequence)
        self._evaluation.submit(sequence)

    def _request_drain(self) -> None:
        if not self._drain_event.is_set():
            # A partial FPGA RLE batch may not yet be USB-readable. Allow one
            # flush period, then drain until the source actually catches up.
            self._drain_not_before = (
                time.monotonic() + (self.settings.rle_batch_frames + 1) * self.settings.frame_period_ms / 1000
            )
            self._drain_event.set()

    @contextmanager
    def _exclusive_io(self) -> Iterator[None]:
        with self._state_lock:
            if (
                self._exporting
                or self._spool.is_alive
                or self._state is _ApolloState.RUNNING
                or (self._worker is not None and self._worker.is_alive())
            ):
                raise ApolloLifecycleError("stop Apollo acquisition before exporting video")
            self._exporting = True
        try:
            yield
        finally:
            with self._state_lock:
                self._exporting = False

    def _materialize_result(self, result: _CapturedSequence) -> np.ndarray:
        try:
            if (
                self._capture.is_done or self.stats.sequences_captured >= self._max_sequences
            ) and not self._worker_done.wait(self._STOP_TIMEOUT_S):
                raise ApolloLifecycleError(
                    f"Apollo acquisition did not finalize within {self._STOP_TIMEOUT_S}s after its last sequence"
                )
            self._raise_worker_error()
        except Exception as error:
            # Preserve a complete shot if failure races with its FIFO dequeue.
            completed = getattr(error, "completed_sequences", [])
            completed.insert(0, result.materialize())
            error.completed_sequences = completed  # type: ignore[attr-defined]
            raise
        return result.materialize()
