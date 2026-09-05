"""The existing recorder API, backed only by the Dropwatch Apollo engine.

This adapter never retries physical dispensing. Errors keep their original type.
Its real supervisor thread remains alive through camera cleanup and AVI export.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from math import ceil
from pathlib import Path
from typing import Any

from dropwatch_apollo._compat_config import CameraSettings
from dropwatch_apollo._compat_config import Recorder
from dropwatch_apollo.apollo import DropwatchApollo
from dropwatch_apollo.models import ApolloFrameSource
from dropwatch_apollo.models import ApolloLifecycleError
from dropwatch_apollo.models import ApolloSequenceEvaluator
from dropwatch_apollo.models import ApolloSettings
from dropwatch_apollo.models import require_finite
from dropwatch_apollo.models import require_integer

logger = logging.getLogger(__name__)


class Dropwatch:
    """Compatibility for the A1 recording context, not the old camera engine."""

    def __init__(
        self,
        recorder: Recorder,
        *,
        decoder_max_images: int = 1000,
        frame_source: ApolloFrameSource | None = None,
        evaluator: ApolloSequenceEvaluator | None = None,
        evaluation_finalizer: Callable[[Any], Any] | None = None,
        spool_chunk_frames: int = 100,
        spool_buffer_count: int = 8,
        max_buffer_bytes: int = 2 * 1024**3,
        max_spool_bytes: int = 64 * 1024**3,
    ) -> None:
        require_integer("decoder_max_images", decoder_max_images)
        if not 100 <= decoder_max_images <= 1000:
            raise ValueError("decoder_max_images must be 100..1000")
        if not isinstance(recorder, Recorder):
            raise TypeError("recorder must come from the Dropwatch Apollo compatibility package")
        self.recorder = recorder
        self._decoder_max_images = decoder_max_images
        self._source = frame_source
        self._evaluator = evaluator
        self._finalizer = evaluation_finalizer
        self._storage = dict(
            spool_chunk_frames=spool_chunk_frames,
            spool_buffer_count=spool_buffer_count,
            max_buffer_bytes=max_buffer_bytes,
            max_spool_bytes=max_spool_bytes,
        )
        self._camera_settings = CameraSettings()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._core: DropwatchApollo | None = None
        self._error: BaseException | None = None
        self._cleanup_ok = True
        self._cleanup_error: BaseException | None = None
        self.evaluations: Any = None

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def recording_directory(self) -> Path | None:
        return self._core.recording_directory if self._core is not None else None

    @property
    def read_error_count(self) -> int:
        return self._core.stats.zero_byte_reads if self._core is not None else 0

    @property
    def last_read_error(self) -> str | None:
        return self._core.stats.last_vendor_error if self._core is not None else None

    @property
    def stream_discontinuity_count(self) -> int:
        return self._core.stats.frame_gaps if self._core is not None else 0

    def setup_camera(self, settings: CameraSettings | None = None) -> None:
        """Configure the next session; hardware opens only in the recording worker."""
        with self._lock:
            if self.is_recording:
                raise ApolloLifecycleError("cannot configure a live recording")
            selected = settings or CameraSettings()
            if selected.num_img_flush > self._decoder_max_images:
                raise ValueError("num_img_flush exceeds decoder_max_images")
            self._camera_settings = selected

    def start_background(self, *, max_num_triggers: int | None = None, max_duration: float | None = None) -> None:
        self._launch(max_num_triggers, max_duration, 60.0)

    def start(self, *, max_num_triggers: int | None = None, max_duration: float | None = None) -> None:
        self.start_background(max_num_triggers=max_num_triggers, max_duration=max_duration)
        self.join()

    def _launch(self, max_num_triggers: int | None, max_duration: float | None, ready_timeout: float) -> None:
        if max_duration is None and max_num_triggers is None:
            max_duration = 300.0
        if max_duration is not None:
            require_finite("max_duration", max_duration)
            if max_duration <= 0:
                raise ValueError("max_duration must be > 0")
        camera = self._camera_settings
        capture = self.recorder.capture_state
        if max_num_triggers is None:
            assert max_duration is not None
            frames = max_duration * 1000 / camera.frame_period + camera.num_img_flush + 1
            max_num_triggers = ceil(frames / capture.capture_len) + 1
        require_integer("max_num_triggers", max_num_triggers)
        if max_num_triggers < 1:
            raise ValueError("max_num_triggers must be >= 1")
        settings = ApolloSettings(
            max_number_frames=capture.capture_len + capture.lookback_len,
            pre_trigger=capture.lookback_len,
            trigger_roi=capture.detector.roi,
            trigger_on_pixels=capture.detector.detection_threshold_px,
            trigger_off_pixels=0,
            rearm_clear_frames=1,
            trigger_policy="level",
            frame_period_ms=camera.frame_period,
            exposure_time_ms=camera.exposure_time,
            threshold=camera.bin_threshold,
            rle_batch_frames=camera.num_img_flush,
            spool_directory=self.recorder.output_dir / "raw",
            spool_chunk_frames=self._storage["spool_chunk_frames"],
            spool_buffer_count=self._storage["spool_buffer_count"],
            max_buffer_bytes=self._storage["max_buffer_bytes"],
            max_spool_bytes=self._storage["max_spool_bytes"],
        )
        with self._lock:
            if self.is_recording:
                raise ApolloLifecycleError("recording or finalization is still running")
            if not self._cleanup_ok:
                raise ApolloLifecycleError("previous camera cleanup failed; close() must succeed before reuse")
            for sink in self.recorder.sinks:
                if sink.frame_period != camera.frame_period:
                    raise ValueError("video frame_period must match CameraSettings.frame_period")
                path = sink.output_path or self.recorder.output_dir / "recording.avi"
                if path.exists():
                    raise FileExistsError(f"choose a new video output path before recording: {path}")
            self.recorder.sequences.clear()
            self.evaluations = None
            self._core = DropwatchApollo(
                settings, frame_source=self._source, evaluator=self._evaluator, evaluation_finalizer=self._finalizer
            )
            core = self._core
            capture._stats = lambda: core.stats
            self._ready.clear()
            self._stop.clear()
            self._error = None
            self._cleanup_ok = False
            self._cleanup_error = None
            self._thread = threading.Thread(
                target=self._run,
                args=(max_num_triggers, max_duration, ready_timeout),
                name="dropwatch-apollo-recording",
                daemon=False,
            )
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                self._cleanup_ok = True  # No worker started and no hardware was opened.
                raise

    def _run(self, limit: int, duration: float | None, ready_timeout: float) -> None:
        core = self._core
        assert core is not None
        sequences = []
        try:
            core.start(timeout_s=ready_timeout, max_sequences=limit, max_duration_s=duration, cancel_event=self._stop)
            if self._stop.is_set():
                raise ApolloLifecycleError("recording cancelled before readiness")
            if not core.is_running:
                sequences = core.stop()
                raise ApolloLifecycleError("camera stopped before readiness could be delivered; do not dispense")
            self._ready.set()
            while core.is_running and not self._stop.wait(0.01):
                pass
            sequences = core.stop()
            if self._evaluator is not None:
                self.evaluations = core.get_evaluations()
        except BaseException as error:
            self._error = error.with_traceback(None)
            sequences = getattr(error, "completed_sequences", sequences)
        finally:
            # A timed-out worker is still real work. Keep this supervisor alive
            # so the caller's rig lease cannot be released while USB is active.
            worker = core._worker
            if worker is not None and worker.is_alive():
                core.request_stop(drain=False)
                worker.join()
            for child in (core._spool._worker, core._evaluation._worker):
                if child is not None and child.is_alive():
                    child.join()
            try:
                core.close()
            except BaseException as error:
                self._cleanup_error = error.with_traceback(None)
                self._error = self._error or self._cleanup_error
            self._cleanup_ok = core._state.name == "CLOSED"
            try:
                for sink in self.recorder.sinks:
                    sink.save(sequences, self.recorder.output_dir)
            except BaseException as error:
                self._error = self._error or error.with_traceback(None)
            if self.recorder.keep_sequences:
                self.recorder.sequences = sequences

    def stop(self) -> None:
        """Nonblocking request; join() owns finalization and error propagation."""
        self._stop.set()

    def close(self) -> None:
        """Finalize, or retry a failed vendor close without losing its handle."""
        self.stop()
        if self._thread is not None:
            self._thread.join()
        if self._core is not None and not self._cleanup_ok:
            self._core.close()
            self._cleanup_ok = True
            if self._error is self._cleanup_error:
                self._error = None
        if self._error is not None:
            raise self._error

    def join(self, timeout: float | None = None) -> None:
        if timeout is not None:
            _validate_timeout("timeout", timeout)
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("recording/finalization is still running; keep the rig lease until _thread exits")
        if self._error is not None:
            raise self._error

    def wait_until_ready(self, timeout: float = 5.0) -> None:
        import time

        _validate_timeout("timeout", timeout)
        deadline = time.monotonic() + timeout
        while not self._ready.is_set():
            if not self.is_recording:
                self.join()
                raise ApolloLifecycleError("recording stopped before readiness")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("camera did not become ready; do not dispense")
            self._ready.wait(min(remaining, 0.01))
        if not self.is_recording or self._core is None or not self._core.is_running:
            raise ApolloLifecycleError("camera already stopped; do not dispense")

    @contextmanager
    def recording(
        self,
        *,
        max_num_triggers: int | None = None,
        max_duration: float | None = None,
        ready_timeout: float = 60.0,
        join_timeout: float = 10.0,
    ) -> Iterator[None]:
        _validate_timeout("ready_timeout", ready_timeout)
        _validate_timeout("join_timeout", join_timeout)
        self._launch(max_num_triggers, max_duration, ready_timeout)
        try:
            self.wait_until_ready(ready_timeout)
            yield
        except BaseException:
            self.stop()
            try:
                self.join(join_timeout)
            except BaseException:
                logger.exception("recording cleanup also failed; inspect _thread before releasing the rig lease")
            raise
        else:
            self.stop()
            self.join(join_timeout)


def _validate_timeout(name: str, value: float) -> None:
    require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
