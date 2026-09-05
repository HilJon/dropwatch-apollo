"""FastEye RLE source adapted to Apollo's single-view frame contract."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING

import numpy as np

from dropwatch.models import ApolloSettings
from dropwatch.models import ApolloTransportError

if TYPE_CHECKING:
    from dropwatch._hardware import FastEyeRLE
    from dropwatch._hardware import RLEDecoder
    from dropwatch._hardware import RLEReadGate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RLESourceStats:
    daq_reads: int = 0
    zero_byte_reads: int = 0
    recovered_zero_byte_reads: int = 0
    transport_failures: int = 0
    last_daq_read_ms: float = 0.0
    max_daq_read_ms: float = 0.0
    last_vendor_error: str | None = None
    last_frame_counter: int | None = None


class _FastEyeApolloSource:
    """Thin FastEye RLE adapter with lazy vendor-DLL imports."""

    def __init__(self, settings: ApolloSettings) -> None:
        self._settings = settings
        self._camera: FastEyeRLE | None = None
        self._decoder: RLEDecoder | None = None
        self._read_gate: RLEReadGate | None = None
        self._decode_buffer: np.ndarray | None = None
        self._started = False
        self._needs_flush = False
        self._previous_frame_counter: int | None = None
        self._poisoned = False
        self._stats_lock = threading.Lock()
        self._stats = _RLESourceStats()

    @property
    def frame_shape(self) -> tuple[int, int]:
        from dropwatch._hardware import LEFT_VIEW_WIDTH
        from dropwatch._hardware import RAW_FRAME_HEIGHT

        return RAW_FRAME_HEIGHT, LEFT_VIEW_WIDTH

    @property
    def frame_dtype(self) -> type[np.uint8]:
        return np.uint8

    @property
    def reserved_buffer_bytes(self) -> int:
        from dropwatch._hardware import ENCODED_BUFFER_BYTES
        from dropwatch._hardware import RAW_FRAME_HEIGHT
        from dropwatch._hardware import RAW_FRAME_WIDTH

        decoded = self._settings.rle_batch_frames * RAW_FRAME_HEIGHT * RAW_FRAME_WIDTH
        encoded = ENCODED_BUFFER_BYTES
        return decoded + encoded

    @property
    def diagnostics(self) -> _RLESourceStats:
        with self._stats_lock:
            return self._stats

    def open(self) -> None:
        if self._camera is not None:
            return

        from dropwatch._hardware import FastEyeRLE

        camera = FastEyeRLE()
        try:
            camera.open()
        except Exception:
            try:
                camera.close()
            except Exception:
                logger.exception("failed to close FastEye after an incomplete open")
            raise
        self._camera = camera

    def start(self) -> None:
        if self._started:
            return
        if self._poisoned and self._camera is not None:
            self._camera.close()
            self._camera = None
        if self._camera is None:
            self.open()
        if self._camera is None:
            raise RuntimeError("camera could not be opened")

        from dropwatch._hardware import RAW_FRAME_HEIGHT
        from dropwatch._hardware import RAW_FRAME_WIDTH
        from dropwatch._hardware import RLEDecoder
        from dropwatch._hardware import RLEReadGate

        camera = self._camera
        try:
            self._needs_flush = True
            self._poisoned = False
            self._reset_stats()
            camera.setup(read_timeout_ms=self._settings.effective_read_timeout_ms)
            camera.configure(
                frame_period_ms=self._settings.frame_period_ms,
                exposure_time_ms=self._settings.exposure_time_ms,
                threshold=self._settings.threshold,
                rle_batch_frames=self._settings.rle_batch_frames,
            )

            if self._decoder is None:
                self._decoder = RLEDecoder()
            expected_shape = (self._settings.rle_batch_frames, RAW_FRAME_HEIGHT, RAW_FRAME_WIDTH)
            if self._decode_buffer is None or self._decode_buffer.shape != expected_shape:
                self._decode_buffer = np.full(expected_shape, fill_value=255, dtype=np.uint8)
            else:
                self._decode_buffer.fill(255)
            self._previous_frame_counter = None
            self._read_gate = RLEReadGate(camera)
            # Flush resets RLE acquisition, so it must precede enabling it.
            camera.flush()
            camera.set_enc_mode()
            camera.trig_frame()
            self._started = True
        except Exception:
            self._poisoned = True
            try:
                self.stop()
            except Exception:
                logger.exception("failed to flush FastEye after an incomplete start")
            raise

    def read(self) -> np.ndarray | None:
        if not self._started or self._camera is None or self._decoder is None:
            raise RuntimeError("camera source is not started")
        try:
            return self._read_started()
        except Exception:
            self._poisoned = True
            if self._read_gate is not None:
                self._read_gate.reset()
            raise

    def _read_started(self) -> np.ndarray | None:
        from dropwatch._hardware import DAQError
        from dropwatch._hardware import validate_rle_batch

        if self._camera is None or self._decoder is None or self._read_gate is None:
            raise RuntimeError("camera source is not available")
        camera = self._camera
        decoder = self._decoder
        try:
            ready = self._read_gate.poll()
        except DAQError as exc:
            self._record_transport_failure(last_vendor_error=str(exc))
            raise ApolloTransportError(f"could not query camera stream status: {exc}") from exc
        if not ready:
            return None
        if self._decode_buffer is None:
            raise RuntimeError("decode buffer is not available")

        recovered = self._read_image_with_retry(camera)
        try:
            encoded_data = camera.get_enc_data()
        except DAQError as exc:
            self._record_transport_failure(last_vendor_error=str(exc))
            raise ApolloTransportError(f"could not access the camera RLE buffer: {exc}") from exc
        frames, last_counter = validate_rle_batch(
            decoder, encoded_data, self._decode_buffer, self._previous_frame_counter
        )
        self._previous_frame_counter = last_counter
        with self._stats_lock:
            self._stats = dataclass_replace(self._stats, last_frame_counter=last_counter)
        if recovered:
            self._update_stats(recovered_zero_byte_reads=1)

        from dropwatch._hardware import LEFT_VIEW_WIDTH

        return self._decode_buffer[:frames, :, :LEFT_VIEW_WIDTH]

    def _read_image_with_retry(self, camera: FastEyeRLE) -> bool:
        from dropwatch._hardware import DAQError
        from dropwatch._hardware import DAQReadError

        saw_zero_byte_read = False
        total_attempts = self._settings.zero_byte_read_retries + 1
        for attempt in range(total_attempts):
            read_started = time.perf_counter()
            try:
                camera.read_image()
            except DAQReadError as exc:
                if self._read_gate is not None:
                    self._read_gate.reset()
                if exc.actual_bytes != 0:
                    self._record_transport_failure(last_vendor_error=exc.vendor_error)
                    raise ApolloTransportError(
                        f"partial camera transfer: received {exc.actual_bytes} of "
                        f"{exc.expected_bytes} bytes (vendor error: {exc.vendor_error})"
                    ) from exc

                saw_zero_byte_read = True
                self._update_stats(zero_byte_reads=1, last_vendor_error=exc.vendor_error)
                if attempt + 1 >= total_attempts:
                    self._record_transport_failure(last_vendor_error=exc.vendor_error)
                    raise ApolloTransportError(
                        f"camera returned 0 bytes on {total_attempts} consecutive reads "
                        f"(vendor error: {exc.vendor_error})"
                    ) from exc
                # Stats provide diagnostics without calling arbitrary logging
                # handlers on the latency-sensitive camera thread.
                if self._settings.zero_byte_retry_delay_ms:
                    time.sleep(self._settings.zero_byte_retry_delay_ms / 1000)
            except DAQError as exc:
                self._record_transport_failure(last_vendor_error=str(exc))
                raise ApolloTransportError(f"camera read failed: {exc}") from exc
            else:
                return saw_zero_byte_read
            finally:
                self._update_stats(daq_reads=1, daq_read_ms=(time.perf_counter() - read_started) * 1000)
        raise AssertionError("unreachable RLE read retry state")

    def stop(self) -> None:
        if not self._needs_flush:
            return
        camera = self._camera
        flush_error: Exception | None = None
        try:
            if camera is not None:
                camera.flush()
        except Exception as exc:  # noqa: BLE001 - reset and poisoned-handle close must still run
            flush_error = exc
            self._poisoned = True
        finally:
            self._started = False
            self._needs_flush = False
            self._previous_frame_counter = None
            self._read_gate = None
        if self._poisoned and camera is not None:
            try:
                camera.close()
            except Exception:
                if flush_error is None:
                    raise
                logger.exception("failed to close poisoned FastEye handle")
            else:
                self._camera = None
        if flush_error is not None:
            raise flush_error

    def close(self) -> None:
        stop_error: Exception | None = None
        try:
            self.stop()
        except Exception as exc:  # noqa: BLE001 - the camera handle still has to be closed
            stop_error = exc
        if self._camera is not None:
            self._camera.close()
        self._camera = None
        self._decode_buffer = None
        self._decoder = None
        self._read_gate = None
        self._poisoned = False
        if stop_error is not None:
            raise stop_error

    def _reset_stats(self) -> None:
        with self._stats_lock:
            self._stats = _RLESourceStats()

    def _record_transport_failure(self, *, last_vendor_error: str | None = None) -> None:
        self._update_stats(transport_failures=1, last_vendor_error=last_vendor_error)

    def _update_stats(
        self,
        *,
        daq_reads: int = 0,
        zero_byte_reads: int = 0,
        recovered_zero_byte_reads: int = 0,
        transport_failures: int = 0,
        daq_read_ms: float | None = None,
        last_vendor_error: str | None = None,
    ) -> None:
        with self._stats_lock:
            current = self._stats
            self._stats = _RLESourceStats(
                daq_reads=current.daq_reads + daq_reads,
                zero_byte_reads=current.zero_byte_reads + zero_byte_reads,
                recovered_zero_byte_reads=current.recovered_zero_byte_reads + recovered_zero_byte_reads,
                transport_failures=current.transport_failures + transport_failures,
                last_daq_read_ms=current.last_daq_read_ms if daq_read_ms is None else daq_read_ms,
                max_daq_read_ms=(
                    current.max_daq_read_ms if daq_read_ms is None else max(current.max_daq_read_ms, daq_read_ms)
                ),
                last_vendor_error=last_vendor_error if last_vendor_error is not None else current.last_vendor_error,
                last_frame_counter=current.last_frame_counter,
            )
