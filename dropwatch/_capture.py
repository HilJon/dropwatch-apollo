"""Trigger state and explicitly owned, preallocated image buffers."""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from math import ceil
from math import prod
from typing import Any

import numpy as np

from dropwatch.models import ApolloLifecycleError
from dropwatch.models import ApolloSettings


@dataclass(eq=False)
class _FrameLease:
    """Keep a block pinned while any lookback frame still refers to it."""

    buffer: np.ndarray
    release: Callable[[np.ndarray], None] | None = None

    def __del__(self) -> None:
        if self.release is not None:
            self.release(self.buffer)


@dataclass(eq=False)
class _CapturedSequence:
    frames: np.ndarray
    history: tuple[tuple[np.ndarray, _FrameLease], ...] = ()
    lease: _FrameLease | None = None
    _materialized: bool = field(default=False, init=False, repr=False)
    _materialize_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def materialize(self) -> np.ndarray:
        """Copy only the lookback prefix, outside camera intake; never copy the shot."""
        with self._materialize_lock:
            if not self._materialized:
                for index, (frame, _owner) in enumerate(self.history):
                    self.frames[index] = frame
                self.history = ()
                self.frames.flags.writeable = False
                self._materialized = True
        return self.frames


class _SequenceCapture:
    """One active window, continuous lookback, and hysteretic re-arming."""

    def __init__(self, settings: ApolloSettings) -> None:
        self._settings = settings
        self._max_sequences = 1
        self.reset()

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def is_done(self) -> bool:
        return self._sequence_index >= self._max_sequences

    @property
    def is_capturing(self) -> bool:
        return self._sequence_pos > 0

    @property
    def frames_remaining(self) -> int:
        return self._settings.max_number_frames - self._sequence_pos if self.is_capturing else 0

    def reset(self, *, max_sequences: int | None = None) -> None:
        if max_sequences is not None:
            self._max_sequences = max_sequences
        self._sequence_pool: queue.Queue[np.ndarray] = queue.Queue()
        self._sequence_index = 0
        self._lease: _FrameLease | None = None
        self._sequence: np.ndarray | None = None
        self._pre_trigger_buffer: np.ndarray | None = None
        self._pre_trigger_pos = 0
        self._sequence_pos = 0
        self._history: deque[tuple[np.ndarray, _FrameLease]] = deque(maxlen=self._settings.pre_trigger)
        self._trigger_history: tuple[tuple[np.ndarray, _FrameLease], ...] = ()
        self._clear_frames = 0
        self._armed = False
        self._frame_shape: tuple[int, int] | None = None
        self._frame_dtype: np.dtype[Any] | None = None

    def prepare(
        self,
        frame_shape: tuple[int, int],
        frame_dtype: np.dtype[Any] | type[Any],
        *,
        additional_buffer_bytes: int = 0,
    ) -> None:
        if len(frame_shape) != 2 or any(d < 1 for d in frame_shape):
            raise ValueError(f"Apollo expects a positive 2D frame shape, got {frame_shape}")
        if additional_buffer_bytes < 0:
            raise ValueError("additional_buffer_bytes must be >= 0")
        if self._settings.trigger_position_px + self._settings.trigger_width_px > frame_shape[1]:
            raise ValueError("trigger ROI extends outside the image width")
        dtype = np.dtype(frame_dtype)
        self._frame_shape, self._frame_dtype = frame_shape, dtype
        count = self._max_sequences
        if self._settings.spool_directory is not None:
            count = min(count, self._settings.spool_buffer_count)
            post = self._settings.max_number_frames - self._settings.pre_trigger
            minimum = min(self._max_sequences, ceil(self._settings.pre_trigger / post) + 2)
            if count < minimum:
                raise ValueError(f"spool_buffer_count must be at least {minimum} for this pre_trigger")
        # Separate waiting history lets consumers fill the final prefix while
        # intake still holds immutable references to that history.
        block_frames = self._settings.max_number_frames + self._settings.pre_trigger
        sequence_bytes = count * block_frames * prod(frame_shape) * dtype.itemsize
        required = sequence_bytes + additional_buffer_bytes
        if required > self._settings.max_buffer_bytes:
            raise MemoryError(
                f"Apollo needs {required} bytes of acquisition buffers ({sequence_bytes} sequence, "
                f"{additional_buffer_bytes} camera), exceeding max_buffer_bytes={self._settings.max_buffer_bytes}"
            )
        for _ in range(count):
            self._sequence_pool.put_nowait(np.full((block_frames, *frame_shape), 255, dtype=dtype))
        self._activate_sequence()

    def _activate_sequence(self) -> None:
        try:
            block = self._sequence_pool.get_nowait()
        except queue.Empty as exc:
            raise ApolloLifecycleError(
                "recording writer cannot keep up: no free sequence buffer; "
                "completed shots are preserved, but acquisition must stop"
            ) from exc
        release = self._sequence_pool.put_nowait if self._settings.spool_directory is not None else None
        self._lease = _FrameLease(block, release)
        self._sequence = block[: self._settings.max_number_frames]
        self._sequence.flags.writeable = True
        self._pre_trigger_buffer = block[self._settings.max_number_frames :]
        self._pre_trigger_pos = 0
        self._sequence_pos = 0

    def push(self, frame: np.ndarray) -> _CapturedSequence | None:
        frame = np.asarray(frame)
        if self._frame_shape is None:
            self.prepare(frame.shape, frame.dtype)
        if frame.shape != self._frame_shape or frame.dtype != self._frame_dtype:
            raise ValueError("frame shape and dtype must remain constant during an acquisition")
        if self.is_done:
            raise RuntimeError("capture already completed")
        if self._sequence is None:
            self._activate_sequence()
        assert self._sequence is not None and self._lease is not None
        pixels = self._foreground_pixels(frame)
        if not self.is_capturing and self._armed and pixels > self._settings.trigger_on_pixels:
            self._trigger_history = tuple(self._history)
            self._sequence_pos = self._settings.pre_trigger
            return self._record_frame(frame, pixels)
        if self.is_capturing:
            return self._record_frame(frame, pixels)
        if self._settings.pre_trigger:
            assert self._pre_trigger_buffer is not None
            slot = self._pre_trigger_buffer[self._pre_trigger_pos]
            slot[:] = frame
            self._history.append((slot, self._lease))
            self._pre_trigger_pos = (self._pre_trigger_pos + 1) % self._settings.pre_trigger
        self._update_arm(pixels)
        return None

    def _update_arm(self, pixels: int) -> None:
        if pixels <= self._settings.trigger_off_pixels:
            self._clear_frames += 1
            if (
                self._clear_frames >= self._settings.rearm_clear_frames
                and len(self._history) == self._settings.pre_trigger
            ):
                self._armed = True
        else:
            self._clear_frames = 0
            if pixels > self._settings.trigger_on_pixels:
                self._armed = False

    def _record_frame(self, frame: np.ndarray, pixels: int) -> _CapturedSequence | None:
        assert self._sequence is not None and self._lease is not None
        slot = self._sequence[self._sequence_pos]
        slot[:] = frame
        self._history.append((slot, self._lease))
        self._sequence_pos += 1
        self._update_arm(pixels)
        if self._sequence_pos < self._settings.max_number_frames:
            return None
        result = _CapturedSequence(self._sequence, self._trigger_history, self._lease)
        self._trigger_history = ()
        self._sequence_index += 1
        self._sequence_pos = 0
        self._sequence = self._pre_trigger_buffer = None
        self._lease = None
        if self.is_done:
            self._history.clear()
            self._sequence_pool = queue.Queue()
        return result

    def _foreground_pixels(self, frame: np.ndarray) -> int:
        start = self._settings.trigger_position_px
        if not self._settings.trigger_from_top:
            start = frame.shape[1] - start - self._settings.trigger_width_px
        roi = frame[:, start : start + self._settings.trigger_width_px]
        return int(roi.size - np.count_nonzero(roi))


def _find_bottom_object_position(frame: np.ndarray) -> int:
    """Find the highest foreground connected to the physical bottom edge."""
    import cv2

    if frame.ndim != 2:
        raise ValueError(f"Apollo expects a single 2D image, got shape {frame.shape}")
    _, labels, stats, _ = cv2.connectedComponentsWithStats((frame == 0).astype(np.uint8), connectivity=4)
    edge_labels = np.unique(labels[:, -1])
    edge_labels = edge_labels[edge_labels != 0]
    if len(edge_labels) == 0:
        raise ValueError("no foreground object is connected to the bottom image edge")
    return frame.shape[1] - int(np.min(stats[edge_labels, 0]))
