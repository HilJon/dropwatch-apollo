"""Public configuration, protocols, errors, and statistics for Apollo."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from math import isfinite
from pathlib import Path
from typing import Any
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ApolloSettings:
    """Settings for fixed-length, single-view drop recordings."""

    max_number_frames: int
    pre_trigger: int = 0
    trigger_position_px: int = 300
    trigger_from_top: bool = False
    trigger_width_px: int = 10
    trigger_on_pixels: int = 25
    trigger_off_pixels: int = 10
    rearm_clear_frames: int = 2
    rle_batch_frames: int = 100
    frame_period_ms: float = 1.0
    exposure_time_ms: float = 0.05
    threshold: int = 127
    read_timeout_ms: int | None = None
    zero_byte_read_retries: int = 1
    zero_byte_retry_delay_ms: float = 1.0
    max_buffer_bytes: int = 2 * 1024**3
    spool_directory: Path | str | None = None
    spool_buffer_count: int = 3

    def __post_init__(self) -> None:
        for name in (
            "max_number_frames",
            "pre_trigger",
            "trigger_position_px",
            "trigger_width_px",
            "trigger_on_pixels",
            "trigger_off_pixels",
            "rearm_clear_frames",
            "rle_batch_frames",
            "threshold",
            "zero_byte_read_retries",
            "max_buffer_bytes",
            "spool_buffer_count",
        ):
            require_integer(name, getattr(self, name))
        for name in ("frame_period_ms", "exposure_time_ms", "zero_byte_retry_delay_ms"):
            require_finite(name, getattr(self, name))
        if self.read_timeout_ms is not None:
            require_integer("read_timeout_ms", self.read_timeout_ms)
        if self.spool_buffer_count < 1:
            raise ValueError("spool_buffer_count must be >= 1")
        if not 20 <= self.max_number_frames <= 2000:
            raise ValueError("max_number_frames must be between 20 and 2000")
        if not 0 <= self.pre_trigger < self.max_number_frames:
            raise ValueError("pre_trigger must be >= 0 and smaller than max_number_frames")
        if self.trigger_position_px < 0:
            raise ValueError("trigger_position_px must be >= 0")
        if self.trigger_width_px < 1:
            raise ValueError("trigger_width_px must be >= 1")
        if self.trigger_off_pixels < 0:
            raise ValueError("trigger_off_pixels must be >= 0")
        if self.trigger_on_pixels <= self.trigger_off_pixels:
            raise ValueError("trigger_on_pixels must be greater than trigger_off_pixels")
        if self.rearm_clear_frames < 1:
            raise ValueError("rearm_clear_frames must be >= 1")
        if not 100 <= self.rle_batch_frames <= 1000:
            raise ValueError("rle_batch_frames must be between 100 and 1000")
        if self.frame_period_ms <= 0:
            raise ValueError("frame_period_ms must be > 0")
        if not 0 < self.exposure_time_ms <= self.frame_period_ms:
            raise ValueError("exposure_time_ms must be > 0 and not exceed frame_period_ms")
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        minimum_read_timeout_ms = ceil(self.rle_batch_frames * self.frame_period_ms) + 50
        if self.read_timeout_ms is not None and self.read_timeout_ms < minimum_read_timeout_ms:
            raise ValueError(
                f"read_timeout_ms must be at least {minimum_read_timeout_ms} ms for the configured RLE batch"
            )
        if not 0 <= self.zero_byte_read_retries <= 3:
            raise ValueError("zero_byte_read_retries must be between 0 and 3")
        if self.zero_byte_retry_delay_ms < 0:
            raise ValueError("zero_byte_retry_delay_ms must be >= 0")
        if self.max_buffer_bytes < 1:
            raise ValueError("max_buffer_bytes must be >= 1")

    @property
    def effective_read_timeout_ms(self) -> int:
        """Bound a vendor read while allowing one complete RLE batch plus margin."""
        if self.read_timeout_ms is not None:
            return self.read_timeout_ms
        batch_duration_ms = self.rle_batch_frames * self.frame_period_ms
        return max(500, ceil(2 * batch_duration_ms + 100))


@dataclass(frozen=True)
class ApolloVideoSettings:
    """Options for post-acquisition left-view AVI or MP4 export."""

    playback_fps: float = 25.0
    codec: str | None = None
    annotate: bool = True
    invert: bool = True
    trim_start: int = 0
    trim_end: int | None = None
    crop_bottom: int = 0
    separator_frames: int = 0

    def __post_init__(self) -> None:
        require_finite("playback_fps", self.playback_fps)
        for name in ("trim_start", "crop_bottom", "separator_frames"):
            require_integer(name, getattr(self, name))
        if self.trim_end is not None:
            require_integer("trim_end", self.trim_end)
        if self.playback_fps <= 0:
            raise ValueError("playback_fps must be > 0")
        if self.codec is not None and (len(self.codec) != 4 or not self.codec.isascii()):
            raise ValueError("codec must contain exactly four ASCII characters")
        if self.trim_start < 0:
            raise ValueError("trim_start must be >= 0")
        if self.trim_end is not None and self.trim_end <= self.trim_start:
            raise ValueError("trim_end must be greater than trim_start")
        if self.crop_bottom < 0:
            raise ValueError("crop_bottom must be >= 0")
        if self.separator_frames < 0:
            raise ValueError("separator_frames must be >= 0")


class ApolloFrameSource(Protocol):
    """Camera boundary returning single-view batches shaped ``(frames, height, width)``."""

    @property
    def frame_shape(self) -> tuple[int, int]: ...

    @property
    def frame_dtype(self) -> np.dtype[Any] | type[Any]: ...

    def open(self) -> None: ...

    def start(self) -> None: ...

    def read(self) -> np.ndarray | None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class ApolloSequenceEvaluator(Protocol):
    """Callable that evaluates a list containing one completed sequence."""

    def __call__(self, sequences: list[np.ndarray]) -> Any: ...


class ApolloLifecycleError(RuntimeError):
    """Raised when an Apollo worker cannot be stopped cleanly."""


class ApolloFrameLossError(RuntimeError):
    """Raised when the RLE stream proves that camera frames were lost."""


class ApolloTransportError(ApolloFrameLossError):
    """Raised when a camera transfer cannot be trusted after bounded recovery."""


class ApolloIncompleteSequenceError(ApolloLifecycleError):
    """Raised when a triggered sequence cannot be drained before stopping."""

    def __init__(self, message: str, completed_sequences: list[np.ndarray]) -> None:
        super().__init__(message)
        self.completed_sequences = completed_sequences


class ApolloEvaluationError(RuntimeError):
    """Raised when the optional sequence evaluator fails."""


@dataclass(frozen=True)
class ApolloStats:
    """Counters for the current or most recently completed acquisition."""

    frames_received: int = 0
    sequences_captured: int = 0
    frame_gaps: int = 0
    daq_reads: int = 0
    zero_byte_reads: int = 0
    recovered_zero_byte_reads: int = 0
    transport_failures: int = 0
    last_daq_read_ms: float = 0.0
    max_daq_read_ms: float = 0.0
    incomplete_sequences: int = 0
    last_vendor_error: str | None = None
    last_frame_counter: int | None = None


def require_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")


def require_finite(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
