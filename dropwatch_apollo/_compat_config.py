"""Small configuration objects for the reviewed dropwatch-recorder interface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dropwatch_apollo._video import save_video
from dropwatch_apollo.models import ApolloSettings
from dropwatch_apollo.models import ApolloStats
from dropwatch_apollo.models import ApolloVideoSettings
from dropwatch_apollo.models import DisplayRoi2D
from dropwatch_apollo.models import require_integer


@dataclass(frozen=True)
class CameraSettings:
    frame_period: float = 1.0
    exposure_time: float = 0.05
    bin_threshold: int = 127
    max_bytes_per_img: int = 10_000
    num_img_flush: int = 100

    def __post_init__(self) -> None:
        require_integer("max_bytes_per_img", self.max_bytes_per_img)
        if self.max_bytes_per_img != 10_000:
            raise ValueError("the verified transport requires max_bytes_per_img=10000")
        if not 100 <= self.num_img_flush <= 122:
            raise ValueError("num_img_flush must be 100..122; decoder capacity is not the USB flush size")
        ApolloSettings(
            max_number_frames=20,
            frame_period_ms=self.frame_period,
            exposure_time_ms=self.exposure_time,
            threshold=self.bin_threshold,
            rle_batch_frames=self.num_img_flush,
        )


@dataclass(frozen=True)
class RoiPixelDetector:
    roi: DisplayRoi2D
    detection_threshold_px: int = 15
    view: str = "left"

    def __post_init__(self) -> None:
        if self.view != "left":
            raise ValueError("Dropwatch Apollo records only the left view")
        if not isinstance(self.roi, DisplayRoi2D):
            raise TypeError("roi must be DisplayRoi2D")
        require_integer("detection_threshold_px", self.detection_threshold_px)
        if self.detection_threshold_px < 1:
            raise ValueError("detection_threshold_px must be >= 1")


class CaptureState:
    """Window configuration and session counters; never retains frame lists."""

    def __init__(self, detector: RoiPixelDetector, capture_len: int, copy_frames: bool = True) -> None:
        require_integer("capture_len", capture_len)
        if capture_len < 1:
            raise ValueError("capture_len must be >= 1")
        if not isinstance(detector, RoiPixelDetector):
            raise TypeError("the compatibility layer requires RoiPixelDetector")
        self.detector = detector
        self.capture_len = capture_len
        self.copy_frames = copy_frames  # Ownership is always safe, even if the old caller passes False.
        self.lookback_len = 0
        self._stats: Callable[[], ApolloStats] = ApolloStats

    @property
    def image_count(self) -> int:
        return self._stats().frames_received

    @property
    def trigger_count(self) -> int:
        return self._stats().triggers_detected


class BufferedCaptureState(CaptureState):
    """Legacy lookback is additive; capture_len includes the trigger frame."""

    def __init__(
        self, detector: RoiPixelDetector, capture_len: int, lookback_len: int = 0, copy_frames: bool = True
    ) -> None:
        super().__init__(detector, capture_len, copy_frames)
        require_integer("lookback_len", lookback_len)
        if lookback_len < 0:
            raise ValueError("lookback_len must be >= 0")
        self.lookback_len = lookback_len


class LegacyVideoSaver:
    """Post-acquisition, atomic left-only AVI with the legacy header/separators."""

    def __init__(
        self,
        output_path: str | Path | None = None,
        fps: float = 16,
        frame_period: float = 1,
        cut_bottom: int | None = None,
        trim_start: int | None = None,
        trim_end: int | None = None,
        check_white_stripe: bool = False,
        split_x: int = 1120,
        codec: str = "XVID",
        add_blank_frame_between_sequences: bool = True,
        invert_bw: bool = False,
        stack: bool = True,
    ) -> None:
        if check_white_stripe or split_x != 1120 or not stack:
            raise ValueError("white-stripe filtering and alternate/split layouts are not supported")
        if cut_bottom is not None and cut_bottom % 2:
            raise ValueError("cut_bottom must be even to avoid codec truncation")
        self.output_path = Path(output_path) if output_path is not None else None
        self.frame_period = frame_period
        # Legacy trim_end is a count *after* applying trim_start.
        start = 0 if trim_start is None else trim_start
        self.options = ApolloVideoSettings(
            playback_fps=fps,
            codec=codec,
            invert=invert_bw,
            trim_start=start,
            trim_end=None if trim_end is None else start + trim_end,
            crop_bottom=abs(cut_bottom or 0),
            separator_frames=int(add_blank_frame_between_sequences),
            legacy_layout=True,
        )
        from dropwatch_apollo.models import require_finite

        require_finite("frame_period", frame_period)
        if frame_period <= 0:
            raise ValueError("frame_period must be > 0")
        if self.output_path is not None and self.output_path.suffix.lower() != ".avi":
            raise ValueError("LegacyVideoSaver requires an .avi output path")

    def save(self, sequences: list[np.ndarray], output_dir: Path) -> None:
        if sequences:
            self.output_path = self.output_path or output_dir / "recording.avi"
            save_video(
                sequences, self.output_path, frame_period_ms=self.frame_period, pre_trigger=0, options=self.options
            )


class Recorder:
    """Pipeline configuration; all capture and storage run in DropwatchApollo."""

    def __init__(
        self,
        capture_state: CaptureState,
        output_dir: str | Path,
        sinks: list[LegacyVideoSaver] | None = None,
        keep_sequences: bool = False,
    ) -> None:
        if not isinstance(capture_state, CaptureState):
            raise TypeError("capture_state must be CaptureState")
        self.capture_state = capture_state
        self.output_dir = Path(output_dir)
        self.sinks = list(sinks or [])
        if any(not isinstance(sink, LegacyVideoSaver) for sink in self.sinks):
            raise TypeError("compatibility sinks must be LegacyVideoSaver; use the native API for other exports")
        self.keep_sequences = keep_sequences
        self.sequences: list[np.ndarray] = []

    @property
    def completed_windows(self) -> int:
        return self.capture_state._stats().sequences_captured

    @property
    def partial_windows(self) -> int:
        return self.capture_state._stats().incomplete_sequences


class FastPostTriggerRecorder(Recorder):
    def __init__(
        self, capture_state: CaptureState, output_dir: str | Path, sinks: list[LegacyVideoSaver] | None = None
    ) -> None:
        super().__init__(capture_state, output_dir, sinks, keep_sequences=False)
