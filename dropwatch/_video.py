"""Atomic, post-acquisition video export for Apollo's left camera view."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from dropwatch.models import ApolloVideoSettings


def save_png(frame: np.ndarray, path: str | Path) -> Path:
    """Save one binary image in physical orientation, without annotations."""
    import cv2

    output = Path(path)
    if frame.ndim != 2 or output.suffix.lower() != ".png":
        raise ValueError("snapshot requires a 2D image and a .png path")
    image = np.ascontiguousarray((frame != 0).T, dtype=np.uint8) * 255
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("PNG encoder failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.part")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def save_video(
    sequences: np.ndarray | Iterable[np.ndarray],
    path: str | Path,
    *,
    frame_period_ms: float,
    pre_trigger: int,
    options: ApolloVideoSettings,
    shot_offset: int = 0,
) -> Path:
    """Write one or more shots without modifying their acquisition buffers."""
    import cv2

    shots = _normalise_sequences(sequences, options)
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix not in {".avi", ".mp4"}:
        raise ValueError("video path must end in .avi or .mp4")

    first_frame = shots[0][options.trim_start]
    output_height = first_frame.shape[1] - options.crop_bottom
    output_width = first_frame.shape[0]
    if output_height < 1:
        raise ValueError(
            f"crop_bottom={options.crop_bottom} removes the complete physical image height "
            f"of {first_frame.shape[1]} pixels"
        )
    if output_width % 2 or output_height % 2:
        raise ValueError("video dimensions must be even; choose an even crop_bottom to avoid codec truncation")
    frame_size = (output_width, output_height)
    codec = options.codec or ("MJPG" if suffix == ".avi" else "mp4v")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.{uuid4().hex}.part{suffix}")
    try:
        writer = cv2.VideoWriter(
            str(temporary_path),
            cv2.VideoWriter_fourcc(*codec),  # type: ignore[attr-defined]
            options.playback_fps,
            frame_size,
            False,
        )
        try:
            if not writer.isOpened():
                raise OSError(f"could not open {codec} video writer for {output_path}")

            background = 0 if options.invert else 255
            separator = np.full((output_height, output_width), background, dtype=np.uint8)
            for shot_index, sequence in enumerate(shots):
                shot = shot_offset + shot_index
                stop = len(sequence) if options.trim_end is None else min(options.trim_end, len(sequence))
                for frame_index in range(options.trim_start, stop):
                    image = _render_frame(
                        sequence[frame_index],
                        shot=shot,
                        frame_index=frame_index,
                        frame_period_ms=frame_period_ms,
                        pre_trigger=pre_trigger,
                        options=options,
                        cv2=cv2,
                    )
                    writer.write(image)
                if shot_index + 1 < len(shots):
                    for _ in range(options.separator_frames):
                        writer.write(separator)
        finally:
            writer.release()

        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise OSError(f"video writer produced no data for {output_path}")
        expected_frames = (
            sum(
                (len(shot) if options.trim_end is None else min(options.trim_end, len(shot))) - options.trim_start
                for shot in shots
            )
            + (len(shots) - 1) * options.separator_frames
        )
        _verify_video(temporary_path, expected_frames, frame_size)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _verify_video(path: Path, expected_frames: int, size: tuple[int, int]) -> None:
    """Catch silent writer failures before replacing the destination file."""
    import cv2

    reader = cv2.VideoCapture(str(path))
    try:
        dimensions = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not reader.isOpened() or dimensions != size or int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) != expected_frames:
            raise OSError("video verification failed: incorrect dimensions or frame count")
        if not reader.read()[0]:
            raise OSError("video verification failed: first frame cannot be decoded")
        if expected_frames > 1 and (
            not reader.set(cv2.CAP_PROP_POS_FRAMES, expected_frames - 1) or not reader.read()[0]
        ):
            raise OSError("video verification failed: last frame cannot be decoded")
    finally:
        reader.release()


def save_sequence_videos(
    sequences: np.ndarray | Iterable[np.ndarray],
    directory: str | Path,
    *,
    prefix: str,
    suffix: str,
    frame_period_ms: float,
    pre_trigger: int,
    options: ApolloVideoSettings,
) -> list[Path]:
    """Write one independently finalized video per shot."""
    shots = _normalise_sequences(sequences, options)
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty file-name component")
    normalized_suffix = suffix.lower()
    if normalized_suffix not in {".avi", ".mp4"}:
        raise ValueError("suffix must be .avi or .mp4")

    output_directory = Path(directory)
    per_shot_options = replace(options, separator_frames=0)
    paths: list[Path] = []
    for shot, sequence in enumerate(shots):
        paths.append(
            save_video(
                sequence,
                output_directory / f"{prefix}_{shot:03d}{normalized_suffix}",
                frame_period_ms=frame_period_ms,
                pre_trigger=pre_trigger,
                options=per_shot_options,
                shot_offset=shot,
            )
        )
    return paths


def _normalise_sequences(
    sequences: np.ndarray | Iterable[np.ndarray],
    options: ApolloVideoSettings,
) -> tuple[np.ndarray, ...]:
    shots: tuple[np.ndarray, ...]
    if isinstance(sequences, np.ndarray):
        shots = (np.asarray(sequences),)
    else:
        shots = tuple(np.asarray(sequence) for sequence in sequences)
    if not shots:
        raise ValueError("at least one sequence is required")

    frame_shape: tuple[int, int] | None = None
    for shot, sequence in enumerate(shots):
        if sequence.ndim != 3 or len(sequence) == 0:
            raise ValueError(f"sequence {shot} must have shape (frames, height, width), got {sequence.shape}")
        if options.trim_start >= len(sequence):
            raise ValueError(
                f"trim_start={options.trim_start} removes every frame from sequence {shot} with {len(sequence)} frames"
            )
        if frame_shape is None:
            frame_shape = sequence.shape[1:]
        elif sequence.shape[1:] != frame_shape:
            raise ValueError("all video sequences must have the same frame dimensions")
    return shots


def _render_frame(
    frame: np.ndarray,
    *,
    shot: int,
    frame_index: int,
    frame_period_ms: float,
    pre_trigger: int,
    options: ApolloVideoSettings,
    cv2: object,
) -> np.ndarray:
    foreground = np.asarray(frame) == 0
    if options.invert:
        image = foreground.T.astype(np.uint8) * 255
        text_color = 255
    else:
        image = (~foreground).T.astype(np.uint8) * 255
        text_color = 0
    if options.crop_bottom:
        image = image[: -options.crop_bottom]
    image = np.ascontiguousarray(image)
    if options.annotate:
        time_ms = (frame_index - pre_trigger) * frame_period_ms
        label = f"shot {shot}  frame {frame_index}  t={time_ms:+.3f} ms"
        cv2.putText(  # type: ignore[attr-defined]
            image,
            label,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore[attr-defined]
            0.55,
            text_color,
            1,
            cv2.LINE_AA,  # type: ignore[attr-defined]
        )
    return image
