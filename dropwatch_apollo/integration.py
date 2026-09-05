"""Optional A1 consumer helpers. No dispenser, calibration or tracking code lives here.

The existing fast_seq_eval module is imported only when evaluation is requested.
Raw evaluation avoids video compression; AVI reading supports our left-only
LegacyVideoSaver format, not arbitrary or old split-view videos.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import overload

import numpy as np

from dropwatch_apollo._video import legacy_header_height
from dropwatch_apollo.models import require_finite


def extract_video(path: str | Path, *, invert_bw: bool = True) -> Iterator[Iterator[tuple[np.ndarray]]]:
    """Stream annotated, left-only AVI windows; consume each window in order.

    Return singleton-view tuples for the old crop-helper convention. Header and
    codec padding are removed, so crop coordinates are physical image pixels.
    Only the current decoded frame is held here, not a whole video or window.
    """
    import cv2

    reader = cv2.VideoCapture(str(path))
    try:
        header = legacy_header_height()
        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if not reader.isOpened() or width != 512 or height <= header + header % 2 or expected < 1:
            raise ValueError("expected a Dropwatch Apollo left-only LegacyVideoSaver AVI (width 512)")

        def frames() -> Iterator[np.ndarray | None]:
            for _ in range(expected):
                ok, image = reader.read()
                if not ok:
                    raise OSError("AVI ended before its declared frame count")
                # Test the encoded image, not its binary mask: a valid empty
                # frame still has a coloured label, unlike the black separator.
                if image.max() <= 16:
                    yield None
                else:
                    label = image[:header].astype(np.int16)
                    coloured = (label[:, :, 0] - label[:, :, 2] > 32) & (label[:, :, 1] - label[:, :, 2] > 32)
                    if np.count_nonzero(coloured) < 3:
                        raise ValueError("AVI frame is missing the expected LegacyVideoSaver label")
                    body = image[header : height - header % 2]
                    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
                    yield gray > 127 if invert_bw else gray < 127

        for separator, group in groupby(frames(), key=lambda frame: frame is None):
            if not separator:
                yield ((frame,) for frame in group if frame is not None)
    finally:
        reader.release()


def crop_sequences(
    sequences: Iterable[Iterable[Any]],
    v_fov: slice | None = None,
    h_fov: slice | None = None,
    flip_vertical: bool = False,
) -> list[list[np.ndarray]]:
    """Accept legacy view tuples or 2D frames; retain only the cropped pixels."""
    result = []
    try:
        for sequence in sequences:
            cropped = []
            for frame in sequence:
                image = frame if isinstance(frame, np.ndarray) and frame.ndim == 2 else np.hstack(frame)
                image = image[v_fov or slice(None), h_fov or slice(None)]
                if image.ndim != 2 or not image.size:
                    raise ValueError("evaluation crop must contain a non-empty 2D image")
                # copy() is intentional: slices must not retain a full decoded image.
                cropped.append(np.flipud(image).copy() if flip_vertical else image.copy())
            result.append(cropped)
    finally:
        close = getattr(sequences, "close", None)
        if close is not None:
            close()
    return result


class _EvaluationFrames(Sequence[np.ndarray]):
    """Lazy physical, cropped masks over one read-only raw memmap."""

    def __init__(self, raw: np.ndarray, rows: slice, cols: slice) -> None:
        self.raw, self.rows, self.cols = raw, rows, cols

    def __len__(self) -> int:
        return len(self.raw)

    @overload
    def __getitem__(self, index: int) -> np.ndarray: ...

    @overload
    def __getitem__(self, index: slice) -> list[np.ndarray]: ...

    def __getitem__(self, index: int | slice) -> np.ndarray | list[np.ndarray]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        mask = self.raw[index].T[self.rows, self.cols] == 0
        if not mask.size:
            raise ValueError("evaluation crop is empty")
        return mask


def _fast_eval_module() -> Any:
    from a1_experiment_lib import fast_seq_eval  # type: ignore[import-not-found]

    return fast_seq_eval


def _finalize(data: Any, module: Any, frame_period_ms: float) -> Any:
    require_finite("frame_period_ms", frame_period_ms)
    if frame_period_ms <= 0:
        raise ValueError("frame_period_ms must be > 0")
    if data.empty:
        return data.copy()
    connected, _groups = module.connect_shots(data)
    result = module.postproc_full_data(connected)
    # The reviewed evaluator returns mm/frame. Dividing by ms/frame gives m/s.
    for column in ("speed", "speed_start"):
        if column in result:
            result[column] = result[column] / frame_period_ms
    return result


def evaluate_sequences(sequences: Any, *, frame_period_ms: float = 1.0, **options: Any) -> Any:
    """Sequential compatibility path, retaining global cross-window tracking."""
    module = _fast_eval_module()
    return _finalize(module.fast_eval_sequences(sequences, **options), module, frame_period_ms)


def make_evaluation_callbacks(
    *, rows: slice = slice(200, 1100), cols: slice = slice(200, 400), frame_period_ms: float = 1.0, **options: Any
) -> tuple[Callable[[list[np.ndarray]], Any], Callable[[Any], Any]]:
    """Per-window raw observations in parallel; connect/postprocess once at the end.

    The consumer owns the evaluator's memory use and tracking parameters. This
    does not evaluate already postprocessed per-window tables and concatenate
    them: that would lose droplets spanning two windows.
    """
    require_finite("frame_period_ms", frame_period_ms)
    if frame_period_ms <= 0:
        raise ValueError("frame_period_ms must be > 0")
    module = _fast_eval_module()  # Fail before acquisition if the consumer dependency is unavailable.

    def evaluate(sequences: list[np.ndarray]) -> Any:
        return module.fast_eval_sequences([_EvaluationFrames(seq, rows, cols) for seq in sequences], **options)

    def finalize(data: Any) -> Any:
        return _finalize(data, module, frame_period_ms)

    return evaluate, finalize
