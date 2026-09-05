"""Hardware-free, bounded-batch replay of left-view NPY shots or FastEye RLE files."""

from __future__ import annotations

import time
from collections.abc import Iterable
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from dropwatch._hardware import LEFT_VIEW_WIDTH
from dropwatch._hardware import RAW_FRAME_HEIGHT
from dropwatch._hardware import RAW_FRAME_WIDTH
from dropwatch._hardware import RLEDecoder
from dropwatch.models import ApolloFrameLossError
from dropwatch.models import require_finite
from dropwatch.models import require_integer


def _rle_frames(data: bytes) -> Iterator[np.ndarray]:
    """Reference decoder, used only offline; reject partial/malformed frames."""
    if len(data) % 40:
        raise ApolloFrameLossError("RLE file ends in a partial 40-byte packet")
    start: int | None = None
    previous: int | None = None
    for offset in range(0, len(data), 40):
        header = data[offset : offset + 6] == RLEDecoder._HEADER
        if header or data[offset] == 0:
            if start is not None:
                counter = (data[start + 7] & 127) * 256 + data[start + 6]
                if previous is not None and counter != (previous + 1) & 32767:
                    raise ApolloFrameLossError(f"RLE replay counter jumped from {previous} to {counter}")
                yield _decode_frame(memoryview(data)[start + 8 : offset])
                previous = counter
            start = offset if header else None
    if start is not None:
        raise ApolloFrameLossError("RLE file ends before its final frame delimiter")


def _decode_frame(payload: memoryview) -> np.ndarray:
    line = np.empty(RAW_FRAME_HEIGHT * RAW_FRAME_WIDTH, dtype=np.uint8)
    pos = length = shift = 0
    value = 1
    for byte in payload:
        if byte == 0 and pos > 0:
            break
        length |= (byte & 127) << shift
        shift += 7
        if shift > 21 or pos + length > len(line):
            raise ApolloFrameLossError("invalid RLE run length")
        if byte & 128:
            continue
        line[pos : pos + length] = value
        pos += length
        value = 1 - value
        length = shift = 0
    if pos != len(line) or shift:
        raise ApolloFrameLossError(f"incomplete RLE image: decoded {pos} of {len(line)} pixels")
    return line.reshape(RAW_FRAME_HEIGHT, RAW_FRAME_WIDTH)[:, :LEFT_VIEW_WIDTH]


class ReplayFrameSource:
    """Replay files through exactly the same trigger/capture API as the camera.

    BIN files are separate recordings (counters are checked within each file).
    NPY files must have the same raw left-view shape and dtype. Replay runs as
    fast as possible unless frame_period_ms is given. EOF is explicit.
    """

    def __init__(
        self, paths: str | Path | Iterable[str | Path], *, batch_frames: int = 100, frame_period_ms: float | None = None
    ) -> None:
        self.paths = (Path(paths),) if isinstance(paths, (str, Path)) else tuple(Path(p) for p in paths)
        if not self.paths or any(p.suffix.lower() not in {".bin", ".npy"} for p in self.paths):
            raise ValueError("replay requires .bin or .npy files")
        require_integer("batch_frames", batch_frames)
        if not 1 <= batch_frames <= 1000:
            raise ValueError("batch_frames must be between 1 and 1000")
        if frame_period_ms is not None:
            require_finite("frame_period_ms", frame_period_ms)
            if frame_period_ms <= 0:
                raise ValueError("frame_period_ms must be > 0")
        self.frame_shape = (RAW_FRAME_HEIGHT, LEFT_VIEW_WIDTH)
        self.frame_dtype = np.dtype(np.uint8)
        if self.paths[0].suffix.lower() == ".npy":
            first = np.load(self.paths[0], mmap_mode="r", allow_pickle=False)
            if first.ndim != 3 or not len(first):
                raise ValueError("NPY replay requires a non-empty (frames, height, width) array")
            self.frame_shape = first.shape[1:]
            self.frame_dtype = first.dtype
        self._batch_frames = batch_frames
        self._period_ms = frame_period_ms
        self._iterator: Iterator[np.ndarray] | None = None
        self._buffer: np.ndarray | None = None
        self.exhausted = False
        self._next_read = 0.0

    @property
    def reserved_buffer_bytes(self) -> int:
        # Reference decoding also holds one full image and a source file.
        file_bytes = max((p.stat().st_size for p in self.paths if p.suffix.lower() == ".bin"), default=0)
        return (
            self._batch_frames * int(np.prod(self.frame_shape)) * self.frame_dtype.itemsize
            + RAW_FRAME_HEIGHT * RAW_FRAME_WIDTH
            + file_bytes
        )

    def open(self) -> None:
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(path)

    def start(self) -> None:
        self.open()
        self.stop()
        self.exhausted = False
        self._buffer = np.empty((self._batch_frames, *self.frame_shape), dtype=self.frame_dtype)
        self._iterator = self._frames()
        self._next_read = time.monotonic()

    def _frames(self) -> Iterator[np.ndarray]:
        for path in self.paths:
            if path.suffix.lower() == ".bin":
                yield from _rle_frames(path.read_bytes())
            else:
                data = np.load(path, mmap_mode="r", allow_pickle=False)
                if data.ndim != 3:
                    raise ValueError("NPY replay requires 3D arrays")
                yield from data

    def read(self) -> np.ndarray | None:
        if self._iterator is None or self._buffer is None:
            raise RuntimeError("replay is not started")
        count = 0
        while count < self._batch_frames:
            try:
                frame = next(self._iterator)
            except StopIteration:
                self.exhausted = True
                break
            if frame.shape != self.frame_shape or frame.dtype != self.frame_dtype:
                raise ValueError("replay frame shape and dtype must remain constant")
            self._buffer[count] = frame
            count += 1
        if not count:
            return None
        if self._period_ms is not None:
            self._next_read += count * self._period_ms / 1000
            time.sleep(max(0.0, self._next_read - time.monotonic()))
        return self._buffer[:count]

    def stop(self) -> None:
        if self._iterator is not None:
            self._iterator.close()  # type: ignore[attr-defined]
        self._iterator = None
        self._buffer = None

    def close(self) -> None:
        self.stop()
