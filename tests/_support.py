from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from dropwatch_apollo import ApolloSettings


class FakeFrameSource:
    frame_shape = (4, 8)
    frame_dtype = np.uint16

    def __init__(self) -> None:
        self._batches: queue.Queue[np.ndarray | Exception] = queue.Queue()
        self.open_count = 0
        self.start_count = 0
        self.stop_count = 0
        self.close_count = 0
        self.opened = False
        self.started = False

    def open(self) -> None:
        if not self.opened:
            self.opened = True
            self.open_count += 1

    def start(self) -> None:
        if not self.opened:
            raise RuntimeError("source is closed")
        self.started = True
        self.start_count += 1

    def read(self) -> np.ndarray | None:
        if not self.started:
            raise RuntimeError("source is not started")
        try:
            item = self._batches.get(timeout=0.01)
        except queue.Empty:
            return None
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self) -> None:
        if self.started:
            self.started = False
            self.stop_count += 1

    def close(self) -> None:
        self.stop()
        if self.opened:
            self.opened = False
            self.close_count += 1

    def feed(self, frames: list[np.ndarray]) -> None:
        self._batches.put(np.stack(frames))

    def fail(self, error: Exception) -> None:
        self._batches.put(error)


class FakeVideoWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False
        self.output_path: Path | None = None

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True
        if self.output_path is not None:
            self.output_path.write_bytes(b"fake video")


def settings(pre_trigger: int = 3) -> ApolloSettings:
    return ApolloSettings(
        max_number_frames=20,
        pre_trigger=pre_trigger,
        trigger_position_px=2,
        trigger_from_top=True,
        trigger_width_px=2,
        trigger_on_pixels=2,
        trigger_off_pixels=0,
        rearm_clear_frames=2,
    )


def frame(frame_id: int, *, drop: bool = False) -> np.ndarray:
    image = np.ones((4, 8), dtype=np.uint16)
    image[0, 0] = frame_id
    if drop:
        image[:, 2:4] = 0
    return image


def plate_frame(*, plate_top: int = 150) -> np.ndarray:
    image = np.ones((20, 200), dtype=np.uint8)
    image[4:16, plate_top:] = 0
    image[8:12, 50:55] = 0
    return image


def one_sequence_frames(pre_trigger: int = 3, start_id: int = 1) -> list[np.ndarray]:
    warmup_frames = max(pre_trigger, 2)
    frames = [frame(start_id + index) for index in range(warmup_frames)]
    trigger_id = start_id + warmup_frames
    frames.append(frame(trigger_id, drop=True))
    remaining = 20 - pre_trigger - 1
    frames.extend(frame(trigger_id + index + 1, drop=True) for index in range(remaining))
    return frames


def expected_sequence_ids(pre_trigger: int = 3, start_id: int = 1) -> list[int]:
    warmup_frames = max(pre_trigger, 2)
    first_id = start_id + warmup_frames - pre_trigger
    return list(range(first_id, first_id + 20))


def multiple_sequence_frames(count: int, pre_trigger: int = 3) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for sequence_index in range(count):
        result.extend(one_sequence_frames(pre_trigger, start_id=sequence_index * 20 + 1))
    return result


def frame_ids(sequence: np.ndarray) -> list[int]:
    return [int(item[0, 0]) for item in sequence]


def apollo_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "dropwatch-apollo-acquisition"]


def evaluation_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "dropwatch-apollo-evaluation"]
