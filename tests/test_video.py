from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dropwatch import ApolloLifecycleError
from dropwatch import ApolloVideoSettings
from dropwatch import DropwatchApollo

from ._support import FakeFrameSource
from ._support import FakeVideoWriter
from ._support import frame
from ._support import settings


@pytest.fixture(autouse=True)
def mocked_video_verification(monkeypatch):
    # These tests exercise writer orchestration. Real codecs/readback are
    # covered separately in test_regressions.py, without these patches.
    monkeypatch.setattr("dropwatch._video._verify_video", lambda *_args: None)


def _sequence(offset: int = 0) -> np.ndarray:
    result = np.ones((5, 4, 4), dtype=np.uint8)
    for index in range(5):
        result[index, index % 3, (index + offset) % 4] = 0
    result.flags.writeable = False
    return result


def test_combined_video_preserves_shot_order_and_supports_recorder_options(monkeypatch, tmp_path):
    writers: list[FakeVideoWriter] = []
    writer_args: list[tuple[object, ...]] = []
    labels: list[str] = []

    def create_writer(*args):
        writer = FakeVideoWriter()
        writer.output_path = Path(args[0])
        writers.append(writer)
        writer_args.append(args)
        return writer

    def put_text(image, label, *_args):
        labels.append(label)
        return image

    monkeypatch.setattr("cv2.VideoWriter_fourcc", lambda *codec: codec)
    monkeypatch.setattr("cv2.VideoWriter", create_writer)
    monkeypatch.setattr("cv2.putText", put_text)
    first = _sequence()
    second = _sequence(offset=1)
    first_copy = first.copy()
    second_copy = second.copy()
    apollo = DropwatchApollo(settings(pre_trigger=3), frame_source=FakeFrameSource())
    options = ApolloVideoSettings(
        playback_fps=50,
        annotate=True,
        invert=False,
        trim_start=1,
        trim_end=4,
        crop_bottom=2,
        separator_frames=2,
    )

    output = apollo.save_video([first, second], tmp_path / "combined.mp4", options=options)

    assert output.read_bytes() == b"fake video"
    assert writer_args[0][1:] == (("m", "p", "4", "v"), 50, (4, 2), False)
    assert len(writers[0].frames) == 8
    assert labels == [
        "shot 0  frame 1  t=-2.000 ms",
        "shot 0  frame 2  t=-1.000 ms",
        "shot 0  frame 3  t=+0.000 ms",
        "shot 1  frame 1  t=-2.000 ms",
        "shot 1  frame 2  t=-1.000 ms",
        "shot 1  frame 3  t=+0.000 ms",
    ]
    assert np.all(np.stack(writers[0].frames[3:5]) == 255)
    assert np.array_equal(first, first_copy)
    assert np.array_equal(second, second_copy)
    assert not first.flags.writeable
    assert not second.flags.writeable
    assert not list(tmp_path.glob(".*.part.mp4"))


def test_per_shot_videos_are_independently_finalized(monkeypatch, tmp_path):
    writers: list[FakeVideoWriter] = []
    labels: list[str] = []

    def create_writer(*args):
        writer = FakeVideoWriter()
        writer.output_path = Path(args[0])
        writers.append(writer)
        return writer

    monkeypatch.setattr("cv2.VideoWriter_fourcc", lambda *_codec: 42)
    monkeypatch.setattr("cv2.VideoWriter", create_writer)
    monkeypatch.setattr("cv2.putText", lambda image, label, *_args: labels.append(label) or image)
    apollo = DropwatchApollo(settings(), frame_source=FakeFrameSource())

    paths = apollo.save_videos(
        [_sequence(), _sequence(offset=1)],
        tmp_path,
        prefix="drop",
        options=ApolloVideoSettings(annotate=True, separator_frames=10),
    )

    assert paths == [tmp_path / "drop_000.avi", tmp_path / "drop_001.avi"]
    assert all(path.read_bytes() == b"fake video" for path in paths)
    assert [len(writer.frames) for writer in writers] == [5, 5]
    assert labels[0].startswith("shot 0")
    assert labels[5].startswith("shot 1")


def test_video_writer_failure_never_replaces_an_existing_file(monkeypatch, tmp_path):
    class ClosedWriter(FakeVideoWriter):
        def isOpened(self) -> bool:
            return False

    writer = ClosedWriter()

    def create_writer(*args):
        writer.output_path = Path(args[0])
        return writer

    monkeypatch.setattr("cv2.VideoWriter_fourcc", lambda *_codec: 42)
    monkeypatch.setattr("cv2.VideoWriter", create_writer)
    output = tmp_path / "recording.avi"
    output.write_bytes(b"previous recording")
    apollo = DropwatchApollo(settings(), frame_source=FakeFrameSource())

    with pytest.raises(OSError, match="could not open"):
        apollo.save_avi(_sequence(), output)

    assert writer.released
    assert output.read_bytes() == b"previous recording"
    assert not list(tmp_path.glob(".*.part.avi"))


def test_video_write_failure_releases_writer_and_removes_partial_file(monkeypatch, tmp_path):
    class FailingWriter(FakeVideoWriter):
        def write(self, frame: np.ndarray) -> None:
            super().write(frame)
            if len(self.frames) == 2:
                raise OSError("disk full")

    writer = FailingWriter()

    def create_writer(*args):
        writer.output_path = Path(args[0])
        return writer

    monkeypatch.setattr("cv2.VideoWriter_fourcc", lambda *_codec: 42)
    monkeypatch.setattr("cv2.VideoWriter", create_writer)
    output = tmp_path / "recording.avi"
    apollo = DropwatchApollo(settings(), frame_source=FakeFrameSource())

    with pytest.raises(OSError, match="disk full"):
        apollo.save_video(_sequence(), output, options=ApolloVideoSettings(annotate=False))

    assert writer.released
    assert not output.exists()
    assert not list(tmp_path.glob(".*.part.avi"))


def test_instance_video_export_is_rejected_during_acquisition(monkeypatch, tmp_path):
    monkeypatch.setattr("cv2.VideoWriter", lambda *_args: pytest.fail("writer must not be opened"))
    source = FakeFrameSource()
    source.feed([frame(1), frame(2), frame(3)])
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.start()

    with pytest.raises(ApolloLifecycleError, match="before exporting video"):
        apollo.save_video(_sequence(), tmp_path / "unsafe.avi")

    apollo.abort()
    apollo.close()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (ApolloVideoSettings(trim_start=5), "removes every frame"),
        (ApolloVideoSettings(crop_bottom=4), "complete physical image height"),
    ],
)
def test_video_rejects_empty_output(options, message, tmp_path):
    apollo = DropwatchApollo(settings(), frame_source=FakeFrameSource())

    with pytest.raises(ValueError, match=message):
        apollo.save_video(_sequence(), tmp_path / "invalid.avi", options=options)
