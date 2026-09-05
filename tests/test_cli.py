import sys

import cv2
import numpy as np
import pytest

from dropwatch_apollo import DropwatchApollo
from dropwatch_apollo.__main__ import main
from tests._support import FakeFrameSource


@pytest.mark.parametrize("command", ["record", "replay", "snapshot", "preview"])
def test_cli_real_capture_export_and_replay(command, tmp_path, monkeypatch):
    plate = np.full((512, 1120), 255, dtype=np.uint8)
    plate[:, 1000:] = 0
    frames = np.stack([plate.copy() for _ in range(25)])
    frames[5:, :, 900:1000] = 0
    source = FakeFrameSource()
    source.frame_shape = (512, 1120)
    source.frame_dtype = np.uint8
    if command == "record":
        source.feed([plate])
    source.feed(list(frames))
    recorder = DropwatchApollo
    monkeypatch.setattr(
        "dropwatch_apollo.__main__.DropwatchApollo",
        lambda settings, frame_source=None: recorder(
            settings, frame_source=frame_source if frame_source is not None else source
        ),
    )
    monkeypatch.setattr(cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *_: 27)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    argv = ["dwa", command]
    if command == "replay":
        np.save(tmp_path / "input.npy", frames)
        argv.append(str(tmp_path / "input.npy"))
    if command == "record":
        argv.append("--auto-trigger")
    output = tmp_path / "output"
    argv += [
        "--output",
        str(output),
        "--frames",
        "20",
        "--pre-trigger",
        "5",
        "--trigger-position",
        "120",
        "--trigger-width",
        "100",
        "--duration",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    if command in {"record", "replay"}:
        videos = list(output.rglob("*.avi"))
        assert len(videos) == 1
        capture = cv2.VideoCapture(str(videos[0]))
        try:
            assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 20
        finally:
            capture.release()
        if command == "record":
            assert len(list(output.rglob("*.npy"))) == 1
    elif command == "snapshot":
        assert (output / "snapshot.png").is_file()
    if command != "replay":
        assert source.close_count == 1


def test_cli_replay_requires_input(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dwa", "replay"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_failed_close_can_retry_same_source():
    class RetryCloseSource(FakeFrameSource):
        attempts = 0

        def close(self):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("busy closing device")
            super().close()

    from tests._support import settings

    source = RetryCloseSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.open()
    with pytest.raises(OSError, match="busy closing"):
        apollo.close()
    apollo.close()
    assert source.attempts == 2
    assert source.close_count == 1
