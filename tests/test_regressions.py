"""Production-shaped regressions found in the second review."""

import base64
import gc
import hashlib
import json
import sys
import threading
import weakref
import zlib
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from dropwatch_apollo import ApolloVideoSettings
from dropwatch_apollo import DropwatchApollo
from dropwatch_apollo import ReplayFrameSource
from dropwatch_apollo._capture import _CapturedSequence
from dropwatch_apollo._capture import _SequenceCapture
from dropwatch_apollo._evaluation import _EvaluationRunner
from dropwatch_apollo._hardware import RLEDecoder
from dropwatch_apollo.replay import _rle_frames
from tests._support import FakeFrameSource
from tests._support import frame
from tests._support import frame_ids
from tests._support import one_sequence_frames
from tests._support import settings


def reference_recordings():
    fixtures = json.loads((Path(__file__).parent / "fixtures/rle_reference.json").read_text())
    return [
        (name, zlib.decompress(base64.b64decode(fixture["rle_zlib_base64"])), fixture)
        for name, fixture in fixtures.items()
    ]


@pytest.mark.parametrize("name,data,fixture", reference_recordings())
def test_real_rle_padding_and_reference_pixels(name, data, fixture):
    counters = RLEDecoder.completed_frame_counters(bytearray(data))
    assert len(counters) == fixture["frames"] == 4
    assert counters == list(range(counters[0], counters[0] + 4))
    frames = np.stack(list(_rle_frames(data)))
    assert hashlib.sha256(((frames != 0).astype(np.uint8) * 255).tobytes()).hexdigest() == fixture["left_uint8_sha256"]
    assert frames.shape == (4, 512, 1120)


@pytest.mark.skipif(sys.platform != "win32", reason="real vendor DLL requires Windows; no camera required")
@pytest.mark.parametrize("name,data,fixture", reference_recordings())
def test_real_vendor_decoder_contract(name, data, fixture):
    decoder = RLEDecoder()
    output = np.full((100, 512, 2240), 255, dtype=np.uint8)
    result = decoder.decode_buffer(bytearray(data), output)
    assert result.frames == 4
    assert result.generated_bytes == 4 * 512 * 2240
    assert 0 < result.consumed_bytes <= len(data)
    assert not any(data[result.consumed_bytes :])
    binary = (output[:4, :, :1120] != 0).astype(np.uint8) * 255
    assert hashlib.sha256(binary.tobytes()).hexdigest() == fixture["left_uint8_sha256"]


@pytest.mark.parametrize("suffix", [".avi", ".mp4"])
def test_actual_annotated_video_roundtrip(suffix, tmp_path):
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    frames = np.full((20, 512, 1120), 255, dtype=np.uint8)
    frames[:, 200:210, 300:320] = 0
    original = frames.copy()
    output = apollo.save_video(frames, tmp_path / f"annotated{suffix}")
    reader = cv2.VideoCapture(str(output))
    try:
        assert reader.isOpened()
        assert int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
        assert (reader.get(cv2.CAP_PROP_FRAME_WIDTH), reader.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (512, 1120)
        ok, image = reader.read()
        assert ok
        assert image[300:320, 200:210].mean() > 240
        assert image[:30].max() > 100  # actual text was drawn
    finally:
        reader.release()
    assert np.array_equal(frames, original)
    with pytest.raises(ValueError, match="even"):
        apollo.save_video(frames, tmp_path / f"odd{suffix}", options=ApolloVideoSettings(crop_bottom=1))


def test_stop_drains_a_droplet_behind_a_clear_inflight_batch():
    class BackloggedSource(FakeFrameSource):
        def __init__(self):
            super().__init__()
            self.reads = 0
            self.inflight = threading.Event()
            self.release = threading.Event()

        def read(self):
            self.reads += 1
            if self.reads == 1:
                return np.stack([frame(i) for i in (1, 2, 3)])
            if self.reads == 2:
                self.inflight.set()
                assert self.release.wait(2)
                return np.stack([frame(i) for i in (4, 5, 6)])
            return super().read()

    source = BackloggedSource()
    with DropwatchApollo(settings(), frame_source=source) as apollo:
        apollo.start(max_sequences=2)
        assert source.inflight.wait(1)
        source.feed([frame(i, drop=True) for i in range(7, 24)])
        holder = []
        worker = threading.Thread(target=lambda: holder.append(apollo.stop(timeout_s=1)))
        worker.start()
        assert apollo._drain_event.wait(1)
        source.release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert [frame_ids(s) for s in holder[0]] == [list(range(4, 24))]
        assert source._batches.empty()


@pytest.mark.parametrize("pre_trigger,second_trigger", [(5, 23), (19, 23)])
def test_history_survives_window_boundaries_and_materialization(pre_trigger, second_trigger):
    capture = _SequenceCapture(settings(pre_trigger))
    capture.reset(max_sequences=2)
    capture.prepare((4, 8), np.uint16)
    first_trigger = pre_trigger + 1
    shots = []
    for index in range(1, second_trigger + 20 - pre_trigger):
        result = capture.push(frame(index, drop=index in (first_trigger, second_trigger)))
        if result is not None:
            # Materializing the first prefix while it is still in history must
            # neither corrupt the second prefix nor re-arm after a blind gap.
            shots.append(result.materialize())
    assert len(shots) == 2
    assert frame_ids(shots[0]) == list(range(1, 21))
    assert frame_ids(shots[1]) == list(range(second_trigger - pre_trigger, second_trigger + 20 - pre_trigger))


def test_completed_buffer_is_released_when_it_leaves_history():
    capture = _SequenceCapture(settings(5))
    capture.reset(max_sequences=3)
    capture.prepare((4, 8), np.uint16)
    result = None
    for index in range(1, 21):
        result = capture.push(frame(index, drop=index == 6))
    assert result is not None
    ref = weakref.ref(result.frames.base)
    result.materialize()
    del result
    for index in range(21, 27):
        capture.push(frame(index))
    gc.collect()
    assert ref() is None


def test_hung_evaluator_releases_all_pending_buffers_on_cancel():
    entered, release = threading.Event(), threading.Event()

    def evaluator(_sequences):
        entered.set()
        release.wait(2)
        return pd.DataFrame({"shot": [0]})

    runner = _EvaluationRunner(evaluator)
    runner._STOP_TIMEOUT_S = 0.01
    runner.prepare(3)
    done = threading.Event()
    runner.start(done)
    refs = []
    try:
        for _ in range(3):
            result = _CapturedSequence(np.ones((20, 4, 8), dtype=np.uint8))
            refs.append(weakref.ref(result.frames))
            runner.submit(result)
        del result
        assert entered.wait(1)
        done.set()
        assert runner.finish() is not None
        gc.collect()
        assert refs[0]() is not None  # arbitrary running code still owns this
        assert refs[1]() is refs[2]() is None
    finally:
        release.set()
        runner._worker.join(2)
        runner.finish()


def test_snapshot_raw_png_and_npy_replay(tmp_path):
    source = FakeFrameSource()
    source.feed([frame(1)])
    with DropwatchApollo(settings(), frame_source=source) as apollo:
        image = apollo.snapshot(tmp_path / "snapshot.png")
        assert np.array_equal(image, frame(1))
        assert cv2.imread(str(tmp_path / "snapshot.png"), cv2.IMREAD_GRAYSCALE).shape == (8, 4)
        raw = np.stack(one_sequence_frames())
        paths = apollo.save_raw([raw], tmp_path / "raw")
        pngs = apollo.save_frames([raw[:2]], tmp_path / "png")
        assert len(pngs) == 2
    replay = ReplayFrameSource(paths, batch_frames=4)
    with DropwatchApollo(settings(), frame_source=replay) as apollo:
        apollo.start()
        assert frame_ids(apollo.get_sequence(1)) == list(range(1, 21))
        apollo.start()
        assert frame_ids(apollo.get_sequence(1)) == list(range(1, 21))


def test_duration_limited_recording_finishes_without_a_trigger():
    source = FakeFrameSource()
    source.feed([frame(1), frame(2), frame(3)])
    with DropwatchApollo(settings(), frame_source=source) as apollo:
        apollo.start(max_sequences=2, max_duration_s=0.02)
        assert apollo.get_sequences(1) == []


@pytest.mark.parametrize(
    "field,value", [("pre_trigger", 1.2), ("zero_byte_read_retries", True), ("zero_byte_retry_delay_ms", float("nan"))]
)
def test_invalid_settings_are_rejected_before_hardware(field, value):
    with pytest.raises(ValueError):
        replace(settings(), **{field: value})


def test_error_racing_with_result_dequeue_preserves_that_result(monkeypatch):
    apollo = DropwatchApollo(settings(), frame_source=FakeFrameSource())
    completed = np.stack([frame(i) for i in range(1, 21)])
    apollo._results.put_nowait(_CapturedSequence(completed))
    original_get = apollo._results.get_nowait

    def racing_get():
        item = original_get()
        apollo._worker_error = RuntimeError("concurrent transport failure")
        return item

    monkeypatch.setattr(apollo._results, "get_nowait", racing_get)
    with pytest.raises(RuntimeError, match="concurrent transport failure") as error:
        apollo.get_sequence(0)
    assert [frame_ids(seq) for seq in error.value.completed_sequences] == [list(range(1, 21))]


def test_export_guard_also_prevents_start_during_export(tmp_path, monkeypatch):
    apollo = DropwatchApollo(settings(), frame_source=FakeFrameSource())

    def encode(*args, **_kwargs):
        with pytest.raises(Exception, match="export or raw recording"):
            apollo.start()
        return args[1]

    monkeypatch.setattr("dropwatch_apollo.apollo._save_video", encode)
    apollo.save_video(np.ones((20, 4, 8), dtype=np.uint8), tmp_path / "video.avi")
    assert not apollo._exporting


def test_failed_session_cannot_silently_discard_uncollected_shots_on_restart():
    source = FakeFrameSource()
    with DropwatchApollo(settings(), frame_source=source) as apollo:
        source.feed(one_sequence_frames())
        apollo.start(max_sequences=2)
        source.fail(RuntimeError("late failure"))
        assert apollo._worker_done.wait(1)
        with pytest.raises(RuntimeError, match="late failure") as error:
            apollo.start()
        assert len(error.value.completed_sequences) == 1
        source.feed(one_sequence_frames(start_id=21))
        apollo.start()
        assert frame_ids(apollo.get_sequence(1)) == list(range(21, 41))
