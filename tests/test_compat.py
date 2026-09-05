import gc
import threading
import time
import weakref
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest

from dropwatch_apollo import ApolloLifecycleError
from dropwatch_apollo import ApolloTransportError
from dropwatch_apollo import DropwatchApollo
from dropwatch_apollo._capture import _SequenceCapture
from dropwatch_apollo._chunked import _ChunkedCapture
from dropwatch_apollo._chunked import _ChunkedSpool
from dropwatch_apollo._video import legacy_header_height
from recorder.api.dropwatch import Dropwatch
from recorder.core.capture import BufferedCaptureState
from recorder.core.capture import CaptureState
from recorder.core.detectors import DisplayRoi2D
from recorder.core.detectors import RoiPixelDetector
from recorder.core.recorder import FastPostTriggerRecorder
from recorder.core.recorder import Recorder
from recorder.core.sinks import LegacyVideoSaver
from recorder.settings import CameraSettings
from tests._support import FakeFrameSource
from tests._support import frame
from tests._support import frame_ids
from tests._support import multiple_sequence_frames
from tests._support import settings


def pipeline(tmp_path, *, capture_len=20, lookback=0, sinks=None, **kwargs):
    detector = RoiPixelDetector(DisplayRoi2D(y0=2, y1=4, x0=0, x1=4), 2, "left")
    capture = BufferedCaptureState(detector, capture_len, lookback) if lookback else CaptureState(detector, capture_len)
    recorder = Recorder(capture, tmp_path, sinks=sinks, keep_sequences=True)
    source = FakeFrameSource()
    return Dropwatch(recorder, frame_source=source, **kwargs), source, capture, recorder


def eventually(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.002)


def test_existing_context_records_multiple_windows_and_keeps_decoder_independent(tmp_path):
    dw, source, capture, recorder = pipeline(tmp_path, decoder_max_images=1000)
    source.feed([frame(1), frame(2)])
    with dw.recording(max_duration=30, ready_timeout=1, join_timeout=2):
        assert dw.is_recording
        assert dw._core.settings.rle_batch_frames == 100
        # Level trigger: a continuously occupied ROI begins a second window.
        source.feed([frame(i, drop=True) for i in range(40)])
        eventually(lambda: capture.trigger_count == 2 and recorder.completed_windows == 2)
    assert not dw.is_recording
    assert source.close_count == 1
    assert [frame_ids(s) for s in recorder.sequences] == [list(range(20)), list(range(20, 40))]
    assert all(isinstance(s, np.memmap) and not s.flags.writeable for s in recorder.sequences)
    assert len(list(dw.recording_directory.glob("shot_*.npy"))) == 2


def test_pretrigger_is_additive_in_legacy_interface(tmp_path):
    dw, source, _, recorder = pipeline(tmp_path, capture_len=5, lookback=3)
    source.feed([frame(i) for i in range(3)])
    with dw.recording(max_num_triggers=1, ready_timeout=1, join_timeout=2):
        source.feed([frame(i, drop=True) for i in range(3, 8)])
        eventually(lambda: recorder.completed_windows == 1)
    assert frame_ids(recorder.sequences[0]) == list(range(8))


def test_long_window_crosses_chunks_without_allocating_the_whole_shot(tmp_path, monkeypatch):
    allocated = []
    real_empty = np.empty

    def allocate(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 3:
            allocated.append(shape)
        return real_empty(shape, *args, **kwargs)

    dw, source, _, recorder = pipeline(tmp_path, capture_len=3201, spool_chunk_frames=64)
    source.feed([frame(1)])
    monkeypatch.setattr("dropwatch_apollo._chunked.np.empty", allocate)
    with dw.recording(max_num_triggers=1, ready_timeout=1, join_timeout=3):
        for offset in range(0, 3201, 64):
            source.feed([frame(i, drop=True) for i in range(offset, min(offset + 64, 3201))])
            eventually(lambda: source._batches.empty() and dw._core._spool._tasks.empty())
        eventually(lambda: recorder.completed_windows == 1)
    assert max(s[0] for s in allocated) == 64
    assert frame_ids(recorder.sequences[0]) == list(range(3201))


@pytest.mark.parametrize("chunked", [False, True])
def test_rectangular_roi_excludes_objects_outside_short_axis(tmp_path, chunked):
    config = replace(settings(0), trigger_roi=DisplayRoi2D(y0=2, y1=4, x0=2, x1=4))
    if chunked:
        config = replace(config, spool_directory=tmp_path, spool_chunk_frames=8)

        def discard(chunk):
            chunk.release(chunk.buffer)

        capture = _ChunkedCapture(config, discard)
    else:
        capture = _SequenceCapture(config)
    capture.prepare((4, 8), np.uint16)
    for i in range(2):
        capture.push(frame(i))
    outside = frame(3)
    outside[:2, 2:4] = 0
    capture.push(outside)
    assert capture.trigger_count == 0
    capture.push(frame(4, drop=True))
    assert capture.trigger_count == 1


def test_startup_timeout_never_enters_dispense_body_and_keeps_live_thread(tmp_path):
    dw, source, _, _ = pipeline(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def blocked_read():
        entered.set()
        release.wait(3)
        return None

    source.read = blocked_read
    try:
        with pytest.raises(TimeoutError), dw.recording(ready_timeout=0.02, join_timeout=0.02):
            pytest.fail("unsafe: dispense body entered before readiness")
        assert entered.is_set()
        assert isinstance(dw._thread, threading.Thread) and dw._thread.is_alive()
        assert source.close_count == 0
        with pytest.raises(ApolloLifecycleError, match="still running"):
            dw.start_background()
    finally:
        release.set()
        eventually(lambda: not dw.is_recording)
    assert source.close_count == 1


def test_video_finalization_is_included_in_real_thread_lifetime(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    sink = LegacyVideoSaver(tmp_path / "exact.avi", codec="MJPG", invert_bw=True)
    dw, source, _, recorder = pipeline(tmp_path, sinks=[sink])

    def save(*args):
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(sink, "save", save)
    source.feed([frame(1)])
    dw.start_background(max_num_triggers=1)
    dw.wait_until_ready(1)
    source.feed([frame(i, drop=True) for i in range(20)])
    try:
        assert entered.wait(2)
        assert source.close_count == 1
        with pytest.raises(ApolloLifecycleError, match="camera already stopped"):
            dw.wait_until_ready(0.01)
        with pytest.raises(TimeoutError, match="still running"):
            dw.join(0.01)
        assert dw._thread.is_alive()
    finally:
        release.set()
        dw.join(2)
    assert recorder.completed_windows == 1


def test_stream_failure_preserves_only_complete_windows_and_original_error(tmp_path):
    dw, source, capture, recorder = pipeline(tmp_path)
    source.feed([frame(1)])
    failure = ApolloTransportError("unrecoverable read")
    with pytest.raises(ApolloTransportError) as caught, dw.recording(max_duration=30, ready_timeout=1, join_timeout=2):
        source.feed([frame(i, drop=True) for i in range(23)])
        eventually(lambda: capture.trigger_count == 2)
        source.fail(failure)
        eventually(lambda: not dw.is_recording)
    assert caught.value is failure
    assert len(recorder.sequences) == 1
    assert recorder.partial_windows == 1
    assert len(list(dw.recording_directory.glob("shot_*.npy"))) == 1
    assert not list(dw.recording_directory.glob("*.part"))
    assert source.close_count == 1


def test_chunk_quota_and_slow_writer_fail_explicitly(tmp_path):
    config = replace(settings(0), spool_directory=tmp_path, spool_chunk_frames=4, max_spool_bytes=100)
    spool = _ChunkedSpool(config)
    capture = _ChunkedCapture(config, spool.submit)
    capture.prepare((4, 8), np.uint16)
    with pytest.raises(OSError, match="max_spool_bytes"):
        spool.start(tmp_path, 3, 1280, lambda _: None)
    spool = _ChunkedSpool(replace(config, max_spool_bytes=500))
    capture = _ChunkedCapture(config, spool.submit)
    capture.prepare((4, 8), np.uint16)
    spool.start(tmp_path, 3, 0, lambda _: pytest.fail("must not publish a partial shot"))
    for i in range(2):
        capture.push(frame(i))
    for i in range(4):
        capture.push(frame(i, drop=True))
    spool.finish()
    assert isinstance(spool.error, OSError) and "max_spool_bytes" in str(spool.error)
    assert list(spool.directory.iterdir()) == []
    held = []
    capture = _ChunkedCapture(config, held.append)
    capture.prepare((4, 8), np.uint16)
    for i in range(2):
        capture.push(frame(i))
    with pytest.raises(ApolloLifecycleError, match="cannot keep up"):
        for i in range(20):
            capture.push(frame(i, drop=True))
    assert len(held) == 3  # Never waits for the writer or drops an old chunk.


def test_repeated_contexts_release_all_chunk_buffers_and_threads(tmp_path, monkeypatch):
    buffers = []
    real_empty = np.empty

    def allocate(*args, **kwargs):
        result = real_empty(*args, **kwargs)
        buffers.append(weakref.ref(result))
        return result

    dw, source, _, recorder = pipeline(tmp_path)
    monkeypatch.setattr("dropwatch_apollo._chunked.np.empty", allocate)
    for _ in range(100):
        source.feed([frame(1)])
        with dw.recording(max_num_triggers=1, ready_timeout=1, join_timeout=2):
            source.feed([frame(i, drop=True) for i in range(20)])
            eventually(lambda: recorder.completed_windows == 1)
        assert len(recorder.sequences) == 1
    gc.collect()
    assert all(ref() is None for ref in buffers)
    assert source.close_count == source.start_count == 100
    assert not [t for t in threading.enumerate() if t.name.startswith("dropwatch-apollo-")]


def test_global_evaluation_finalizer_sees_original_shot_ids_in_order():
    finalized = []

    def evaluate(sequences):
        return pd.DataFrame({"shot": [0], "value": [int(sequences[0][0, 0, 0])]})

    def finalize(data):
        finalized.append(data.copy())
        return pd.DataFrame({"sum": [data.value.sum()]})

    source = FakeFrameSource()
    source.feed(multiple_sequence_frames(2))
    with DropwatchApollo(settings(), frame_source=source, evaluator=evaluate, evaluation_finalizer=finalize) as dw:
        dw.start(max_sequences=2)
        dw.get_sequences(2)
        result = dw.get_evaluations(2)
    assert result.to_dict("records") == [{"sum": 22}]
    assert finalized[0].shot.tolist() == [0, 1]


def test_real_legacy_avi_has_left_orientation_header_polarity_and_final_separator(tmp_path):
    image = np.ones((64, 96), np.uint8)
    image[16:32, 40:60] = 0
    sequences = [np.stack([image] * 3), np.stack([image] * 2)]
    path = tmp_path / "requested.avi"
    saver = LegacyVideoSaver(path, codec="MJPG", invert_bw=True)
    saver.save(sequences, tmp_path)
    reader = cv2.VideoCapture(str(path))
    try:
        assert reader.get(cv2.CAP_PROP_FRAME_COUNT) == 7
        assert reader.get(cv2.CAP_PROP_FPS) == 16
        frames = [reader.read()[1] for _ in range(7)]
        header = legacy_header_height()
        assert frames[0].shape == (96 + header + header % 2, 64, 3)
        np.testing.assert_array_equal(frames[0][header : header + 96, :, 0] > 127, image.T == 0)
        assert np.max(frames[3]) < 10 and np.max(frames[-1]) < 10
        assert np.max(frames[0][:header]) > 100
    finally:
        reader.release()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CameraSettings(num_img_flush=1000),
        lambda: RoiPixelDetector(DisplayRoi2D(1, 2, 1, 2), view="both"),
        lambda: LegacyVideoSaver(check_white_stripe=True),
    ],
)
def test_unsupported_unsafe_legacy_options_fail_before_hardware(factory):
    with pytest.raises(ValueError):
        factory()


def test_fast_recorder_keeps_no_raw_references(tmp_path):
    detector = RoiPixelDetector(DisplayRoi2D(2, 4, 0, 4), 2)
    recorder = FastPostTriggerRecorder(CaptureState(detector, 20), tmp_path)
    source = FakeFrameSource()
    source.feed([frame(1)])
    dw = Dropwatch(recorder, frame_source=source)
    with dw.recording(max_num_triggers=1, ready_timeout=1, join_timeout=2):
        source.feed([frame(i, drop=True) for i in range(20)])
        eventually(lambda: recorder.completed_windows == 1)
    assert recorder.sequences == []
    assert dw._core._results.empty()


def test_failed_vendor_close_retains_handle_and_blocks_restart_until_retry(tmp_path):
    dw, source, _, recorder = pipeline(tmp_path)
    original_close = source.close
    fail = True

    def close():
        if fail:
            raise OSError("vendor close failed")
        original_close()

    source.close = close
    source.feed([frame(1)])
    with (
        pytest.raises(OSError, match="vendor close failed"),
        dw.recording(max_num_triggers=1, ready_timeout=1, join_timeout=2),
    ):
        source.feed([frame(i, drop=True) for i in range(20)])
        eventually(lambda: recorder.completed_windows == 1)
    assert source.opened
    with pytest.raises(ApolloLifecycleError, match="cleanup failed"):
        dw.start_background()
    fail = False
    dw.close()
    assert not source.opened
    assert dw._cleanup_ok


def test_evaluation_failure_does_not_discard_completed_images(tmp_path):
    from dropwatch_apollo import ApolloEvaluationError

    def evaluate(_sequences):
        raise ValueError("tracking failed")

    dw, source, _, recorder = pipeline(tmp_path, evaluator=evaluate)
    source.feed([frame(1)])
    with pytest.raises(ApolloEvaluationError), dw.recording(max_num_triggers=1, ready_timeout=1, join_timeout=2):
        source.feed([frame(i, drop=True) for i in range(20)])
        eventually(lambda: recorder.completed_windows == 1)
    assert len(recorder.sequences) == 1
    assert frame_ids(recorder.sequences[0]) == list(range(20))


def test_partial_file_is_removed_when_stream_fails_mid_chunked_window(tmp_path):
    dw, source, capture, recorder = pipeline(tmp_path, capture_len=200, spool_chunk_frames=64)
    source.feed([frame(1)])
    with pytest.raises(ApolloTransportError), dw.recording(max_duration=30, ready_timeout=1, join_timeout=2):
        source.feed([frame(i, drop=True) for i in range(200)])
        eventually(lambda: len(list(dw.recording_directory.glob("shot_*.npy"))) == 1)
        source.feed([frame(i, drop=True) for i in range(80)])
        eventually(lambda: bool(list(dw.recording_directory.glob("*.part"))))
        source.fail(ApolloTransportError("stream lost"))
        eventually(lambda: not dw.is_recording)
    assert capture.trigger_count == 2
    assert len(recorder.sequences) == 1
    assert not list(dw.recording_directory.glob("*.part"))


def test_production_long_window_allocation_is_independent_of_window_length(tmp_path, monkeypatch):
    allocated = []
    real_empty = np.empty

    def allocate(shape, *args, **kwargs):
        allocated.append(shape)
        return real_empty((shape[0], 4, 8), *args, **kwargs)

    config = replace(
        settings(3),
        pre_trigger=20,
        max_number_frames=55_020,
        spool_directory=tmp_path,
        spool_chunk_frames=100,
        spool_buffer_count=8,
    )
    capture = _ChunkedCapture(config, lambda _: None)
    monkeypatch.setattr("dropwatch_apollo._chunked.np.empty", allocate)
    camera_bytes = 100 * 512 * 2240 + 1_228_800
    capture.prepare((512, 1120), np.uint8, additional_buffer_bytes=camera_bytes)
    assert allocated == [(100, 512, 1120)] * 8 + [(20, 512, 1120)]
    assert sum(np.prod(shape) for shape in allocated) + camera_bytes < 600 * 1024**2
