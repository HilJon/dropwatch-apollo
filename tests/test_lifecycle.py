from __future__ import annotations

import gc
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest

from dropwatch_apollo import ApolloFrameLossError
from dropwatch_apollo import ApolloIncompleteSequenceError
from dropwatch_apollo import ApolloLifecycleError
from dropwatch_apollo import ApolloSettings
from dropwatch_apollo import DropwatchApollo

from ._support import FakeFrameSource
from ._support import FakeVideoWriter
from ._support import apollo_threads
from ._support import frame
from ._support import frame_ids
from ._support import multiple_sequence_frames
from ._support import one_sequence_frames
from ._support import settings


def test_settings_validate_public_frame_contract():
    defaults = ApolloSettings(max_number_frames=20)
    assert defaults.trigger_from_top is False
    assert defaults.frame_period_ms == 1.0
    assert defaults.exposure_time_ms == 0.05
    assert defaults.threshold == 127
    assert defaults.zero_byte_read_retries == 1
    assert defaults.effective_read_timeout_ms == 500
    with pytest.raises(ValueError, match="between 20 and 2000"):
        ApolloSettings(max_number_frames=19)
    with pytest.raises(ValueError, match="smaller than max_number_frames"):
        ApolloSettings(max_number_frames=20, pre_trigger=20)
    with pytest.raises(ValueError, match="greater than trigger_off_pixels"):
        ApolloSettings(
            max_number_frames=20,
            trigger_on_pixels=10,
            trigger_off_pixels=10,
        )
    with pytest.raises(ValueError, match="rle_batch_frames"):
        ApolloSettings(max_number_frames=20, rle_batch_frames=99)
    with pytest.raises(ValueError, match="frame_period_ms"):
        ApolloSettings(max_number_frames=20, frame_period_ms=0)
    with pytest.raises(ValueError, match="exposure_time_ms"):
        ApolloSettings(
            max_number_frames=20,
            frame_period_ms=0.5,
            exposure_time_ms=0.6,
        )
    with pytest.raises(ValueError, match="threshold"):
        ApolloSettings(max_number_frames=20, threshold=256)
    with pytest.raises(ValueError, match="read_timeout_ms"):
        ApolloSettings(max_number_frames=20, read_timeout_ms=149)
    with pytest.raises(ValueError, match="zero_byte_read_retries"):
        ApolloSettings(max_number_frames=20, zero_byte_read_retries=-1)
    with pytest.raises(ValueError, match="zero_byte_read_retries"):
        ApolloSettings(max_number_frames=20, zero_byte_read_retries=4)
    with pytest.raises(ValueError, match="zero_byte_retry_delay_ms"):
        ApolloSettings(max_number_frames=20, zero_byte_retry_delay_ms=-0.1)
    with pytest.raises(ValueError, match="max_buffer_bytes"):
        ApolloSettings(max_number_frames=20, max_buffer_bytes=0)

    source = FakeFrameSource()
    apollo = DropwatchApollo(defaults, frame_source=source)
    with pytest.raises(ValueError, match="max_sequences"):
        apollo.start(max_sequences=0)


def test_save_avi_writes_binary_frames_and_releases_writer(monkeypatch, tmp_path):
    monkeypatch.setattr("dropwatch_apollo._video._verify_video", lambda *_args: None)
    writer = FakeVideoWriter()
    writer_args: list[object] = []

    def create_writer(*args):
        writer_args.extend(args)
        writer.output_path = Path(args[0])
        return writer

    monkeypatch.setattr("cv2.VideoWriter_fourcc", lambda *codec: 42)
    monkeypatch.setattr("cv2.VideoWriter", create_writer)
    sequence = np.asarray(
        [
            [[0, 1, 1, 1], [1, 0, 1, 1]],
            [[1, 1, 0, 1], [0, 0, 1, 1]],
        ],
        dtype=np.uint8,
    )

    apollo = DropwatchApollo(settings(), frame_source=FakeFrameSource())
    result = apollo.save_avi(sequence, tmp_path / "recording.avi")

    assert result == Path(tmp_path / "recording.avi")
    assert Path(writer_args[0]).name.startswith(".recording.")
    assert Path(writer_args[0]).suffix == ".avi"
    assert writer_args[1:] == [42, 25.0, (2, 4), False]
    assert np.array_equal(
        writer.frames[0],
        np.asarray([[255, 0], [0, 255], [0, 0], [0, 0]], dtype=np.uint8),
    )
    assert len(writer.frames) == 2
    assert writer.released
    assert result.read_bytes() == b"fake video"
    assert list(tmp_path.glob("*.part.avi")) == []


def test_start_timeout_stops_and_releases_acquisition_buffers():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)

    with pytest.raises(TimeoutError, match="did not become armed"):
        apollo.start(timeout_s=0.02)

    assert not apollo.is_running
    assert apollo._capture._sequence is None
    assert apollo._capture._pre_trigger_buffer is None
    assert source.stop_count == 1
    apollo.close()


def test_one_start_captures_one_sequence_and_stops_camera_intake():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    source.feed(one_sequence_frames())
    source.feed(one_sequence_frames(start_id=21))
    apollo.start()

    first_sequence = apollo.get_sequence(timeout_s=1)
    apollo.stop()
    apollo.close()

    assert frame_ids(first_sequence) == list(range(1, 21))
    assert source._batches.qsize() == 1
    assert source.stop_count == 1


def test_bounded_multi_trigger_returns_all_sequences_without_queue_backpressure():
    source = FakeFrameSource()
    source.feed(multiple_sequence_frames(3))
    apollo = DropwatchApollo(settings(), frame_source=source)

    apollo.start(max_sequences=3)
    sequences = apollo.get_sequences(timeout_s=1)

    assert [frame_ids(sequence) for sequence in sequences] == [
        list(range(1, 21)),
        list(range(21, 41)),
        list(range(41, 61)),
    ]
    assert apollo.stats.sequences_captured == 3
    assert source.stop_count == 1
    assert not apollo.is_running
    apollo.close()


def test_multi_trigger_reserves_every_sequence_buffer_before_arming():
    source = FakeFrameSource()
    source.feed([frame(1), frame(2), frame(3)])
    apollo = DropwatchApollo(settings(), frame_source=source)

    apollo.start(max_sequences=3)

    assert apollo._capture._sequence_pool.qsize() == 2
    assert apollo._capture._sequence.shape == (20, 4, 8)
    source.feed([frame(index, drop=True) for index in range(4, 21)])
    source.feed(multiple_sequence_frames(2))
    sequences = apollo.get_sequences(timeout_s=1)
    apollo.close()

    assert len(sequences) == 3


def test_multi_trigger_can_be_consumed_one_sequence_at_a_time():
    source = FakeFrameSource()
    source.feed(one_sequence_frames())
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.start(max_sequences=2)

    first = apollo.get_sequence(timeout_s=1)
    source.feed(one_sequence_frames(start_id=21))
    second = apollo.get_sequence(timeout_s=1)
    remaining = apollo.stop()
    apollo.close()

    assert frame_ids(first) == list(range(1, 21))
    assert frame_ids(second) == list(range(21, 41))
    assert remaining == []


def test_early_stop_returns_only_complete_multi_trigger_sequences():
    source = FakeFrameSource()
    source.feed(one_sequence_frames())
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.start(max_sequences=3)

    deadline = time.monotonic() + 1
    while apollo.stats.sequences_captured < 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    sequences = apollo.stop()

    assert len(sequences) == 1
    assert frame_ids(sequences[0]) == list(range(1, 21))
    assert apollo._capture._sequence_pool.empty()
    apollo.close()


def test_stop_drains_an_already_triggered_sequence_without_waiting_for_another():
    source = FakeFrameSource()
    source.feed([frame(1), frame(2), frame(3), frame(4, drop=True), frame(5, drop=True)])
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.start(max_sequences=3)

    deadline = time.monotonic() + 1
    while not apollo._capture.is_capturing and time.monotonic() < deadline:
        time.sleep(0.001)

    stopped = threading.Event()
    holder: list[list[np.ndarray]] = []

    def stop_apollo() -> None:
        holder.append(apollo.stop(timeout_s=1))
        stopped.set()

    stop_thread = threading.Thread(target=stop_apollo)
    stop_thread.start()
    assert apollo._drain_event.wait(1)
    assert not stopped.is_set()
    source.feed([frame(index, drop=True) for index in range(6, 21)])
    stop_thread.join(1)

    assert not stop_thread.is_alive()
    assert [frame_ids(sequence) for sequence in holder[0]] == [list(range(1, 21))]
    assert apollo.stats.incomplete_sequences == 0
    assert source.stop_count == 1
    apollo.close()


def test_stop_processes_an_inflight_batch_before_deciding_there_is_no_trigger():
    class InflightSource(FakeFrameSource):
        def __init__(self) -> None:
            super().__init__()
            self.read_count = 0
            self.inflight = threading.Event()
            self.release = threading.Event()

        def read(self) -> np.ndarray | None:
            self.read_count += 1
            if self.read_count == 1:
                return np.stack([frame(1), frame(2), frame(3)])
            if self.read_count == 2:
                self.inflight.set()
                assert self.release.wait(1)
                return np.stack([frame(index, drop=True) for index in range(4, 21)])
            return None

    source = InflightSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.start(max_sequences=2)
    assert source.inflight.wait(1)

    holder: list[list[np.ndarray]] = []
    stop_thread = threading.Thread(target=lambda: holder.append(apollo.stop(timeout_s=1)))
    stop_thread.start()
    assert apollo._drain_event.wait(1)
    source.release.set()
    stop_thread.join(1)

    assert not stop_thread.is_alive()
    assert [frame_ids(sequence) for sequence in holder[0]] == [list(range(1, 21))]
    apollo.close()


def test_stop_reports_a_triggered_sequence_that_cannot_be_drained():
    source = FakeFrameSource()
    source.feed([frame(1), frame(2), frame(3), frame(4, drop=True)])
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.start(max_sequences=2)

    deadline = time.monotonic() + 1
    while not apollo._capture.is_capturing and time.monotonic() < deadline:
        time.sleep(0.001)
    with pytest.raises(ApolloIncompleteSequenceError, match="could not drain") as error:
        apollo.stop(timeout_s=0.01)

    assert error.value.completed_sequences == []
    assert apollo.stats.incomplete_sequences == 1
    assert not apollo.is_running
    apollo.close()


def test_drain_timeout_preserves_earlier_complete_sequences_on_the_error():
    source = FakeFrameSource()
    frames = one_sequence_frames()
    frames.extend([frame(21), frame(22), frame(23), frame(24, drop=True)])
    source.feed(frames)
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.start(max_sequences=3)

    deadline = time.monotonic() + 1
    while apollo.stats.sequences_captured < 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    while not apollo._capture.is_capturing and time.monotonic() < deadline:
        time.sleep(0.001)

    with pytest.raises(ApolloIncompleteSequenceError) as error:
        apollo.stop(timeout_s=0.01)

    assert [frame_ids(sequence) for sequence in error.value.completed_sequences] == [list(range(1, 21))]
    assert apollo.stats.sequences_captured == 1
    assert apollo.stats.incomplete_sequences == 1
    apollo.close()


def test_abort_explicitly_discards_an_active_sequence():
    source = FakeFrameSource()
    source.feed([frame(1), frame(2), frame(3), frame(4, drop=True)])
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.start(max_sequences=2)

    deadline = time.monotonic() + 1
    while not apollo._capture.is_capturing and time.monotonic() < deadline:
        time.sleep(0.001)
    assert apollo.abort() == []
    assert apollo.stats.incomplete_sequences == 1
    apollo.close()


def test_get_sequences_timeout_keeps_multi_trigger_session_running():
    source = FakeFrameSource()
    source.feed(one_sequence_frames())
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.start(max_sequences=2)

    with pytest.raises(TimeoutError, match="1 of 2 sequences"):
        apollo.get_sequences(timeout_s=0.02)
    assert apollo.is_running
    sequences = apollo.stop()
    apollo.close()

    assert len(sequences) == 1


def test_timeout_does_not_stop_acquisition():
    source = FakeFrameSource()
    with DropwatchApollo(settings(), frame_source=source) as apollo:
        source.feed([frame(1), frame(2), frame(3)])
        apollo.start()
        with pytest.raises(TimeoutError, match="no triggered sequence"):
            apollo.get_sequence(timeout_s=0.02)
        assert apollo.is_running
        apollo.stop()


def test_read_error_is_propagated_and_source_is_stopped():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    source.feed([frame(1), frame(2), frame(3)])
    apollo.start()
    source.fail(RuntimeError("read failed"))

    with pytest.raises(RuntimeError, match="read failed"):
        apollo.get_sequence(timeout_s=1)
    with pytest.raises(RuntimeError, match="read failed"):
        apollo.stop()
    with pytest.raises(RuntimeError, match="read failed"):
        apollo.close()

    assert source.stop_count == 1
    assert source.close_count == 1
    assert not apollo_threads()


def test_frame_loss_error_is_counted_and_propagated():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    source.feed([frame(1), frame(2), frame(3)])
    apollo.start()
    source.fail(ApolloFrameLossError("counter gap"))

    with pytest.raises(ApolloFrameLossError, match="counter gap"):
        apollo.get_sequence(timeout_s=1)
    with pytest.raises(ApolloFrameLossError, match="counter gap"):
        apollo.stop()

    assert apollo.stats.frame_gaps == 1
    assert apollo.stats.sequences_captured == 0
    with pytest.raises(ApolloFrameLossError, match="counter gap"):
        apollo.close()


def test_failed_multi_trigger_preserves_results_and_next_start_recovers():
    source = FakeFrameSource()
    source.feed(one_sequence_frames())
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo.start(max_sequences=2)
    source.fail(RuntimeError("late read failure"))
    assert apollo._worker_done.wait(1)

    with pytest.raises(RuntimeError, match="late read failure"):
        apollo.get_sequence(timeout_s=1)
    with pytest.raises(RuntimeError, match="late read failure") as error:
        apollo.stop()
    assert apollo._results.empty()
    assert [frame_ids(seq) for seq in error.value.completed_sequences] == [list(range(1, 21))]

    source.feed(one_sequence_frames(start_id=21))
    apollo.start()
    sequence = apollo.get_sequence(timeout_s=1)
    apollo.stop()
    apollo.close()

    assert frame_ids(sequence) == list(range(21, 41))


def test_decode_shape_error_is_propagated():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    source._batches.put(np.ones((4, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="invalid batch shape"):
        apollo.start()
    with pytest.raises(ValueError, match="invalid batch shape"):
        apollo.close()


def test_incomplete_source_start_is_stopped_and_instance_remains_reusable():
    class FailingStartSource(FakeFrameSource):
        def __init__(self) -> None:
            super().__init__()
            self.fail_start = True

        def start(self) -> None:
            super().start()
            if self.fail_start:
                self.fail_start = False
                raise RuntimeError("start failed")

    source = FailingStartSource()
    apollo = DropwatchApollo(settings(), frame_source=source)

    with pytest.raises(RuntimeError, match="start failed"):
        apollo.start()
    assert source.stop_count == 1

    source.feed(one_sequence_frames())
    apollo.start()
    sequence = apollo.get_sequence(timeout_s=1)
    apollo.stop()
    apollo.close()

    assert sequence.shape == (20, 4, 8)
    assert source.start_count == 2
    assert source.stop_count == 2


def test_close_never_closes_source_while_worker_is_still_reading():
    class BlockingSource(FakeFrameSource):
        def __init__(self) -> None:
            super().__init__()
            self.release_read = threading.Event()
            self.read_count = 0

        def read(self) -> np.ndarray | None:
            self.read_count += 1
            if self.read_count == 1:
                return np.stack([frame(1), frame(2), frame(3)])
            self.release_read.wait()
            return None

    source = BlockingSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    apollo._STOP_TIMEOUT_S = 0.01
    apollo.start()

    with pytest.raises(ApolloLifecycleError, match="did not stop"):
        apollo.close()
    assert source.close_count == 0

    source.release_read.set()
    apollo.stop()
    apollo.close()
    assert source.close_count == 1


def test_start_stop_cycles_reuse_one_instance_without_worker_leaks():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    baseline_thread_ids = {thread.ident for thread in apollo_threads()}

    for cycle in range(100):
        source.feed(one_sequence_frames(start_id=cycle * 20 + 1))
        apollo.start()
        sequence = apollo.get_sequence(timeout_s=1)
        apollo.stop()
        assert sequence.shape == (20, 4, 8)
        assert apollo.stats.sequences_captured == 1
        assert apollo.stats.frame_gaps == 0
        assert {thread.ident for thread in apollo_threads()} == baseline_thread_ids

    apollo.close()
    apollo.close()

    assert source.open_count == 1
    assert source.start_count == 100
    assert source.stop_count == 100
    assert source.close_count == 1
    assert not apollo_threads()


def test_completed_one_shot_can_restart_without_manual_stop():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)

    source.feed(one_sequence_frames())
    apollo.start()
    first = apollo.get_sequence(timeout_s=1)
    source.feed(one_sequence_frames(start_id=21))
    apollo.start()
    second = apollo.get_sequence(timeout_s=1)
    apollo.close()

    assert frame_ids(first) == list(range(1, 21))
    assert frame_ids(second) == list(range(21, 41))
    assert source.start_count == 2
    assert source.stop_count == 2


def test_fetched_sequence_is_not_retained_by_apollo():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    source.feed(one_sequence_frames())
    apollo.start()
    sequence = apollo.get_sequence(timeout_s=1)
    sequence_ref = weakref.ref(sequence)

    del sequence
    for _ in range(10):
        gc.collect()
        if sequence_ref() is None:
            break
        time.sleep(0.01)

    apollo.stop()
    apollo.close()
    assert sequence_ref() is None


def test_lifecycle_methods_are_idempotent_and_reject_double_start():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)

    apollo.stop()
    apollo.open()
    apollo.open()
    source.feed([frame(1), frame(2), frame(3)])
    apollo.start()
    with pytest.raises(ApolloLifecycleError, match="already running"):
        apollo.start()
    apollo.stop()
    apollo.stop()
    apollo.close()
    apollo.close()

    assert source.open_count == 1
    assert source.start_count == 1
    assert source.stop_count == 1
    assert source.close_count == 1


def test_get_sequence_requires_an_active_or_completed_acquisition():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)

    with pytest.raises(ApolloLifecycleError, match="not running"):
        apollo.get_sequence(timeout_s=0)
