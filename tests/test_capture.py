from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dropwatch_apollo import ApolloLifecycleError
from dropwatch_apollo import ApolloSettings
from dropwatch_apollo import DropwatchApollo
from dropwatch_apollo._capture import _SequenceCapture

from ._support import FakeFrameSource
from ._support import expected_sequence_ids
from ._support import frame
from ._support import frame_ids
from ._support import one_sequence_frames
from ._support import plate_frame
from ._support import settings


def test_default_trigger_position_is_measured_from_bottom():
    source = FakeFrameSource()
    bottom_settings = ApolloSettings(
        max_number_frames=20,
        pre_trigger=0,
        trigger_position_px=2,
        trigger_width_px=2,
        trigger_on_pixels=2,
        trigger_off_pixels=0,
    )
    frames = [frame(1), frame(2)]
    for frame_id in range(3, 23):
        image = frame(frame_id)
        image[:, 4:6] = 0
        frames.append(image)

    with DropwatchApollo(bottom_settings, frame_source=source) as apollo:
        source.feed(frames)
        apollo.start()
        sequence = apollo.get_sequence(timeout_s=1)
        apollo.stop()

    assert frame_ids(sequence) == list(range(3, 23))
    assert np.all(sequence[0, :, 4:6] == 0)


def test_set_trigger_size_places_band_above_bottom_connected_plate():
    source = FakeFrameSource()
    source.frame_shape = (20, 200)
    source.frame_dtype = np.uint8
    source.feed([plate_frame()])
    apollo = DropwatchApollo(ApolloSettings(max_number_frames=20), frame_source=source)

    trigger_position_px = apollo.set_trigger_size(width=20, timeout_s=1)

    assert trigger_position_px == 50
    assert apollo.settings.trigger_position_px == 50
    assert apollo.settings.trigger_width_px == 20
    assert apollo.settings.trigger_from_top is False
    assert source.open_count == 1
    assert source.start_count == 1
    assert source.stop_count == 1

    trigger_frames = [plate_frame(), plate_frame()]
    for frame_id in range(1, 21):
        image = plate_frame()
        image[0, 0] = frame_id
        image[:, 130:150] = 0
        trigger_frames.append(image)
    source.feed(trigger_frames)
    apollo.start()
    sequence = apollo.get_sequence(timeout_s=1)
    apollo.stop()

    assert frame_ids(sequence) == list(range(1, 21))
    apollo.close()


def test_set_trigger_size_rejects_missing_plate_and_stops_source():
    source = FakeFrameSource()
    source.feed([np.ones((20, 200), dtype=np.uint8)])
    apollo = DropwatchApollo(ApolloSettings(max_number_frames=20), frame_source=source)

    with pytest.raises(ValueError, match="no foreground object"):
        apollo.set_trigger_size(timeout_s=1)

    assert source.stop_count == 1
    apollo.close()


def test_set_trigger_size_rejects_running_acquisition():
    source = FakeFrameSource()
    with DropwatchApollo(settings(), frame_source=source) as apollo:
        source.feed([frame(1), frame(2), frame(3)])
        apollo.start()
        with pytest.raises(
            ApolloLifecycleError,
            match="while Dropwatch Apollo acquisition is running",
        ):
            apollo.set_trigger_size()
        apollo.stop()


def test_pre_trigger_is_preserved_across_source_batches():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)

    source.feed([frame(1), frame(2)])
    source.feed([frame(3), frame(4, drop=True)])
    source.feed([frame(index, drop=True) for index in range(5, 21)])
    apollo.start()

    sequence = apollo.get_sequence(timeout_s=1)
    apollo.stop()
    apollo.close()

    assert isinstance(sequence, np.ndarray)
    assert sequence.shape == (20, 4, 8)
    assert frame_ids(sequence) == list(range(1, 21))
    assert np.count_nonzero(sequence[3, :, 2:4]) == 0


def test_pre_trigger_rotation_is_deferred_until_sequence_is_consumed():
    capture = _SequenceCapture(settings())
    capture.prepare((4, 8), np.uint16)

    for frame_id in range(1, 5):
        assert capture.push(frame(frame_id)) is None
    assert capture.is_armed
    assert capture.push(frame(5, drop=True)) is None
    result = None
    for frame_id in range(6, 22):
        result = capture.push(frame(frame_id, drop=True))

    assert result is not None
    assert frame_ids(result.frames[:3]) == [255, 255, 255]
    assert [int(item[0][0, 0]) for item in result.history] == [2, 3, 4]
    assert frame_ids(result.materialize()) == list(range(2, 22))


def test_memory_budget_is_checked_before_source_start():
    class SizedSource(FakeFrameSource):
        frame_shape = (4, 8)
        frame_dtype = np.uint16
        reserved_buffer_bytes = 100

    source = SizedSource()
    required_bytes = 3 * 20 * 4 * 8 * np.dtype(np.uint16).itemsize + source.reserved_buffer_bytes
    constrained = replace(settings(), max_buffer_bytes=required_bytes - 1)
    apollo = DropwatchApollo(constrained, frame_source=source)

    with pytest.raises(MemoryError, match="exceeding max_buffer_bytes"):
        apollo.start(max_sequences=3)

    assert source.start_count == 0
    apollo.close()


def test_sequence_buffers_are_initialized_before_arming():
    capture = _SequenceCapture(settings())
    capture.reset(max_sequences=2)
    capture.prepare((4, 8), np.uint8)

    assert capture._sequence_pool.qsize() == 1
    assert np.all(capture._sequence == 255)
    assert np.all(capture._sequence_pool.get_nowait() == 255)


def test_start_returns_only_after_clear_pre_trigger_buffer_is_armed():
    source = FakeFrameSource()
    apollo = DropwatchApollo(settings(), frame_source=source)
    source.feed([frame(0, drop=True), frame(1), frame(2), frame(3)])

    apollo.start()

    assert apollo.is_running
    assert apollo._capture.is_armed
    source.feed([frame(index, drop=True) for index in range(4, 21)])
    sequence = apollo.get_sequence(timeout_s=1)
    apollo.stop()
    apollo.close()

    assert frame_ids(sequence) == list(range(1, 21))
    assert np.all(sequence[:3, :, 2:4] == 1)


@pytest.mark.parametrize("pre_trigger", [0, 19])
def test_trigger_index_and_exact_sequence_length(pre_trigger):
    source = FakeFrameSource()
    with DropwatchApollo(settings(pre_trigger), frame_source=source) as apollo:
        source.feed(one_sequence_frames(pre_trigger))
        apollo.start()
        sequence = apollo.get_sequence(timeout_s=1)
        apollo.stop()

    assert sequence.shape[0] == 20
    assert frame_ids(sequence) == expected_sequence_ids(pre_trigger)
    assert np.count_nonzero(sequence[pre_trigger, :, 2:4]) == 0
    if pre_trigger:
        assert np.all(sequence[:pre_trigger, :, 2:4] == 1)
