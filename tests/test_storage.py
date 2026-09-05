import gc
import threading
import weakref
from dataclasses import replace

import numpy as np
import pytest

from dropwatch import ApolloLifecycleError
from dropwatch import ApolloSettings
from dropwatch import DropwatchApollo
from dropwatch._capture import _SequenceCapture
from dropwatch._storage import save_npy
from tests._support import FakeFrameSource
from tests._support import frame
from tests._support import frame_ids
from tests._support import multiple_sequence_frames
from tests._support import one_sequence_frames
from tests._support import settings


def test_forty_shots_use_three_recycled_blocks_and_survive_close(tmp_path, monkeypatch):
    real_full = np.full
    buffers = []

    def allocate(*args, **kwargs):
        buffer = real_full(*args, **kwargs)
        buffers.append(weakref.ref(buffer))
        return buffer

    source = FakeFrameSource()
    config = replace(settings(), spool_directory=tmp_path)
    with DropwatchApollo(config, frame_source=source) as apollo:
        source.feed([frame(1), frame(2), frame(3)])
        with monkeypatch.context() as m:
            m.setattr("dropwatch._capture.np.full", allocate)
            apollo.start(max_sequences=40)
        sequences = []
        for shot in range(40):
            source.feed(one_sequence_frames(start_id=shot * 20 + 1))
            sequences.append(apollo.get_sequence(2))
        apollo.stop()
        assert apollo.stats.sequences_captured == 40
        assert len(buffers) == 3
    gc.collect()
    assert all(ref() is None for ref in buffers)
    assert all(isinstance(seq, np.memmap) and not seq.flags.writeable for seq in sequences)
    assert len(list(tmp_path.glob("acquisition_*/shot_*.npy"))) == 40
    assert [frame_ids(seq) for seq in sequences] == [list(range(i * 20 + 1, i * 20 + 21)) for i in range(40)]


def test_production_40_by_1000_memory_plan_is_bounded(tmp_path, monkeypatch):
    shapes = []
    real_full = np.full

    def allocate(shape, *args, **kwargs):
        shapes.append(shape)
        # Test the real allocation plan without occupying 1.8 GB on the test host.
        return real_full((shape[0], 4, 8), *args, **kwargs)

    config = ApolloSettings(max_number_frames=1000, pre_trigger=20, spool_directory=tmp_path)
    capture = _SequenceCapture(config)
    capture.reset(max_sequences=40)
    monkeypatch.setattr("dropwatch._capture.np.full", allocate)
    camera_bytes = 100 * 512 * 2240 + 1228800
    capture.prepare((512, 1120), np.uint8, additional_buffer_bytes=camera_bytes)
    assert shapes == [(1020, 512, 1120)] * 3
    assert sum(np.prod(shape) for shape in shapes) + camera_bytes < config.max_buffer_bytes


def test_slow_writer_fails_explicitly_and_keeps_complete_shots(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    actual_save = save_npy

    def slow_save(*args):
        entered.set()
        assert release.wait(3)
        return actual_save(*args)

    monkeypatch.setattr("dropwatch._storage.save_npy", slow_save)
    source = FakeFrameSource()
    apollo = DropwatchApollo(replace(settings(), spool_directory=tmp_path), frame_source=source)
    source.feed([frame(1), frame(2), frame(3)])
    apollo.start(max_sequences=10)
    source.feed(multiple_sequence_frames(10))
    try:
        assert entered.wait(1)
        # Intake must fail, never wait on a writer and silently overflow FPGA.
        for _ in range(100):
            if apollo._worker_error is not None:
                break
            threading.Event().wait(0.005)
        assert isinstance(apollo._worker_error, ApolloLifecycleError)
    finally:
        release.set()
    with pytest.raises(ApolloLifecycleError, match="cannot keep up") as error:
        apollo.stop()
    assert len(error.value.completed_sequences) == 3
    assert all(len(seq) == 20 for seq in error.value.completed_sequences)
    assert len(list(tmp_path.glob("acquisition_*/shot_*.npy"))) == 3
    with pytest.raises(ApolloLifecycleError):
        apollo.close()


def test_disk_failure_preserves_the_validated_memory_sequence(tmp_path, monkeypatch):
    def broken_save(*_args):
        raise OSError("disk disconnected")

    monkeypatch.setattr("dropwatch._storage.save_npy", broken_save)
    source = FakeFrameSource()
    apollo = DropwatchApollo(replace(settings(), spool_directory=tmp_path), frame_source=source)
    source.feed([frame(1), frame(2), frame(3)])
    apollo.start(max_sequences=4)
    source.feed(one_sequence_frames())
    assert apollo._worker_done.wait(2)
    with pytest.raises(OSError, match="disk disconnected") as error:
        apollo.stop()
    assert [frame_ids(seq) for seq in error.value.completed_sequences] == [list(range(1, 21))]
    with pytest.raises(OSError):
        apollo.close()


def test_atomic_npy_does_not_replace_previous_file_on_failure(tmp_path, monkeypatch):
    output = tmp_path / "shot.npy"
    output.write_bytes(b"previous")

    def fail(handle, *_args, **_kwargs):
        handle.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr("numpy.save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_npy(np.ones((20, 4, 8), dtype=np.uint8), output)
    assert output.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [output]
