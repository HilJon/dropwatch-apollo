from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from dropwatch import ApolloEvaluationError
from dropwatch import ApolloLifecycleError
from dropwatch import DropwatchApollo

from ._support import FakeFrameSource
from ._support import evaluation_threads
from ._support import frame_ids
from ._support import multiple_sequence_frames
from ._support import one_sequence_frames
from ._support import settings


def test_evaluator_runs_on_single_sequences_without_blocking_acquisition():
    evaluation_started = threading.Event()
    release_evaluation = threading.Event()
    evaluated_ids: list[int] = []

    def evaluator(sequences):
        assert len(sequences) == 1
        sequence_id = int(sequences[0][0, 0, 0])
        evaluated_ids.append(sequence_id)
        if len(evaluated_ids) == 1:
            evaluation_started.set()
            assert release_evaluation.wait(2)
        return pd.DataFrame(
            [
                {
                    "sequence_id": sequence_id,
                    "shot": 0,
                    "speed": sequence_id / 10,
                }
            ]
        )

    source = FakeFrameSource()
    source.feed(multiple_sequence_frames(2))
    apollo = DropwatchApollo(settings(), frame_source=source, evaluator=evaluator)
    apollo.start(max_sequences=2)

    assert evaluation_started.wait(1)
    assert apollo._worker_done.wait(1)
    sequences = apollo.stop()
    assert not apollo._evaluation.done.is_set()

    release_evaluation.set()
    evaluations = apollo.get_evaluations(timeout_s=2)
    apollo.close()

    assert [frame_ids(sequence) for sequence in sequences] == [
        list(range(1, 21)),
        list(range(21, 41)),
    ]
    assert all(not sequence.flags.writeable for sequence in sequences)
    assert evaluations.to_dict("records") == [
        {"sequence_id": 1, "shot": 0, "speed": 0.1},
        {"sequence_id": 21, "shot": 1, "speed": 2.1},
    ]
    assert not evaluation_threads()


def test_evaluation_failure_does_not_invalidate_captured_sequence():
    def evaluator(_sequences):
        raise ValueError("tracking failed")

    source = FakeFrameSource()
    source.feed(one_sequence_frames())
    apollo = DropwatchApollo(settings(), frame_source=source, evaluator=evaluator)
    apollo.start()

    sequence = apollo.get_sequence(timeout_s=1)
    apollo.stop()
    with pytest.raises(ApolloEvaluationError, match="sequence evaluation failed") as error:
        apollo.get_evaluations(timeout_s=1)
    apollo.close()

    assert isinstance(error.value.__cause__, ValueError)
    assert frame_ids(sequence) == list(range(1, 21))
    assert not evaluation_threads()


def test_evaluations_must_be_collected_before_next_start():
    def evaluator(sequences):
        return pd.DataFrame([{"sequence_id": int(sequences[0][0, 0, 0])}])

    source = FakeFrameSource()
    source.feed(one_sequence_frames())
    apollo = DropwatchApollo(settings(), frame_source=source, evaluator=evaluator)
    apollo.start()
    apollo.get_sequence(timeout_s=1)
    apollo.stop()
    assert apollo._evaluation.done.wait(1)

    with pytest.raises(ApolloLifecycleError, match=r"call get_evaluations\(\)"):
        apollo.start()
    first_evaluation = apollo.get_evaluations(timeout_s=1)

    source.feed(one_sequence_frames(start_id=21))
    apollo.start()
    apollo.get_sequence(timeout_s=1)
    apollo.stop()
    second_evaluation = apollo.get_evaluations(timeout_s=1)
    apollo.close()

    assert first_evaluation["sequence_id"].tolist() == [1]
    assert second_evaluation["sequence_id"].tolist() == [21]
    assert source.start_count == 2


def test_get_evaluations_requires_configured_evaluator():
    apollo = DropwatchApollo(settings(), frame_source=FakeFrameSource())

    with pytest.raises(ApolloLifecycleError, match="no sequence evaluator"):
        apollo.get_evaluations()


def test_hung_evaluator_cannot_block_camera_close_forever():
    evaluation_started = threading.Event()
    release_evaluation = threading.Event()

    def evaluator(_sequences):
        evaluation_started.set()
        release_evaluation.wait()
        return pd.DataFrame([{"ok": True}])

    source = FakeFrameSource()
    source.feed(one_sequence_frames())
    apollo = DropwatchApollo(settings(), frame_source=source, evaluator=evaluator)
    apollo._evaluation._STOP_TIMEOUT_S = 0.01
    apollo.start()
    apollo.get_sequence(timeout_s=1)
    assert evaluation_started.wait(1)

    started = time.monotonic()
    with pytest.raises(ApolloLifecycleError, match="evaluator did not stop"):
        apollo.close()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert source.close_count == 1
    release_evaluation.set()
    deadline = time.monotonic() + 1
    while evaluation_threads() and time.monotonic() < deadline:
        time.sleep(0.001)
    apollo.close()
    assert not evaluation_threads()
