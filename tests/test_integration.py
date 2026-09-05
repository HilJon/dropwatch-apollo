import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dropwatch_apollo._compat_config import LegacyVideoSaver
from dropwatch_apollo._video import save_video
from dropwatch_apollo.integration import crop_sequences
from dropwatch_apollo.integration import evaluate_sequences
from dropwatch_apollo.integration import extract_video
from dropwatch_apollo.integration import make_evaluation_callbacks
from dropwatch_apollo.models import ApolloVideoSettings


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("codec", ["MJPG", "XVID"])
def test_avi_readback_preserves_windows_empty_frames_orientation_and_crop(tmp_path, invert, codec):
    raw = np.ones((512, 96), np.uint8)
    raw[200:240, 40:64] = 0
    empty = np.ones_like(raw)
    sequences = [np.stack([raw, empty]), np.stack([raw])]
    path = tmp_path / "recording.avi"
    LegacyVideoSaver(path, invert_bw=invert, codec=codec).save(sequences, tmp_path)
    cropped = crop_sequences(extract_video(path, invert_bw=invert), slice(32, 80), slice(192, 256))
    assert [len(seq) for seq in cropped] == [2, 1]
    expected = raw.T[32:80, 192:256] == 0
    np.testing.assert_array_equal(cropped[0][0], expected)
    assert not np.any(cropped[0][1])  # An empty frame is not a separator.
    np.testing.assert_array_equal(cropped[1][0], expected)
    assert cropped[0][0].base is None  # Does not retain the full decoded frame.


def test_parallel_raw_observations_have_same_global_finalization_as_sequential(monkeypatch):
    calls = []

    def fast(sequences, **options):
        calls.append(options)
        return pd.DataFrame(
            [
                {"shot": shot, "frame": frame, "area": int(np.count_nonzero(image))}
                for shot, seq in enumerate(sequences)
                for frame, image in enumerate(seq)
            ]
        )

    def connect(data):
        assert data.shot.unique().tolist() == [0, 1]
        data = data.copy()
        data["shot"] = 0  # Simulate a track spanning both windows.
        return data, [(0, 1)]

    def postproc(data):
        return pd.DataFrame({"shot": [0], "volume": [data.area.sum()], "speed": [1.0], "speed_start": [2.0]})

    module = SimpleNamespace(fast_eval_sequences=fast, connect_shots=connect, postproc_full_data=postproc)
    monkeypatch.setattr("dropwatch_apollo.integration._fast_eval_module", lambda: module)
    raw = np.ones((2, 4, 8), np.uint8)
    raw[:, 1:3, 2:5] = 0
    rows, cols = slice(1, 7), slice(1, 4)
    evaluator, finalizer = make_evaluation_callbacks(rows=rows, cols=cols, frame_period_ms=0.5, clear_border=False)
    observations = []
    for shot, seq in enumerate([raw, raw]):
        result = evaluator([seq])
        result["shot"] = shot  # Same indexing contract as _EvaluationRunner.
        observations.append(result)
    parallel = finalizer(pd.concat(observations, ignore_index=True))
    sequential = evaluate_sequences([[f.T[rows, cols] == 0 for f in raw]] * 2, frame_period_ms=0.5, clear_border=False)
    pd.testing.assert_frame_equal(parallel, sequential)
    assert parallel.speed.iloc[0] == 2.0
    assert parallel.speed_start.iloc[0] == 4.0
    assert all(options == {"clear_border": False} for options in calls)


def test_empty_finalizer_does_not_call_tracking(monkeypatch):
    module = SimpleNamespace()
    monkeypatch.setattr("dropwatch_apollo.integration._fast_eval_module", lambda: module)
    _, finalize = make_evaluation_callbacks()
    assert finalize(pd.DataFrame()).empty


def test_reader_rejects_old_split_or_unannotated_video(tmp_path):
    raw = np.ones((1024, 96), np.uint8)
    path = tmp_path / "split.avi"
    LegacyVideoSaver(path, codec="MJPG").save([np.stack([raw])], tmp_path)
    with pytest.raises(ValueError, match="left-only"):
        list(extract_video(path))


def test_consumer_patch_applies_to_reviewed_import_block(tmp_path):
    source = tmp_path / "src/a1_dispense_executor/run_dispense.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n" * 32
        + "\n".join(
            [
                "    run_low_level_dispense,",
                "    run_low_level_survey,",
                ")",
                "from a1_experiment_lib.dropwatch_helper import _crop_sequences, _evaluate_sequences",
                "from dropeval.utils.io import extract_video",
                "from fasteye import FastEyeReadError",
                "from recorder.api.dropwatch import Dropwatch",
                "from recorder.core.capture import CaptureState",
            ]
        )
        + "\n"
    )
    patch = Path(__file__).resolve().parents[1] / "integration/a1-consumer-imports.patch"
    result = subprocess.run(["git", "apply", "--check", str(patch)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reader_does_not_silently_crop_an_unannotated_native_video(tmp_path):
    raw = np.ones((2, 512, 96), np.uint8)
    path = tmp_path / "native.avi"
    save_video(raw, path, frame_period_ms=1, pre_trigger=0, options=ApolloVideoSettings(annotate=False, invert=False))
    with pytest.raises(ValueError, match="missing.*label"):
        crop_sequences(extract_video(path, invert_bw=False))
