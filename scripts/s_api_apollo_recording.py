import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dropwatch_apollo import ApolloSettings
from dropwatch_apollo import ApolloVideoSettings
from dropwatch_apollo import DropwatchApollo

logger = logging.getLogger(__name__)


def evaluate_sequences(sequences: list[np.ndarray]) -> pd.DataFrame:
    """Placeholder: replace the NaN values with the production evaluation."""
    return pd.DataFrame(
        [
            {
                "shot": shot,
                "number_frames": len(sequence),
                "volume_nl": np.nan,
                "speed_m_s": np.nan,
            }
            for shot, sequence in enumerate(sequences)
        ]
    )


def finalize_evaluations(observations: pd.DataFrame) -> pd.DataFrame:
    """Called once with global shot IDs; placeholder leaves the table unchanged.

    For the existing A1 tracker, use make_evaluation_callbacks() from
    dropwatch_apollo.integration: connect_shots/postproc belong here, not in
    the per-window worker. See MIGRATION.md.
    """
    return observations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output_dir = Path("dropwatch_recordings")

    settings = ApolloSettings(
        max_number_frames=50,
        pre_trigger=10,
        frame_period_ms=1.0,  # 1000 fps
        exposure_time_ms=0.05,
        rle_batch_frames=100,
        # Optional: bounded RAM + durable NPY shots, suitable for e.g. 40 x
        # 1000 frames. Use a fast local SSD and qualify it at the target rate.
        spool_directory=output_dir / "raw",
        spool_chunk_frames=100,
        spool_buffer_count=8,
    )

    number_sequences = 2
    with DropwatchApollo(
        settings, evaluator=evaluate_sequences, evaluation_finalizer=finalize_evaluations
    ) as dropwatch:
        trigger_position_px = dropwatch.set_trigger_size(width=100)
        logger.info("Plate detected; trigger position is %d px from the bottom", trigger_position_px)

        dropwatch.start(max_sequences=number_sequences)
        logger.info("Acquisition armed; dispense %d droplets", number_sequences)
        sequences = dropwatch.get_sequences(timeout_s=30)
        evaluations = dropwatch.get_evaluations(timeout_s=30)
        session_dir = dropwatch.recording_directory
        assert session_dir is not None

        # One atomically finalized file per shot limits the impact of an
        # interrupted export. Encoding starts only after camera intake stopped.
        dropwatch.save_videos(
            sequences,
            session_dir / "video",
            options=ApolloVideoSettings(playback_fps=25, annotate=True),
        )
        evaluations.to_csv(session_dir / "evaluations.csv", index=False)
        logger.info("Acquisition statistics: %s", dropwatch.stats)
        logger.info("Evaluations:\n%s", evaluations)
