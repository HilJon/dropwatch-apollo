"""Small recorder CLI: python -m dropwatch --help (or dwa --help)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dropwatch import ApolloSettings
from dropwatch import ApolloVideoSettings
from dropwatch import DropwatchApollo
from dropwatch import ReplayFrameSource


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        from dropwatch.diagnostics import main as diagnose

        diagnose(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="Single-view Apollo recording and replay")
    parser.add_argument("command", choices=("record", "snapshot", "preview", "replay", "diagnose"))
    parser.add_argument("files", nargs="*", type=Path, help="RLE BIN or raw NPY files for replay")
    parser.add_argument("--output", type=Path, default=Path("apollo_recordings"))
    parser.add_argument("--frames", type=int, default=200, help="total frames per shot, including lookback")
    parser.add_argument("--pre-trigger", type=int, default=20)
    parser.add_argument("--shots", type=int, default=1, help="maximum number of shots")
    parser.add_argument("--duration", type=float, help="recording/preview duration in seconds")
    parser.add_argument("--frame-period-ms", type=float, default=1.0)
    parser.add_argument("--trigger-position", type=int, default=300, help="pixels from physical bottom")
    parser.add_argument("--trigger-width", type=int, default=100)
    parser.add_argument("--auto-trigger", action="store_true", help="detect plate using an idle snapshot")
    parser.add_argument("--playback-fps", type=float, default=25)
    parser.add_argument("--mp4", action="store_true", help="export MP4 instead of AVI")
    args = parser.parse_args()
    if args.command == "replay" and not args.files:
        parser.error("replay requires at least one BIN or NPY file")
    logging.basicConfig(level=logging.INFO)
    config = ApolloSettings(
        max_number_frames=args.frames,
        pre_trigger=args.pre_trigger,
        frame_period_ms=args.frame_period_ms,
        trigger_position_px=args.trigger_position,
        trigger_width_px=args.trigger_width,
        spool_directory=args.output / "raw" if args.command == "record" else None,
    )
    source = ReplayFrameSource(args.files) if args.command == "replay" else None
    with DropwatchApollo(config, frame_source=source) as recorder:
        if args.command == "snapshot":
            recorder.snapshot(args.output / "snapshot.png")
            return
        if args.command == "preview":
            recorder.preview(args.duration if args.duration is not None else 10.0)
            return
        if args.auto_trigger:
            recorder.set_trigger_size(args.trigger_width)
        recorder.start(max_sequences=args.shots, max_duration_s=args.duration)
        logging.info("Armed; ready to dispense")
        try:
            sequences = recorder.get_sequences()
        except Exception as error:
            logging.exception("Recording failed; completed raw shots remain in %s", recorder.recording_directory)
            # Preserve the original acquisition failure, not a successful exit.
            raise error
        output = recorder.recording_directory or args.output
        if sequences:
            recorder.save_videos(
                sequences,
                output / "video",
                suffix=".mp4" if args.mp4 else ".avi",
                options=ApolloVideoSettings(playback_fps=args.playback_fps),
            )
        logging.info("Captured %d shots; output: %s; statistics: %s", len(sequences), output, recorder.stats)


if __name__ == "__main__":
    main()
