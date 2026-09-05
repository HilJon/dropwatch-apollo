"""Bounded, fail-fast camera diagnostics; no recording or acquisition threads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import deque
from datetime import datetime
from datetime import timezone
from pathlib import Path
from time import perf_counter
from time import sleep
from typing import Any
from uuid import uuid4

import numpy as np

from dropwatch import __version__
from dropwatch._hardware import CORE_PATH
from dropwatch._hardware import ENCODED_BUFFER_BYTES
from dropwatch._hardware import MAX_BYTES_PER_IMAGE
from dropwatch._hardware import RAW_FRAME_HEIGHT
from dropwatch._hardware import RAW_FRAME_WIDTH
from dropwatch._hardware import DAQReadError
from dropwatch._hardware import FastEyeRLE
from dropwatch._hardware import RLEDecoder
from dropwatch._hardware import validate_rle_batch
from dropwatch.models import ApolloSettings
from dropwatch.models import require_finite
from dropwatch.models import require_integer


def _error(error: BaseException) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(error).__name__, "message": str(error)}
    if isinstance(error, DAQReadError):
        result.update(
            actual_bytes=error.actual_bytes,
            expected_bytes=error.expected_bytes,
            vendor_error=error.vendor_error,
        )
    return result


def _environment() -> dict[str, Any]:
    hashes = {}
    for name in ("PLabDAQCore.dll", "rlDecodeLib.dll", "cfg/enc_viamos", "cfg/viamos/fc/fc.xml"):
        try:
            hashes[name] = hashlib.sha256((CORE_PATH / name).read_bytes()).hexdigest()
        except OSError as error:
            hashes[name] = f"unavailable: {error}"
    return {
        "dropwatch_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "bundle_sha256": hashes,
    }


def _failure_status(camera: FastEyeRLE) -> dict[str, Any]:
    # Only after preserving the primary error: another vendor call can overwrite it.
    result: dict[str, Any] = {}
    for name in ("encoder_status", "num_stored_images", "num_available_images"):
        try:
            result[name] = getattr(camera, name)
        except (Exception, KeyboardInterrupt) as error:
            result[name] = _error(error)
    return result


def _acquire(
    camera: FastEyeRLE,
    decoder: RLEDecoder | None,
    destination: np.ndarray | None,
    report: dict[str, Any],
    events: deque[dict[str, Any]],
    duration_s: float,
    idle_timeout_s: float,
    extended_status: bool,
) -> None:
    summary = report["summary"]
    started = last_data = perf_counter()
    previous_read_end: float | None = None
    previous_counter: int | None = None
    try:
        while perf_counter() - started < duration_s:
            event: dict[str, Any] = {"at_s": perf_counter() - started}
            appended = False
            try:
                report["phase"] = "status"
                event["encoder_status"] = camera.encoder_status
                event["num_stored_images"] = camera.num_stored_images
                if event["encoder_status"] >= 4:
                    raise RuntimeError(f"camera RLE encoder reported status {event['encoder_status']}")
                if perf_counter() - last_data >= idle_timeout_s:
                    raise TimeoutError(f"no successful transfer for {idle_timeout_s:g} seconds")
                if event["num_stored_images"] < 1:
                    summary["empty_polls"] += 1
                    sleep(0.001)
                    continue
                if extended_status:
                    event["num_available_images"] = camera.num_available_images
                summary["read_attempts"] += 1
                event.update(read=summary["read_attempts"], expected_bytes=ENCODED_BUFFER_BYTES)
                events.append(event)
                appended = True
                report["phase"] = "read"
                read_started = perf_counter()
                event["read_started_s"] = read_started - started
                event["gap_since_previous_read_ms"] = (
                    None if previous_read_end is None else (read_started - previous_read_end) * 1000
                )
                if previous_read_end is not None:
                    summary["max_read_gap_ms"] = max(summary["max_read_gap_ms"], event["gap_since_previous_read_ms"])
                try:
                    camera.read_image()
                except DAQReadError as error:
                    event["actual_bytes"] = error.actual_bytes
                    summary["zero_byte_reads"] += int(error.actual_bytes == 0)
                    summary["short_reads"] += int(error.actual_bytes != 0)
                    raise
                else:
                    event["actual_bytes"] = ENCODED_BUFFER_BYTES
                    summary["successful_reads"] += 1
                    summary["bytes_received"] += ENCODED_BUFFER_BYTES
                finally:
                    previous_read_end = perf_counter()
                    event["read_ms"] = (previous_read_end - read_started) * 1000
                    summary["max_read_ms"] = max(summary["max_read_ms"], event["read_ms"])
                    summary["total_read_ms"] += event["read_ms"]
                last_data = previous_read_end
                if decoder is not None and destination is not None:
                    report["phase"] = "decode"
                    decode_started = perf_counter()
                    try:
                        frames, previous_counter = validate_rle_batch(
                            decoder, camera.get_enc_data(), destination, previous_counter
                        )
                    finally:
                        event["decode_ms"] = (perf_counter() - decode_started) * 1000
                        summary["max_decode_ms"] = max(summary["max_decode_ms"], event["decode_ms"])
                    event.update(frames=frames, last_frame_counter=previous_counter)
                    summary["decoded_frames"] += frames
                    summary["last_frame_counter"] = previous_counter
                if extended_status:
                    report["phase"] = "status_after_read"
                    event["status_after_read"] = {
                        "encoder_status": camera.encoder_status,
                        "num_stored_images": camera.num_stored_images,
                        "num_available_images": camera.num_available_images,
                    }
                    if event["status_after_read"]["encoder_status"] >= 4:
                        raise RuntimeError("camera RLE encoder reported an error after read")
                event["outcome"] = "ok"
            except (Exception, KeyboardInterrupt) as error:
                event.update(
                    outcome="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=_error(error)
                )
                if not appended:
                    events.append(event)
                raise
        if not summary["successful_reads"]:
            empty_error = TimeoutError("test ended without any successful camera transfer")
            events.append({"at_s": perf_counter() - started, "outcome": "failed", "error": _error(empty_error)})
            raise empty_error
    finally:
        summary["acquisition_elapsed_s"] = perf_counter() - started


def run_diagnostics(
    output: Path,
    settings: ApolloSettings,
    *,
    mode: str = "transport",
    duration_s: float = 600,
    history_size: int = 500,
    idle_timeout_s: float = 5,
    extended_status: bool = False,
    label: str = "",
) -> tuple[Path, int]:
    """Write one unique JSON report and return (path, exit code: 0, 1, or 130).

    Always stop on the first error, regardless of settings.zero_byte_read_retries.
    Settings/permission errors fail before opening hardware. No images are retained.
    """
    if mode not in {"transport", "decode"}:
        raise ValueError("mode must be transport or decode")
    for name, value in (("duration_s", duration_s), ("idle_timeout_s", idle_timeout_s)):
        require_finite(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    require_integer("history_size", history_size)
    if not 1 <= history_size <= 10_000:
        raise ValueError("history_size must be between 1 and 10000")
    if settings.rle_batch_frames * MAX_BYTES_PER_IMAGE > ENCODED_BUFFER_BYTES:
        raise ValueError("RLE batch exceeds the safe encoded buffer capacity")
    decode_bytes = settings.rle_batch_frames * RAW_FRAME_HEIGHT * RAW_FRAME_WIDTH if mode == "decode" else 0
    if decode_bytes + ENCODED_BUFFER_BYTES > settings.max_buffer_bytes:
        raise ValueError("diagnostic buffers exceed max_buffer_bytes")

    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "label": label,
        "environment": _environment(),
        "settings": {
            "fps": 1000 / settings.frame_period_ms,
            "frame_period_ms": settings.frame_period_ms,
            "exposure_time_ms": settings.exposure_time_ms,
            "threshold": settings.threshold,
            "rle_batch_frames": settings.rle_batch_frames,
            "read_timeout_ms": settings.effective_read_timeout_ms,
            "zero_byte_read_retries": 0,
            "duration_s": duration_s,
            "idle_timeout_s": idle_timeout_s,
            "history_size": history_size,
            "extended_status": extended_status,
            "readiness_rule": "numImgStored > 0 (same as Apollo; vendor semantics unverified)",
            "reserved_image_buffer_bytes": decode_bytes + ENCODED_BUFFER_BYTES,
        },
        "summary": {
            "read_attempts": 0,
            "successful_reads": 0,
            "bytes_received": 0,
            "zero_byte_reads": 0,
            "short_reads": 0,
            "empty_polls": 0,
            "decoded_frames": 0,
            "last_frame_counter": None,
            "max_read_ms": 0.0,
            "total_read_ms": 0.0,
            "max_read_gap_ms": 0.0,
            "max_decode_ms": 0.0,
            "acquisition_elapsed_s": 0.0,
            "frame_continuity": "not_checked" if mode == "transport" else "checked_in_received_batches",
        },
        "outcome": "failed",
        "error": None,
        "cleanup_errors": [],
        "camera_closed": None,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:12]}.json"
    # Check output access before touching the camera. No writes while acquiring.
    with path.open("x", encoding="utf-8") as stream:
        camera: FastEyeRLE | None = None
        decoder: RLEDecoder | None = None
        destination: np.ndarray | None = None
        needs_flush = False
        events: deque[dict[str, Any]] = deque(maxlen=history_size)
        report["phase"] = "open"
        try:
            camera = FastEyeRLE()
            report["camera_closed"] = False
            camera.open()
            report["phase"] = "setup"
            needs_flush = True
            camera.setup(read_timeout_ms=settings.effective_read_timeout_ms)
            camera.configure(
                frame_period_ms=settings.frame_period_ms,
                exposure_time_ms=settings.exposure_time_ms,
                threshold=settings.threshold,
                rle_batch_frames=settings.rle_batch_frames,
            )
            if mode == "decode":
                decoder = RLEDecoder()
                destination = np.full(
                    (settings.rle_batch_frames, RAW_FRAME_HEIGHT, RAW_FRAME_WIDTH), 255, dtype=np.uint8
                )
            report["phase"] = "start"
            camera.set_enc_mode()
            camera.flush()
            camera.trig_frame()
            _acquire(camera, decoder, destination, report, events, duration_s, idle_timeout_s, extended_status)
            report.update(outcome="passed", phase="complete")
        except (Exception, KeyboardInterrupt) as error:
            report["error"] = {"phase": report["phase"], **_error(error)}
            report["outcome"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            if camera is not None and needs_flush and not isinstance(error, KeyboardInterrupt):
                report["status_after_failure"] = _failure_status(camera)
        finally:
            if camera is not None:
                try:
                    if needs_flush:
                        camera.flush()
                except (Exception, KeyboardInterrupt) as error:
                    report["cleanup_errors"].append({"phase": "flush", **_error(error)})
                finally:
                    # A failed vendor close keeps its handle; retry only cleanup,
                    # never acquisition. Report the first failure even if recovered.
                    for attempt in (1, 2):
                        try:
                            camera.close()
                        except (Exception, KeyboardInterrupt) as error:
                            report["cleanup_errors"].append({"phase": "close", "attempt": attempt, **_error(error)})
                        else:
                            report["camera_closed"] = True
                            break
            camera = decoder = destination = None
            if report["cleanup_errors"] and report["outcome"] == "passed":
                report["outcome"] = "failed"
            report["events"] = list(events)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            events.clear()
    return path, {"passed": 0, "failed": 1, "interrupted": 130}[report["outcome"]]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dwa diagnose", description=__doc__)
    parser.add_argument("--mode", choices=("transport", "decode"), default="transport")
    parser.add_argument("--fps", type=float, default=1000)
    parser.add_argument("--duration", type=float, default=600, help="acquisition duration in seconds")
    parser.add_argument("--output", type=Path, default=Path("apollo_diagnostics"))
    parser.add_argument("--read-timeout-ms", type=int, help="default: Apollo's batch-aware timeout")
    parser.add_argument("--rle-batch-frames", type=int, default=100)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--exposure-time-ms", type=float, default=0.05)
    parser.add_argument("--history-size", type=int, default=500, help="maximum retained events (1..10000)")
    parser.add_argument("--idle-timeout", type=float, default=5, help="fail after this many seconds without a transfer")
    parser.add_argument("--extended-status", action="store_true", help="extra pre/post-read queries; changes timing")
    parser.add_argument("--label", default="", help="notes: camera, firmware/driver, USB port/cable, scene")
    args = parser.parse_args(argv)
    try:
        require_finite("fps", args.fps)
        if args.fps <= 0:
            raise ValueError("fps must be > 0")
        settings = ApolloSettings(
            max_number_frames=20,
            frame_period_ms=1000 / args.fps,
            read_timeout_ms=args.read_timeout_ms,
            rle_batch_frames=args.rle_batch_frames,
            exposure_time_ms=args.exposure_time_ms,
            threshold=args.threshold,
            zero_byte_read_retries=0,
        )
        print("Camera diagnostic: first error stops the test; Ctrl+C stops and saves a report.", flush=True)
        path, code = run_diagnostics(
            args.output,
            settings,
            mode=args.mode,
            duration_s=args.duration,
            history_size=args.history_size,
            idle_timeout_s=args.idle_timeout,
            extended_status=args.extended_status,
            label=args.label,
        )
    except ValueError as error:
        parser.error(str(error))
    except OSError as error:
        parser.exit(1, f"Diagnostic output failed: {error}\n")
    print(f"Diagnostic {'passed' if code == 0 else 'did not pass'}; report: {path.resolve()}")
    raise SystemExit(code)
