"""Deterministic transport faults; no Windows DLLs or camera required."""

import gc
import json
import sys
import weakref
from dataclasses import replace

import pytest

from dropwatch import diagnostics
from dropwatch.__main__ import main
from dropwatch._hardware import ENCODED_BUFFER_BYTES
from dropwatch._hardware import DAQError
from dropwatch._hardware import DAQReadError
from dropwatch._hardware import FastEyeRLE
from dropwatch._hardware import RLEDecoder
from dropwatch._hardware import RLEDecodeResult
from dropwatch.models import ApolloSettings
from tests.test_hardware import FakeFastEye
from tests.test_hardware import rle_stream


class Clock:
    now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Camera(FakeFastEye):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.calls = []
        self.counter = 32765
        self.encoder_status = 0
        self.num_available_images = 100
        self.payload = bytearray(ENCODED_BUFFER_BYTES)

    def read_image(self):
        self.calls.append("read")
        self.clock.sleep(0.01)
        super().read_image()
        self.counter = (self.counter + 1) & 0x7FFF

    def get_enc_data(self):
        data = rle_stream([self.counter])
        self.payload[: len(data)] = data
        return self.payload

    def flush(self):
        self.calls.append("flush")
        super().flush()

    def close(self):
        self.calls.append("close")
        self.payload = None
        super().close()


class Decoder:
    def __init__(self, clock):
        self.clock = clock
        self.destinations = []
        self.ids = set()

    def decode_buffer(self, data, destination):
        self.clock.sleep(0.002)
        self.destinations.append(weakref.ref(destination))
        self.ids.add(id(destination))
        return RLEDecodeResult(1, destination.shape[1] * destination.shape[2], 80)

    completed_frame_counters = RLEDecoder.completed_frame_counters


@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    camera = Camera(clock)
    decoder = Decoder(clock)
    monkeypatch.setattr(diagnostics, "perf_counter", clock.time)
    monkeypatch.setattr(diagnostics, "sleep", clock.sleep)
    monkeypatch.setattr(diagnostics, "FastEyeRLE", lambda: camera)
    monkeypatch.setattr(diagnostics, "RLEDecoder", lambda: decoder)
    monkeypatch.setattr(diagnostics, "_environment", lambda: {"platform": "test"})
    # Small destination for fast lifecycle tests; real shape validated in the
    # existing vendor-fixture and source contract tests.
    monkeypatch.setattr(diagnostics, "RAW_FRAME_HEIGHT", 4)
    monkeypatch.setattr(diagnostics, "RAW_FRAME_WIDTH", 8)
    return camera, decoder, clock


def run(tmp_path, **kwargs):
    path, code = diagnostics.run_diagnostics(
        tmp_path, ApolloSettings(max_number_frames=20), duration_s=kwargs.pop("duration_s", 0.12), **kwargs
    )
    return json.loads(path.read_text()), code


@pytest.mark.parametrize("mode", ["transport", "decode"])
@pytest.mark.parametrize("frame_period_ms", [1.0, 0.5])
def test_diagnostics_flushes_before_enabling_rle(tmp_path, rig, mode, frame_period_ms):
    camera, _decoder, _clock = rig
    path, code = diagnostics.run_diagnostics(
        tmp_path,
        ApolloSettings(max_number_frames=20, frame_period_ms=frame_period_ms),
        mode=mode,
        duration_s=0.05,
    )
    report = json.loads(path.read_text())
    assert code == 0
    assert report["summary"]["successful_reads"] > 0
    assert report["summary"]["empty_polls"] == 0
    assert camera.sync_commands == ["flush", "rle", "frame", "flush"]
    assert report["camera_closed"] is True
    assert not camera.stream_running
    assert camera.payload is None


def test_transport_is_bounded_does_not_decode_or_write_during_reads(tmp_path, rig, monkeypatch):
    camera, decoder, _clock = rig

    def forbidden():
        pytest.fail("transport mode must not load the decoder DLL")

    monkeypatch.setattr(diagnostics, "RLEDecoder", forbidden)
    original_read = camera.read_image

    def read():
        assert len(list(tmp_path.glob("*.json"))) == 1
        assert next(tmp_path.glob("*.json")).stat().st_size == 0
        original_read()

    camera.read_image = read
    report, code = run(tmp_path, history_size=3, duration_s=10)
    assert code == 0
    assert report["outcome"] == "passed"
    assert report["summary"]["read_attempts"] >= 999
    assert len(report["events"]) == 3
    assert report["events"][-1]["read"] == report["summary"]["read_attempts"]
    assert report["summary"]["bytes_received"] == camera.read_count * ENCODED_BUFFER_BYTES
    assert report["summary"]["frame_continuity"] == "not_checked"
    assert report["events"][-1]["read_ms"] == pytest.approx(10)
    assert camera.flush_count == 2
    assert camera.close_count == 1
    assert camera.payload is None
    assert not decoder.destinations


@pytest.mark.parametrize("actual", [0, 40, -1])
def test_first_bad_read_stops_and_preserves_vendor_error(tmp_path, rig, actual):
    camera, _decoder, _clock = rig
    camera.read_results = [None, DAQReadError(actual, ENCODED_BUFFER_BYTES, "USB timeout: original"), None]
    report, code = run(tmp_path)
    assert code == 1
    assert camera.read_count == 2  # no retry, third buffer untouched
    assert report["error"] == {
        "phase": "read",
        "type": "DAQReadError",
        "message": f"camera returned {actual} of {ENCODED_BUFFER_BYTES} bytes (vendor error: USB timeout: original)",
        "actual_bytes": actual,
        "expected_bytes": ENCODED_BUFFER_BYTES,
        "vendor_error": "USB timeout: original",
    }
    assert report["events"][-1]["actual_bytes"] == actual
    assert report["events"][-1]["read_ms"] == pytest.approx(10)
    assert report["summary"]["zero_byte_reads"] == int(actual == 0)
    assert report["summary"]["short_reads"] == int(actual != 0)
    assert report["status_after_failure"]["encoder_status"] == 0
    assert camera.calls == ["flush", "read", "read", "flush", "close"]


def test_decode_checks_wrap_and_reuses_then_releases_buffer(tmp_path, rig):
    camera, decoder, _clock = rig
    report, code = run(tmp_path, mode="decode", extended_status=True)
    assert code == 0
    assert report["summary"]["decoded_frames"] == camera.read_count
    assert report["summary"]["last_frame_counter"] == camera.counter
    assert camera.counter < 32765  # crossed 32767 -> 0
    assert report["summary"]["max_decode_ms"] == pytest.approx(2)
    assert report["summary"]["max_read_gap_ms"] == pytest.approx(2)
    assert report["events"][0]["status_after_read"]["num_available_images"] == 100
    assert len(decoder.ids) == 1
    gc.collect()
    assert all(ref() is None for ref in decoder.destinations)
    assert camera.payload is None


def test_counter_gap_fails_even_after_full_transfers(tmp_path, rig):
    camera, decoder, _clock = rig
    original_read = camera.read_image

    def read():
        original_read()
        if camera.read_count == 2:
            camera.counter = (camera.counter + 1) & 0x7FFF

    camera.read_image = read
    report, code = run(tmp_path, mode="decode")
    assert code == 1
    assert camera.read_count == 2
    assert report["error"]["phase"] == "decode"
    assert report["error"]["type"] == "ApolloFrameLossError"
    assert "jumped" in report["error"]["message"]
    assert report["summary"]["successful_reads"] == 2
    assert report["summary"]["decoded_frames"] == 1
    gc.collect()
    assert all(ref() is None for ref in decoder.destinations)


def test_no_data_never_passes_and_polling_is_bounded(tmp_path, rig):
    camera, _decoder, clock = rig
    camera.num_stored_images = 0
    report, code = run(tmp_path, idle_timeout_s=0.03, history_size=2)
    assert code == 1
    assert report["error"]["type"] == "TimeoutError"
    assert len(report["events"]) == 1
    assert 0.03 <= clock.now < 0.04
    assert camera.read_count == 0
    assert camera.close_count == 1


def test_short_test_with_no_transfers_does_not_pass(tmp_path, rig):
    rig[0].num_stored_images = 0
    report, code = run(tmp_path)
    assert code == 1
    assert "without any successful" in report["error"]["message"]


def test_encoder_status_prevents_read(tmp_path, rig):
    camera = rig[0]
    camera.encoder_status = 5
    report, code = run(tmp_path)
    assert code == 1
    assert report["events"][0]["encoder_status"] == 5
    assert camera.read_count == 0
    assert camera.close_count == 1


def test_extended_status_detects_encoder_error_after_read(tmp_path, rig):
    camera = rig[0]
    original_read = camera.read_image

    def read():
        original_read()
        camera.encoder_status = 4

    camera.read_image = read
    report, code = run(tmp_path, extended_status=True)
    assert code == 1
    assert report["error"]["phase"] == "status_after_read"
    assert report["events"][0]["status_after_read"]["encoder_status"] == 4
    assert camera.read_count == 1


def test_keyboard_interrupt_still_closes_and_writes_report(tmp_path, rig):
    camera = rig[0]
    camera.read_results = [KeyboardInterrupt()]
    report, code = run(tmp_path)
    assert code == 130
    assert report["outcome"] == "interrupted"
    assert report["error"]["type"] == "KeyboardInterrupt"
    assert "status_after_failure" not in report
    assert camera.flush_count == 2
    assert camera.close_count == 1


def test_cleanup_and_status_errors_do_not_mask_primary_error(tmp_path, rig):
    camera = rig[0]
    camera.read_results = [DAQReadError(0, ENCODED_BUFFER_BYTES, "original vendor error")]
    original_flush = camera.flush

    def flush():
        original_flush()
        if camera.flush_count > 1:
            raise DAQError("failed flush")

    def close():
        camera.close_count += 1
        raise DAQError("failed close")

    camera.flush = flush
    camera.close = close
    del camera.num_available_images
    report, code = run(tmp_path)
    assert code == 1
    assert report["error"]["vendor_error"] == "original vendor error"
    assert report["status_after_failure"]["num_available_images"]["type"] == "AttributeError"
    assert [e["phase"] for e in report["cleanup_errors"]] == ["flush", "close", "close"]
    assert camera.close_count == 2
    assert report["camera_closed"] is False


def test_failed_close_retries_cleanup_only(tmp_path, rig):
    camera = rig[0]
    original_close = camera.close

    def close():
        if not camera.close_count:
            camera.close_count += 1
            raise DAQError("busy close")
        original_close()

    camera.close = close
    report, code = run(tmp_path)
    assert code == 1  # flaky cleanup is not a passed diagnostic
    assert report["camera_closed"] is True
    assert camera.close_count == 2
    assert camera.open_count == 1
    assert report["cleanup_errors"][0]["attempt"] == 1


def test_cleanup_failure_turns_success_into_failure(tmp_path, rig):
    def fail_close():
        raise OSError("close failed")

    rig[0].close = fail_close
    report, code = run(tmp_path)
    assert code == 1
    assert report["error"] is None
    assert report["cleanup_errors"][0]["message"] == "close failed"


@pytest.mark.parametrize("phase", ["open", "setup", "configure", "set_enc_mode", "trig_frame"])
def test_start_failure_closes_camera_and_creates_report(tmp_path, rig, phase):
    camera = rig[0]

    def fail(*_args, **_kwargs):
        raise DAQError(f"failed {phase}")

    setattr(camera, phase, fail)
    report, code = run(tmp_path)
    assert code == 1
    assert report["error"]["message"] == f"failed {phase}"
    assert camera.close_count == 1


def test_start_flush_failure_never_enables_or_triggers(tmp_path, rig, monkeypatch):
    camera = rig[0]
    original_flush = camera.flush

    def fail_first_flush():
        original_flush()
        if camera.flush_count == 1:
            raise DAQError("startup flush failed")

    monkeypatch.setattr(camera, "flush", fail_first_flush)
    report, code = run(tmp_path)
    assert code == 1
    assert report["error"] == {"phase": "start", "type": "DAQError", "message": "startup flush failed"}
    assert camera.sync_commands == ["flush", "flush"]  # Failure cleanup only.
    assert report["summary"]["read_attempts"] == 0
    assert report["camera_closed"] is True
    assert report["cleanup_errors"] == []
    assert camera.close_count == 1
    assert camera.payload is None


def test_dll_load_failure_is_reported_without_camera(tmp_path, rig, monkeypatch):
    def load():
        raise OSError("Windows DLL unavailable")

    monkeypatch.setattr(diagnostics, "FastEyeRLE", load)
    report, code = run(tmp_path)
    assert code == 1
    assert report["error"]["phase"] == "open"
    assert report["error"]["message"] == "Windows DLL unavailable"
    assert rig[0].open_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "unknown"},
        {"duration_s": 0},
        {"duration_s": float("nan")},
        {"duration_s": float("inf")},
        {"idle_timeout_s": 0},
        {"history_size": 0},
        {"history_size": 10_001},
        {"history_size": 1.5},
    ],
)
def test_invalid_arguments_do_not_open_hardware(tmp_path, rig, kwargs):
    with pytest.raises(ValueError):
        run(tmp_path, **kwargs)
    assert rig[0].open_count == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("changes", [{"rle_batch_frames": 123}, {"max_buffer_bytes": 1}])
def test_buffer_preflight_before_camera(tmp_path, rig, changes):
    settings = replace(ApolloSettings(max_number_frames=20), **changes)
    with pytest.raises(ValueError):
        diagnostics.run_diagnostics(tmp_path, settings)
    assert rig[0].open_count == 0


def test_output_preflight_and_unique_reports(tmp_path, rig):
    invalid = tmp_path / "file"
    invalid.write_text("keep")
    with pytest.raises(OSError):
        run(invalid)
    assert invalid.read_text() == "keep"
    assert rig[0].open_count == 0
    run(tmp_path)
    first = next(tmp_path.glob("*.json"))
    content = first.read_bytes()
    run(tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert first.read_bytes() == content


def test_report_write_failure_happens_after_cleanup_and_buffer_release(tmp_path, rig, monkeypatch):
    def fail_write(*_args, **_kwargs):
        assert rig[0].close_count == 1
        assert rig[0].payload is None
        assert all(ref() is None for ref in rig[1].destinations)
        raise OSError("disk full")

    monkeypatch.setattr(diagnostics.json, "dump", fail_write)
    with pytest.raises(OSError, match="disk full"):
        run(tmp_path, mode="decode")


@pytest.mark.parametrize("bad", [False, True])
def test_cli_dispatch_and_exit_codes(tmp_path, rig, monkeypatch, capsys, bad):
    if bad:
        rig[0].read_results = [DAQReadError(0, ENCODED_BUFFER_BYTES, "timeout")]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dwa",
            "diagnose",
            "--fps",
            "2000",
            "--duration",
            "0.1",
            "--output",
            str(tmp_path),
            "--read-timeout-ms",
            "1000",
            "--label",
            "camera A / port 1",
        ],
    )
    with pytest.raises(SystemExit) as exit_error:
        main()
    assert exit_error.value.code == int(bad)
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert report["settings"]["fps"] == 2000
    assert report["settings"]["read_timeout_ms"] == 1000
    assert report["settings"]["zero_byte_read_retries"] == 0
    assert report["label"] == "camera A / port 1"
    assert "report:" in capsys.readouterr().out


def test_cli_invalid_fps_never_opens_camera(rig, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dwa", "diagnose", "--fps", "nan"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert rig[0].open_count == 0


def test_cli_zero_fps_never_opens_camera(rig, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dwa", "diagnose", "--fps", "0"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert rig[0].open_count == 0


def test_cli_output_failure_is_not_a_success(tmp_path, rig, monkeypatch, capsys):
    invalid = tmp_path / "file"
    invalid.write_text("keep")
    monkeypatch.setattr(sys, "argv", ["dwa", "diagnose", "--output", str(invalid)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "Diagnostic output failed" in capsys.readouterr().err
    assert rig[0].open_count == 0


def test_raw_status_accessors_keep_numeric_vendor_values():
    camera = FastEyeRLE.__new__(FastEyeRLE)

    class DAQ:
        def get_int(self, name):
            return {"encoderStatus": 5, "numImgAvail": 9, "numImgStored": 10}[name]

    camera._daq = DAQ()
    assert camera.encoder_status == 5
    assert camera.get_enc_error()
    assert camera.num_available_images == 9
    assert camera.num_stored_images == 10


def test_environment_fingerprints_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostics, "CORE_PATH", tmp_path)
    (tmp_path / "PLabDAQCore.dll").write_bytes(b"test")
    info = diagnostics._environment()
    assert len(info["bundle_sha256"]["PLabDAQCore.dll"]) == 64
    assert info["bundle_sha256"]["rlDecodeLib.dll"].startswith("unavailable")
    assert info["python"]
