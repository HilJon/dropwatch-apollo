from __future__ import annotations

import ctypes
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from dropwatch import ApolloFrameLossError
from dropwatch import ApolloSettings
from dropwatch import ApolloTransportError
from dropwatch._hardware import _DAQ
from dropwatch._hardware import DAQError
from dropwatch._hardware import DAQReadError
from dropwatch._hardware import FastEyeRLE
from dropwatch._hardware import RLEDecoder
from dropwatch._hardware import RLEDecodeResult
from dropwatch._source import _FastEyeApolloSource


class FakeFastEye:
    def __init__(self) -> None:
        self.serial_nr = 123
        self.num_stored_images = 1
        self.frame_period = None
        self.exposure_time = None
        self.bin_threshold = None
        self.max_bytes_per_image = None
        self.num_img_flush = None
        self.open_count = 0
        self.setup_count = 0
        self.flush_count = 0
        self.close_count = 0
        self.frame_counters = [1]
        self.read_timeout_ms = None
        self.read_results: list[Exception | None] = []
        self.read_count = 0

    def open(self) -> None:
        self.open_count += 1

    def setup(self, *, read_timeout_ms: int = 2000) -> None:
        self.setup_count += 1
        self.read_timeout_ms = read_timeout_ms

    def configure(
        self,
        *,
        frame_period_ms: float,
        exposure_time_ms: float,
        threshold: int,
        rle_batch_frames: int,
    ) -> None:
        self.frame_period = frame_period_ms
        self.exposure_time = exposure_time_ms
        self.bin_threshold = threshold
        self.max_bytes_per_image = 10_000
        self.num_img_flush = rle_batch_frames

    def set_enc_mode(self) -> None:
        pass

    def trig_frame(self) -> None:
        pass

    def get_enc_error(self) -> bool:
        return False

    def read_image(self) -> None:
        self.read_count += 1
        if self.read_results:
            result = self.read_results.pop(0)
            if result is not None:
                raise result

    def get_enc_data(self) -> bytearray:
        return rle_stream(self.frame_counters)

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1


class FakeDecoder:
    instances = 0
    destination_ids: ClassVar[list[int]] = []
    return_code = 1
    generated_bytes_override: int | None = None
    consumed_bytes_override: int | None = None

    def __init__(self) -> None:
        type(self).instances += 1

    @staticmethod
    def completed_frame_counters(source: bytearray) -> list[int]:
        return RLEDecoder.completed_frame_counters(source)

    def decode_buffer(self, source: bytearray, destination: np.ndarray) -> RLEDecodeResult:
        source_bytes = len(source)
        type(self).destination_ids.append(id(destination))
        if self.return_code < 0:
            return RLEDecodeResult(self.return_code, 0, 0)
        destination[: self.return_code, :, :1120] = 7
        # Simulate foreground in the discarded right view. It must never reach
        # Apollo's trigger, sequence, or video boundary.
        destination[: self.return_code, :, 1120:] = 0
        generated_bytes = self.return_code * 512 * 2240
        if self.generated_bytes_override is not None:
            generated_bytes = self.generated_bytes_override
        consumed_bytes = source_bytes
        if self.consumed_bytes_override is not None:
            consumed_bytes = self.consumed_bytes_override
        return RLEDecodeResult(self.return_code, generated_bytes, consumed_bytes)


class FakeCFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


def rle_stream(counters: list[int]) -> bytearray:
    data = bytearray()
    for counter in counters:
        header = bytearray(40)
        header[:6] = b"\x00\x00RLE1"
        header[6] = counter & 0xFF
        header[7] = (counter >> 8) & 0x7F
        data.extend(header)
    data.extend(bytearray(40))
    return data


def test_fasteye_adapter_returns_one_view_and_reuses_decode_buffer(
    monkeypatch,
):
    fake_camera = FakeFastEye()
    FakeDecoder.instances = 0
    FakeDecoder.destination_ids = []
    FakeDecoder.return_code = 1
    FakeDecoder.generated_bytes_override = None
    FakeDecoder.consumed_bytes_override = None
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20))

    source.open()
    source.start()
    first_batch = source.read()
    source.stop()
    assert source._decode_buffer is not None
    source._decode_buffer[0, 0, 0] = 0
    source.start()
    assert source._decode_buffer[0, 0, 0] == 255
    second_batch = source.read()
    source.stop()
    source.close()

    assert first_batch is not None
    assert second_batch is not None
    assert first_batch.shape == (1, 512, 1120)
    assert np.all(first_batch == 7)
    assert np.all(second_batch == 7)
    assert fake_camera.frame_period == 1.0
    assert fake_camera.exposure_time == 0.05
    assert fake_camera.bin_threshold == 127
    assert fake_camera.max_bytes_per_image == 10_000
    assert fake_camera.num_img_flush == 100
    assert fake_camera.read_timeout_ms == 500
    assert FakeDecoder.instances == 1
    assert len(set(FakeDecoder.destination_ids)) == 1
    assert fake_camera.open_count == 1
    assert fake_camera.setup_count == 2
    assert fake_camera.flush_count == 4
    assert fake_camera.close_count == 1
    assert source._camera is None
    assert source._decoder is None
    assert source._decode_buffer is None


def test_fasteye_adapter_recovers_one_zero_byte_read_without_flushing(monkeypatch):
    fake_camera = FakeFastEye()
    fake_camera.read_results = [DAQReadError(0, 80, "USB timeout"), None]
    FakeDecoder.return_code = 1
    FakeDecoder.generated_bytes_override = None
    FakeDecoder.consumed_bytes_override = None
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(
        ApolloSettings(max_number_frames=20, zero_byte_read_retries=1, zero_byte_retry_delay_ms=0)
    )

    source.open()
    source.start()
    batch = source.read()

    assert batch is not None
    assert fake_camera.read_count == 2
    assert fake_camera.flush_count == 1
    assert source.diagnostics.daq_reads == 2
    assert source.diagnostics.zero_byte_reads == 1
    assert source.diagnostics.recovered_zero_byte_reads == 1
    assert source.diagnostics.transport_failures == 0
    assert source.diagnostics.last_daq_read_ms >= 0
    assert source.diagnostics.max_daq_read_ms >= source.diagnostics.last_daq_read_ms
    assert source.diagnostics.last_vendor_error == "USB timeout"
    assert source.diagnostics.last_frame_counter == 1
    source.close()


def test_fasteye_adapter_rejects_repeated_zero_byte_reads_and_reopens(monkeypatch):
    failed_camera = FakeFastEye()
    failed_camera.read_results = [
        DAQReadError(0, 80, "first timeout"),
        DAQReadError(0, 80, "second timeout"),
    ]
    recovered_camera = FakeFastEye()
    cameras = iter([failed_camera, recovered_camera])
    FakeDecoder.return_code = 1
    FakeDecoder.generated_bytes_override = None
    FakeDecoder.consumed_bytes_override = None
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: next(cameras))
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(
        ApolloSettings(max_number_frames=20, zero_byte_read_retries=1, zero_byte_retry_delay_ms=0)
    )

    source.open()
    source.start()
    with pytest.raises(ApolloTransportError, match="0 bytes on 2 consecutive reads"):
        source.read()
    assert source.diagnostics.transport_failures == 1
    source.stop()
    assert failed_camera.close_count == 1

    source.start()
    assert source.read() is not None
    source.close()
    assert recovered_camera.open_count == 1
    assert recovered_camera.close_count == 1


def test_fasteye_adapter_never_retries_a_positive_partial_read(monkeypatch):
    fake_camera = FakeFastEye()
    fake_camera.read_results = [DAQReadError(40, 80, "short USB transfer"), None]
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20, zero_byte_read_retries=3))

    source.open()
    source.start()
    with pytest.raises(ApolloTransportError, match="partial camera transfer"):
        source.read()

    assert fake_camera.read_count == 1
    assert source.diagnostics.zero_byte_reads == 0
    assert source.diagnostics.transport_failures == 1
    source.close()


def test_fasteye_adapter_rejects_zero_decoded_frames(monkeypatch):
    fake_camera = FakeFastEye()
    fake_camera.frame_counters = []
    FakeDecoder.return_code = 0
    FakeDecoder.generated_bytes_override = None
    FakeDecoder.consumed_bytes_override = None
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20))

    source.open()
    source.start()
    with pytest.raises(ApolloFrameLossError, match="no complete frames"):
        source.read()
    source.close()
    FakeDecoder.return_code = 1


def test_fasteye_adapter_reports_rle_decode_error(monkeypatch):
    fake_camera = FakeFastEye()
    FakeDecoder.return_code = -7
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20))

    source.open()
    source.start()
    with pytest.raises(ApolloFrameLossError, match="RLE decoder failed with status -7"):
        source.read()
    source.close()
    FakeDecoder.return_code = 1


def test_fasteye_adapter_flushes_after_incomplete_start(monkeypatch):
    fake_camera = FakeFastEye()

    def fail_configuration(**_settings) -> None:
        raise RuntimeError("configuration failed")

    fake_camera.configure = fail_configuration  # type: ignore[method-assign]
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20))

    source.open()
    with pytest.raises(RuntimeError, match="configuration failed"):
        source.start()
    source.close()

    assert fake_camera.flush_count == 1
    assert fake_camera.close_count == 1


def test_fasteye_adapter_rejects_frame_counter_gap(monkeypatch):
    fake_camera = FakeFastEye()
    fake_camera.frame_counters = [10, 12]
    FakeDecoder.return_code = 2
    FakeDecoder.generated_bytes_override = None
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20))

    source.open()
    source.start()
    with pytest.raises(ApolloFrameLossError, match="jumped from 10 to 12"):
        source.read()
    source.close()
    FakeDecoder.return_code = 1


def test_zero_byte_retry_still_rejects_a_frame_counter_gap(monkeypatch):
    fake_camera = FakeFastEye()
    fake_camera.frame_counters = [10]
    FakeDecoder.return_code = 1
    FakeDecoder.generated_bytes_override = None
    FakeDecoder.consumed_bytes_override = None
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(
        ApolloSettings(max_number_frames=20, zero_byte_read_retries=1, zero_byte_retry_delay_ms=0)
    )

    source.open()
    source.start()
    assert source.read() is not None
    fake_camera.frame_counters = [12]
    fake_camera.read_results = [DAQReadError(0, 80, "timeout"), None]

    with pytest.raises(ApolloFrameLossError, match="jumped from 10 to 12"):
        source.read()
    source.close()


def test_fasteye_adapter_rejects_incomplete_decoder_output(monkeypatch):
    fake_camera = FakeFastEye()
    FakeDecoder.return_code = 1
    FakeDecoder.generated_bytes_override = 0
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20))

    source.open()
    source.start()
    with pytest.raises(ApolloFrameLossError, match="produced 0 bytes"):
        source.read()
    source.close()
    FakeDecoder.generated_bytes_override = None


def test_fasteye_adapter_rejects_unconsumed_rle_input(monkeypatch):
    fake_camera = FakeFastEye()
    fake_camera.frame_counters = [1, 2]
    FakeDecoder.return_code = 1
    FakeDecoder.generated_bytes_override = None
    FakeDecoder.consumed_bytes_override = 40
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20))

    source.open()
    source.start()
    with pytest.raises(ApolloFrameLossError, match="consumed 40 of 120 encoded bytes"):
        source.read()
    source.close()
    FakeDecoder.consumed_bytes_override = None


def test_fasteye_adapter_accepts_only_zero_unconsumed_padding(monkeypatch):
    fake_camera = FakeFastEye()
    monkeypatch.setattr(FakeDecoder, "return_code", 1)
    monkeypatch.setattr(FakeDecoder, "generated_bytes_override", None)
    monkeypatch.setattr(FakeDecoder, "consumed_bytes_override", 40)
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20))
    source.start()
    assert len(source.read()) == 1
    source.close()


def test_failed_poisoned_handle_close_is_retained_for_retry(monkeypatch):
    class RetryCloseCamera(FakeFastEye):
        def close(self):
            self.close_count += 1
            if self.close_count == 1:
                raise DAQError("close failed")

    camera = RetryCloseCamera()
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20))
    source.start()
    source._poisoned = True
    with pytest.raises(DAQError, match="close failed"):
        source.stop()
    assert source._camera is camera
    source.close()
    assert camera.close_count == 2
    assert source._camera is source._decode_buffer is None


def test_fasteye_rejects_rle_batch_larger_than_encoded_buffer():
    class FakeDAQ:
        def __init__(self) -> None:
            self.settings: list[tuple[str, str | float]] = []

        def set(self, name: str, value: str | float) -> None:
            self.settings.append((name, value))

    camera = FastEyeRLE.__new__(FastEyeRLE)
    camera._daq = FakeDAQ()
    camera._encoded_data = bytearray(1_000_000)

    with pytest.raises(DAQError, match="safe maximum: 100"):
        camera.configure(
            frame_period_ms=1.0,
            exposure_time_ms=0.05,
            threshold=127,
            rle_batch_frames=101,
        )
    assert camera._daq.settings == []

    camera.configure(
        frame_period_ms=1.0,
        exposure_time_ms=0.05,
        threshold=127,
        rle_batch_frames=100,
    )
    assert camera._daq.settings[-1] == ("numImgFlush", 100)


def test_fasteye_setup_validates_geometry_and_applies_bounded_timeout():
    expected_size = 960 * 1024 * 10 // 8

    class FakeSetupDAQ:
        def __init__(self) -> None:
            self.frame = None
            self.settings: list[tuple[str, str | float]] = []

        @staticmethod
        def get_int(name: str) -> int:
            return {"width": 960, "height": 1024, "reqBufSize": expected_size}[name]

        def set(self, name: str, value: str | float) -> None:
            self.settings.append((name, value))

    camera = FastEyeRLE.__new__(FastEyeRLE)
    camera._daq = FakeSetupDAQ()
    camera._encoded_data = None

    camera.setup(read_timeout_ms=500)

    assert len(camera._encoded_data) == expected_size
    assert len(camera._daq.frame) == expected_size
    assert camera._daq.settings == [("timeOut", 500), ("APP_MODE", 0)]


def test_fasteye_setup_rejects_an_unexpected_vendor_geometry():
    class WrongGeometryDAQ:
        frame = None

        @staticmethod
        def get_int(name: str) -> int:
            return {"width": 1120, "height": 512}[name]

    camera = FastEyeRLE.__new__(FastEyeRLE)
    camera._daq = WrongGeometryDAQ()
    camera._encoded_data = None

    with pytest.raises(DAQError, match="unexpected camera geometry 1120x512"):
        camera.setup()


def test_fasteye_clears_stale_rle_bytes_before_every_vendor_read():
    class FakeReadDAQ:
        def __init__(self) -> None:
            self.frame = (ctypes.c_ubyte * 16)(*([123] * 16))

        @staticmethod
        def read() -> int:
            return 0

        @staticmethod
        def last_error() -> str:
            return "timeout"

    camera = FastEyeRLE.__new__(FastEyeRLE)
    camera._daq = FakeReadDAQ()

    with pytest.raises(DAQReadError, match="0 of 16 bytes") as error:
        camera.read_image()

    assert error.value.actual_bytes == 0
    assert error.value.expected_bytes == 16
    assert error.value.vendor_error == "timeout"
    assert bytes(camera._daq.frame) == bytes(16)


def test_daq_declares_vendor_abi_and_uses_integer_handle(monkeypatch):
    class FakeLibrary:
        def __init__(self) -> None:
            self.set_calls: list[tuple[int, bytes, bytes]] = []
            self.daq_open = FakeCFunction(lambda _device, _config: 17)
            self.daq_close = FakeCFunction(lambda _handle: 1)
            self.daq_get = FakeCFunction(self.get)
            self.daq_set = FakeCFunction(self.set)
            self.daq_read = FakeCFunction(lambda _handle, _frame, length: length)
            self.daq_lastErrorMessage = FakeCFunction(self.last_error)

        @staticmethod
        def get(_handle, name, output, _length):
            output.value = b"1120" if name == b"width" else b"0"
            return 1

        def set(self, handle, name, value):
            self.set_calls.append((handle, name, value))
            return 1

        @staticmethod
        def last_error(_handle, output, _length):
            output.value = b"fake error"
            return 1

    library = FakeLibrary()
    monkeypatch.setattr(ctypes.cdll, "LoadLibrary", lambda _path: library)
    daq = _DAQ(Path("unused.dll"))

    assert library.daq_open.argtypes == (
        ctypes.c_char_p,
        ctypes.c_char_p,
    )
    assert library.daq_open.restype is ctypes.c_int
    assert library.daq_read.argtypes == (
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    assert daq.open("enc_viamos", "rle")
    assert daq.get_int("width") == 1120
    daq.set("sync", "frame")
    daq.frame = (ctypes.c_ubyte * 16)()
    assert daq.read() == 16
    daq.close()
    assert library.set_calls == [(17, b"sync", b"frame")]


def test_rle_decoder_validates_buffers_and_reports_vendor_counts(
    monkeypatch,
):
    class FakeDecodeLibrary:
        def __init__(self) -> None:
            self.rlDecode = FakeCFunction(self.decode)

        @staticmethod
        def decode(
            _destination,
            _source,
            destination_size,
            source_size,
            generated,
            consumed,
        ):
            generated._obj.value = destination_size
            consumed._obj.value = source_size
            return 1

    library = FakeDecodeLibrary()
    monkeypatch.setattr(ctypes, "WinDLL", lambda _path: library, raising=False)
    decoder = RLEDecoder(Path("unused.dll"))
    encoded = bytearray(80)
    destination = np.empty((1, 4, 8), dtype=np.uint8)

    assert decoder.decode_buffer(encoded, destination) == RLEDecodeResult(1, 32, 80)
    with pytest.raises(ValueError, match="C-contiguous uint8"):
        decoder.decode_buffer(encoded, destination.astype(np.uint16))
    with pytest.raises(ValueError, match="multiple of 40"):
        decoder.decode_buffer(bytearray(41), destination)


def test_rle_counter_parser_requires_complete_frames():
    complete = rle_stream([32767, 0])
    incomplete = complete[:-40]

    assert RLEDecoder.completed_frame_counters(complete) == [32767, 0]
    assert RLEDecoder.completed_frame_counters(incomplete) == [32767]


def test_fasteye_frame_counter_wrap_is_contiguous(monkeypatch):
    fake_camera = FakeFastEye()
    fake_camera.frame_counters = [32767]
    FakeDecoder.return_code = 1
    FakeDecoder.generated_bytes_override = None
    monkeypatch.setattr("dropwatch._hardware.FastEyeRLE", lambda: fake_camera)
    monkeypatch.setattr("dropwatch._hardware.RLEDecoder", FakeDecoder)
    source = _FastEyeApolloSource(ApolloSettings(max_number_frames=20))

    source.open()
    source.start()
    source.read()
    fake_camera.frame_counters = [0]
    source.read()
    source.close()
