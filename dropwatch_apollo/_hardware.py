"""Minimal Windows FastEye RLE hardware adapter used by Dropwatch Apollo."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from dropwatch_apollo.models import ApolloFrameLossError

PACKAGE_ROOT = Path(__file__).parent
CORE_PATH = PACKAGE_ROOT / "data" / "core"
RAW_FRAME_HEIGHT = 512
RAW_FRAME_WIDTH = 2240
LEFT_VIEW_WIDTH = RAW_FRAME_WIDTH // 2
SENSOR_WIDTH = 960
SENSOR_HEIGHT = 1024
ENCODED_BUFFER_BYTES = SENSOR_WIDTH * SENSOR_HEIGHT * 10 // 8
MAX_BYTES_PER_IMAGE = 10_000


class DAQError(RuntimeError):
    """PLabDAQ operation failed."""


class DAQReadError(DAQError):
    """A vendor read returned a byte count that cannot form a complete buffer."""

    def __init__(self, actual_bytes: int, expected_bytes: int, vendor_error: str) -> None:
        super().__init__(f"camera returned {actual_bytes} of {expected_bytes} bytes (vendor error: {vendor_error})")
        self.actual_bytes = actual_bytes
        self.expected_bytes = expected_bytes
        self.vendor_error = vendor_error


@dataclass(frozen=True)
class RLEDecodeResult:
    """Metadata returned by one vendor decoder call."""

    frames: int
    generated_bytes: int
    consumed_bytes: int


class _DAQ:
    def __init__(self, library_path: Path = CORE_PATH / "PLabDAQCore.dll") -> None:
        self._library = ctypes.cdll.LoadLibrary(str(library_path))
        self._library.daq_open.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
        self._library.daq_open.restype = ctypes.c_int
        self._library.daq_close.argtypes = (ctypes.c_int,)
        self._library.daq_close.restype = ctypes.c_int
        self._library.daq_get.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
        self._library.daq_get.restype = ctypes.c_int
        self._library.daq_set.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
        self._library.daq_set.restype = ctypes.c_int
        self._library.daq_read.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
        self._library.daq_read.restype = ctypes.c_int
        self._library.daq_lastErrorMessage.argtypes = (
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        self._library.daq_lastErrorMessage.restype = ctypes.c_int
        self._device = 0
        self._buffer = ctypes.create_string_buffer(500)
        self.frame: Any | None = None

    def open(self, device_name: str, config_name: str) -> bool:
        if not self._device:
            self._device = self._library.daq_open(device_name.encode(), config_name.encode())
        return bool(self._device)

    def close(self) -> None:
        if self._device:
            result = self._library.daq_close(self._device)
            error = self.last_error() if not result else None
            if error is not None:
                raise DAQError(f"could not close camera: {error}")
            self._device = 0

    def get_int(self, name: str) -> int:
        if not self._device:
            raise DAQError("no camera is open")
        result = self._library.daq_get(self._device, name.encode(), self._buffer, len(self._buffer))
        if not result:
            raise DAQError(f"could not read {name}: {self.last_error()}")
        return int(self._buffer.value)

    def set(self, name: str, value: str | float) -> None:
        if not self._device:
            raise DAQError("no camera is open")
        result = self._library.daq_set(self._device, name.encode(), str(value).encode())
        if not result:
            raise DAQError(f"could not set {name}={value}: {self.last_error()}")

    def read(self) -> int:
        if not self._device:
            raise DAQError("no camera is open")
        if self.frame is None:
            raise DAQError("camera frame buffer is not allocated")
        return int(self._library.daq_read(self._device, self.frame, len(self.frame)))

    def last_error(self) -> str:
        result = self._library.daq_lastErrorMessage(self._device, self._buffer, len(self._buffer))
        return self._buffer.value.decode(errors="replace") if result else "unknown camera error"


class FastEyeRLE:
    """Only the FastEye operations required by the Dropwatch Apollo RLE stream."""

    def __init__(self) -> None:
        self._daq = _DAQ()
        self._encoded_data: bytearray | None = None

    def open(self) -> None:
        for _ in range(3):
            if self._daq.open("enc_viamos", "rle"):
                self._daq.close()
                if self._daq.open("enc_viamos", "rle"):
                    return
            self._daq.close()
        raise ConnectionError(f"could not open FastEye camera: {self._daq.last_error()}")

    def setup(self, *, read_timeout_ms: int = 2000) -> None:
        if read_timeout_ms < 1:
            raise ValueError("read_timeout_ms must be >= 1")
        camera_width = self._daq.get_int("width")
        camera_height = self._daq.get_int("height")
        frame_size = camera_width * camera_height * 10 // 8
        # The sensor/USB geometry is not the padded decoder output geometry.
        expected_size = ENCODED_BUFFER_BYTES
        if (camera_width, camera_height) != (SENSOR_WIDTH, SENSOR_HEIGHT):
            raise DAQError(
                f"unexpected camera geometry {camera_width}x{camera_height}: "
                f"RLE buffer is {frame_size} bytes, expected {expected_size}"
            )
        requested_size = self._daq.get_int("reqBufSize")
        if requested_size != frame_size:
            raise DAQError(f"camera requested {requested_size} bytes, expected {frame_size}")

        encoded_data = self._encoded_data
        if encoded_data is None or len(encoded_data) != frame_size:
            encoded_data = bytearray(frame_size)
        frame_type = ctypes.c_ubyte * frame_size
        self._daq.frame = frame_type.from_buffer(encoded_data)
        ctypes.memset(self._daq.frame, 0, frame_size)
        self._encoded_data = encoded_data
        self._daq.set("timeOut", read_timeout_ms)
        self._daq.set("APP_MODE", 0)

    @property
    def num_stored_images(self) -> int:
        return self._daq.get_int("numImgStored")

    @property
    def num_available_images(self) -> int:
        return self._daq.get_int("numImgAvail")

    @property
    def minimum_ready_buffers(self) -> int:
        """Read a conservative buffer count, normally with one register query.

        The vendor's numImgStored reads all three bytes over USB. With one host
        reader, the FPGA can only add buffers during this query. The first
        nonzero byte is therefore a lower bound; never combine independently
        sampled bytes, which could overestimate a counter crossing 255/65535.
        Read upper bytes when needed so multiples of 256 are not mistaken for 0.
        """
        for index in range(3):
            value = self._daq.get_int(f"MEM_NR_IMG_STORED_{index}")
            if not 0 <= value <= 255:
                raise DAQError(f"invalid stored-buffer counter byte {index}: {value}")
            if value:
                return value << (8 * index)
        return 0

    @property
    def encoder_status(self) -> int:
        return self._daq.get_int("encoderStatus")

    @property
    def acquisition_mode(self) -> int:
        return self._daq.get_int("APP_MODE")

    @property
    def accelerator_control(self) -> int:
        return self._daq.get_int("ACCELERATOR_CTRL")

    def configure(
        self,
        *,
        frame_period_ms: float,
        exposure_time_ms: float,
        threshold: int,
        rle_batch_frames: int,
    ) -> None:
        encoded_data = self._encoded_data
        if encoded_data is None:
            raise DAQError("camera frame buffer is not available; call setup() before configure()")
        required_bytes = rle_batch_frames * MAX_BYTES_PER_IMAGE
        if required_bytes > len(encoded_data):
            max_frames = len(encoded_data) // MAX_BYTES_PER_IMAGE
            raise DAQError(
                f"rle_batch_frames={rle_batch_frames} can require {required_bytes} encoded bytes, "
                f"but the camera buffer has {len(encoded_data)} bytes (safe maximum: {max_frames})"
            )
        self._daq.set("framePeriod", frame_period_ms)
        self._daq.set("exposureTime", exposure_time_ms)
        self._daq.set("binarizationThreshold", threshold)
        self._daq.set("maxBytesPerImage", MAX_BYTES_PER_IMAGE)
        self._daq.set("numImgFlush", rle_batch_frames)

    def set_enc_mode(self) -> None:
        self._daq.set("sync", "rle")

    def trig_frame(self) -> None:
        self._daq.set("sync", "frame")

    def flush(self) -> None:
        """Stop/reset RLE acquisition and clear FPGA buffers; re-enable before triggering."""
        self._daq.set("sync", "flush")

    def get_enc_error(self) -> bool:
        return self.encoder_status >= 4

    def read_image(self) -> None:
        if self._daq.frame is None:
            raise DAQError("camera frame buffer is not allocated")
        # A failed vendor read may leave the previous RLE payload untouched.
        # Clearing first makes it impossible to decode stale data as a new batch.
        ctypes.memset(self._daq.frame, 0, len(self._daq.frame))
        number_bytes = self._daq.read()
        expected_bytes = len(self._daq.frame)
        if number_bytes != expected_bytes:
            raise DAQReadError(number_bytes, expected_bytes, self._daq.last_error())

    def get_enc_data(self) -> bytearray:
        if self._encoded_data is None:
            raise DAQError("camera frame buffer is not available")
        return self._encoded_data

    def close(self) -> None:
        try:
            self._daq.close()
        finally:
            self._daq.frame = None
            self._encoded_data = None


class RLEReadGate:
    """Bounded readiness credits shared by the recorder and diagnostic loop.

    Only confirmed buffers may be read. There is exactly one camera reader and
    no automatic reset/retrigger here. Encoder errors are checked on EVERY poll,
    including reads using cached credits. Reset credits after a read failure.
    """

    def __init__(self, camera: FastEyeRLE, *, legacy: bool = False, clock: Callable[[], float] | None = None) -> None:
        self._camera = camera
        self._legacy = legacy
        self._clock = perf_counter if clock is None else clock
        self._ready = 0

    def reset(self) -> None:
        self._ready = 0

    def query(self, name: str, event: dict[str, Any] | None) -> int:
        """Sample one status property and optionally time the complete vendor call."""
        started = self._clock()
        try:
            value: int = getattr(self._camera, name)
            if event is not None:
                event[name] = value
            return value
        finally:
            if event is not None:
                event.setdefault("status_query_ms", {})[name] = (self._clock() - started) * 1000

    def poll(self, event: dict[str, Any] | None = None) -> bool:
        try:
            status = self.query("encoder_status", event)
            if status < 0 or status >= 4:
                raise DAQError(f"camera RLE encoder reported status {status}")
            if self._legacy or not self._ready:
                name = "num_stored_images" if self._legacy else "minimum_ready_buffers"
                available = self.query(name, event)
                if not 0 <= available <= 0xFFFFFF:
                    raise DAQError(f"invalid ready-buffer count: {available}")
                # Bound each drain to eight confirmed buffers, regardless of backlog.
                self._ready = min(available, 8)
                readiness_source = "full_counter" if self._legacy else "counter_bytes"
            else:
                readiness_source = "cached_credit"
            if event is not None:
                event.update(readiness_source=readiness_source, ready_buffer_credit=self._ready)
            if not self._ready:
                return False
            self._ready -= 1
            return True
        except BaseException:
            self.reset()
            raise


class RLEDecoder:
    """Thin wrapper around the camera vendor's RLE decoder DLL."""

    _PACKET_BYTES = 40
    _HEADER = b"\x00\x00RLE1"

    def __init__(self, library_path: Path = CORE_PATH / "rlDecodeLib.dll") -> None:
        self._library = ctypes.WinDLL(str(library_path))  # type: ignore[attr-defined]
        self._library.rlDecode.argtypes = (
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        )
        self._library.rlDecode.restype = ctypes.c_int64

    def decode_buffer(self, encoded_data: bytearray, destination: np.ndarray) -> RLEDecodeResult:
        if destination.dtype != np.uint8 or not destination.flags.c_contiguous:
            raise ValueError("RLE destination must be a C-contiguous uint8 array")
        if len(encoded_data) % self._PACKET_BYTES:
            raise ValueError(f"RLE source length must be a multiple of {self._PACKET_BYTES} bytes")
        generated_bytes = ctypes.c_uint64()
        consumed_bytes = ctypes.c_uint64()
        source = (ctypes.c_uint8 * len(encoded_data)).from_buffer(encoded_data)
        frames = int(
            self._library.rlDecode(
                destination.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                source,
                destination.size,
                len(encoded_data),
                ctypes.byref(generated_bytes),
                ctypes.byref(consumed_bytes),
            )
        )
        return RLEDecodeResult(
            frames=frames,
            generated_bytes=int(generated_bytes.value),
            consumed_bytes=int(consumed_bytes.value),
        )

    @classmethod
    def completed_frame_counters(cls, encoded_data: bytearray) -> list[int]:
        """Read counters for complete images from the vendor's 40-byte packets."""
        counters: list[int] = []
        active_counter: int | None = None
        complete_length = len(encoded_data) - len(encoded_data) % cls._PACKET_BYTES

        for offset in range(0, complete_length, cls._PACKET_BYTES):
            is_header = encoded_data[offset : offset + len(cls._HEADER)] == cls._HEADER
            if is_header:
                if active_counter is not None:
                    counters.append(active_counter)
                active_counter = ((encoded_data[offset + 7] & 0x7F) << 8) | encoded_data[offset + 6]
            elif encoded_data[offset] == 0:
                if active_counter is not None:
                    counters.append(active_counter)
                    active_counter = None
                # Real vendor buffers contain leading and inter-frame padding.
                # Only a subsequent header starts another frame.

        return counters


def validate_rle_batch(
    decoder: RLEDecoder,
    encoded_data: bytearray,
    destination: np.ndarray,
    previous_counter: int | None,
) -> tuple[int, int]:
    """Decode and validate one batch; shared by acquisition and diagnostics."""
    decoded = decoder.decode_buffer(encoded_data, destination)
    if decoded.frames < 0:
        raise ApolloFrameLossError(f"RLE decoder failed with status {decoded.frames}")
    if decoded.frames == 0:
        raise ApolloFrameLossError("RLE decoder returned no complete frames")
    if decoded.frames > len(destination):
        raise ApolloFrameLossError(f"RLE decoder returned {decoded.frames} frames for a buffer of {len(destination)}")
    if not 0 < decoded.consumed_bytes <= len(encoded_data) or np.count_nonzero(
        np.frombuffer(encoded_data, dtype=np.uint8, offset=min(decoded.consumed_bytes, len(encoded_data)))
    ):
        raise ApolloFrameLossError(
            f"RLE decoder consumed {decoded.consumed_bytes} of {len(encoded_data)} encoded bytes"
        )
    expected_bytes = decoded.frames * destination.shape[1] * destination.shape[2]
    if decoded.generated_bytes != expected_bytes:
        raise ApolloFrameLossError(
            f"RLE decoder produced {decoded.generated_bytes} bytes; expected {expected_bytes} "
            f"for {decoded.frames} frames"
        )
    counters = decoder.completed_frame_counters(encoded_data)
    if len(counters) != decoded.frames:
        raise ApolloFrameLossError(
            f"RLE stream contains {len(counters)} completed frame counters, "
            f"but the decoder returned {decoded.frames} frames"
        )
    for counter in counters:
        if previous_counter is not None:
            expected = (previous_counter + 1) & 0x7FFF
            if counter != expected:
                raise ApolloFrameLossError(
                    f"RLE frame counter jumped from {previous_counter} to {counter}; expected {expected}"
                )
        previous_counter = counter
    return decoded.frames, counters[-1]
