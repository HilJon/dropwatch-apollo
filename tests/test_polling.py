"""Conservative readiness and a deterministic model of measured USB overhead."""

import ctypes

import pytest

from dropwatch._hardware import DAQError
from dropwatch._hardware import DAQReadError
from dropwatch._hardware import FastEyeRLE
from dropwatch._hardware import RLEReadGate


@pytest.mark.parametrize(
    "counter,expected,queries",
    [(0, 0, 3), (1, 1, 1), (255, 255, 1), (256, 256, 2), (257, 1, 1), (65536, 65536, 3), (0xFFFFFF, 255, 1)],
)
def test_fast_counter_is_a_lower_bound_and_handles_zero_low_bytes(counter, expected, queries):
    class DAQ:
        calls = []

        def get_int(self, name):
            self.calls.append(name)
            index = int(name[-1])
            return (counter >> (8 * index)) & 255

    camera = FastEyeRLE.__new__(FastEyeRLE)
    camera._daq = DAQ()
    assert camera.minimum_ready_buffers == expected
    assert expected <= counter
    assert camera._daq.calls == [f"MEM_NR_IMG_STORED_{i}" for i in range(queries)]


def test_fast_counter_does_not_combine_bytes_across_a_rollover():
    class DAQ:
        calls = []

        def get_int(self, name):
            self.calls.append(name)
            # Buffer count grows from 255 to 256 after the low byte was sampled.
            return {"MEM_NR_IMG_STORED_0": 255, "MEM_NR_IMG_STORED_1": 1}[name]

    camera = FastEyeRLE.__new__(FastEyeRLE)
    camera._daq = DAQ()
    assert camera.minimum_ready_buffers == 255  # Not the unsafe composite 511.
    assert len(camera._daq.calls) == 1


@pytest.mark.parametrize("value", [-1, 256])
def test_fast_counter_rejects_invalid_register_values(value):
    class DAQ:
        def get_int(self, _name):
            return value

    camera = FastEyeRLE.__new__(FastEyeRLE)
    camera._daq = DAQ()
    with pytest.raises(DAQError, match="invalid stored-buffer counter byte"):
        _ = camera.minimum_ready_buffers


class BufferedCamera:
    available = 3
    encoder_status = 2
    full_queries = 0
    fast_queries = 0

    @property
    def num_stored_images(self):
        self.full_queries += 1
        return self.available

    @property
    def minimum_ready_buffers(self):
        self.fast_queries += 1
        return self.available


def test_gate_drains_only_confirmed_buffers_then_rechecks():
    camera = BufferedCamera()
    gate = RLEReadGate(camera)
    assert gate.poll()
    camera.available = 0  # The remaining credits still refer to the original snapshot.
    assert gate.poll()
    assert gate.poll()
    assert not gate.poll()
    assert camera.fast_queries == 2
    assert camera.full_queries == 0


def test_gate_rechecks_after_at_most_eight_credits():
    camera = BufferedCamera()
    camera.available = 255
    gate = RLEReadGate(camera)
    for _ in range(8):
        assert gate.poll()
    assert camera.fast_queries == 1
    assert gate.poll()
    assert camera.fast_queries == 2


def test_gate_reset_discards_all_old_credits():
    camera = BufferedCamera()
    gate = RLEReadGate(camera)
    assert gate.poll()
    gate.reset()
    camera.available = 0
    assert not gate.poll()


def test_encoder_error_is_checked_even_with_cached_ready_buffers():
    camera = BufferedCamera()
    gate = RLEReadGate(camera)
    assert gate.poll()
    camera.encoder_status = 4
    with pytest.raises(DAQError, match="encoder reported status 4"):
        gate.poll()
    camera.encoder_status = 2
    camera.available = 0
    assert not gate.poll()  # The failed poll invalidated the credits too.


def test_legacy_gate_queries_the_full_counter_before_every_read():
    camera = BufferedCamera()
    gate = RLEReadGate(camera, legacy=True)
    assert gate.poll()
    assert gate.poll()
    camera.available = 0
    assert not gate.poll()
    assert camera.full_queries == 3
    assert camera.fast_queries == 0


def test_gate_rejects_invalid_credit_count():
    camera = BufferedCamera()
    camera.available = -1
    with pytest.raises(DAQError, match="invalid ready-buffer count"):
        RLEReadGate(camera).poll()


class Clock:
    now = 0.0

    def time(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class TimedDAQ:
    """Synthetic four-buffer queue, not a claim about the physical FPGA capacity."""

    def __init__(self, clock, fps):
        self.clock = clock
        self.period = 100 / fps
        self.consumed = 0
        self.frame = (ctypes.c_ubyte * 8)()
        self.peak_backlog = 0

    @property
    def stored(self):
        return int(self.clock.now / self.period) - self.consumed

    def get_int(self, name):
        # Reports show ~42 ms for four control reads, ~29 ms for a bulk read.
        self.clock.sleep(0.0105 * (3 if name == "numImgStored" else 1))
        if name == "encoderStatus":
            return 2
        if name == "numImgStored":
            return self.stored
        return (self.stored >> (8 * int(name[-1]))) & 255

    def read(self):
        self.peak_backlog = max(self.peak_backlog, self.stored)
        assert self.stored > 0, "attempted an unconfirmed read"
        if self.stored >= 4:
            self.clock.sleep(0.5)
            return 0
        self.clock.sleep(0.029)
        self.consumed += 1
        return len(self.frame)

    def last_error(self):
        return "USB Bulk Read failed"


@pytest.mark.parametrize("fps", [1000, 2000])
def test_reduced_polling_sustains_measured_delays_in_bounded_queue_model(fps):
    clock = Clock()
    camera = FastEyeRLE.__new__(FastEyeRLE)
    camera._daq = TimedDAQ(clock, fps)
    gate = RLEReadGate(camera, clock=clock.time)
    while clock.now < 30:
        if gate.poll():
            camera.read_image()
            clock.sleep(0.002)  # Small decode/validation budget as well as USB time.
        else:
            clock.sleep(0.001)
    assert camera._daq.consumed >= int(30 * fps / 100) - 3
    assert camera._daq.peak_backlog < 4


def test_legacy_polling_reproduces_backlog_failure_in_same_queue_model():
    clock = Clock()
    camera = FastEyeRLE.__new__(FastEyeRLE)
    camera._daq = TimedDAQ(clock, 2000)
    gate = RLEReadGate(camera, legacy=True, clock=clock.time)
    with pytest.raises(DAQReadError, match="USB Bulk Read failed"):
        while clock.now < 3:
            if gate.poll():
                camera.read_image()
                clock.sleep(0.002)
            else:
                clock.sleep(0.001)
    assert camera._daq.peak_backlog >= 4
