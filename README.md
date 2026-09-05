# dropwatch-apollo

A small, single-view FastEye RLE recorder. No calibration, split-view processing,
tip detector, or dropeval dependency. Camera frames arrive on the PC continuously;
triggering and recording happen there. Hardware acquisition requires Windows x64.
Tests and reference-file replay also run without hardware.

The product is **dropwatch-apollo**; its Python namespace is `dropwatch_apollo`.
Apollo alone names the dispenser, not the camera or this library.
For existing `recorder.*` scripts, see [MIGRATION.md](MIGRATION.md): version 0.3
includes a small compatibility facade, not a second capture engine.

## Recording

```python
from dropwatch_apollo import ApolloSettings, DropwatchApollo

settings = ApolloSettings(
    max_number_frames=200,  # TOTAL length, including pre-trigger
    pre_trigger=20,
    frame_period_ms=1.0,  # 1000 fps; use 0.5 for 2000 fps
    exposure_time_ms=0.05,
)

with DropwatchApollo(settings) as dw:
    dw.snapshot("plate.png")  # optional: inspect the physical image
    dw.set_trigger_size(width=100)
    dw.start(max_sequences=5)  # returns after clear ROI + full lookback
    # ... dispense here ...
    sequences = dw.stop()  # drains queued data and finishes active shot
    dw.save_videos(sequences, "videos")
```

Each sequence is one read-only NumPy array, shaped
`(max_number_frames, 512, 1120)`. Only the left half is retained.
Zero denotes black foreground; nonzero denotes background. The trigger frame
is at index `pre_trigger`. Arrays keep their raw orientation; videos/PNGs are
transposed into physical orientation. A PC-side crop does **not** reduce the
FPGA/USB transfer or the full vendor decode.

Dropwatch Apollo defaults to `trigger_from_top=False`, no tip detector (equivalent to
`IgnoreTipDetector`), and no debug plots or split-view trigger.
`set_trigger_size()` finds the highest foreground connected to the physical
bottom edge and puts a band immediately above it. Inspect the snapshot: noise
connected to an edge can affect this placement.

## Trigger and stop semantics

- `start()` defaults to one shot. `max_sequences` is a hard upper bound;
  reaching it stops intake, even if more droplets arrive.
- A second droplet inside an active window is included in that window, not
  duplicated into a separate shot.
- Lookback and clear-frame history continue throughout capture. There is no
  additional lookback-refill interval between shots. The ROI must still clear
  for `rearm_clear_frames` before another trigger is accepted.
- `get_sequence(timeout_s=...)` consumes one completed shot without stopping.
  `get_sequences(timeout_s=...)` waits for the limit, duration stop, or replay
  EOF, then consumes all remaining shots. A retrieval timeout leaves capture running.
- `stop()` allows one RLE flush interval after the request, drains until the
  source reports it has caught up, then finishes any active shot. This includes
  queued droplets and partially flushed FPGA data, and may include droplets
  arriving just after the request. At defaults, the flush grace is about 101 ms.
  This boundary must be qualified on the real camera.
- A drain timeout raises `ApolloIncompleteSequenceError`, even if Dropwatch Apollo could
  not prove the queue empty and no shot was active. Its `completed_sequences`
  contains the finished shots.
- `abort()` intentionally discards unread/in-progress data. It still waits for
  an outstanding vendor read and cleanup; it cannot interrupt arbitrary native code.
- `start(max_sequences=40, max_duration_s=60)` also requests a graceful stop at
  the duration limit. Draining/finalization can extend beyond that duration.

Use the same instance for subsequent acquisitions. Collect sequences and
evaluation results first. Immutable configuration cannot be replaced behind the
camera adapter's back; create another instance for different camera settings.

## Large recordings and memory ownership

In-memory mode reserves one block per possible shot before arming. Each block
contains the final sequence plus a separate `pre_trigger`-frame waiting area.
This lets consumers put the lookback prefix in chronological order without a
large copy in the camera thread. Completed blocks are no longer retained by the
allocation pool; only callers, evaluation, and still-needed lookback references
can keep them alive.

40 shots × 1000 frames need about 21.4 GiB for the returned pixels alone.
For this workload, use bounded raw spooling:

```python
settings = ApolloSettings(
    max_number_frames=1000,
    pre_trigger=20,
    spool_directory="recordings/raw",
    spool_buffer_count=3,
)
with DropwatchApollo(settings) as dw:
    dw.set_trigger_size(width=100)
    dw.start(max_sequences=40)
    # ... dispense ...
    sequences = dw.stop()
    session = dw.recording_directory
    dw.save_videos(sequences, session / "video")
```

One background writer saves each complete shot as an atomic, lossless NPY file
in a unique acquisition directory. Only after finalization is its RAM block
eligible for recycling; lookback references also pin blocks until safe.
Returned sequences are read-only `np.memmap` arrays. Fetching them does not
read or copy all image pixels into RAM, and files remain after `close()`.

For 40 × 1000 frames with a 20-frame lookback and three blocks, explicit capture
and camera buffers total about **1.75 GiB**, below the default
`max_buffer_bytes=2 * 1024**3`. Very long lookbacks need more pinned blocks;
impossible configurations are rejected before recording. Three 2000-frame blocks
need a larger memory budget, e.g. 4 GiB.

The memory budget covers Dropwatch Apollo's explicit acquisition buffers, **not** OS file
cache, mapped pages accessed by consumers, or arbitrary evaluator allocations.
Release your own arrays/DataFrames when finished. No library can release memory
still referenced by application code.

Use a fast local SSD with enough free disk space. The queue and buffer count
are bounded: if writing cannot keep up, Dropwatch Apollo raises an error rather than
blocking camera intake or silently overwriting a shot. Existing completed shots
remain recoverable. Disk spooling, evaluation, and camera throughput must be
benchmarked together at 1000/2000 fps.

For windows longer than 2000 frames, or a smaller fixed memory footprint, set
`spool_chunk_frames=100` together with `spool_directory`. This mode allocates
`spool_buffer_count` chunks plus the pre-trigger ring, independent of window
length. Chunks are written in order and only a complete, fsynced NPY is published.
Unfinished files are removed on failure; completed windows remain. A session
quota (`max_spool_bytes`, default 64 GiB) and per-window free-space checks fail
explicitly. This is the default storage mode of the `recorder.*` facade.

## Failure handling

One zero-byte vendor read is retried by default, without an intervening flush
or retrigger. Positive partial reads, exhausted retries, invalid decoder output,
or discontinuous 15-bit frame counters fail the session. A retry counts as
recovered only after its decoded frames pass integrity checks.

The transport geometry is `960 × 1024` (1,228,800 encoded bytes), distinct from
the padded decoder layout `512 × 2240`. Leading/inter-frame RLE padding is
accepted. Unconsumed decoder input is allowed only when it is zero padding.
Real reference excerpts and independent pixel-mask hashes are included in tests;
Windows tests additionally exercise the actual decoder DLL without a camera.

A later read failure does **not** destroy earlier completed recordings:

```python
try:
    sequences = dw.stop()
except Exception as error:
    completed = getattr(error, "completed_sequences", [])
    # Only previously completed, validated shots; never the incomplete window.
    if completed:
        dw.save_videos(completed, "recovered_video")
    raise
```

Use `stats` for transfer counts, zero-byte reads/recoveries, vendor errors,
latency, integrity-failure events, and incomplete sequences. Transport failure
does not by itself prove the FPGA is the cause. A poisoned handle is closed and
reopened on the next acquisition. If vendor close fails, the handle is retained
so another `close()` can retry it.

For an isolated, fail-fast zero-byte investigation, use `dwa diagnose`:

```shell
dwa diagnose --mode transport --fps 1000 --duration 600
dwa diagnose --mode decode --fps 2000 --duration 600
```

The first mode reads camera buffers only; the second also validates decoded
frames and counter continuity. Neither records videos or evaluates droplets.
Both stop on the first error, keep bounded event history, and write a unique
JSON report after cleanup. See [DIAGNOSTICS.md](DIAGNOSTICS.md) for the test
sequence, report fields, timeout comparisons and limitations.

Camera buffers are reused while open and released on successful close. Pending
evaluation tasks/results are cleared on cancellation. An arbitrary hung
evaluator or native call cannot be force-killed by Python threads: shutdown
reports a lifecycle error, and the active call retains its own data until it
returns. A blocked disk writer is not allowed to publish into a closed instance.
Do not treat an errored close as successful cleanup.

## Parallel evaluation

Install `dropwatch-apollo[evaluation]`, then pass the existing function:

```python
import numpy as np
import pandas as pd


def evaluate_sequences(sequences: list[np.ndarray]) -> pd.DataFrame:
    # Placeholder: replace with the existing volume/speed implementation.
    return pd.DataFrame(
        [{"shot": shot, "volume_nl": np.nan, "speed_m_s": np.nan} for shot, sequence in enumerate(sequences)]
    )


with DropwatchApollo(settings, evaluator=evaluate_sequences) as dw:
    dw.start(max_sequences=5)
    sequences = dw.get_sequences(timeout_s=30)
    evaluations = dw.get_evaluations(timeout_s=30)
```

The FIFO worker calls `evaluator([sequence])` once per shot. Results are
concatenated with global `shot=0,1,2,...`, replacing singleton `shot=0`.
Evaluation exceptions do not invalidate completed images. Retrieve evaluations
before restarting. With spooling, evaluation starts from the finalized NPY,
not a buffer that the writer will recycle. Inputs are read-only; copy only if
your evaluator must modify them.

Evaluation is opt-in: a Python-heavy callback can contend for the GIL, CPU,
memory bandwidth, and file cache. Leave it disabled until the production
function has been tested on the acquisition PC.

If tracking joins droplets across windows, supply `evaluation_finalizer=...`.
It receives the concatenated raw-observation DataFrame with global shot IDs
once, after per-sequence evaluation. Do not independently postprocess each
window before joining tracks. The optional A1 bridge in `MIGRATION.md` implements
this separation using the existing external `fast_seq_eval` functions.

## Export, preview, and replay

`save_video(sequences, "all.avi", options=...)` combines shots;
`save_videos(sequences, directory, options=...)` writes one file per shot.
`ApolloVideoSettings` supports playback FPS, four-character codec override,
shot/frame/trigger-relative-time annotations, inversion, trimming,
`crop_bottom`, and separator frames. AVI defaults to MJPG; MP4 to mp4v.
`save_avi(sequence, path)` remains the unannotated compatibility helper.

Encoding is rejected while acquisition is active; starting acquisition during
an export is also rejected. Temporary videos are checked for dimensions, frame
count, and decodable first/last frames before replacing the target. Odd output
dimensions are rejected to prevent silent codec cropping. Video is potentially
lossy: use NPY or PNG for pixel-exact data.

- `snapshot("plate.png")`: idle RLE snapshot plus optional PNG.
- `preview(duration_s=10)`: idle live view; Escape closes it; requires GUI OpenCV.
- `save_raw(sequences, directory)`: unrotated lossless NPY per shot.
- `save_frames(sequences, directory)`: physical-orientation PNG per frame.

```python
from dropwatch_apollo import ReplayFrameSource

source = ReplayFrameSource(["recording.bin"], batch_frames=100)
with DropwatchApollo(settings, frame_source=source) as dw:
    dw.start(max_sequences=40)
    sequences = dw.get_sequences()
```

Replay accepts BIN or NPY files and uses the same trigger state machine.
BIN uses a strict, offline Python reference decoder; counter continuity is
checked within each file (separate files may be separate acquisitions).
NPY preserves the stored shape/dtype. EOF is explicit; an unfinished triggered
window is an error. Already-triggered NPY shots can instead be exported directly
without re-triggering. Replay is fast by default; `frame_period_ms` optionally
paces it. It is a diagnostic tool, not the live high-speed backend.

CLI (also available as `python -m dropwatch_apollo`):

```shell
dwa snapshot --output inspection
dwa preview --duration 10
dwa record --frames 1000 --pre-trigger 20 --shots 40 --duration 60 --auto-trigger
dwa replay recording.bin --frames 200 --pre-trigger 20 --shots 40
```

The CLI records to unique raw acquisition directories and generates videos
after capture. See `scripts/s_api_apollo_recording.py` for evaluation integration.

## Development and qualification

```shell
pip install -e . --group dev
python -m pytest --cov=dropwatch_apollo --cov-fail-under=80
python -m ruff check .
python -m ruff format --check .
python -m mypy dropwatch_apollo recorder
python -m build --wheel
```

Modules separate lifecycle, capture buffers, hardware/stream validation,
evaluation, disk storage, export, and replay. No camera is needed for software
tests. The real-DLL tests require Windows and are skipped elsewhere.

Passing tests is not hardware qualification. Complete
`HARDWARE_ACCEPTANCE.md` on the actual camera, firmware, USB controller, disk,
and acquisition PC before relying on a run. No implementation can guarantee
that hardware faults or an undersized recording limit never lose a droplet.
