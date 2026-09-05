# Existing scripts → dropwatch-apollo 0.3

`dropwatch-apollo` is the distribution/product name; `dropwatch_apollo` is the
native Python namespace. Apollo alone refers to the dispenser. The native
`DropwatchApollo`, `ApolloSettings`, and exception class names remain available.
The old `dropwatch` import namespace is no longer occupied by this package.

## What remains unchanged

The reviewed A1 constructor/import contract works through configuration-only
facades. No original recorder acquisition, calibration, or tracking engine is
included:

```python
from pathlib import Path
from recorder.api.dropwatch import Dropwatch
from recorder.core.capture import CaptureState
from recorder.core.detectors import DisplayRoi2D, RoiPixelDetector
from recorder.core.recorder import FastPostTriggerRecorder
from recorder.core.sinks import LegacyVideoSaver
from recorder.settings import CameraSettings

run_dir = Path("new_run")  # use a new output path for every experiment
camera = CameraSettings()  # 1 ms period, 100-frame hardware flush
detector = RoiPixelDetector(
    roi=DisplayRoi2D(y0=800, y1=950, x0=200, x1=512),
    detection_threshold_px=15,
    view="left",
)
capture = CaptureState(detector=detector, capture_len=30, copy_frames=True)
recorder = FastPostTriggerRecorder(
    capture_state=capture,
    output_dir=run_dir,
    sinks=[
        LegacyVideoSaver(
            output_path=run_dir / "recording.avi",
            frame_period=camera.frame_period,
            invert_bw=True,
        )
    ],
)
dropwatch = Dropwatch(recorder, decoder_max_images=1000)

with dropwatch.recording(max_duration=300, ready_timeout=60, join_timeout=300):
    # Existing dispense workflow goes here, only after camera readiness.
    ...
```

The facade supports `start`, `start_background`, `wait_until_ready`, nonblocking
`stop`, `join`, `recording`, `is_recording`, and the real `_thread`. `setup_camera`
configures the next recording without opening hardware early. `close()` can
retry a failed vendor close. Native snapshot, auto-trigger placement, preview,
MP4/AVI/PNG export and replay remain on `DropwatchApollo`.

## One-time central consumer update

Your experiment scripts need not change, but the **A1 library and environment**
need these updates. Their repositories are not in this checkout; the source on
Drive is a reference snapshot, and has not been modified.

1. In A1's dependency declarations, replace the `dropwatch-recorder` requirement
   with `dropwatch-apollo==0.3.0`. In development requirements, replace
   `-e ../dropwatch-recorder` with the path to this checkout. Refresh its lockfile.
2. Keep the existing pinned `fasteye-sdk` dependency while A1 still imports
   `FastEyeReadError`. This library itself does not require that SDK.
3. Uninstall `dropwatch-recorder`, then install/reinstall `dropwatch-apollo`.
   Installing both into one environment can mix two implementations under the
   same `recorder` namespace; the facade explicitly rejects this situation.
4. Apply [the import patch](integration/a1-consumer-imports.patch) in the A1
   repository, first checking it:

   ```shell
   git apply --check /path/to/dropwatch-main/integration/a1-consumer-imports.patch
   git apply /path/to/dropwatch-main/integration/a1-consumer-imports.patch
   ```

The patch changes only two imports in `run_dispense.py`. It avoids eager
imports of obsolete camera/calibration code in `dropwatch_helper.py` and uses
the left-only AVI reader. It leaves public A1 functions, settings, physical
actuation, rig leasing, and re-dispense policy untouched. The bridge calls the
existing `a1_experiment_lib.fast_seq_eval` module; no copy of that code is shipped.

## Intentional compatibility boundaries

- Multiple windows are recorded until context exit, the duration limit, or
  `max_num_triggers`. With neither bound supplied, the facade uses 300 seconds.
  It does **not** use the native API's default one-shot limit.
- The legacy facade uses level triggering: after a window, an occupied ROI can
  immediately start another. Startup still requires a clear ROI and a full
  lookback buffer before entering the dispense body. Native mode keeps its
  hysteretic edge-trigger default.
- `CaptureState.capture_len` includes the trigger frame.
  `BufferedCaptureState.lookback_len` is additional; native `max_number_frames`
  instead counts the total including lookback. Both support rectangular ROIs.
- `decoder_max_images` is accepted as a capacity ceiling, **not** a request to
  flush 1000 images from the camera. Default flush is 100; the verified fixed
  transport buffer permits at most 122 with the current encoded-image limit.
- The facade uses eight 100-frame chunks by default (short windows use shorter
  chunks). Explicit RAM at full geometry is approximately 548 MiB including
  camera buffers, plus lookback; not tens of GiB for a long window. Its 64 GiB
  raw-data quota can be configured with `max_spool_bytes`. A fast SSD remains
  necessary; insufficient throughput stops with an error, never silent eviction.
- `trigger_count` counts started windows, including an incomplete final one.
  Only complete windows are exported. Counters reset for each new session.
- Graceful stop drains queued frames and completes an active window; it is not
  an instantaneous cutoff. If cleanup/export blocks, `join` raises `TimeoutError`
  while `_thread` stays alive. The caller must continue to hold the rig lease.
- The same Dropwatch object is reusable after successful cleanup. Choose a fresh
  video path for every session; an existing destination is rejected before arming.
  Raw windows always live in unique directories and survive `close()`.
- Supported facade sinks are `LegacyVideoSaver`. Alternate views, arbitrary
  detectors/sinks, white-stripe filtering and unrotated layouts fail explicitly.
  There are no silent no-op settings and no calibration or dropeval dependency.

## AVI and crop coordinates

`LegacyVideoSaver` keeps the requested path, physical transpose, inversion,
XVID/16 fps defaults, label text/position and black separator after **every**
window (including the last). Only the left view is written. An extra bottom row
is padding when needed to prevent codec truncation of the original odd-height
label layout. Measurement pixels are never cropped to satisfy the codec.

The supplied reader streams frames, removes the label/padding, and returns
singleton-view tuples that the crop helper understands. It supports this
left-only annotated format, not old split-view recordings or arbitrary videos.
`crop_sequences` retains only cropped pixels, but its returned list and the
external evaluator still use application memory.

**Check the crop once against a real snapshot:** rows/columns are physical left
image coordinates with the label removed. The old `dropeval.extract_video`
implementation was not included in the Drive snapshot; exact equivalence to
its header handling and compression-dependent measurement results is not yet
verified. Do not mix that old reader with the new single-view AVI.

## Optional parallel evaluation

The default A1 path continues to evaluate after recording. To evaluate during
acquisition, the following callbacks use raw NPY data without an AVI round-trip:

```python
from dropwatch_apollo.integration import make_evaluation_callbacks

evaluate, finalize = make_evaluation_callbacks(
    rows=slice(200, 1100),
    cols=slice(200, 400),
    frame_period_ms=1.0,
)
dropwatch = Dropwatch(
    recorder,
    evaluator=evaluate,
    evaluation_finalizer=finalize,
)
# After recording exits successfully:
# drops_df = dropwatch.evaluations
```

Per-window raw observations run on one FIFO worker during acquisition. Global
shot IDs are assigned in order; `connect_shots` and `postproc_full_data` run
once at the end, preserving cross-window tracks. Speeds from the reviewed
evaluator (mm/frame) are converted to m/s using `frame_period_ms`. At 1 ms the
numbers are unchanged; at 0.5 ms they double. The sequential helper defaults
to 1 ms, matching A1's reviewed fixed `CameraSettings()`; pass the actual period
if central camera configuration changes.
The reviewed A1 `capture_len` calculation also contains a fixed 1000 fps factor;
update that centrally when changing the actual camera period. Changing only
`setup_camera()` would not preserve the requested recording duration.

Consuming `dropwatch.evaluations` instead of evaluating the AVI requires a small
additional change inside A1's result-building code. The provided import patch
does not make that optional behavioral change. The numerical production
evaluation still needs verification with the actual installed A1 package/data.

## Safety and remaining qualification

No physical re-dispense retry was added. New transport errors retain their
types; they are deliberately not disguised as the old `FastEyeReadError` that
causes A1 to repeat a physical experiment. Review that policy separately before
enabling any automatic re-dispensing after actuation.

Offline tests cover the exact constructor/context contract, 100 reusable
sessions, buffer release, long-window chunking, readiness failures, real thread
ownership on timeout, interrupted streams, quotas, evaluation finalization,
and AVI readback. The vendor DLL tests require Windows. Real triggered video
runs and simultaneous disk/evaluation throughput still need hardware testing;
the previously observed 2000 fps transport stall is not claimed fixed by this
interface work.
