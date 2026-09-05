# Investigating zero-byte camera reads

`dwa diagnose` isolates the camera transport from triggering, recording, video
export and evaluation. It never retries a read or restarts acquisition after a
fault. It changes the usual acquisition settings, not the camera firmware.

## Run on the Windows acquisition PC

Install version **0.2.3 or later** for reduced polling and query timing. Version
0.2.2 fixed the startup-order bug in 0.2.1 but still read four status registers
before every buffer. From the updated source
folder: `python -m pip install --upgrade .`. Verify the installed version with
`python -c "import dropwatch; print(dropwatch.__version__)"`.
Close other programs holding the camera, including another Dropwatch process.
Use the normal illumination, threshold and exposure. Keep the plate in view.

Start with transport only, separately at each production frame rate:

```powershell
dwa diagnose --mode transport --fps 1000 --duration 600 --label "stationary plate; normal cable/port"
dwa diagnose --mode transport --fps 2000 --duration 600 --label "stationary plate; normal cable/port"
```

Then add the same RLE decoder and integrity checks used by the recorder:

```powershell
dwa diagnose --mode decode --fps 1000 --duration 600
dwa diagnose --mode decode --fps 2000 --duration 600
```

Each command runs for up to ten minutes of acquisition, stopping at the first
fault. Repeat with representative droplets. These commands do **not** operate
the dispenser or save droplet videos. Rare faults need longer tests, e.g.
`--duration 3600`. Repeat separate commands to exercise camera open/start/close.

The acquisition duration excludes setup and cleanup; an in-progress read can
extend it. `Ctrl+C` requests interruption and report writing after the current
native call returns. A vendor DLL that hangs or ignores its timeout cannot be
force-stopped safely by this tool. A forced process kill can leave an empty
report file; that is not a passed test. There are no background worker threads.

## Reports

Each run creates a unique JSON file in `apollo_diagnostics` (change with
`--output`). Send the reports from successful runs too, together with the actual
camera firmware, driver version and USB/cable arrangement. `--label` records
these notes. Bundle hashes identify the host DLL/config/FPGA/FX3 files, **not** the
firmware actually loaded on the camera.

The report includes:

- Settings, host/Python/package information and DLL/config/FPGA/FX3 SHA-256 hashes.
- Cumulative read counts, byte counts, zero/short reads and empty status polls.
- The last 500 read/error events, including the failing one; use
  `--history-size` to change this bound (1..10000). No image arrays are retained.
- Requested/returned bytes, read duration and delay since the previous read.
  `read_ms` times `read_image()`, including buffer clearing and vendor error
  retrieval, not just USB bus time. `decode_ms` includes integrity validation.
- Numeric encoder status before every read, including reads using cached readiness.
  Schema 2 reports `readiness_source` and `ready_buffer_credit` (a lower bound,
  **not** an exact backlog). Fresh queries include `minimum_ready_buffers` in
  reduced mode or `num_stored_images` in legacy mode. Cached values are never
  labelled as a fresh full-counter sample.
- Per-query `status_query_ms` and cumulative `summary.status_queries` (count,
  total and maximum duration), including empty polls and failed queries. These
  time Python-to-vendor property calls, not individual USB packets; for example
  legacy `num_stored_images` includes three register reads. `cached_ready_reads`
  counts read attempts that used an earlier confirmed buffer count.
- A best-effort status snapshot after a failure, before cleanup, including the
  full stored count, `APP_MODE` and `ACCELERATOR_CTRL`. The original vendor error
  is captured **before** these additional queries can overwrite it.
- In decode mode: validated frame count and last 15-bit frame counter per batch.
- Original failure, cleanup errors and whether camera close succeeded.

Only an empty output file is reserved before opening the camera. The bounded
history stays in RAM; JSON is written and flushed after cleanup. Existing reports
are never overwritten. The transport buffer is reused; decode mode adds one
reused destination (~109 MiB at 100 frames). Both are released on return. If the
vendor refuses to close its handle, the report explicitly says so; one additional
close attempt is made during cleanup, without restarting or rereading.
Readiness credits hold no image data and are discarded after a fault or stop;
the recorder also clears them before a retry/restart can reuse stale readiness.

Exit codes: **0** = the selected test passed for this run; **1** = hardware,
stream or cleanup failure; **130** = interrupted; **2** = invalid arguments.
Output permission errors fail before opening hardware. Report-write failures
after acquisition are surfaced as errors, not reported as successful tests.

Transport-only success does **not** establish frame continuity. Decode mode
checks continuity within and between received batches, including counter wrap,
but cannot prove images were never lost before the first received frame. This
test discards images and stops at its deadline without draining the final batch;
it is not a substitute for recorder/stop qualification in `HARDWARE_ACCEPTANCE.md`.

## Controlled follow-up tests

### Compare polling overhead

Reduced polling is the default in both diagnostics and normal acquisition.
It reads the low byte of the FPGA stored-buffer counter first. A nonzero byte
provides a conservative lower bound with one query. When it is zero, upper bytes
are checked too, so a count of 256 is not treated as empty. Independently sampled
bytes are never combined into a potentially overestimated count. This requires
one host reader; do not share the camera with another program.

Up to eight confirmed buffers may be drained before refreshing readiness, one
read per loop iteration. **Encoder status is still checked before every read.**
Byte counts, read timeouts and decoder/counter checks are unchanged. No extra
read trigger, reset, retry or firmware change is introduced.

Use `--polling legacy` to compare the full counter query before every read with
the 0.2.2 strategy. Keep all other conditions the same and compare both frame rates:

```powershell
dwa diagnose --mode transport --fps 2000 --duration 60 --polling reduced --label "reduced polling"
dwa diagnose --mode transport --fps 2000 --duration 60 --polling legacy --label "legacy comparison"
```

At 2,000 fps / 100-frame flushes, a new buffer arrives about every 50 ms. The
0.2.2 hardware reports measured about 43 ms for status queries plus 29 ms for a
read. This is a throughput problem; a longer read timeout does not increase the
loop's capacity. The reduced path passed a synthetic bounded-queue timing test,
but must still be qualified on the acquisition PC, including the separate
1,000 fps stall. A transport-only pass does not qualify decoded frames or videos.

### Timeout before the first image read

If `read_attempts` and `bytes_received` are both zero, a status/idle timeout is
**not** a zero-byte read: the diagnostic never called the image-read function.
In version 0.2.1, startup used `enable RLE -> flush -> trigger`. The vendor flush
command disables RLE and clears the FPGA buffers; triggering alone does not
re-enable it. Version 0.2.2 uses `flush -> enable RLE -> trigger` in both the
recorder and diagnostics, preserving startup cleanup without disabling capture.

After upgrading, repeat the two transport commands above at 1,000 and 2,000 fps.
Only proceed to decode/recording qualification once image transfers occur and
the transport runs pass. Increasing read retries or the read timeout does not
address a run with zero read attempts. `numImgAvail = 1` alone is not proof that
a buffer is ready: the trigger also updates this host-side counter.

### Failures during transfers

Change only one factor per comparison. For example, repeat a failing run with
the same frame rate and a different timeout:

```powershell
dwa diagnose --mode transport --fps 2000 --duration 600 --read-timeout-ms 2000 --label "timeout comparison"
```

Defaults match Apollo: 100 frames per RLE flush, 500 ms read timeout at 1–2 kfps,
threshold 127 and exposure 0.05 ms. Options `--threshold`, `--exposure-time-ms`
and `--rle-batch-frames` allow matching the production setup. Unsafe batch sizes
are rejected before opening hardware. `--idle-timeout` defaults to 5 seconds
without a successful transfer; a run receiving no data never passes.

`--extended-status` adds `numImgAvail` before each read and all three status
values after each read. These extra vendor calls change timing, so use this as a
separate comparison, not the initial baseline. `numImgAvail` is host-side trigger/
read bookkeeping and is never used as a readiness predicate. Negative values
are expected in continuous RLE mode: one initial trigger, then many reads.

An error near the read timeout suggests investigating readiness/flush timing or
missing data. A long gap before the failing read suggests checking host delays
and backlog. Neither observation by itself proves an FPGA, USB or driver fault.
After isolated transport/decode runs, compare with normal recording and then
the production evaluator. Full recorder diagnostics remain available via `stats`.
