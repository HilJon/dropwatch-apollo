# Dropwatch Apollo hardware acceptance

The Apollo implementation is unit-tested without a camera. The following items
must be verified on the Windows acquisition PC before Apollo is considered
hardware-qualified.

## Image and trigger geometry

- Confirm that the Apollo image is the first 1120-pixel half of the decoded
  `512 x 2240` FastEye RLE frame.
- Put a foreground target in only the second half and confirm it neither
  triggers Apollo nor appears in a returned sequence or video.
- Confirm raw NumPy orientation and zero/nonzero polarity. Videos and PNGs
  transpose once into physical orientation.
- Verify 960 x 1024 transport geometry and reqBufSize=1228800, distinct from
  the padded 512 x 2240 decoded layout.
- Capture a stationary drop and verify `trigger_position_px`,
  `trigger_width_px`, `trigger_on_pixels`, and `trigger_off_pixels`.
- Check that the trigger image is returned at index `pre_trigger`.

Apollo defaults to `trigger_from_top=False` and
`trigger_position_px=300`. This reproduces the previous bottom-oriented Apollo
trigger. It is an implementation starting point, not a calibrated hardware
value.

Call `set_trigger_size(width=100)` while Apollo is idle to replace that starting
value automatically. It takes one RLE snapshot, finds the highest foreground
connected to the physical lower image edge, and places a 100-pixel
trigger band directly above it. On the target setup, verify that:

- the detected object is the microtiter plate rather than an edge artifact;
- the returned `trigger_position_px` agrees with a visual RLE snapshot;
- an absent plate produces a clear error rather than an incorrect trigger
  position.

## Throughput

First run the isolated transport and decode comparisons in
[DIAGNOSTICS.md](DIAGNOSTICS.md), retaining their JSON reports. They provide a
baseline without trigger/storage/evaluation load; they do not replace the full
recording tests below.

Run acquisitions at both 1000 fps and 2000 fps with
`rle_batch_frames=100`.

- Run the included real-DLL golden-fixture tests on Windows (no camera needed).
  Also retain a full encoded batch from this camera. Verify that any unconsumed
  input is only zero padding, and the decoder produces
  exactly `frames * 512 * 2240` bytes, and that the parsed header count matches
  the decoded frame count.
- Verify that RLE decoding is faster than incoming acquisition.
- Verify that no camera encoder error is reported.
- Verify that no source batch contains more than `rle_batch_frames`.
- Deliberately select an RLE batch larger than the encoded buffer permits and
  confirm that `start()` fails before changing camera acquisition settings.
- Force or simulate one frame-counter gap and confirm that the acquisition
  fails with `ApolloFrameLossError` rather than returning a partial sequence.
- Inject a single zero-byte `daq_read` followed by a complete contiguous RLE
  buffer. Confirm recovery without a runtime flush/retrigger, with
  `zero_byte_reads == recovered_zero_byte_reads == 1`.
- Inject two consecutive zero-byte reads and one positive partial read. Confirm
  both sessions fail with `ApolloTransportError`, no uncertain sequence is
  returned, and the next `start()` opens a fresh camera handle.
- After a recovered zero-byte read, inject a counter jump. Confirm recovery is
  rejected rather than hiding the missing interval.
- Run continuously beyond one complete 15-bit counter wrap (at least 16.384 s
  at 2000 fps) and confirm the wrap is accepted with no gap.
- Log vendor-read latency, `last_vendor_error`, encoder status, counters, and
  retry statistics during a long soak. A high zero-byte rate is a hardware/USB
  fault to investigate, not a reason to increase retries without a latency and
  overflow study.
- Test `max_number_frames` at 20, a typical production value, and 2000.
- Confirm that `start()` returns only after the live trigger area has been
  clear and all pre-trigger slots have been filled.
- Run two consecutive one-shot acquisitions and verify that the second
  `start()` begins from a flushed camera buffer.
- Call `stop()` immediately after dispensing, including when the droplet is
  behind a clear unread batch or inside a not-yet-flushed RLE batch. Verify the
  flush grace and caught-up poll retain it. Finish any active window.
  Remove incoming frames mid-window and confirm `ApolloIncompleteSequenceError`.
  `abort()` intentionally discards unread/in-progress frames but still waits
  for any outstanding native read and cleanup.
- Run `start(max_sequences=5)` without consuming intermediate results and
  verify that five separated droplets are returned in order without camera
  backpressure. Confirm that a trigger-area clear interval is required between
  droplets.
- Place the next drop immediately after a completed window, including long
  lookbacks. Verify there is no lookback-refill gap once the ROI has cleared.
- Exercise `max_duration_s`; verify it requests graceful draining, not an
  immediate truncation of an active shot.
- Run 40 x 1000 frames with a 20-frame pre-trigger and three spool buffers on
  the target SSD. Check all 40 NPY files, trigger positions, and videos.
  Throttle or disconnect the disk: expect an explicit failure, no overwritten
  buffers, and recovery of earlier complete shots. Record disk throughput.
- Run the production evaluator with `start(max_sequences=5)`. Artificially
  delay the first `evaluator([sequence])` call and confirm that all five
  sequences are still acquired without frame-counter gaps. Verify that the
  combined DataFrame follows trigger order and that evaluator failure does not
  invalidate a successfully captured sequence.
- Save representative combined and per-shot AVI/MP4 files. Verify frame count,
  physical orientation, left-only content, annotation timestamps, playback
  speed, trimming, crop, polarity, and separators in the target player.
- Interrupt an export and confirm the prior target remains intact and only a
  temporary `.part` file is affected. Confirm export is rejected while camera
  acquisition is active.

## Lifecycle and memory

Run at least 100 `start()` / `get_sequence()` / `stop()` cycles on the same
`DropwatchApollo` instance.

- Confirm that one acquisition worker exists only while recording.
- Monitor process RSS. Each block contains final frames plus a separate waiting
  lookback area; memory mode reserves one block per shot, spooling recycles a
  bounded pool. Lookback references pin blocks until safe. The decoder is reused
  while open. Explicit buffers must plateau; OS file cache and consumer-accessed
  memory-mapped pages are additional reclaimable memory, not covered by the budget.
- Confirm that sequence, decode, encoded, result, counter, and trigger state do
  not carry stale data into the next acquisition.
- Verify that a configuration exceeding `max_buffer_bytes` is rejected before
  the camera starts. Record both the configured limit and measured peak RSS.
- Use a long pre-trigger near the production maximum and confirm that completion
  of a sequence does not pause camera intake while its lookback prefix is put
  into chronological order.
- Confirm that the camera can start again after a timeout, read error, and
  explicit stop.
- Complete one sequence in a multi-trigger run, then inject a source failure.
  Confirm that the completed-but-unread sequence is available through
  `error.completed_sequences`, without a partial failed window, and that the
  original error remains visible until the next `start()` or `close()`.
- Delay reading the returned sequence and confirm that camera intake has
  already stopped and is never blocked by an output queue.
- Confirm that `close()` releases the camera handle and that a new process can
  open the camera immediately.
- Deliberately hang an evaluator. Confirm camera close returns with a lifecycle
  error after the bounded wait, pending tasks/results release their references,
  and the Python process can still exit. The running callback's own input cannot
  be force-freed. Native code holding the GIL may defeat Python-level timeouts.
- Simulate one failed vendor close; verify another close retries the same handle.

## Release decision

For the 0.2.3 polling change, run the reduced/legacy comparison in `DIAGNOSTICS.md`
at 1,000 and 2,000 fps. Confirm stable transfers, bounded backlog and continuous
decoded counters, then repeat the same-instance start/stop and production
recording tests. A passing synthetic timing model is not a hardware pass.

Record the tested camera firmware, PLabDAQ DLL version, frame rates, final
trigger settings, `rle_batch_frames`, memory limit, peak RSS, and cycle count.
Keep the qualified DLL/configuration bundle together with its recorded version.
Hardware qualification is complete only after every check above has passed on
the target camera and PC.

The bundled RLE backend still transfers both views and decodes `512 x 2240`; the left crop
only reduces downstream RAM and CPU. Do not switch to one of the bundled ROI
profiles merely based on its filename. A reduced FPGA transfer is a separate
backend change requiring vendor format documentation and this full acceptance
suite on real hardware.
