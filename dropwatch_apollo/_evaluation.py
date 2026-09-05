"""Bounded single-worker evaluation for completed Dropwatch Apollo sequences."""

from __future__ import annotations

import importlib
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from dropwatch_apollo._capture import _CapturedSequence
from dropwatch_apollo.models import ApolloEvaluationError
from dropwatch_apollo.models import ApolloLifecycleError
from dropwatch_apollo.models import ApolloSequenceEvaluator
from dropwatch_apollo.models import require_finite


class _EvaluationRunner:
    """Evaluate completed buffers in FIFO order without blocking acquisition."""

    _POLL_INTERVAL_S = 0.05
    _STOP_TIMEOUT_S = 5.0

    def __init__(
        self, evaluator: ApolloSequenceEvaluator | None, finalizer: Callable[[Any], Any] | None = None
    ) -> None:
        self._evaluator = evaluator
        self._finalizer = finalizer
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._cancel = threading.Event()
        self._done.set()
        self._acquisition_done: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._error: Exception | None = None
        self._session_active = False
        self._submitted = 0
        self._tasks: queue.Queue[_CapturedSequence] = queue.Queue(maxsize=1)
        self._results: queue.Queue[Any] = queue.Queue(maxsize=1)

    @property
    def enabled(self) -> bool:
        return self._evaluator is not None

    @property
    def done(self) -> threading.Event:
        return self._done

    @property
    def is_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def prepare(self, max_sequences: int) -> None:
        if not self.enabled:
            return
        _load_pandas()

        if self.is_alive:
            raise ApolloLifecycleError("previous sequence evaluations are still running")
        with self._lock:
            session_active = self._session_active
            submitted = self._submitted
            error = self._error
        if session_active and submitted:
            if error is not None:
                raise ApolloEvaluationError("previous sequence evaluation failed") from error
            raise ApolloLifecycleError("call get_evaluations() before starting another acquisition")

        self.reset()
        with self._lock:
            self._tasks = queue.Queue(maxsize=max_sequences)
            self._results = queue.Queue(maxsize=max_sequences)
            self._done.clear()
            self._cancel.clear()
            self._session_active = True

    def start(self, acquisition_done: threading.Event) -> None:
        if not self.enabled:
            return
        self._acquisition_done = acquisition_done
        worker = threading.Thread(
            target=self._run,
            name="dropwatch-apollo-evaluation",
            # User evaluators are arbitrary code and cannot be force-killed in
            # Python. A daemon prevents one broken evaluator from keeping the
            # whole acquisition process alive forever.
            daemon=True,
        )
        worker.start()
        self._worker = worker

    def submit(self, sequence: _CapturedSequence) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._cancel.is_set() or self._error is not None:
                return
            self._tasks.put_nowait(sequence)
            self._submitted += 1

    def cancel(self) -> None:
        with self._lock:
            self._cancel.set()
            self._clear_tasks()
            self._take_results()

    def wait(self, acquisition_done: threading.Event, timeout_s: float | None) -> None:
        if not self.enabled:
            raise ApolloLifecycleError("no sequence evaluator is configured")
        if timeout_s is not None:
            require_finite("timeout_s", timeout_s)
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        with self._lock:
            if not self._session_active:
                raise ApolloLifecycleError("no evaluation session is available")

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        for event in (acquisition_done, self._done):
            if event.is_set():
                continue
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"evaluations were not available after {timeout_s}s")
            if not event.wait(remaining):
                raise TimeoutError(f"evaluations were not available after {timeout_s}s")

    def collect(self) -> Any:
        with self._lock:
            error = self._error
        if error is not None:
            self.reset()
            raise ApolloEvaluationError("sequence evaluation failed") from error

        pandas = _load_pandas()
        result_frames = self._take_results()
        try:
            indexed_frames = []
            for shot, result_frame in enumerate(result_frames):
                indexed_frame = result_frame.copy()
                indexed_frame["shot"] = shot
                indexed_frames.append(indexed_frame)
            result = pandas.concat(indexed_frames, ignore_index=True) if indexed_frames else pandas.DataFrame()
            if self._finalizer is not None:
                result = self._finalizer(result)
                if not isinstance(result, pandas.DataFrame):
                    raise TypeError("evaluation finalizer must return pandas.DataFrame")
        except Exception as exc:
            self.reset()
            raise ApolloEvaluationError("could not combine sequence evaluation results") from exc
        self.reset()
        return result

    def finish(self) -> Exception | None:
        worker = self._worker
        if worker is not None:
            self.cancel()
            worker.join(timeout=self._STOP_TIMEOUT_S)
            if worker.is_alive():
                return ApolloLifecycleError(
                    f"sequence evaluator did not stop within {self._STOP_TIMEOUT_S}s; "
                    "the camera was closed, but this Dropwatch Apollo instance cannot be reused"
                )
        with self._lock:
            error = self._error
        self.reset()
        return error

    def reset(self) -> None:
        with self._lock:
            self._acquisition_done = None
            self._worker = None
            self._error = None
            self._session_active = False
            self._submitted = 0
            self._tasks = queue.Queue(maxsize=1)
            self._results = queue.Queue(maxsize=1)
            self._cancel.clear()
            self._done.set()

    def _run(self) -> None:
        evaluator = self._evaluator
        acquisition_done = self._acquisition_done
        if evaluator is None or acquisition_done is None:
            self._done.set()
            return

        try:
            pandas = _load_pandas()
            while not self._cancel.is_set():
                try:
                    sequence = self._tasks.get(timeout=self._POLL_INTERVAL_S)
                except queue.Empty:
                    if acquisition_done.is_set():
                        return
                    continue
                if self._cancel.is_set():
                    return

                result = evaluator([sequence.materialize()])
                if not isinstance(result, pandas.DataFrame):
                    raise TypeError(f"sequence evaluator must return pandas.DataFrame, got {type(result).__name__}")
                with self._lock:
                    if not self._cancel.is_set():
                        self._results.put_nowait(result)
                del sequence, result
        except Exception as exc:  # noqa: BLE001 - evaluator failures are returned to the caller
            with self._lock:
                self._error = exc.with_traceback(None)
                self._clear_tasks()
        finally:
            self._done.set()

    def _clear_tasks(self) -> None:
        while True:
            try:
                self._tasks.get_nowait()
            except queue.Empty:
                return

    def _take_results(self) -> list[Any]:
        results: list[Any] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results


def _load_pandas() -> Any:
    try:
        return importlib.import_module("pandas")
    except ImportError as exc:
        raise ApolloEvaluationError(
            "parallel evaluation requires pandas; install dropwatch-apollo[evaluation]"
        ) from exc
