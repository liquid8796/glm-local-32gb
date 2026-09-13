"""One exclusively owned reader task and at most two charged encoded bands.

The consumer borrows each packet only until release(). It must clear its own
payload references before release/advancing. No reader calls may occur on the
consumer while the stream is active. Teardown joins the producer before reader
ownership is returned; an OS read may still need to finish before that join.
"""
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import BoundedSemaphore, Event, Lock, RLock, get_ident
import time
import traceback

from .safetensor_reader import MAX_READ_SPAN_BYTES

READ_AHEAD_BUFFER_BYTES = 2 * MAX_READ_SPAN_BYTES
_END = object()


def clear_error_frames(error):
    """Drop inactive I/O/kernel frame locals without changing the primary error."""
    pending, seen = [error], set()
    while pending and len(seen) < 16:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)


class _Failure:
    def __init__(self, error):
        from .residency import BudgetExceededError, MAX_BYTES
        self.kind = type(error)
        self.message = error.args[0][:2000] if error.args and type(error.args[0]) is str else "Producer read failed"
        self.budget = None
        if (isinstance(error, BudgetExceededError) and error.device in ("cpu", "cuda")
                and type(error.required_bytes) is int and 0 <= error.required_bytes <= 2 * MAX_BYTES
                and type(error.budget_bytes) is int and 0 <= error.budget_bytes <= MAX_BYTES):
            self.budget = error.device, error.required_bytes, error.budget_bytes
        clear_error_frames(error)

    def exception(self):
        if self.budget is not None:
            from .residency import BudgetExceededError
            return BudgetExceededError(*self.budget)
        try:
            return self.kind(self.message)
        except Exception:
            return RuntimeError(f"{self.kind.__name__}: {self.message}")


class _Packet:
    __slots__ = ("item", "payload", "_lease", "_stream", "_released")

    def __init__(self, item, payload, lease, stream):
        self.item, self.payload, self._lease, self._stream = item, payload, lease, stream
        self._released = False

    def release(self):
        if self._released:
            return
        self._released = True
        self.payload = None
        if self._lease is not None:
            self._lease.release()
            self._lease = None
        self._stream._release_slot()


class ReadAheadPool:
    """Lazy single producer per RuntimeWeights; close before closing its reader."""
    def __init__(self):
        self._lock = RLock()
        self._executor = self._active = None
        self._closed = False
        self._shutdown_complete = Event()

    def bands(self, items, load, size, *, ledger):
        if not isinstance(items, (list, tuple, range)) or not 1 <= len(items) <= 8192:
            raise ValueError("Read-ahead requires 1..8192 sized band positions")
        if not callable(load) or not callable(size) or not callable(getattr(ledger, "reserve", None)):
            raise ValueError("Read-ahead requires load/size callbacks and an allocation ledger")
        return _BandStream(self, items, load, size, ledger)

    def _start(self, stream):
        with self._lock:
            if self._closed:
                raise RuntimeError("Read-ahead pool is closed")
            if self._active is not None:
                raise RuntimeError("Read-ahead pool already owns an active reader projection")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glm-bounded-reader")
            self._active = stream
            try:
                return self._executor.submit(stream._produce)
            except BaseException:
                self._active = None
                raise

    def _finish(self, stream):
        with self._lock:
            if self._active is stream:
                self._active = None
            stream._closed_event.set()

    def close(self):
        with self._lock:
            active = self._active
            if active is not None and get_ident() in (active._owner, active._producer_thread):
                raise RuntimeError("Close the active band stream before its read-ahead pool")
            initiate = not self._closed
            if initiate:
                self._closed = True
                if active is not None:
                    active._stop.set()
            executor = self._executor
        if not initiate:
            self._shutdown_complete.wait()
            return
        try:
            if active is not None:
                active._closed_event.wait()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            with self._lock:
                self._executor = None
        finally:
            self._shutdown_complete.set()

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Read-ahead pool is closed")
        return self

    def __exit__(self, *_):
        self.close()


class _BandStream:
    def __init__(self, pool, items, load, size, ledger):
        self._pool, self._items, self._load, self._size, self._ledger = pool, items, load, size, ledger
        self._queue = Queue()  # Payload admission is bounded before allocation by the semaphore.
        self._slots, self._stop, self._closed_event = BoundedSemaphore(2), Event(), Event()
        self._counts_lock = Lock()
        self._future = self._current = self._failure = None
        self._owner = self._producer_thread = None
        self._closed = False
        self._counts = dict(produced_bands=0, consumed_bands=0, live_bands=0, peak_live_bands=0,
                            producer_read_seconds=0.0, producer_wait_seconds=0.0, consumer_wait_seconds=0.0)

    def _release_slot(self):
        with self._counts_lock:
            self._counts["live_bands"] -= 1
        self._slots.release()

    def _produce(self):
        self._producer_thread = get_ident()
        try:
            for item in self._items:
                began = time.perf_counter()
                while not self._slots.acquire(timeout=0.05):
                    if self._stop.is_set():
                        return
                with self._counts_lock:
                    self._counts["producer_wait_seconds"] += time.perf_counter() - began
                lease = payload = packet = None
                counted, transferred = False, False
                try:
                    if self._stop.is_set():
                        return
                    size = self._size(item)
                    if type(size) is not int or not 1 <= size <= MAX_READ_SPAN_BYTES:
                        raise ValueError("Read-ahead band payload must be 1 byte..8 MiB")
                    lease = self._ledger.reserve(size, label="read_ahead_encoded_band")
                    with self._counts_lock:
                        self._counts["live_bands"] += 1
                        self._counts["peak_live_bands"] = max(self._counts["peak_live_bands"], self._counts["live_bands"])
                    counted = True
                    began = time.perf_counter()
                    payload = self._load(item)
                    with self._counts_lock:
                        self._counts["producer_read_seconds"] += time.perf_counter() - began
                    if self._stop.is_set():
                        return
                    packet = _Packet(item, payload, lease, self)
                    # The packet must be the sole producer-side payload owner
                    # before publication: a fast consumer may release at once.
                    payload = lease = None
                    self._queue.put(packet)
                    transferred = True
                    with self._counts_lock:
                        self._counts["produced_bands"] += 1
                except BaseException as error:
                    self._failure = _Failure(error)
                    return
                finally:
                    payload = None
                    if not transferred:
                        if packet is not None:
                            packet.release()
                        else:
                            if lease is not None:
                                lease.release()
                            if counted:
                                self._release_slot()
                            else:
                                self._slots.release()
                    lease = packet = None
        except BaseException as error:
            self._failure = _Failure(error)
        finally:
            self._queue.put(_END)

    def __enter__(self):
        if self._future is not None or self._closed:
            raise RuntimeError("Read-ahead band stream cannot be reused")
        self._owner = get_ident()
        self._future = self._pool._start(self)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self._future is None or self._closed:
            raise RuntimeError("Read-ahead band stream is not active")
        if self._current is not None and not self._current._released:
            raise RuntimeError("Release the current read-ahead packet before advancing")
        self._current = None
        began = time.perf_counter()
        while True:
            if self._stop.is_set():
                raise RuntimeError("Read-ahead projection was stopped")
            try:
                packet = self._queue.get(timeout=0.05)
                break
            except Empty:
                if self._future.done():
                    self._future.result()
                    raise RuntimeError("Read-ahead producer ended without a terminal marker")
        with self._counts_lock:
            self._counts["consumer_wait_seconds"] += time.perf_counter() - began
        if packet is _END:
            if self._failure is not None:
                raise self._failure.exception()
            raise StopIteration
        self._current = packet
        with self._counts_lock:
            self._counts["consumed_bands"] += 1
        return packet

    def stats(self):
        with self._counts_lock:
            return {"enabled": True, "maximum_producer_threads": 1, **self._counts}

    def __exit__(self, exc_type, error, _):
        self._stop.set()
        if error is not None:
            clear_error_frames(error)
        if self._current is not None:
            self._current.release()
            self._current = None
        unexpected = None
        try:
            try:
                self._future.result()
            except BaseException as failure:
                unexpected = _Failure(failure)
        finally:
            while True:
                try:
                    packet = self._queue.get_nowait()
                except Empty:
                    break
                if packet is not _END:
                    packet.release()
                packet = None
            self._closed = True
            self._future = None
            self._load = self._size = self._items = None
            self._pool._finish(self)
        if error is None and (unexpected is not None or self._failure is not None):
            raise (unexpected or self._failure).exception()
        return False
