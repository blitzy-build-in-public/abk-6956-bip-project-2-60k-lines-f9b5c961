"""The bounded capture of a child process's output streams, shared by the two layers that spawn one.

A ``cli`` target and the SVG rasterizer are both third-party programs whose stdout and stderr this
framework reads without letting either grow the process: the adapter drives a target that "may
return garbage, lie about metadata, hang or crash", and
:func:`qrcheck.checks.render.rasterize_svg` drives a tool reading a target's untrusted document.
Both spawn the child with a pipe per stream and drain the other end under a byte bound, and this
module holds the one implementation of that capture.

It lives in neither layer because neither may reach the other: ``qrcheck/checks/**`` may import
``qrcheck.targets.base`` and nothing else of the adapter layer, and ``qrcheck/targets/**`` may not
import ``qrcheck.checks``, the inward-only direction ``tests/unit/test_layering.py`` holds.
No adapter error vocabulary appears here either: the byte bound, the read-chunk size and the drain
timeout are the caller's own numbers, and every failure is the plain ``OSError`` or ``RuntimeError``
the caller converts into its layer's error type.
"""

from __future__ import annotations

import contextlib
import os
import threading

__all__ = ["BoundedStreamCapture", "open_capture_pair"]


class BoundedStreamCapture:
    """Capture at most `limit` bytes of one output stream of a child process.

    One instance owns one ``os.pipe()`` and one daemon thread: ``subprocess.run`` is handed
    :attr:`write_fd` as its ``stdout=`` or ``stderr=``, so the child writes into the pipe and the
    thread reads the other end until the child closes it, keeping at most `limit` bytes. A stream
    that passes the bound sets :attr:`overflowed` and closes its read end, so the child's next write
    fails rather than this process buffering output it will never use. :meth:`finish` releases this
    process's ends once the child has exited and is idempotent, so the success, timeout and
    spawn-failure paths all call it without knowing which one they are on.
    """

    def __init__(
        self, name: str, *, limit: int, chunk_bytes: int, join_timeout_s: float
    ) -> None:
        """Create the pipe for the `name` stream and start draining it.

        Args:
            name: ``"stdout"`` or ``"stderr"``, the stream this object holds, for diagnostics.
            limit: The most bytes kept; the ones past it are counted and dropped, never stored.
            chunk_bytes: How many bytes one read asks for, so a flood costs a bounded number of
                reads rather than one call per byte.
            join_timeout_s: How long :meth:`finish` waits for the drain thread.

        Raises:
            OSError: The pipe itself could not be created.
            RuntimeError: The drain thread could not be started; both descriptors are released
                first, so a caller sees one failure rather than a leaked pipe.
        """
        self._name: str = name
        self._limit: int = limit
        self._chunk_bytes: int = chunk_bytes
        self._join_timeout_s: float = join_timeout_s
        self._buffer: bytearray = bytearray()
        self._received: int = 0
        self._overflowed: bool = False
        self._read_fd: int = -1
        self._write_fd: int = -1
        self._read_fd, self._write_fd = os.pipe()
        self._thread: threading.Thread = threading.Thread(
            target=self._drain,
            name=f"qrcheck-capture-{name}",
            daemon=True,
        )
        try:
            self._thread.start()
        except RuntimeError:
            self._close_read_fd()
            self._close_write_fd()
            raise

    @property
    def write_fd(self) -> int:
        """Return the descriptor to hand ``subprocess.run`` as this stream's destination."""
        return self._write_fd

    @property
    def name(self) -> str:
        """Return the stream's name, ``"stdout"`` or ``"stderr"``."""
        return self._name

    @property
    def bytes(self) -> bytes:
        """Return the bytes captured from this stream, at most the `limit` it was given."""
        return bytes(self._buffer)

    @property
    def received(self) -> int:
        """Return how many bytes arrived on this stream, the over-bound ones included."""
        return self._received

    @property
    def overflowed(self) -> bool:
        """Return whether this stream delivered more than the `limit` bytes it was given."""
        return self._overflowed

    def finish(self) -> None:
        """Release this process's ends of the pipe once the child has exited.

        The write end is closed first, so a drain thread still reading sees end-of-file and returns,
        and the thread is then joined for the caller's drain timeout. The read end is closed only
        once the thread has ended, because closing a descriptor another thread may still be reading
        is not safe.
        """
        self._close_write_fd()
        self._thread.join(self._join_timeout_s)
        if not self._thread.is_alive():
            self._close_read_fd()

    def _drain(self) -> None:
        """Read this stream until the child closes it or the bound is passed."""
        while True:
            try:
                chunk = os.read(self._read_fd, self._chunk_bytes)
            except OSError:
                # A read end closed on the over-bound path, or a descriptor the kernel refused.
                return
            if not chunk:
                return
            self._received += len(chunk)
            room = self._limit - len(self._buffer)
            if room > 0:
                self._buffer += chunk[:room]
            if self._received > self._limit:
                self._overflowed = True
                self._close_read_fd()
                return

    def _close_read_fd(self) -> None:
        """Close the read end once, if it is still open."""
        read_fd, self._read_fd = self._read_fd, -1
        _closed(read_fd)

    def _close_write_fd(self) -> None:
        """Close this process's copy of the write end once, if it is still open."""
        write_fd, self._write_fd = self._write_fd, -1
        _closed(write_fd)


def _closed(descriptor: int) -> None:
    """Close one descriptor, ignoring the error a descriptor already closed elsewhere reports."""
    if descriptor < 0:
        return
    with contextlib.suppress(OSError):
        os.close(descriptor)


def open_capture_pair(
    *, limit: int, chunk_bytes: int, join_timeout_s: float
) -> tuple[BoundedStreamCapture, BoundedStreamCapture]:
    """Return one bounded capture per output stream of a child about to be spawned, stdout first.

    Opening them is part of spawning: with nowhere to write, the child is not run at all, so a
    capture this call already opened is released before the failure is raised. The three numbers
    reach each capture as :class:`BoundedStreamCapture` documents them.

    Raises:
        OSError: A pipe could not be created.
        RuntimeError: A drain thread could not be started.
    """
    captures: list[BoundedStreamCapture] = []
    try:
        for name in ("stdout", "stderr"):
            captures.append(
                BoundedStreamCapture(
                    name, limit=limit, chunk_bytes=chunk_bytes, join_timeout_s=join_timeout_s
                )
            )
    except (OSError, RuntimeError):
        for capture in captures:
            capture.finish()
        raise
    return captures[0], captures[1]
