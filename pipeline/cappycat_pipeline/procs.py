"""Child-process registry and the parent-death watchdog.

Every ffmpeg / ffprobe the pipeline starts goes through :func:`popen` / :func:`run`, which record
the ``Popen`` object while it is alive. :func:`start_stdin_watchdog` starts a daemon thread that
reads stdin until EOF; the Rust core keeps the pipeline's stdin open as a pipe for the lifetime of
the job, so EOF means the parent died or cancelled the job. The watchdog then kills every live
child process (so no ffmpeg is left writing a file) and exits the process immediately.

The watchdog is opt-in (``--watch-stdin`` or ``CAPPYCAT_WATCH_STDIN=1``) so running the CLI from a
terminal is not affected.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from typing import Any, Callable, List, Optional, Sequence

log = logging.getLogger("cappycat.procs")

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
WATCHDOG_EXIT_CODE = 3

_LOCK = threading.Lock()
_CHILDREN: List[subprocess.Popen] = []
_WATCHDOG: Optional[threading.Thread] = None


def _register(p: subprocess.Popen) -> None:
    with _LOCK:
        _CHILDREN[:] = [c for c in _CHILDREN if c.poll() is None]
        _CHILDREN.append(p)


def _unregister(p: subprocess.Popen) -> None:
    with _LOCK:
        if p in _CHILDREN:
            _CHILDREN.remove(p)


def live_children() -> List[subprocess.Popen]:
    with _LOCK:
        return [c for c in _CHILDREN if c.poll() is None]


def popen(cmd: Sequence[str], **kw: Any) -> subprocess.Popen:
    """``subprocess.Popen`` (no console window on Windows) registered for the watchdog."""
    kw.setdefault("creationflags", CREATE_NO_WINDOW)
    p = subprocess.Popen(list(cmd), **kw)
    _register(p)
    return p


def run(cmd: Sequence[str], *, input: Optional[bytes] = None, timeout: Optional[float] = None,
        **kw: Any) -> subprocess.CompletedProcess:
    """``subprocess.run`` equivalent whose process is visible to the watchdog."""
    kw.setdefault("stdout", subprocess.PIPE)
    kw.setdefault("stderr", subprocess.PIPE)
    if input is None:
        kw.setdefault("stdin", subprocess.DEVNULL)
    else:
        kw["stdin"] = subprocess.PIPE
    p = popen(cmd, **kw)
    try:
        out, err = p.communicate(input=input, timeout=timeout)
    except BaseException:
        p.kill()
        p.wait()
        raise
    finally:
        _unregister(p)
    return subprocess.CompletedProcess(list(cmd), p.returncode, out, err)


def kill_children() -> int:
    """Kill every live registered child; returns how many were killed."""
    n = 0
    for c in live_children():
        try:
            c.kill()
            n += 1
        except Exception:  # already gone
            pass
    for c in live_children():
        try:
            c.wait(timeout=2.0)
        except Exception:
            pass
    return n


def watch_stdin_enabled(argv_flag: bool = False) -> bool:
    return argv_flag or os.environ.get("CAPPYCAT_WATCH_STDIN", "").strip().lower() in ("1", "true", "yes", "on")


def start_stdin_watchdog(stream: Any = None, on_eof: Optional[Callable[[], None]] = None,
                         exit_code: int = WATCHDOG_EXIT_CODE) -> threading.Thread:
    """Start the daemon thread (idempotent). ``stream`` defaults to the binary stdin; ``on_eof``
    (tests) replaces the default action (kill children + ``os._exit``)."""
    global _WATCHDOG
    if _WATCHDOG is not None and _WATCHDOG.is_alive():
        return _WATCHDOG
    src = stream if stream is not None else getattr(sys.stdin, "buffer", sys.stdin)

    # Read the real stdin through a duplicated raw file descriptor with os.read: a daemon thread
    # blocked inside the *buffered* sys.stdin object crashes the interpreter at shutdown
    # ("Fatal Python error: _enter_buffered_busy"), so every normal exit would look like a crash.
    raw_fd: Optional[int] = None
    if stream is None:
        try:
            raw_fd = os.dup(sys.stdin.fileno())
        except (AttributeError, OSError, ValueError):
            raw_fd = None

    def _watch() -> None:
        try:
            while True:
                if raw_fd is not None:
                    chunk = os.read(raw_fd, 65536)
                else:
                    chunk = src.read(1) if not hasattr(src, "read1") else src.read1(65536)
                if not chunk:
                    break
        except Exception:
            pass  # a broken stdin pipe means the same thing as EOF
        if on_eof is not None:
            on_eof()
            return
        n = kill_children()
        try:
            sys.stderr.write(f"cappycat: stdin closed (parent gone); killed {n} child process(es), exiting\n")
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(exit_code)

    _WATCHDOG = threading.Thread(target=_watch, name="cappycat-stdin-watchdog", daemon=True)
    _WATCHDOG.start()
    return _WATCHDOG
