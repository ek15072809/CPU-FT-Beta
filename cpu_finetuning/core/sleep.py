"""Keep the machine awake during training (requirement 9).

On Windows this calls ``SetThreadExecutionState`` with
``ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED`` so the OS will
not sleep, hibernate or turn off the display lock while training runs.  A
keepalive thread re-asserts the flags every 50 seconds (recommended by the
Win32 docs because some events reset them).

On other platforms the guard is a no-op and says so in the logs.
"""

from __future__ import annotations

import threading
import time


class SleepGuard:
    def __init__(self, log=None):
        self._log = log or (lambda msg: print(msg, flush=True))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active = False

    # ------------------------------------------------------------------
    def _windows_flags(self) -> bool:
        try:
            import ctypes

            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_AWAYMODE_REQUIRED = 0x00000040
            flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            # SetThreadExecutionState returns NULL (0) on failure
            result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
            return result != 0
        except Exception:
            return False

    def _loop(self):
        while not self._stop.wait(50.0):
            self._windows_flags()

    # ------------------------------------------------------------------
    def start(self, lang: str = "ja"):
        from .i18n import t

        import sys
        if sys.platform == "win32":
            ok = self._windows_flags()
            self._active = ok
            self._log(t("sleep_guard_on" if ok else "sleep_guard_off", lang))
        else:
            self._active = True  # no-op guard keeps the interface uniform
            self._log(t("sleep_guard_off", lang))
        if self._active:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            import sys
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS only
        except Exception:
            pass
        self._active = False
