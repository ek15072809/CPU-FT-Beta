"""Watchdog: keep training alive across process crashes (requirement 9).

The ``train`` CLI command runs this supervisor, which spawns the actual
training child in the SAME console (so the CUI output streams directly),
monitors it, and restarts it with automatic checkpoint resume if it dies
unexpectedly.  Ctrl+C is forwarded once so the child can save an emergency
checkpoint; a second Ctrl+C force-stops.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


def run_with_watchdog(config_path: str, max_restarts: int = 5, lang: str = "ja",
                      hold_console: bool = False) -> int:
    from .i18n import t

    pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    attempts = 0
    code = 1
    while attempts <= max_restarts:
        cmd = [sys.executable, "-m", "cpu_finetuning", "_train_child",
               "--config", config_path, "--lang", lang]
        proc = subprocess.Popen(cmd, cwd=pkg_dir)
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            # first Ctrl+C: forward, give the child a chance to save
            try:
                if os.name == "nt":
                    signal_ctrl(proc)
                else:
                    proc.send_signal(subprocess.signal.SIGINT)
            except Exception:
                pass
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = 130
            break
        if code == 0:
            break
        if code == 2:
            # deterministic configuration error (e.g. empty dataset source);
            # restarting cannot fix it and only reloads the model needlessly
            break
        attempts += 1
        if attempts > max_restarts:
            print(f"[watchdog] giving up after {max_restarts} restarts", flush=True)
            break
        print(t("watchdog_restart", lang, n=attempts, max=max_restarts), flush=True)
        time.sleep(3)
    else:
        code = 1
    if hold_console:
        # Requirement 9: when the Web UI launches training, keep the console
        # open after the run so the complete log remains visible.  The same log
        # is also persisted in logs/<run>.log.
        try:
            if sys.stdin and sys.stdin.isatty():
                input("\n学習ログを確認したら Enter を押してください / "
                      "Press Enter to close this console...")
            else:
                # A non-interactive parent would otherwise leave the window
                # open forever; the log file is still available.
                pass
        except (EOFError, KeyboardInterrupt):
            pass
    return code


def signal_ctrl(proc):
    # Windows: send CTRL_C_EVENT to the process group
    try:
        import ctypes
        ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, proc.pid)
    except Exception:
        proc.terminate()
