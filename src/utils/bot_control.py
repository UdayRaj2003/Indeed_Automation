"""Stop controls, global Ctrl+Shift+Q / corner failsafe, and screen keep-awake for Indeed bot."""

from __future__ import annotations

import ctypes
import gc
import os
import threading
import time

from src.dashboard.runtime import update_control_state
from src.utils.runtime_control import runtime_controller

_stop_event = threading.Event()
_ui_thread: threading.Thread | None = None
_hotkey_thread: threading.Thread | None = None
_root = None
_started = False


class BotStopped(Exception):
    """Raised when user triggers STOP BOT window, Ctrl+Shift+Q, or corner failsafe."""


def request_stop(reason: str = "user") -> None:
    _stop_event.set()
    runtime_controller.request_shutdown(reason)
    try:
        if os.environ.get("DASHBOARD_RUN_ID"):
            update_control_state(stop_requested=True)
    except Exception:
        pass


def clear_stop() -> None:
    _stop_event.clear()


def should_stop() -> bool:
    return _stop_event.is_set() or runtime_controller.is_shutdown_requested()


def keep_screen_awake(enable: bool = True) -> None:
    """Prevent Windows display/system sleep during non-stop bot runs."""
    if os.name != "nt":
        return
    try:
        es_continuous = 0x80000000
        es_system_required = 0x00000001
        es_display_required = 0x00000002
        flags = (
            es_continuous | es_system_required | es_display_required
            if enable
            else es_continuous
        )
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def _on_stop_click() -> None:
    request_stop("stop_button")
    global _root
    if _root is not None:
        try:
            _root.after(0, _root.destroy)
        except Exception:
            pass


def _run_ui() -> None:
    global _root
    root = None
    _poll_stop = None
    title_font = None
    hint_font = None
    btn_font = None
    try:
        import tkinter as tk
        from tkinter import font as tkfont

        root = tk.Tk()
        _root = root
        root.title("Indeed Job Bot")
        root.attributes("-topmost", True)
        root.resizable(False, False)

        title_font = tkfont.Font(size=9)
        hint_font = tkfont.Font(size=7)
        btn_font = tkfont.Font(size=12, weight="bold")
        tk.Label(root, text="Indeed Job Bot", font=title_font).pack(pady=(8, 2))
        tk.Label(
            root,
            text="Ctrl+Shift+Q | corner failsafe",
            font=hint_font,
            fg="#555555",
        ).pack()
        tk.Button(
            root,
            text="STOP BOT",
            fg="white",
            bg="#C62828",
            activebackground="#B71C1C",
            activeforeground="white",
            font=btn_font,
            command=_on_stop_click,
            height=1,
            width=14,
        ).pack(pady=6, padx=10, fill="x")

        root.update_idletasks()
        w = max(200, root.winfo_reqwidth() + 16)
        h = max(90, root.winfo_reqheight() + 8)
        x = max(0, root.winfo_screenwidth() - w - 24)
        root.minsize(w, h)
        root.geometry(f"{w}x{h}+{x}+24")
        root.deiconify()
        root.lift()

        root.protocol("WM_DELETE_WINDOW", _on_stop_click)
        root.bind("<Control-q>", lambda _e: _on_stop_click())

        def _poll_stop() -> None:
            if should_stop():
                try:
                    root.destroy()
                except Exception:
                    pass
                return
            try:
                root.after(150, _poll_stop)
            except Exception:
                pass

        root.after(150, _poll_stop)
        root.mainloop()
    except Exception:
        pass
    finally:
        _root = None
        _poll_stop = None
        title_font = None
        hint_font = None
        btn_font = None
        root = None
        gc.collect()


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _run_hotkey_and_corner_watcher() -> None:
    """Watch Win32 Ctrl+Shift+Q hotkey and top-left (0,0) corner failsafe."""
    if os.name != "nt":
        return
    try:
        user32 = ctypes.windll.user32
    except Exception:
        return

    vk_ctrl, vk_shift, vk_q = 0x11, 0x10, 0x51
    corner_ticks = 0
    while not should_stop():
        try:
            ctrl = user32.GetAsyncKeyState(vk_ctrl) & 0x8000
            shift = user32.GetAsyncKeyState(vk_shift) & 0x8000
            q_key = user32.GetAsyncKeyState(vk_q) & 0x8000
            if ctrl and shift and q_key:
                request_stop("hotkey_ctrl_shift_q")
                break

            pt = _POINT()
            if user32.GetCursorPos(ctypes.byref(pt)):
                if pt.x <= 1 and pt.y <= 1:
                    corner_ticks += 1
                    if corner_ticks >= 3:
                        request_stop("corner_failsafe")
                        break
                else:
                    corner_ticks = 0
        except Exception:
            break
        time.sleep(0.15)


def start_stop_controls() -> None:
    """Start always-on-top STOP BOT window and Win32 hotkey/corner failsafe once."""
    global _ui_thread, _hotkey_thread, _started
    if _started or "PYTEST_CURRENT_TEST" in os.environ:
        return
    _started = True
    clear_stop()
    _ui_thread = threading.Thread(target=_run_ui, name="indeed-stop-ui", daemon=True)
    _ui_thread.start()
    _hotkey_thread = threading.Thread(
        target=_run_hotkey_and_corner_watcher, name="indeed-stop-hotkey", daemon=True
    )
    _hotkey_thread.start()


def stop_stop_controls() -> None:
    global _ui_thread
    request_stop("shutdown")
    keep_screen_awake(False)
    thread = _ui_thread
    _ui_thread = None
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1.5)
