"""
launcher.py — Workshop Launcher
================================
A clean GUI with buttons to launch the demo or main project.
No terminal needed — just run this file and click.

  python launcher.py
"""

import subprocess
import sys
import tkinter as tk
from tkinter import font as tkfont

# ── Colours (dark theme) ──────────────────────────────────────────────────────
BG        = "#1a1a2e"
CARD_BG   = "#16213e"
ACCENT    = "#0f3460"
GREEN     = "#00e676"
RED       = "#ff1744"
WHITE     = "#e0e0e0"
GRAY      = "#888888"
TITLE_CLR = "#e94560"

# ── State ─────────────────────────────────────────────────────────────────────
running_proc = None
running_name = ""


def launch(script: str, name: str) -> None:
    """Launch a Python script as a subprocess."""
    global running_proc, running_name

    # Kill any already-running process first
    stop()

    running_name = name
    running_proc = subprocess.Popen(
        [sys.executable, script],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    update_status()


def stop() -> None:
    """Kill the currently running subprocess."""
    global running_proc, running_name
    if running_proc and running_proc.poll() is None:
        running_proc.terminate()
        try:
            running_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            running_proc.kill()
    running_proc = None
    running_name = ""
    update_status()


def update_status() -> None:
    """Update the status indicator."""
    if running_proc and running_proc.poll() is None:
        status_dot.config(bg=GREEN)
        status_label.config(text=f"Running: {running_name}", fg=GREEN)
        stop_btn.config(state=tk.NORMAL)
    else:
        status_dot.config(bg=RED)
        status_label.config(text="Not running", fg=GRAY)
        stop_btn.config(state=tk.DISABLED)
    root.after(500, check_alive)


def check_alive() -> None:
    """Periodically check if the subprocess is still alive."""
    if running_proc and running_proc.poll() is not None:
        stop()
    elif running_proc:
        root.after(500, check_alive)


def on_close() -> None:
    """Clean up on window close."""
    stop()
    root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# Build the window
# ══════════════════════════════════════════════════════════════════════════════

root = tk.Tk()
root.title("Spider-verse Workshop")
root.configure(bg=BG)
root.resizable(False, False)
root.protocol("WM_DELETE_WINDOW", on_close)

# Centre on screen
win_w, win_h = 480, 520
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
x = (screen_w - win_w) // 2
y = (screen_h - win_h) // 2
root.geometry(f"{win_w}x{win_h}+{x}+{y}")

# ── Fonts ─────────────────────────────────────────────────────────────────────
title_font = tkfont.Font(family="Segoe UI", size=22, weight="bold")
sub_font   = tkfont.Font(family="Segoe UI", size=11)
btn_font   = tkfont.Font(family="Segoe UI", size=13, weight="bold")
info_font  = tkfont.Font(family="Segoe UI", size=9)
stat_font  = tkfont.Font(family="Segoe UI", size=10)

# ── Title ─────────────────────────────────────────────────────────────────────
tk.Label(root, text="🕷️ Spider-verse", font=title_font, bg=BG, fg=TITLE_CLR).pack(pady=(30, 2))
tk.Label(root, text="MediaPipe + OpenCV Workshop", font=sub_font, bg=BG, fg=GRAY).pack(pady=(0, 25))

# ── Buttons card ──────────────────────────────────────────────────────────────
card = tk.Frame(root, bg=CARD_BG, highlightbackground=ACCENT, highlightthickness=1)
card.pack(padx=30, pady=5, fill=tk.X)

tk.Label(card, text="Choose a mode:", font=sub_font, bg=CARD_BG, fg=WHITE).pack(pady=(18, 12))

# Demo button
demo_btn = tk.Button(
    card,
    text="🔍  Demo — Raw Landmarks",
    font=btn_font,
    bg="#0f3460",
    fg=WHITE,
    activebackground="#1a5276",
    activeforeground=WHITE,
    relief=tk.FLAT,
    cursor="hand2",
    width=28,
    pady=10,
    command=lambda: launch("demo.py", "Demo — Raw Landmarks"),
)
demo_btn.pack(pady=(0, 10))

# Main project button
main_btn = tk.Button(
    card,
    text="🕷️  Main — Spider-Man Filters",
    font=btn_font,
    bg="#e94560",
    fg=WHITE,
    activebackground="#c0392b",
    activeforeground=WHITE,
    relief=tk.FLAT,
    cursor="hand2",
    width=28,
    pady=10,
    command=lambda: launch("main.py", "Main — Spider-Man Filters"),
)
main_btn.pack(pady=(0, 18))

# ── Stop button ───────────────────────────────────────────────────────────────
stop_btn = tk.Button(
    root,
    text="⏹  Stop",
    font=btn_font,
    bg="#2c2c3e",
    fg=RED,
    activebackground="#3c3c4e",
    activeforeground=RED,
    relief=tk.FLAT,
    cursor="hand2",
    width=28,
    pady=8,
    state=tk.DISABLED,
    command=stop,
)
stop_btn.pack(pady=15)

# ── Status bar ────────────────────────────────────────────────────────────────
status_frame = tk.Frame(root, bg=BG)
status_frame.pack(pady=(5, 0))

status_dot = tk.Label(status_frame, text=" ", bg=RED, width=2, height=1)
status_dot.pack(side=tk.LEFT, padx=(0, 8))

status_label = tk.Label(status_frame, text="Not running", font=stat_font, bg=BG, fg=GRAY)
status_label.pack(side=tk.LEFT)

# ── Info footer ───────────────────────────────────────────────────────────────
tk.Label(root, text="", bg=BG).pack(expand=True)  # spacer
footer = tk.Label(
    root,
    text="Demo: shows raw MediaPipe face + hand mapping points\n"
         "Main: full Spider-Man mask + gesture-controlled filters\n\n"
         "Press 'q' inside the webcam window to close it",
    font=info_font,
    bg=BG,
    fg=GRAY,
    justify=tk.CENTER,
)
footer.pack(pady=(0, 20))

# ── Go ────────────────────────────────────────────────────────────────────────
root.mainloop()
