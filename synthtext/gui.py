"""Small Tkinter GUI for launching the SynthText CLI."""

import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import GenerationConfig


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


class SynthTextLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SynthText Launcher")
        self.minsize(980, 720)

        self.proc = None
        self.reader_thread = None
        self.log_queue = queue.Queue()

        self.vars = self._make_vars()
        self._build_ui()
        self._set_running(False)
        self.after(100, self._drain_log_queue)

    def _make_vars(self):
        cfg = GenerationConfig()
        return {
            "input_dir": tk.StringVar(value=cfg.input_dir),
            "fallback_h5": tk.StringVar(value=cfg.fallback_h5),
            "render_data_path": tk.StringVar(value=cfg.render_data_path),
            "output_file": tk.StringVar(value=cfg.output_file),
            "png_dir": tk.StringVar(value=cfg.png_dir),
            "num_img": tk.StringVar(value=str(cfg.num_img)),
            "instances_per_image": tk.StringVar(value=str(cfg.instances_per_image)),
            "secs_per_img": tk.StringVar(value=str(cfg.secs_per_img)),
            "max_global_tries": tk.StringVar(value=str(cfg.max_global_tries)),
            "max_h5_size_gb": tk.StringVar(value=str(cfg.max_h5_size_gb)),
            "region_workers": tk.StringVar(value=str(cfg.region_workers)),
            "ransac_stats": tk.StringVar(value=str(cfg.ransac_stats)),
            "viz": tk.BooleanVar(value=cfg.viz),
            "interactive": tk.BooleanVar(value=cfg.interactive),
            "ransac_debug": tk.BooleanVar(value=cfg.ransac_debug),
            "placement_debug": tk.BooleanVar(value=cfg.placement_debug),
            "debug_progress": tk.BooleanVar(value=cfg.debug_progress),
        }

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        paths = ttk.LabelFrame(root, text="Paths", padding=10)
        paths.grid(row=0, column=0, sticky="ew")
        paths.columnconfigure(1, weight=1)

        self._path_row(paths, 0, "Input dir", "input_dir", "directory")
        self._path_row(paths, 1, "Fallback H5", "fallback_h5", "file")
        self._path_row(paths, 2, "Render data", "render_data_path", "directory")
        self._path_row(paths, 3, "Output H5", "output_file", "save")
        self._path_row(paths, 4, "PNG dir", "png_dir", "directory")

        options = ttk.LabelFrame(root, text="Generation Options", padding=10)
        options.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for col in range(6):
            options.columnconfigure(col, weight=1)

        numeric_specs = [
            ("Images", "num_img"),
            ("Instances/image", "instances_per_image"),
            ("Seconds/image", "secs_per_img"),
            ("Max tries", "max_global_tries"),
            ("Max H5 GB", "max_h5_size_gb"),
            ("Region workers", "region_workers"),
            ("RANSAC stats N", "ransac_stats"),
        ]
        for idx, (label, key) in enumerate(numeric_specs):
            row = idx // 4
            col = (idx % 4) * 2
            ttk.Label(options, text=label).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=4)
            ttk.Entry(options, textvariable=self.vars[key], width=12).grid(
                row=row, column=col + 1, sticky="ew", padx=(0, 12), pady=4
            )

        flags = ttk.LabelFrame(root, text="Flags", padding=10)
        flags.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        flag_specs = [
            ("Visualize", "viz"),
            ("Interactive input", "interactive"),
            ("RANSAC debug", "ransac_debug"),
            ("Placement debug", "placement_debug"),
            ("Debug progress", "debug_progress"),
        ]
        for idx, (label, key) in enumerate(flag_specs):
            ttk.Checkbutton(flags, text=label, variable=self.vars[key], command=self._update_command_preview).grid(
                row=0, column=idx, sticky="w", padx=(0, 20)
            )

        run_frame = ttk.Frame(root)
        run_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        run_frame.columnconfigure(0, weight=1)
        run_frame.rowconfigure(2, weight=1)

        cmd_frame = ttk.LabelFrame(run_frame, text="Command", padding=8)
        cmd_frame.grid(row=0, column=0, sticky="ew")
        cmd_frame.columnconfigure(0, weight=1)
        self.command_var = tk.StringVar()
        self.command_entry = ttk.Entry(cmd_frame, textvariable=self.command_var, state="readonly")
        self.command_entry.grid(row=0, column=0, sticky="ew")

        buttons = ttk.Frame(run_frame)
        buttons.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self.run_button = ttk.Button(buttons, text="Run", command=self._run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop)
        self.stop_button.pack(side="left", padx=(8, 0))
        self.continue_button = ttk.Button(buttons, text="Continue viz", command=self._continue_viz)
        self.continue_button.pack(side="left", padx=(8, 0))
        self.quit_viz_button = ttk.Button(buttons, text="Quit viz", command=self._quit_viz)
        self.quit_viz_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Clear log", command=self._clear_log).pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(buttons, textvariable=self.status_var).pack(side="right")

        log_frame = ttk.LabelFrame(run_frame, text="Log", padding=8)
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", height=20, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        for key, var in self.vars.items():
            if isinstance(var, tk.StringVar):
                var.trace_add("write", lambda *_args: self._update_command_preview())
        self._update_command_preview()

    def _path_row(self, parent, row, label, key, mode):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=self.vars[key]).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=lambda: self._browse_path(key, mode)).grid(
            row=row, column=2, sticky="e", padx=(8, 0), pady=4
        )

    def _browse_path(self, key, mode):
        current = self.vars[key].get().strip()
        initialdir = current if os.path.isdir(current) else os.getcwd()

        if mode == "directory":
            value = filedialog.askdirectory(initialdir=initialdir)
        elif mode == "save":
            value = filedialog.asksaveasfilename(
                initialdir=os.path.dirname(current) or os.getcwd(),
                initialfile=os.path.basename(current) or "SynthText.h5",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
            )
        else:
            value = filedialog.askopenfilename(
                initialdir=os.path.dirname(current) or os.getcwd(),
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
            )
        if value:
            self.vars[key].set(value)

    def _build_command(self):
        self._validate_inputs()
        args = [
            sys.executable,
            "-u",
            "gen.py",
            "--input-dir", self.vars["input_dir"].get().strip(),
            "--fallback-h5", self.vars["fallback_h5"].get().strip(),
            "--render-data-path", self.vars["render_data_path"].get().strip(),
            "--output-file", self.vars["output_file"].get().strip(),
            "--png-dir", self.vars["png_dir"].get().strip(),
            "--num-img", str(self._int_value("num_img", minimum=-1)),
            "--instances-per-image", str(self._int_value("instances_per_image", minimum=1)),
            "--secs-per-img", str(self._int_value("secs_per_img", minimum=1)),
            "--max-global-tries", str(self._int_value("max_global_tries", minimum=1)),
            "--max-h5-size-gb", str(self._float_value("max_h5_size_gb", minimum=0.1)),
            "--region-workers", str(self._int_value("region_workers", minimum=1)),
        ]

        ransac_stats = self._int_value("ransac_stats", minimum=0)
        if ransac_stats > 0:
            args.extend(["--ransac-stats", str(ransac_stats)])

        bool_flags = [
            ("viz", "--viz"),
            ("interactive", "--interactive"),
            ("ransac_debug", "--ransac-debug"),
            ("placement_debug", "--placement-debug"),
            ("debug_progress", "--debug-progress"),
        ]
        for key, flag in bool_flags:
            if self.vars[key].get():
                args.append(flag)
        return args

    def _validate_inputs(self):
        required = ["input_dir", "fallback_h5", "render_data_path", "output_file", "png_dir"]
        missing = [key for key in required if not self.vars[key].get().strip()]
        if missing:
            raise ValueError("Fill required path fields: " + ", ".join(missing))

    def _int_value(self, key, minimum=None):
        try:
            value = int(self.vars[key].get())
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if minimum is not None and value < minimum:
            raise ValueError(f"{key} must be >= {minimum}")
        return value

    def _float_value(self, key, minimum=None):
        try:
            value = float(self.vars[key].get())
        except ValueError as exc:
            raise ValueError(f"{key} must be a number") from exc
        if minimum is not None and value < minimum:
            raise ValueError(f"{key} must be >= {minimum}")
        return value

    def _update_command_preview(self):
        try:
            args = self._build_command()
            self.command_var.set(" ".join(self._quote_arg(a) for a in args))
        except Exception as exc:
            self.command_var.set(f"Invalid options: {exc}")

    @staticmethod
    def _quote_arg(arg):
        arg = str(arg)
        if not arg or any(ch.isspace() for ch in arg):
            return '"' + arg.replace('"', '\\"') + '"'
        return arg

    def _run(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showinfo("SynthText", "Process is already running.")
            return

        try:
            args = self._build_command()
        except Exception as exc:
            messagebox.showerror("Invalid options", str(exc))
            return

        self._append_log("$ " + " ".join(self._quote_arg(a) for a in args) + "\n")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            self.proc = subprocess.Popen(
                args,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self.proc = None
            messagebox.showerror("Launch failed", repr(exc))
            return

        self._set_running(True)
        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

    def _read_process_output(self):
        try:
            while True:
                chunk = self.proc.stdout.read(1)
                if chunk == "":
                    break
                self.log_queue.put(("line", _strip_ansi(chunk)))
            code = self.proc.wait()
            self.log_queue.put(("done", code))
        except Exception as exc:
            self.log_queue.put(("line", f"[GUI] log reader failed: {repr(exc)}\n"))
            self.log_queue.put(("done", None))

    def _stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        self._append_log("[GUI] stopping process...\n")
        try:
            self.proc.terminate()
        except Exception as exc:
            self._append_log(f"[GUI] terminate failed: {repr(exc)}\n")

    def _send_stdin(self, text):
        if self.proc is None or self.proc.poll() is not None or self.proc.stdin is None:
            return False
        try:
            self.proc.stdin.write(text)
            self.proc.stdin.flush()
            return True
        except Exception as exc:
            self._append_log(f"[GUI] stdin write failed: {repr(exc)}\n")
            return False

    def _continue_viz(self):
        if self._send_stdin("\n"):
            self._append_log("[GUI] continue viz\n")

    def _quit_viz(self):
        if self._send_stdin("q\n"):
            self._append_log("[GUI] quit viz requested\n")

    def _drain_log_queue(self):
        while True:
            try:
                kind, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self._append_log(payload)
            elif kind == "done":
                self._append_log(f"[GUI] process finished with code {payload}\n")
                self._set_running(False)
                self.proc = None
        self.after(100, self._drain_log_queue)

    def _append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _set_running(self, running: bool):
        self.run_button.configure(state=("disabled" if running else "normal"))
        self.stop_button.configure(state=("normal" if running else "disabled"))
        self.continue_button.configure(state=("normal" if running else "disabled"))
        self.quit_viz_button.configure(state=("normal" if running else "disabled"))
        self.status_var.set("Running" if running else "Idle")


def main():
    app = SynthTextLauncher()
    app.mainloop()
