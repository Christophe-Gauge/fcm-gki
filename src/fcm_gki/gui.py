"""
Tkinter GUI modelled on the original WinForms app:
input image on the left, live segmentation on the right, the same controls
(clusters, max iterations, precision, spatial factor, Gki window) and a status
bar with iteration, precision and duration.

    fcm-gki-gui [image]        (or: python -m fcm_gki.gui [image])

Needs Pillow (pip install "fcm-gki[gui]"). Tkinter ships
with the python.org installers for Windows and macOS; on Debian/Ubuntu
install it with `sudo apt install python3-tk`.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from . import core as F
from .io import load_image, save_image


RED = (255, 0, 0)  # the GUI works in RGB order


class ImagePanel(ttk.Frame):
    """A titled canvas that shows an image scaled to fit, aspect preserved."""

    def __init__(self, master, title: str):
        super().__init__(master, padding=4)
        ttk.Label(self, text=title, font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(self, width=320, height=320, bg="#2b2b2b",
                                highlightthickness=1, highlightbackground="#555")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self._img: Image.Image | None = None
        self._photo = None

    def set_array(self, arr: np.ndarray | None):
        self._img = None if arr is None else Image.fromarray(
            np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        if self._img is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self._img.size
        s = min(cw / iw, ch / ih)
        size = (max(1, int(iw * s)), max(1, int(ih * s)))
        # Nearest-neighbour when enlarging keeps cluster edges crisp.
        resample = Image.NEAREST if s >= 1 else Image.LANCZOS
        self._photo = ImageTk.PhotoImage(self._img.resize(size, resample))
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")


class App(tk.Tk):
    POLL_MS = 50

    def __init__(self, path: str | None = None):
        super().__init__()
        self.title("Modified Fuzzy C-Means Clustering (Spatial + Gki)")
        self.geometry("1040x620")
        self.minsize(760, 420)

        self.image: np.ndarray | None = None
        self.image_path: str | None = None
        self.result: F.FCMResult | None = None
        self.worker: threading.Thread | None = None
        self.cancel_evt = threading.Event()
        self.msgs: queue.Queue = queue.Queue()
        self.t0 = 0.0

        self._build_menu()
        self._build_status()   # pack the status bar first so it is never squeezed out
        self._build_body()
        self.bind_all("<Control-o>", lambda e: self.open_image())
        self.bind_all("<Control-s>", lambda e: self.save_results())
        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        if path:
            self.after(100, lambda: self.load(path))

    # ---- layout ----------------------------------------------------------

    def _build_menu(self):
        mb = tk.Menu(self)
        fm = tk.Menu(mb, tearoff=False)
        fm.add_command(label="Open…", accelerator="Ctrl+O", command=self.open_image)
        fm.add_command(label="Save Results…", accelerator="Ctrl+S", command=self.save_results)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.on_exit)
        mb.add_cascade(label="File", menu=fm)
        self.config(menu=mb)
        self.file_menu = fm

    def _build_body(self):
        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.in_panel = ImagePanel(body, "Original")
        self.in_panel.grid(row=0, column=0, sticky="nsew")
        out = ttk.Frame(body)
        out.grid(row=0, column=1, sticky="nsew")
        self.out_panel = ImagePanel(out, "Segmented")
        self.out_panel.pack(fill="both", expand=True)
        vrow = ttk.Frame(out, padding=(4, 0))
        vrow.pack(fill="x")
        ttk.Label(vrow, text="Show:").pack(side="left")
        self.view_var = tk.StringVar(value="Segmented")
        self.view_box = ttk.Combobox(vrow, textvariable=self.view_var, state="readonly",
                                     values=["Segmented"], width=16)
        self.view_box.pack(side="left", padx=4)
        self.view_box.bind("<<ComboboxSelected>>", lambda e: self.show_view())

        ctl = ttk.LabelFrame(body, text="Parameters", padding=8)
        ctl.grid(row=0, column=2, sticky="ns", padx=(6, 0))

        self.v_clusters = tk.IntVar(value=5)
        self.v_iter = tk.IntVar(value=20)
        self.v_prec = tk.StringVar(value="0.00001")
        self.v_m = tk.DoubleVar(value=2.0)
        self.v_spatial = tk.BooleanVar(value=False)
        self.v_gki = tk.BooleanVar(value=False)
        self.v_window = tk.IntVar(value=1)
        self.v_maxdim = tk.IntVar(value=0)
        self.v_seed = tk.StringVar(value="")
        self.v_init = tk.StringVar(value="top_colours")

        def row(r, text, widget):
            ttk.Label(ctl, text=text).grid(row=r, column=0, sticky="w", pady=2)
            widget.grid(row=r, column=1, sticky="ew", pady=2, padx=(6, 0))

        row(0, "Number of Clusters:", ttk.Spinbox(ctl, from_=2, to=64, width=8, textvariable=self.v_clusters))
        row(1, "Maximum Iterations:", ttk.Spinbox(ctl, from_=1, to=1000, width=8, textvariable=self.v_iter))
        row(2, "Precision:", ttk.Entry(ctl, width=10, textvariable=self.v_prec))
        row(3, "Fuzziness (m):", ttk.Spinbox(ctl, from_=1.1, to=5.0, increment=0.1, width=8,
                                             textvariable=self.v_m, format="%.1f"))
        ttk.Separator(ctl).grid(row=4, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Checkbutton(ctl, text="Use Spatial Factor", variable=self.v_spatial
                        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=2)
        gk = ttk.Frame(ctl)
        ttk.Checkbutton(gk, text="Use Gki  Window =", variable=self.v_gki).pack(side="left")
        ttk.Spinbox(gk, from_=1, to=10, width=4, textvariable=self.v_window).pack(side="left", padx=4)
        gk.grid(row=6, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Separator(ctl).grid(row=7, column=0, columnspan=2, sticky="ew", pady=6)
        row(8, "Max size px (0 = full):", ttk.Spinbox(ctl, from_=0, to=10000, increment=100,
                                                     width=8, textvariable=self.v_maxdim))
        row(9, "Random seed:", ttk.Entry(ctl, width=10, textvariable=self.v_seed))
        row(10, "Seeding:", ttk.Combobox(ctl, width=10, state="readonly", textvariable=self.v_init,
                                         values=["top_colours", "kmeans++", "random"]))

        self.btn_run = ttk.Button(ctl, text="Fuzzy C-means Clustering", command=self.start,
                                  state="disabled")
        self.btn_run.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        self.btn_cancel = ttk.Button(ctl, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.grid(row=12, column=0, columnspan=2, sticky="ew")
        self.param_widgets = [w for w in ctl.winfo_children()
                              if isinstance(w, (ttk.Spinbox, ttk.Entry, ttk.Checkbutton, ttk.Combobox))]
        self.param_widgets += list(gk.winfo_children())

    def _build_status(self):
        sb = ttk.Frame(self, relief="sunken", padding=(6, 2))
        sb.pack(fill="x", side="bottom")
        self.s_state = ttk.Label(sb, text="Open an image to begin (File ▸ Open).", width=28)
        self.s_state.pack(side="left")
        self.progress = ttk.Progressbar(sb, length=160, mode="determinate")
        self.progress.pack(side="left", padx=8)
        self.s_prec = ttk.Label(sb, text="Precision:", width=24)
        self.s_prec.pack(side="left")
        self.s_time = ttk.Label(sb, text="Duration:")
        self.s_time.pack(side="left")

    # ---- file actions ----------------------------------------------------

    def open_image(self):
        if self.running:
            return
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
                       ("All files", "*.*")])
        if path:
            self.load(path)

    def load(self, path: str):
        try:
            img = load_image(path)
        except Exception as e:  # noqa: BLE001 - show any decode error to the user
            messagebox.showerror("Error", f"Failed loading the image:\n{e}")
            return
        self.image, self.image_path, self.result = img, path, None
        self.in_panel.set_array(img)
        self.out_panel.set_array(None)
        self.view_box["values"] = ["Segmented"]
        self.view_var.set("Segmented")
        h, w = img.shape[:2]
        self.title(f"Modified Fuzzy C-Means — {os.path.basename(path)} ({w}×{h})")
        self.s_state["text"] = "Ready."
        self.btn_run["state"] = "normal"

    def save_results(self):
        if self.result is None or self.running:
            messagebox.showinfo("Save Results", "Run the clustering first.")
            return
        folder = filedialog.askdirectory(title="Save segmented.png and Cluster*.png to…")
        if not folder:
            return
        img = self.run_image
        save_image(F.render_segmented(self.result, self.used_spatial, RED),
                     os.path.join(folder, "segmented.png"))
        for k, c in enumerate(F.render_clusters(img, self.result)):
            save_image(c, os.path.join(folder, f"Cluster{k}.png"))
        self.s_state["text"] = f"Saved {len(self.result.centers) + 1} images."

    def on_exit(self):
        self.cancel_evt.set()
        self.destroy()

    # ---- run / cancel ----------------------------------------------------

    @property
    def running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _working_image(self) -> np.ndarray:
        md = self.v_maxdim.get()
        return load_image(self.image_path, md) if md > 0 else self.image

    def start(self):
        try:
            params = dict(
                n_clusters=int(self.v_clusters.get()),
                fuzziness=float(self.v_m.get()),
                use_spatial=bool(self.v_spatial.get()),
                use_gki=bool(self.v_gki.get()),
                window=int(self.v_window.get()),
                max_iter=int(self.v_iter.get()),
                tol=float(self.v_prec.get()),
                init=self.v_init.get(),
                seed=int(self.v_seed.get()) if self.v_seed.get().strip() else None,
                verbose=False,
            )
            fcm = F.ModifiedFCM(**params)
            img = self._working_image()
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Invalid parameter", str(e))
            return

        self.run_image = img
        self.used_spatial = params["use_spatial"]
        self.max_iter = params["max_iter"]
        self.cancel_evt.clear()
        self._set_running(True)
        self.progress["value"] = 0
        self.s_state["text"] = "Working…"
        self.s_prec["text"] = "Precision:"
        self.view_box["values"] = ["Segmented"]
        self.view_var.set("Segmented")
        self.result = None
        self.t0 = time.perf_counter()

        def cb(res, delta):
            self.msgs.put(("iter", res, delta, F.render_segmented(res, self.used_spatial, RED)))
            return self.cancel_evt.is_set()

        def work():
            try:
                res = fcm.fit(img, callback=cb)
                self.msgs.put(("done", res, self.cancel_evt.is_set()))
            except Exception as e:  # noqa: BLE001 - surface worker errors in the UI
                self.msgs.put(("error", e))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        self.after(self.POLL_MS, self._poll)

    def cancel(self):
        self.cancel_evt.set()
        self.s_state["text"] = "Aborting… please wait!"

    def _set_running(self, on: bool):
        self.btn_run["state"] = "disabled" if on else "normal"
        self.btn_cancel["state"] = "normal" if on else "disabled"
        for w in self.param_widgets:
            w.state(["disabled"] if on else ["!disabled"])
        st = "disabled" if on else "normal"
        self.file_menu.entryconfig("Open…", state=st)
        self.file_menu.entryconfig("Save Results…", state=st)

    def _poll(self):
        self.s_time["text"] = f"Duration: {time.perf_counter() - self.t0:6.2f} s"
        latest = None
        try:
            while True:
                msg = self.msgs.get_nowait()
                if msg[0] == "iter":
                    latest = msg      # only draw the newest frame
                else:
                    if latest:
                        self._show_iter(*latest[1:])
                    self._finish(msg)
                    return
        except queue.Empty:
            pass
        if latest:
            self._show_iter(*latest[1:])
        self.after(self.POLL_MS, self._poll)

    def _show_iter(self, res, delta, seg):
        self.s_state["text"] = f"Iteration {res.iterations}"
        self.s_prec["text"] = f"Precision: {delta:.2e}"
        self.progress["value"] = 100 * res.iterations / self.max_iter
        self.out_panel.set_array(seg)

    def _finish(self, msg):
        self._set_running(False)
        self.s_time["text"] = f"Duration: {time.perf_counter() - self.t0:6.2f} s"
        if msg[0] == "error":
            self.s_state["text"] = f"Error: {msg[1]}"
            messagebox.showerror("Error", str(msg[1]))
            return
        res, cancelled = msg[1], msg[2]
        self.result = res
        self.progress["value"] = 100
        self.s_state["text"] = (f"Canceled after {res.iterations} iterations." if cancelled
                                else f"Done in {res.iterations} iterations.")
        self.view_box["values"] = ["Segmented"] + [f"Cluster {k}" for k in range(len(res.centers))]
        self.show_view()

    def show_view(self):
        if self.result is None:
            return
        v = self.view_var.get()
        if v == "Segmented":
            self.out_panel.set_array(F.render_segmented(self.result, self.used_spatial, RED))
        else:
            k = int(v.split()[-1])
            self.out_panel.set_array(F.render_clusters(self.run_image, self.result)[k])


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    App(argv[0] if argv else None).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
