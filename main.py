"""
main.py - Pose Annotation Tool: window, drawing, input handling and workflow control.

Workflow: choose "Video / Folder" -> Open Source -> Open Pose JSON (or "Estimate" to run YOLO)
          -> review frame by frame -> drag joints / add skeletons -> Save
Mouse:    left-drag a joint | right-click a joint = toggle visible / invisible
          mouse wheel = magnifier zoom
Keys:     Left / Right = previous / next frame | M = magnifier on / off
          Ctrl+S = save | Ctrl+Z = undo

Modules:
  file_reader.py : frame sources, pose JSON / meta / backup file I/O
  tool.py        : pose data model, YOLO estimation, skeleton operations, magnifier
  main.py        : this file
"""
import os
import queue
import threading
from collections import OrderedDict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk

import file_reader
import tool

# ==========================================
# UI settings
# ==========================================
DEFAULT_MODEL = r".\models\yolo26x-pose.pt"
HIT_RADIUS = 12       # mouse hit radius for picking a joint (screen pixels)
CACHE_SIZE = 60       # recent display frames kept in memory
RAW_CACHE_SIZE = 8    # recent full-resolution frames kept for the magnifier
LOUPE_SIZE = 240      # magnifier panel edge length (screen pixels)
PERSON_COLORS = ["#00ff00", "#ff8c00", "#ff00ff", "#00c8ff", "#ffff00"]   # limb color per person_i


def side_color(j):
    """Joint color: nose white, left side blue, right side red."""
    return "#ffffff" if j == 0 else ("#3fa9ff" if j % 2 else "#ff5c5c")


class PoseEditorApp:
    def __init__(self, root):
        self.root = root
        self.mode = tk.StringVar(value="video")
        self.show_var = tk.BooleanVar(value=True)
        self.names_var = tk.BooleanVar(value=True)
        self.mag_var = tk.BooleanVar(value=True)
        self.jump_var = tk.StringVar()
        self.font = ("Segoe UI", 9, "bold")

        self.src, self.src_path = None, None        # file_reader.FrameSource
        self.doc, self.json_path = None, None       # tool.PoseDoc
        self.k, self.img_w, self.img_h, self.scale = 0, 0, 0, 1.0
        self.cache, self.raw_cache = OrderedDict(), OrderedDict()
        self.photo, self.loupe_photo = None, None
        self.mag = tool.Magnifier(size=LOUPE_SIZE, zoom=4)
        self.mouse = None                           # last cursor position on the canvas
        self.sel, self.hint = None, ""
        self.model_path = DEFAULT_MODEL
        self.worker, self.q, self.stop_evt = None, queue.Queue(), threading.Event()

        self._build_ui()
        self._update_title()
        self.update_status()
        self.update_loupe()

    # ==========================================
    # UI layout
    # ==========================================
    def _build_ui(self):
        r1 = ttk.Frame(self.root, padding=(6, 6, 6, 0))
        r1.pack(fill="x")
        self.rb_video = ttk.Radiobutton(r1, text="Video", value="video", variable=self.mode)
        self.rb_folder = ttk.Radiobutton(r1, text="Folder", value="folder", variable=self.mode)
        self.btn_src = ttk.Button(r1, text="Open Source", command=self.open_source)
        self.btn_json = ttk.Button(r1, text="Open Pose JSON", command=self.open_json)
        self.btn_est = ttk.Button(r1, text="Estimate", command=self.toggle_estimate)
        self.btn_save = ttk.Button(r1, text="Save", command=self.save)
        self.rb_video.pack(side="left")
        self.rb_folder.pack(side="left", padx=(0, 10))
        for b in (self.btn_src, self.btn_json, self.btn_est, self.btn_save):
            b.pack(side="left", padx=2)
        ttk.Button(r1, text="Exit", command=self.quit).pack(side="right")

        r2 = ttk.Frame(self.root, padding=(6, 4))
        r2.pack(fill="x")
        ttk.Button(r2, text="◀ Prev", command=lambda: self.step(-1)).pack(side="left", padx=2)
        ttk.Button(r2, text="Next ▶", command=lambda: self.step(1)).pack(side="left", padx=2)
        ttk.Label(r2, text="Frame").pack(side="left", padx=(10, 2))
        ent = ttk.Entry(r2, width=7, textvariable=self.jump_var)
        ent.pack(side="left")
        ent.bind("<Return>", lambda e: self.jump())
        ttk.Button(r2, text="Go", command=self.jump).pack(side="left", padx=2)
        ttk.Button(r2, text="Next Suspicious", command=self.next_suspicious).pack(side="left", padx=2)
        ttk.Separator(r2, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Checkbutton(r2, text="Show Skeleton", variable=self.show_var,
                        command=self.redraw_pose).pack(side="left")
        ttk.Checkbutton(r2, text="Joint Names", variable=self.names_var,
                        command=self.redraw_pose).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(r2, text="Magnifier (M)", variable=self.mag_var,
                        command=self.update_loupe).pack(side="left", padx=(6, 0))
        ttk.Separator(r2, orient="vertical").pack(side="left", fill="y", padx=8)
        self.btn_add = ttk.Button(r2, text="Add Skeleton", command=self.add_skeleton)
        self.btn_del = ttk.Button(r2, text="Delete Skeleton", command=self.delete_skeleton)
        self.btn_main = ttk.Button(r2, text="Set as person_0", command=self.set_main)
        for b in (self.btn_add, self.btn_del, self.btn_main):
            b.pack(side="left", padx=2)
        self.lock_widgets = [self.rb_video, self.rb_folder, self.btn_src, self.btn_json,
                             self.btn_save, self.btn_add, self.btn_del, self.btn_main]

        # Image canvas in the middle; magnifier panel to its right, aligned to the image bottom
        view = ttk.Frame(self.root)
        view.pack(fill="x", padx=6)
        view.columnconfigure(0, weight=1)                                # left spacer
        view.columnconfigure(2, weight=1, minsize=LOUPE_SIZE + 12)       # keeps space when hidden
        self.canvas = tk.Canvas(view, bg="black", highlightthickness=0, width=960, height=540)
        self.canvas.grid(row=0, column=1, sticky="n")
        self.loupe = tk.Canvas(view, width=LOUPE_SIZE, height=LOUPE_SIZE, bg="#282828",
                               highlightthickness=0)
        self.loupe.grid(row=0, column=2, sticky="sw", padx=(8, 0))

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Leave>", self.on_leave)
        for cv in (self.canvas, self.loupe):
            cv.bind("<MouseWheel>", self.on_wheel)             # Windows / macOS
            cv.bind("<Button-4>", self.on_wheel)               # Linux wheel up
            cv.bind("<Button-5>", self.on_wheel)               # Linux wheel down

        bottom = ttk.Frame(self.root, padding=(6, 2, 6, 6))
        bottom.pack(fill="x")
        self.status = ttk.Label(bottom, anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        self.prog = ttk.Progressbar(bottom, length=220, mode="determinate")   # shown only while estimating

        self.root.bind("<Left>", self._key(lambda: self.step(-1)))
        self.root.bind("<Right>", self._key(lambda: self.step(1)))
        self.root.bind("<Key-m>", self._key(self.toggle_magnifier))
        self.root.bind("<Control-s>", lambda e: self.save())
        self.root.bind("<Control-z>", lambda e: self.undo_last())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.max_w = int(self.root.winfo_screenwidth() * 0.85) - LOUPE_SIZE - 20
        self.max_h = int(self.root.winfo_screenheight() * 0.68)

    def _key(self, action):
        def handler(e):
            if not isinstance(e.widget, tk.Entry):         # keys typed into the entry box are ignored
                action()
        return handler

    # ==========================================
    # State helpers
    # ==========================================
    def busy(self):
        return self.worker is not None

    def editable(self):
        return (self.src is not None and self.doc is not None
                and self.show_var.get() and not self.busy())

    def _can_edit(self, need_sel=False):
        if self.src is None:
            self._info("Please open a source first.")
            return False
        if self.busy():
            return False
        if self.doc is None:
            self._info("Please open a pose JSON first, or press \"Estimate\" to generate one with YOLO.")
            return False
        self.show_var.set(True)
        if need_sel and not self.sel:
            self._info("Click any joint of the skeleton first to select it.")
            return False
        return True

    def _info(self, text):
        messagebox.showinfo("Info", text, parent=self.root)

    def _update_title(self):
        name = os.path.basename(os.path.normpath(self.src_path)) if self.src_path else ""
        dirty = self.doc is not None and self.doc.dirty
        self.root.title("Pose Annotation Tool" + (f" - {name}" if name else "") + (" *" if dirty else ""))

    def update_status(self):
        if self.src is None:
            self.status.config(text="Choose \"Video\" or \"Folder\", then press \"Open Source\".")
            return
        parts = [f"Frame {self.k}" + (f" / {self.src.total}" if self.src.total else "")]
        if self.doc is None:
            parts.append("No pose JSON loaded: press \"Open Pose JSON\" or \"Estimate\"")
        else:
            fd = self.doc.frame(self.k)
            if fd is None:
                parts.append(f"frame_{self.k} not in pose JSON")
            elif not fd:
                parts.append("⚠ No person detected")
            else:
                n0 = tool.good_count(fd["person_0"])
                parts.append(f"{len(fd)} person(s) | person_0 reliable joints {n0}/17"
                             + (" ⚠" if n0 < tool.WEAK_TH else ""))
            if self.doc.is_edited(self.k):
                parts.append("✎ Manually edited")
            if self.sel:
                parts.append(f"Selected {self.sel}")
        if self.hint:
            parts.append(self.hint)
        self.status.config(text=" | ".join(parts))

    # ==========================================
    # Flow: open source / pose JSON
    # ==========================================
    def confirm_discard(self):
        if self.doc is None or not self.doc.dirty:
            return True
        ans = messagebox.askyesnocancel("Unsaved changes",
                                        "The pose data has unsaved changes. Save before continuing?",
                                        parent=self.root)
        if ans is None:
            return False
        if ans:
            self.save()
        return True

    def open_source(self):
        if self.busy() or not self.confirm_discard():
            return
        mode = self.mode.get()
        if mode == "video":
            path = filedialog.askopenfilename(
                title="Select a video",
                filetypes=[("Video", " ".join("*" + e for e in file_reader.VIDEO_EXT)),
                           ("All files", "*.*")])
        else:
            path = filedialog.askdirectory(title="Select a frame folder (jpg / png)")
        if not path:
            return
        try:
            src = file_reader.FrameSource(path, mode)
            first = src.get(1)
            if first is None:
                raise IOError("Cannot read the first frame")
        except Exception as e:
            messagebox.showerror("Open failed", str(e), parent=self.root)
            return

        if self.src:
            self.src.close()
        self.src, self.src_path = src, path
        self.doc = self.json_path = self.sel = None
        self.k = 0
        self.img_h, self.img_w = first.shape[:2]
        self.scale = min(self.max_w / self.img_w, self.max_h / self.img_h, 1.0)
        self.canvas.config(width=round(self.img_w * self.scale), height=round(self.img_h * self.scale))
        self.cache, self.raw_cache = OrderedDict(), OrderedDict()
        self._store_frame(1, first)
        if src.bad:
            messagebox.showwarning(
                "Frame numbering",
                f"Frame #{src.bad[0]} is named {src.bad[1]}; the file number does not match its sorted position.\n"
                "This tool uses the sorted position as frame_N. Make sure your training code reads frames the same way.",
                parent=self.root)

        jp = file_reader.find_json(path, mode)
        if jp:
            self.load_json(jp)
        self._update_title()
        self.show_frame(1)

    def open_json(self):
        if self.src is None:
            self._info("Please open a source first.")
            return
        if self.busy() or not self.confirm_discard():
            return
        base = self.src_path if self.src.mode == "folder" else os.path.dirname(self.src_path)
        jp = filedialog.askopenfilename(title="Select a pose JSON", initialdir=base,
                                        filetypes=[("Pose JSON", "*.json")])
        if jp and self.load_json(jp):
            self.show_frame(self.k or 1)

    def load_json(self, jp):
        try:
            pose, edits = file_reader.load_pose(jp)
        except Exception as e:
            messagebox.showerror("Failed to load pose JSON", f"{jp}\n{e}", parent=self.root)
            return False
        self.doc, self.json_path, self.sel = tool.PoseDoc(pose, edits), jp, None
        if self.src.total and len(pose) != self.src.total:
            messagebox.showwarning("Frame count mismatch",
                                   f"The pose JSON has {len(pose)} frames but the source has {self.src.total}. "
                                   "They may not belong together (video frame counts can be slightly off).",
                                   parent=self.root)
        self._update_title()
        return True

    # ==========================================
    # Frame cache and display
    # ==========================================
    @staticmethod
    def _put(cache, k, value, limit):
        cache[k] = value
        cache.move_to_end(k)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _store_frame(self, k, bgr):
        """Cache frame k: display-size RGB for the canvas, full-resolution RGB for the magnifier."""
        full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        disp = full if self.scale == 1.0 else cv2.resize(
            full, (round(self.img_w * self.scale), round(self.img_h * self.scale)),
            interpolation=cv2.INTER_AREA)
        self._put(self.cache, k, disp, CACHE_SIZE)
        self._put(self.raw_cache, k, full, RAW_CACHE_SIZE)
        return disp

    def show_frame(self, k):
        if self.src is None:
            return False
        rgb = self.cache.get(k)
        if rgb is None:
            img = self.src.get(k)
            if img is None:
                return False
            rgb = self._store_frame(k, img)
        else:
            self.cache.move_to_end(k)
        if self.doc is not None and self.doc.dragging:
            self.doc.end_drag()
        self.k = k
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("img")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags="img")
        self.canvas.tag_lower("img")
        self.jump_var.set(str(k))
        self.hint = ""
        fd = self.doc.frame(k) if self.doc else None
        if self.sel not in (fd or {}):
            self.sel = None
        self.redraw_pose()
        return True

    def _text(self, x, y, text, color, anchor="w", tags="pose"):
        for dx, fill in ((1, "black"), (0, color)):          # black shadow keeps text readable
            self.canvas.create_text(x + dx, y + dx, text=text, fill=fill, anchor=anchor,
                                    font=self.font, tags=tags)

    def redraw_pose(self):
        """Filled dot = reliable joint; hollow dot + dashed limb = unreliable or not yet placed."""
        c = self.canvas
        c.delete("pose")
        fd = self.doc.frame(self.k) if self.doc else None
        if fd and self.show_var.get():
            s = self.scale
            for pi, pk in enumerate(file_reader.person_keys(fd)):
                p, sel = fd[pk], pk == self.sel
                col = PERSON_COLORS[pi % len(PERSON_COLORS)]
                pts, vis = {}, {}
                for j in range(tool.NUM_KPTS):
                    bp = p["keypoints"].get(f"bonepoint_{j}")
                    if bp:
                        pts[j] = (bp["x"] * s, bp["y"] * s)
                        vis[j] = tool.is_visible(bp)
                for a, b in tool.COCO_SKELETON:               # every joint is always connected
                    if a in pts and b in pts:
                        opt = {} if vis[a] and vis[b] else {"dash": (4, 4)}
                        c.create_line(*pts[a], *pts[b], fill=col, width=3 if sel else 2, tags="pose", **opt)
                r = 6 if sel else 4
                for j, (x, y) in pts.items():
                    sc = side_color(j)
                    if vis[j]:
                        c.create_oval(x - r, y - r, x + r, y + r, fill=sc, outline="black", tags="pose")
                    else:
                        c.create_oval(x - r, y - r, x + r, y + r, fill="", outline=sc, width=2, tags="pose")
                    if sel and self.names_var.get():
                        self._text(x + r + 3, y - r - 3, tool.KPT_NAMES[j], sc)
                if pts:
                    x, y = min(pts.values(), key=lambda q: q[1])
                    bc = p.get("box_conf")
                    self._text(x, y - 20, pk + (f" {bc:.2f}" if bc is not None else ""), col, anchor="s")
        self.update_status()
        self.update_loupe()

    # ==========================================
    # Magnifier
    # ==========================================
        # ==========================================
    # Magnifier (fixed panel at the bottom-right of the image)
    # ==========================================
    def toggle_magnifier(self):
        self.mag_var.set(not self.mag_var.get())
        self.update_loupe()

    def _loupe_source(self):
        """Prefer the full-resolution frame; fall back to the display image if it was evicted."""
        raw = self.raw_cache.get(self.k)
        if raw is not None:
            return raw, 1.0
        return self.cache.get(self.k), self.scale

    def _loupe_text(self, x, y, text, anchor="nw"):
        for dx, fill in ((1, "black"), (0, "white")):        # black shadow keeps text readable
            self.loupe.create_text(x + dx, y + dx, text=text, fill=fill, anchor=anchor,
                                   justify="center", font=self.font)

    def update_loupe(self):
        if not self.mag_var.get():
            self.loupe.grid_remove()                         # column minsize keeps the layout still
            return
        self.loupe.grid()
        lp, size = self.loupe, self.mag.size
        lp.delete("all")

        img = None
        if self.src is not None and self.k and self.mouse:
            img, img_scale = self._loupe_source()
        if img is None:
            self._loupe_text(size / 2, size / 2, "Magnifier\nmove the mouse over the image", anchor="center")
        else:
            if self.doc is not None and self.doc.dragging:
                cx, cy = self.doc.drag_point()               # centre on the joint being dragged
            else:
                cx, cy = self.mouse[0] / self.scale, self.mouse[1] / self.scale
            fd = self.doc.frame(self.k) if (self.doc and self.show_var.get()) else None
            out = self.mag.render(img, img_scale, cx, cy, fd=fd, sel=self.sel,
                                  person_colors=PERSON_COLORS, joint_color=side_color)
            self.loupe_photo = ImageTk.PhotoImage(Image.fromarray(out))
            lp.create_image(0, 0, anchor="nw", image=self.loupe_photo)
            self._loupe_text(6, 6, f"x {cx:.1f}   y {cy:.1f}")
        self._loupe_text(6, size - 6, f"×{self.mag.zoom}", anchor="sw")
        lp.create_rectangle(1, 1, size - 1, size - 1, outline="#808080", width=2)

    def on_wheel(self, e):
        if not self.mag_var.get():
            return
        self.mag.step_zoom(1 if (e.num == 4 or getattr(e, "delta", 0) > 0) else -1)
        self.update_loupe()

    def on_leave(self, e):
        if not (self.doc and self.doc.dragging):
            self.mouse = None
            self.update_loupe()

    # ==========================================
    # Navigation
    # ==========================================
    def step(self, d):
        if self.src is None:
            return
        if not self.show_frame(self.k + d):
            self.hint = "Already at the first frame" if d < 0 else "Already at the last frame"
            self.update_status()

    def jump(self):
        try:
            k = int(self.jump_var.get())
        except ValueError:
            return
        if not self.show_frame(k):
            self.hint = f"Frame {k} does not exist"
            self.update_status()
        self.canvas.focus_set()

    def next_suspicious(self):
        if self.doc is None:
            return
        k = self.doc.next_suspicious(self.k)
        if k is None:
            self._info("No more suspicious frames after this one.")
        else:
            self.show_frame(k)

    # ==========================================
    # Mouse -> tool operations
    # ==========================================
    def _hit(self, e):
        if not self.editable():
            return None
        return self.doc.hit_test(self.k, e.x / self.scale, e.y / self.scale,
                                 HIT_RADIUS / self.scale, prefer=self.sel)

    def on_hover(self, e):
        self.mouse = (e.x, e.y)
        hit = self._hit(e) if not (self.doc and self.doc.dragging) else None
        self.canvas.config(cursor="fleur" if hit else "")
        self.update_loupe()

    def on_press(self, e):
        self.canvas.focus_set()
        self.mouse = (e.x, e.y)
        if not self.editable():
            return
        hit = self._hit(e)
        if hit is None:
            if self.sel:
                self.sel = None
                self.redraw_pose()
            return
        self.doc.begin_drag(self.k, *hit)
        self.sel = hit[0]
        self.redraw_pose()

    def on_drag(self, e):
        self.mouse = (e.x, e.y)
        if not (self.doc and self.doc.dragging):
            self.update_loupe()
            return
        self.doc.drag_to(e.x / self.scale, e.y / self.scale, self.img_w, self.img_h)
        self._update_title()
        self.redraw_pose()

    def on_release(self, e):
        if self.doc:
            self.doc.end_drag()
        self.update_loupe()

    def on_right(self, e):
        hit = self._hit(e)
        if hit is None:
            return
        self.doc.toggle_visible(self.k, *hit)
        self.sel = hit[0]
        self._update_title()
        self.redraw_pose()

    # ==========================================
    # Buttons -> tool operations
    # ==========================================
    def add_skeleton(self):
        if not self._can_edit():
            return
        self.sel = self.doc.add_skeleton(self.k, self.img_w, self.img_h)
        self._update_title()
        self.hint = f"Added {self.sel}: drag each hollow dot onto its joint (it turns solid once moved)"
        self.redraw_pose()

    def delete_skeleton(self):
        if not self._can_edit(need_sel=True):
            return
        self.doc.delete_person(self.k, self.sel)
        self.sel = None
        self._update_title()
        self.hint = "Deleted (Ctrl+Z to undo)"
        self.redraw_pose()

    def set_main(self):
        if not self._can_edit(need_sel=True):
            return
        self.sel = self.doc.set_main(self.k, self.sel)
        self._update_title()
        self.redraw_pose()

    def undo_last(self):
        if self.busy() or self.doc is None:
            return
        k = self.doc.undo()
        if k is None:
            return
        self._update_title()
        if k != self.k:
            self.show_frame(k)
        else:
            if self.sel not in (self.doc.frame(k) or {}):
                self.sel = None
            self.redraw_pose()

    # ==========================================
    # Flow: save
    # ==========================================
    def save(self):
        if self.busy() or self.doc is None:
            return
        self.doc.prepare_save()
        try:
            backup = file_reader.save_pose(self.json_path, self.doc.pose, self.doc.edited)
        except OSError as e:
            messagebox.showerror("Save failed", str(e), parent=self.root)
            return
        self.doc.mark_saved()
        self._update_title()
        self.hint = (f"Saved {os.path.basename(self.json_path)} "
                     f"(raw YOLO output kept in {os.path.basename(backup)})")
        self.update_status()

    # ==========================================
    # Flow: YOLO estimation (background thread)
    # ==========================================
    def toggle_estimate(self):
        if self.busy():
            self.stop_evt.set()
            self.btn_est.config(text="Stopping...", state="disabled")
            return
        if self.src is None:
            self._info("Please open a source first.")
            return
        if not self.confirm_discard():
            return
        jp = file_reader.default_json_path(self.src_path, self.src.mode)
        if os.path.exists(jp) and not messagebox.askyesno(
                "Overwrite?",
                f"{jp}\nalready exists. Re-estimating will overwrite it and discard manual edits. Continue?",
                parent=self.root):
            return
        if not os.path.isfile(self.model_path):
            mp = filedialog.askopenfilename(title="Select a YOLO pose model",
                                            filetypes=[("PyTorch model", "*.pt")])
            if not mp:
                return
            self.model_path = mp

        self.stop_evt.clear()
        self.worker = threading.Thread(target=self._estimate_thread, daemon=True,
                                       args=(self.src_path, self.src.mode, self.model_path, jp))
        self.worker.start()
        self.btn_est.config(text="Stop")
        for w in self.lock_widgets:
            w.state(["disabled"])
        self.prog.config(value=0, maximum=self.src.total or 1)
        self.prog.pack(side="right")
        self.root.after(100, self.poll_worker)

    def _estimate_thread(self, path, mode, model_path, json_path):
        """Runs in a background thread; talks to the UI only through self.q."""
        q, src = self.q, None
        try:
            q.put(("msg", "Loading YOLO model..."))
            src = file_reader.FrameSource(path, mode)        # separate reader from the UI's
            result = tool.estimate_pose(src, model_path,
                                        progress=lambda k, total: q.put(("progress", k, total)),
                                        should_stop=self.stop_evt.is_set)
            if result is None:
                q.put(("stopped",))
                return
            pose, stats = result
            file_reader.save_estimation(json_path, pose, {
                "source": os.path.abspath(path), "model": os.path.abspath(model_path),
                "imgsz": stats["imgsz"], "conf": tool.YOLO_CONF, "n_frames": len(pose),
                "person_0": "highest box confidence"})
            q.put(("done", json_path, pose, stats))
        except Exception as e:
            q.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            if src:
                src.close()

    def _finish_estimate(self):
        self.worker = None
        self.prog.pack_forget()
        self.btn_est.config(text="Estimate", state="normal")
        for w in self.lock_widgets:
            w.state(["!disabled"])

    def poll_worker(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "msg":
                    self.status.config(text=msg[1])
                elif msg[0] == "progress":
                    k, total = msg[1], msg[2]
                    self.prog.config(maximum=total or k, value=k)
                    self.status.config(text=f"Running YOLO... frame {k}" + (f" / {total}" if total else ""))
                elif msg[0] == "done":
                    _, jp, pose, stats = msg
                    self._finish_estimate()
                    self.doc, self.json_path, self.sel = tool.PoseDoc(pose), jp, None
                    self._update_title()
                    self.show_frame(1)
                    messagebox.showinfo(
                        "Estimation finished",
                        f"Written to {jp}\n{len(pose)} frames | no person: {stats['n_empty']} | "
                        f"person_0 with < {tool.WEAK_TH} reliable joints: {stats['n_weak']}\n"
                        "Use \"Next Suspicious\" to review them one by one.", parent=self.root)
                    return
                elif msg[0] == "stopped":
                    self._finish_estimate()
                    self.hint = "Estimation stopped; no pose JSON written"
                    self.update_status()
                    return
                elif msg[0] == "error":
                    self._finish_estimate()
                    self.update_status()
                    messagebox.showerror("Estimation failed", msg[1], parent=self.root)
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.poll_worker)

    # ==========================================
    # Flow: exit
    # ==========================================
    def quit(self):
        if self.busy():
            if not messagebox.askyesno("Estimation running", "YOLO is still running. Exit anyway?",
                                       parent=self.root):
                return
            self.stop_evt.set()
        elif not self.confirm_discard():
            return
        if self.src:
            self.src.close()
        self.root.destroy()


if __name__ == "__main__":
    try:                         # Windows high DPI: avoids blurry rendering and click offsets
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    PoseEditorApp(root)
    root.mainloop()