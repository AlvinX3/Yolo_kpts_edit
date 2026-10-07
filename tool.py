"""
tool.py - pose data model and every skeleton operation.

  * YOLO pose estimation (output identical to bp_predict)
  * PoseDoc: joint hit-testing, dragging, visibility toggling,
    adding / deleting / reordering skeletons, undo, bp_Dis recomputation

No UI code here. All coordinates are in original-image pixels.
"""
import copy
import math

import cv2
import numpy as np

from file_reader import idx_of, person_keys, ordered_pose


# ==========================================
# Skeleton definition and thresholds
# ==========================================
NUM_KPTS = 17
# COCO 17-keypoint skeleton, identical to yolo26x-pose (ultralytics)
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]
KPT_NAMES = ["nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder",
             "L elbow", "R elbow", "L wrist", "R wrist", "L hip", "R hip",
             "L knee", "R knee", "L ankle", "R ankle"]
# Standing template for add_skeleton (x, y): height = 1, top of head at y = 0.
# The person faces the camera, so their LEFT side appears on the RIGHT of the image.
TEMPLATE = [(0, .08), (.03, .06), (-.03, .06), (.06, .08), (-.06, .08),
            (.13, .22), (-.13, .22), (.18, .38), (-.18, .38), (.20, .52), (-.20, .52),
            (.09, .52), (-.09, .52), (.10, .73), (-.10, .73), (.10, .95), (-.10, .95)]

CONF_TH = 0.5         # joint "reliable" threshold (same as training gt_mask)
WEAK_TH = 12          # frames where person_0 has fewer reliable joints are suspicious
YOLO_CONF = 0.25      # person box confidence threshold (same default as bp_predict)
UNDO_LIMIT = 200


def is_visible(bp):
    return bp.get("Confidence", 0) > CONF_TH


def good_count(person):
    return sum(1 for v in person["keypoints"].values() if is_visible(v))


def bp_displacement(last_bp, now_bp):
    """Same as bp_Displacement in the original main.py."""
    out = {}
    for key, now in now_bp.items():
        if key in last_bp:
            dx, dy = now["x"] - last_bp[key]["x"], now["y"] - last_bp[key]["y"]
            out[key] = round(math.sqrt(dx * dx + dy * dy), 2)
        else:
            out[key] = None
    return out


def recompute_bp_dis(pose):
    """Recompute every bp_Dis (pose must be ordered and contiguous), using the same rule
    as bp_predict: compare with the same person_i in the previous frame."""
    prev = {}
    for k in range(1, len(pose) + 1):
        cur = {}
        for pk, p in pose[f"frame_{k}"].items():
            p["bp_Dis"] = bp_displacement(prev[pk], p["keypoints"]) if pk in prev else {}
            cur[pk] = p["keypoints"]
        prev = cur


# ==========================================
# YOLO estimation
# ==========================================
def extract_persons(r):
    """Same as _extract_persons in the original main.py:
    returns [(box_conf, keypoints)], highest box confidence first."""
    if r.keypoints is None or len(r.keypoints) == 0:
        return []
    kpts = r.keypoints.xy.cpu().numpy().tolist()
    kconf = (r.keypoints.conf.cpu().numpy().tolist() if r.keypoints.conf is not None
             else [[1.0] * len(k) for k in kpts])
    bconf = (r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None and r.boxes.conf is not None
             else [1.0] * len(kpts))
    persons = []
    for pk, pc, bc in zip(kpts, kconf, bconf):
        kp = {f"bonepoint_{i}": {"x": round(pt[0], 2), "y": round(pt[1], 2), "Confidence": round(c, 4)}
              for i, (pt, c) in enumerate(zip(pk, pc))}
        persons.append((bc, kp))
    persons.sort(key=lambda x: -x[0])
    return persons


def estimate_pose(src, model_path, conf=YOLO_CONF, progress=None, should_stop=None):
    """Run YOLO pose on every frame of `src` (a file_reader.FrameSource).
    progress(k, total) is called after each frame; should_stop() is polled before each frame.
    Returns (pose, stats), or None if stopped. Output format is identical to bp_predict."""
    from ultralytics import YOLO          # imported lazily: only needed when estimating
    model = YOLO(model_path)
    pose, prev, imgsz = {}, {}, None
    n_empty = n_weak = k = 0
    while True:
        if should_stop and should_stop():
            return None
        img = src.get(k + 1)
        if img is None:
            break
        k += 1
        if imgsz is None:
            imgsz = math.ceil(max(img.shape[:2]) / 32) * 32
        r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        fd, cur = {}, {}
        for i, (bc, kp) in enumerate(extract_persons(r)):
            pk = f"person_{i}"
            fd[pk] = {"keypoints": kp,
                      "bp_Dis": bp_displacement(prev[pk], kp) if pk in prev else {},
                      "box_conf": round(bc, 4)}
            cur[pk] = kp
        pose[f"frame_{k}"] = fd
        prev = cur
        n_empty += int(not fd)
        n_weak += int(bool(fd) and good_count(fd["person_0"]) < WEAK_TH)
        if progress:
            progress(k, src.total)
    if k == 0:
        raise RuntimeError("The source has no readable frames")
    return pose, {"imgsz": imgsz, "n_empty": n_empty, "n_weak": n_weak}


# ==========================================
# Pose document: data + editing operations
# ==========================================
class PoseDoc:
    """In-memory pose JSON plus editing state (manual-edit record, undo stack, dirty flag)."""

    def __init__(self, pose, manual_edits=()):
        self.pose = pose
        self.edited = set(manual_edits)
        self.dirty = False
        self._undo = []
        self._drag = None          # [frame k, person key, joint index, moved?]

    # ----- queries -----
    @property
    def n_frames(self):
        return max((idx_of(k) for k in self.pose), default=0)

    def frame(self, k):
        return self.pose.get(f"frame_{k}")

    def is_edited(self, k):
        return f"frame_{k}" in self.edited

    def is_suspicious(self, k):
        """No person detected, or person_0 has too few reliable joints."""
        fd = self.frame(k) or {}
        return not fd or good_count(fd["person_0"]) < WEAK_TH

    def next_suspicious(self, from_k):
        """Next suspicious frame after from_k (manually edited frames are skipped), or None."""
        for k in range(from_k + 1, self.n_frames + 1):
            if not self.is_edited(k) and self.is_suspicious(k):
                return k
        return None

    def hit_test(self, k, x, y, radius, prefer=None):
        """Nearest joint to (x, y) within `radius`; joints of person `prefer` win near-ties.
        Returns (person key, joint index) or None."""
        fd = self.frame(k)
        best, best_d = None, radius
        for pk in person_keys(fd):
            bonus = radius / 4 if pk == prefer else 0
            for j in range(NUM_KPTS):
                bp = fd[pk]["keypoints"].get(f"bonepoint_{j}")
                if bp:
                    d = math.hypot(bp["x"] - x, bp["y"] - y) - bonus
                    if d <= best_d:
                        best, best_d = (pk, j), d
        return best

    # ----- undo -----
    def _snapshot(self, k):
        key = f"frame_{k}"
        self._undo.append((key, copy.deepcopy(self.pose.get(key)), key in self.edited))
        if len(self._undo) > UNDO_LIMIT:
            self._undo.pop(0)

    def _touch(self, k):
        self.edited.add(f"frame_{k}")
        self.dirty = True

    def undo(self):
        """Revert the last edit. Returns the frame number it belonged to, or None."""
        if not self._undo:
            return None
        key, fd, was_edited = self._undo.pop()
        if fd is None:
            self.pose.pop(key, None)
        else:
            self.pose[key] = fd
        if not was_edited:
            self.edited.discard(key)
        self.dirty = True
        return idx_of(key)

    # ----- move a joint (drag) -----
    @property
    def dragging(self):
        return self._drag is not None

    def begin_drag(self, k, pk, j):
        self._snapshot(k)
        self._drag = [k, pk, j, False]
    
    def drag_to(self, x, y, img_w, img_h):
        """Move the dragged joint to (x, y), clamped to the image. A hand-placed joint gets Confidence 1.0."""
        if not self._drag:
            return
        k, pk, j, moved = self._drag
        bp = self.frame(k)[pk]["keypoints"][f"bonepoint_{j}"]
        bp["x"] = round(min(max(x, 0), img_w - 1), 2)
        bp["y"] = round(min(max(y, 0), img_h - 1), 2)
        bp["Confidence"] = 1.0
        if not moved:
            self._drag[3] = True
            self._touch(k)
    
    def drag_point(self):
        """Current (x, y) of the joint being dragged, or None. Used to centre the magnifier."""
        if not self._drag:
            return None
        k, pk, j, _ = self._drag
        bp = self.frame(k)[pk]["keypoints"][f"bonepoint_{j}"]
        return bp["x"], bp["y"]

    def end_drag(self):
        """Finish dragging. A click without movement is not recorded as an edit."""
        if self._drag and not self._drag[3]:
            self._undo.pop()
        self._drag = None

    # ----- other edits -----
    def toggle_visible(self, k, pk, j):
        """Toggle a joint between visible (Confidence 1.0) and invisible (0.0)."""
        self._snapshot(k)
        bp = self.frame(k)[pk]["keypoints"][f"bonepoint_{j}"]
        bp["Confidence"] = 0.0 if is_visible(bp) else 1.0
        self._touch(k)

    def add_skeleton(self, k, img_w, img_h):
        """Add a template skeleton in the image center. All joints start at Confidence 0.0
        (= not placed yet) and become 1.0 once dragged. Returns the new person key."""
        self._snapshot(k)
        fd = self.pose.setdefault(f"frame_{k}", {})
        h = img_h * 0.6
        cx, top = img_w / 2, (img_h - h) / 2
        kp = {f"bonepoint_{j}": {"x": round(cx + tx * h, 2), "y": round(top + ty * h, 2), "Confidence": 0.0}
              for j, (tx, ty) in enumerate(TEMPLATE)}
        pk = f"person_{len(person_keys(fd))}"
        fd[pk] = {"keypoints": kp, "bp_Dis": {}, "box_conf": 1.0}
        self._touch(k)
        return pk

    def delete_person(self, k, pk):
        self._snapshot(k)
        fd = self.frame(k)
        self._reorder(fd, [p for p in person_keys(fd) if p != pk])
        self._touch(k)

    def set_main(self, k, pk):
        """Move person `pk` to person_0 (training only uses person_0). Returns 'person_0'."""
        if pk != "person_0":
            self._snapshot(k)
            fd = self.frame(k)
            self._reorder(fd, [pk] + [p for p in person_keys(fd) if p != pk])
            self._touch(k)
        return "person_0"

    @staticmethod
    def _reorder(fd, order):
        """Rebuild the frame with persons in `order`, renumbered person_0, person_1, ..."""
        items = [fd[p] for p in order]
        fd.clear()
        for i, p in enumerate(items):
            fd[f"person_{i}"] = p

    # ----- save support -----
    def prepare_save(self):
        """Order frames / persons and recompute bp_Dis before writing."""
        self.pose = ordered_pose(self.pose)
        recompute_bp_dis(self.pose)

    def mark_saved(self):
        self.dirty = False

# ==========================================
# Magnifier (loupe)
# ==========================================
_SHIFT = 4                     # sub-pixel precision for OpenCV drawing (1/16 px)
_F = 1 << _SHIFT


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class Magnifier:
    """Loupe for precise joint placement.

    Crops a square region around a point of the frame, enlarges it, and draws the skeleton
    and a crosshair on it. Pure numpy / OpenCV; the caller only has to display the result.

    Coordinates are original-image pixels in the same convention as the pose JSON and the
    main canvas (pixel i covers [i, i+1)). The source image may be the full-resolution frame
    (img_scale = 1.0) or a downscaled display image (img_scale < 1.0)."""

    ZOOM_LEVELS = (2, 3, 4, 6, 8, 12)      # loupe pixels per original-image pixel

    def __init__(self, size=240, zoom=4):
        self.size = size
        self.zoom = zoom
        self.cx = self.cy = 0.0

    # ----- zoom -----
    def step_zoom(self, d):
        """d = +1 zoom in one level, -1 zoom out one level."""
        lv = self.ZOOM_LEVELS
        i = min(range(len(lv)), key=lambda n: abs(lv[n] - self.zoom))
        self.zoom = lv[max(0, min(len(lv) - 1, i + d))]

    # ----- coordinate mapping -----
    def to_loupe(self, x, y):
        """Original-image point -> loupe point (continuous coordinates)."""
        return (self.size / 2 + (x - self.cx) * self.zoom,
                self.size / 2 + (y - self.cy) * self.zoom)

    def to_image(self, u, v):
        """Loupe point -> original-image point."""
        return (self.cx + (u - self.size / 2) / self.zoom,
                self.cy + (v - self.size / 2) / self.zoom)

    @staticmethod
    def _fix(p):
        """Continuous loupe point -> OpenCV fixed-point pixel coordinates (pixel centres at integers)."""
        return int(round((p[0] - 0.5) * _F)), int(round((p[1] - 0.5) * _F))

    # ----- rendering -----
    def render(self, img, img_scale, cx, cy, fd=None, sel=None,
               person_colors=("#00ff00",), joint_color=None):
        """Return a (size x size) RGB image centred on original-image point (cx, cy).
        img           : RGB frame
        img_scale     : img pixels per original-image pixel (1.0 for a full-resolution frame)
        fd            : frame data from the pose JSON (None = image only)
        sel           : selected person key (drawn thicker)
        person_colors : limb color (hex) per person_i
        joint_color   : function j -> hex color for joint j (None = white)"""
        self.cx, self.cy = cx, cy
        f = img_scale
        a = f / self.zoom                                     # src pixels per loupe pixel
        bx = f * cx + f * (0.5 - self.size / 2) / self.zoom - 0.5
        by = f * cy + f * (0.5 - self.size / 2) / self.zoom - 0.5
        m = np.float32([[a, 0, bx], [0, a, by]])
        interp = cv2.INTER_NEAREST if self.zoom / f >= 3 else cv2.INTER_LINEAR   # crisp pixels at high zoom
        out = cv2.warpAffine(img, m, (self.size, self.size), flags=interp | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
        if fd:
            self._draw_pose(out, fd, sel, person_colors, joint_color)
        self._draw_crosshair(out)
        return out

    def _line(self, out, p, q, col, th, dashed):
        if not dashed:
            cv2.line(out, self._fix(p), self._fix(q), col, th, cv2.LINE_AA, _SHIFT)
            return
        ok, p1, p2 = cv2.clipLine((0, 0, self.size, self.size),
                                  (int(round(p[0])), int(round(p[1]))),
                                  (int(round(q[0])), int(round(q[1]))))
        if not ok:
            return
        (x1, y1), (x2, y2) = p1, p2
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1:
            return
        for s in range(0, int(length), 8):                    # 5 px dash, 3 px gap
            t0, t1 = s / length, min(s + 5, length) / length
            a = (x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0)
            b = (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)
            cv2.line(out, self._fix(a), self._fix(b), col, th, cv2.LINE_AA, _SHIFT)

    def _draw_pose(self, out, fd, sel, person_colors, joint_color):
        for pi, pk in enumerate(person_keys(fd)):
            kp = fd[pk]["keypoints"]
            col = hex_to_rgb(person_colors[pi % len(person_colors)])
            is_sel = pk == sel
            pts, vis = {}, {}
            for j in range(NUM_KPTS):
                bp = kp.get(f"bonepoint_{j}")
                if bp:
                    pts[j] = self.to_loupe(bp["x"], bp["y"])
                    vis[j] = is_visible(bp)
            for a, b in COCO_SKELETON:
                if a in pts and b in pts:
                    self._line(out, pts[a], pts[b], col, 2 if is_sel else 1, not (vis[a] and vis[b]))
            # Joints are hollow rings so the pixels under the joint stay visible
            r = (7 if is_sel else 6) * _F
            for j, p in pts.items():
                jc = hex_to_rgb(joint_color(j)) if joint_color else (255, 255, 255)
                cv2.circle(out, self._fix(p), r, (0, 0, 0), 4 if vis[j] else 3, cv2.LINE_AA, _SHIFT)
                cv2.circle(out, self._fix(p), r, jc, 2 if vis[j] else 1, cv2.LINE_AA, _SHIFT)

    def _draw_crosshair(self, out):
        c = self.size / 2
        for p, q in (((c - 18, c), (c - 6, c)), ((c + 6, c), (c + 18, c)),
                     ((c, c - 18), (c, c - 6)), ((c, c + 6), (c, c + 18))):
            cv2.line(out, self._fix(p), self._fix(q), (0, 0, 0), 3, cv2.LINE_AA, _SHIFT)
            cv2.line(out, self._fix(p), self._fix(q), (255, 255, 255), 1, cv2.LINE_AA, _SHIFT)