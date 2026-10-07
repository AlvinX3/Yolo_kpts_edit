"""
tool.py - pose data model and every skeleton operation.

  * YOLO pose estimation (output identical to bp_predict)
  * PoseDoc: joint hit-testing, dragging, visibility toggling,
    adding / deleting / reordering skeletons, undo, bp_Dis recomputation

No UI code here. All coordinates are in original-image pixels.
"""
import copy
import math

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