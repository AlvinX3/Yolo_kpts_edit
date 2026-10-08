"""
file_reader.py - all file I/O for the Pose Annotation Tool.

  * Frame sources: video file or image folder, 1-based frame index (matches frame_N)
  * Pose JSON / meta JSON / backup file read & write
  * Pose JSON format normalization and ordering

This module has no UI code and no skeleton-editing logic.
"""
import os
import re
import glob
import json
import shutil

import cv2
import numpy as np

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
IMAGE_EXT = (".png", ".jpg", ".jpeg")


# ==========================================
# Key helpers (shared with tool.py and main.py)
# ==========================================
def natural_key(s):
    """Natural sort key: 'frame_2' comes before 'frame_10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def idx_of(key):
    """'frame_12' -> 12, 'person_3' -> 3"""
    return int(key.rsplit("_", 1)[1])


def person_keys(fd):
    """person_* keys of one frame, sorted as person_0, person_1, ..."""
    return sorted((k for k in (fd or {}) if k.startswith("person_")), key=idx_of)


# ==========================================
# Frame sources
# ==========================================
def imread_unicode(path):
    """cv2.imread fails on non-ASCII paths on Windows; use imdecode instead."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def list_frames(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in IMAGE_EXT]
    return sorted(files, key=lambda f: natural_key(os.path.basename(f)))


class FrameSource:
    """Unified interface for a video file or an image folder.
    get(k) returns frame k as a BGR image (1-based, matches frame_k), or None if unavailable."""

    def __init__(self, path, mode):
        self.path, self.mode = path, mode
        self.cap, self.files, self.bad = None, None, None
        if mode == "folder":
            self.files = list_frames(path)
            if not self.files:
                raise FileNotFoundError(f"No jpg / png frames found in {path}")
            self.total = len(self.files)
            for k, f in enumerate(self.files, 1):          # same numbering check as bp_predict
                m = re.search(r'(\d+)\.\w+$', os.path.basename(f))
                if not m or int(m.group(1)) != k:
                    self.bad = (k, os.path.basename(f))
                    break
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video: {path}")
            self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            self.next_k = 1

    def get(self, k):
        if k < 1:
            return None
        if self.files is not None:
            return imread_unicode(self.files[k - 1]) if k <= self.total else None
        if k != self.next_k:                               # only seek for non-sequential reads
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k - 1)
        ok, img = self.cap.read()
        self.next_k = k + 1 if ok else -1
        return img if ok else None

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ==========================================
# Paths
# ==========================================
def default_json_path(path, mode):
    """Same naming convention as bp_predict."""
    if mode == "folder":
        name = os.path.basename(os.path.normpath(path))
        return os.path.join(path, "bonePoint", name + "_bp.json")
    return os.path.splitext(path)[0] + "_bp.json"


def find_json(path, mode):
    """Look for an existing pose JSON next to the source; returns a path or None."""
    cands = [default_json_path(path, mode)]
    if mode == "folder":
        cands += sorted(glob.glob(os.path.join(path, "bonePoint", "*_bp.json")))
        cands += sorted(glob.glob(os.path.join(path, "*_bp.json")))
    return next((c for c in cands if os.path.isfile(c)), None)


def meta_path(json_path):
    return os.path.splitext(json_path)[0] + "_meta.json"


def backup_path(json_path):
    """Raw YOLO output backup. Does not match *_bp.json, so it is never auto-loaded."""
    return os.path.splitext(json_path)[0] + "_yolo.json"

def find_model(folder):
    """First *.pt in `folder` in natural name order (yolo8 < yolo11 < yolo26), or None."""
    models = glob.glob(os.path.join(folder, "*.pt"))
    models.sort(key=lambda p: natural_key(os.path.basename(p)))
    return models[0] if models else None

# ==========================================
# JSON read / write
# ==========================================
def _write_json_atomic(path, data, indent):
    """Write to a temp file first, then replace, so a crash never corrupts the original."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def read_meta(json_path):
    try:
        with open(meta_path(json_path), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_meta(json_path, updates, merge=True):
    meta = read_meta(json_path) if merge else {}
    meta.update(updates)
    _write_json_atomic(meta_path(json_path), meta, 2)


def normalize_pose(pose):
    """Accept the legacy format (bonepoints directly under person_i, no 'keypoints' layer)."""
    for fd in pose.values():
        if isinstance(fd, dict):
            for pk in person_keys(fd):
                if "keypoints" not in fd[pk]:
                    fd[pk] = {"keypoints": fd[pk], "bp_Dis": {}, "box_conf": 1.0}
    return pose


def ordered_pose(pose):
    """Sort as frame_1..frame_N and person_0..person_n; fill missing frames with {}."""
    last = max((idx_of(k) for k in pose if k.startswith("frame_")), default=0)
    out = {}
    for k in range(1, last + 1):
        fd = pose.get(f"frame_{k}") or {}
        out[f"frame_{k}"] = {pk: fd[pk] for pk in person_keys(fd)}
    return out


def load_pose(json_path):
    """Load a pose JSON. Returns (pose, manual_edits). Raises ValueError if it is not a pose JSON."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not any(k.startswith("frame_") for k in data):
        raise ValueError("No frame_N keys found; this is not a pose JSON")
    pose = ordered_pose(normalize_pose(data))
    return pose, read_meta(json_path).get("manual_edits", [])


def _taken(path):
    """True if this pose JSON name is already used (the JSON itself, its meta or its backup)."""
    return any(os.path.exists(p) for p in (path, meta_path(path), backup_path(path)))


def unique_json_path(json_path):
    """json_path if its name is free, else the first free name_v2_bp.json, name_v3_bp.json, ...
    Keeps the *_bp.json suffix so find_json still recognizes the file."""
    if not _taken(json_path):
        return json_path
    root, ext = os.path.splitext(json_path)
    suffix = "_bp" if root.endswith("_bp") else ""
    stem = re.sub(r"_v\d+$", "", root[:len(root) - len(suffix)])
    n = 2
    while _taken(f"{stem}_v{n}{suffix}{ext}"):
        n += 1
    return f"{stem}_v{n}{suffix}{ext}"


def save_pose(json_path, pose, manual_edits):
    """Overwrite json_path with the edited pose.
    Only used for a file created by save_pose_as in the current session, never for an older file."""
    _write_json_atomic(json_path, pose, 4)
    write_meta(json_path, {"n_frames": len(pose),
                           "manual_edits": sorted(manual_edits, key=idx_of)})


def save_pose_as(new_path, old_path, pose, manual_edits):
    """Save the edited pose to a NEW file; old_path is left untouched.
    Raises FileExistsError if new_path (or its meta / backup) already exists.
    The raw YOLO output and the meta info of old_path are carried over to the new file."""
    if _taken(new_path):
        raise FileExistsError(f"{new_path} (or its _meta / _yolo file) already exists")
    raw = None
    if old_path:
        old_backup = backup_path(old_path)
        raw = old_backup if os.path.isfile(old_backup) else (old_path if os.path.isfile(old_path) else None)
    if raw:
        shutil.copy2(raw, backup_path(new_path))
    meta = read_meta(old_path) if old_path else {}
    meta.update({"n_frames": len(pose),
                 "manual_edits": sorted(manual_edits, key=idx_of),
                 "saved_from": os.path.abspath(old_path) if old_path else None})
    _write_json_atomic(new_path, pose, 4)
    _write_json_atomic(meta_path(new_path), meta, 2)


def save_estimation(json_path, pose, meta):
    """Save a fresh YOLO result. The caller passes a free name from unique_json_path,
    so nothing existing is overwritten."""
    _write_json_atomic(json_path, pose, 4)
    shutil.copy2(json_path, backup_path(json_path))
    write_meta(json_path, meta, merge=False)