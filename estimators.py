"""
estimators.py - pose estimation backends for the Pose Annotation Tool.

  * Model discovery: one sub-folder of models/ per framework
        models/yolo/*.pt               YOLO pose (also any *.pt directly under models/, legacy)
        models/vitpose/<name>/         ViTPose HuggingFace folder (config.json + model.safetensors)
        models/openpose/               reserved, not implemented yet
  * estimate_pose(): runs the chosen backend on every frame; the output format is identical
    for every backend (COCO 17 keypoints, person_0 = highest box confidence)

No UI code here. Heavy libraries (ultralytics / torch / transformers) are imported lazily.
"""
import os
import glob
import json
import math
from dataclasses import dataclass

from file_reader import natural_key
from tool import bp_displacement, good_count, WEAK_TH

YOLO_CONF = 0.25      # person box confidence threshold (same default as bp_predict)


@dataclass(frozen=True)
class ModelInfo:
    backend: str      # key of BACKENDS
    name: str         # display name
    path: str         # .pt file (YOLO) or model folder (ViTPose)


# ==========================================
# YOLO pose
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


class YoloBackend:
    key, label, needs_detector = "yolo", "YOLO", False

    @staticmethod
    def scan(models_dir):
        files = glob.glob(os.path.join(models_dir, "yolo", "*.pt"))
        files += glob.glob(os.path.join(models_dir, "*.pt"))           # legacy layout
        files.sort(key=lambda p: natural_key(os.path.basename(p)))
        return [ModelInfo("yolo", os.path.basename(p), p) for p in files]

    @staticmethod
    def load(model, detector, conf):
        from ultralytics import YOLO
        net = YOLO(model.path)

        def predict(img, imgsz):
            return extract_persons(net.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0])
        return predict


# ==========================================
# ViTPose (top-down: YOLO person boxes -> ViTPose keypoints)
# ==========================================
class VitPoseBackend:
    key, label, needs_detector = "vitpose", "ViTPose", True

    @staticmethod
    def scan(models_dir):
        out = []
        for d in sorted(glob.glob(os.path.join(models_dir, "vitpose", "*", "")),
                        key=lambda p: natural_key(os.path.basename(os.path.normpath(p)))):
            try:
                with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
                    if json.load(f).get("model_type") != "vitpose":
                        continue
            except (OSError, ValueError):
                continue
            d = os.path.normpath(d)
            out.append(ModelInfo("vitpose", os.path.basename(d), d))
        return out

    @staticmethod
    def load(model, detector, conf):
        import cv2
        import torch
        from ultralytics import YOLO
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        det = YOLO(detector.path)
        processor = AutoProcessor.from_pretrained(model.path)
        net = VitPoseForPoseEstimation.from_pretrained(model.path).to(device).eval()
        # ViTPose++ (mixture of experts) needs to know which dataset head to use; 0 = COCO
        moe = getattr(net.config.backbone_config, "num_experts", 1) > 1

        def predict(img, imgsz):
            r = det.predict(img, imgsz=imgsz, conf=conf, classes=[0], verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return []
            xyxy = r.boxes.xyxy.cpu().numpy()
            bconf = r.boxes.conf.cpu().numpy().tolist()
            boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in xyxy.tolist()]   # COCO x, y, w, h
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs = processor(rgb, boxes=[boxes], return_tensors="pt").to(device)
            if moe:
                inputs["dataset_index"] = torch.zeros(len(boxes), dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = net(**inputs)
            res = processor.post_process_pose_estimation(outputs, boxes=[boxes])[0]
            persons = []
            for bc, p in zip(bconf, res):
                pts, sc = p["keypoints"].tolist(), p["scores"].tolist()
                kp = {f"bonepoint_{i}": {"x": round(x, 2), "y": round(y, 2),
                                         "Confidence": round(min(max(c, 0.0), 1.0), 4)}
                      for i, ((x, y), c) in enumerate(zip(pts, sc))}
                persons.append((bc, kp))
            persons.sort(key=lambda x: -x[0])
            return persons
        return predict


# ==========================================
# OpenPose (reserved)
# ==========================================
class OpenPoseBackend:
    """Placeholder. To implement: scan models/openpose/, and in load() return a predict(img, imgsz)
    that maps OpenPose output (COCO-18 / BODY_25, which add a neck joint and use another order)
    onto the 17 COCO keypoints used here, then add this class to BACKENDS."""
    key, label, needs_detector = "openpose", "OpenPose", False

    @staticmethod
    def scan(models_dir):
        return []

    @staticmethod
    def load(model, detector, conf):
        raise NotImplementedError("OpenPose backend is not implemented yet")


BACKENDS = {b.key: b for b in (YoloBackend, VitPoseBackend)}


def scan_models(models_dir):
    """Every usable model under models_dir, grouped by backend in BACKENDS order."""
    return [m for b in BACKENDS.values() for m in b.scan(models_dir)]


# ==========================================
# Estimation loop (shared by every backend)
# ==========================================
def estimate_pose(src, model, detector=None, conf=YOLO_CONF, progress=None, should_stop=None):
    """Run `model` (a ModelInfo) on every frame of `src` (a file_reader.FrameSource).
    detector: YOLO ModelInfo that supplies person boxes, for backends with needs_detector.
    progress(k, total) is called after each frame; should_stop() is polled before each frame.
    Returns (pose, stats), or None if stopped. Output format is identical to bp_predict."""
    backend = BACKENDS[model.backend]
    if backend.needs_detector and detector is None:
        raise ValueError(f"{backend.label} needs a YOLO person detector")
    predict = backend.load(model, detector, conf)
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
        fd, cur = {}, {}
        for i, (bc, kp) in enumerate(predict(img, imgsz)):
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
