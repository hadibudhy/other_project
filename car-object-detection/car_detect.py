#!/usr/bin/env python3
"""
YOLOv5 Vehicle Detection (Fast CLI, ONNX + OpenCV DNN)

- Detects vehicles (car/truck/bus/motorcycle by default) in images, folders, or videos
- Uses ONNX export of YOLOv5 and OpenCV DNN (no PyTorch at inference)
- Saves annotated media and JSON results
- Efficient: single model init, NumPy NMS, minimal copies

Examples:
  python car_detect.py detect --model yolov5s.onnx --source img.jpg --classes car --save --json out.json
  python car_detect.py detect --model yolov5s.onnx --source /path/to/folder --classes car truck bus --save
  python car_detect.py detect --model yolov5s.onnx --source vid.mp4 --classes car motorcycle --save --view

Notes:
- Works with YOLOv5 ONNX export that outputs (1, N, 85) [x,y,w,h,obj,80 cls...] (COCO format).
- Default class map uses COCO indices: car=2, motorcycle=3, bus=5, truck=7 (bicycle=1 if you need it).
- If your model has different class ordering, pass custom --class-ids.
"""
import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

# ---------------------------- Utils ----------------------------
def letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize+pad to square while keeping aspect ratio (YOLOv5 style)."""
    h0, w0 = img.shape[:2]
    r = new_shape / max(h0, w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(img_resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (dw, dh)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_xyxy(boxes[i:i+1], boxes[order[1:]]).reshape(-1)
        inds = np.where(ious <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------------------- Data ----------------------------
COCO_DEFAULT = {
    "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "train": 6, "truck": 7
}
VEHICLE_DEFAULT = ["car", "motorcycle", "bus", "truck"]

@dataclass
class Det:
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    conf: float
    cls: int
    label: str

# ---------------------------- Model ----------------------------
class YoloV5ONNX:
    def __init__(self, path: str, size: int = 640, conf_thr: float = 0.25, iou_thr: float = 0.45,
                 keep_cls: Optional[List[int]] = None, labels: Optional[Dict[int, str]] = None):
        self.net = cv2.dnn.readNetFromONNX(path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.size = size
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.keep_cls = keep_cls
        self.labels = labels or {}

    def infer(self, img_bgr: np.ndarray) -> List[Det]:
        img, r, (dw, dh) = letterbox(img_bgr, self.size)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        pred = np.squeeze(out, axis=0)  # (N, 85)

        # objectness x class conf
        obj = pred[:, 4]
        cls_scores = pred[:, 5:]
        cls_ids = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(pred.shape[0]), cls_ids]
        conf = obj * cls_conf

        mask = conf > self.conf_thr
        if self.keep_cls is not None:
            mask = mask & np.isin(cls_ids, np.array(self.keep_cls))

        pred = pred[mask]
        conf = conf[mask]
        cls_ids = cls_ids[mask]

        if pred.size == 0:
            return []

        # xywh -> xyxy in padded coords
        xywh = pred[:, :4]
        xyxy = np.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

        # NMS
        keep = nms(xyxy, conf, self.iou_thr)
        xyxy = xyxy[keep]
        conf = conf[keep]
        cls_ids = cls_ids[keep]

        # map back to original image coords
        gain = self.size / max(img_bgr.shape[:2])
        boxes = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls_ids):
            # remove pad then scale
            x1 = (x1 - dw) / gain
            y1 = (y1 - dh) / gain
            x2 = (x2 - dw) / gain
            y2 = (y2 - dh) / gain
            x1 = int(np.clip(x1, 0, img_bgr.shape[1] - 1))
            y1 = int(np.clip(y1, 0, img_bgr.shape[0] - 1))
            x2 = int(np.clip(x2, 0, img_bgr.shape[1] - 1))
            y2 = int(np.clip(y2, 0, img_bgr.shape[0] - 1))
            if x2 > x1 and y2 > y1:
                label = self.labels.get(int(k), str(int(k)))
                boxes.append(Det((x1, y1, x2, y2), float(c), int(k), label))
        return boxes

# ---------------------------- Drawing ----------------------------
def draw(img: np.ndarray, dets: List[Det], thickness: int = 2) -> np.ndarray:
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        txt = f"{d.label} {d.conf:.2f}"
        (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - h - 8)), (x1 + w + 6, y1), (0, 255, 0), -1)
        cv2.putText(out, txt, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out

# ---------------------------- I/O ----------------------------
def list_images(folder: str, exts) -> List[str]:
    exts = tuple(e.lower() for e in exts)
    paths = []
    for r, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(r, f))
    return sorted(paths)

def det_to_dict(d: Det) -> Dict[str, Any]:
    return {"bbox": list(d.bbox), "conf": round(d.conf, 4), "class_id": d.cls, "label": d.label}

# ---------------------------- CLI ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="YOLOv5 Vehicle Detection (ONNX + OpenCV DNN)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="Run detection on image/folder/video")
    d.add_argument("--model", required=True, help="Path to YOLOv5 ONNX model")
    d.add_argument("--source", required=True, help="Image/Folder/Video path")
    d.add_argument("--size", type=int, default=640, help="Model input size")
    d.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    d.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    d.add_argument("--classes", nargs="+", default=VEHICLE_DEFAULT, help="Class names to keep (default vehicles)")
    d.add_argument("--class-ids", nargs="+", type=int, default=None, help="Override with explicit class IDs to keep")
    d.add_argument("--save", action="store_true", help="Save annotated outputs")
    d.add_argument("--json", default="", help="Write JSON results to this file")
    d.add_argument("--view", action="store_true", help="Show preview window")
    d.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png"], help="Valid image extensions for folder mode")
    return p.parse_args()

def build_keep_lists(class_names: List[str], class_ids_cli: Optional[List[int]]):
    if class_ids_cli is not None:
        keep_ids = class_ids_cli
        labels = {cid: str(cid) for cid in keep_ids}
        return keep_ids, labels
    # map names through COCO_DEFAULT
    keep_ids = []
    labels = {}
    for name in class_names:
        if name in COCO_DEFAULT:
            cid = COCO_DEFAULT[name]
            keep_ids.append(cid)
            labels[cid] = name
    # fallback: if no valid mapping, keep all classes
    if not keep_ids:
        return None, {}
    return keep_ids, labels

def run_image(model: YoloV5ONNX, path: str, save: bool, view: bool) -> Dict[str, Any]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    t0 = time.time()
    dets = model.infer(img)
    elapsed = (time.time() - t0) * 1000.0
    out = {"source": path, "elapsed_ms": round(elapsed, 3), "detections": [det_to_dict(d) for d in dets]}
    if save or view:
        drawn = draw(img, dets)
        if save:
            out_path = os.path.splitext(path)[0] + "_veh.jpg"
            cv2.imwrite(out_path, drawn)
        if view:
            cv2.imshow("vehicle-detect", drawn)
            cv2.waitKey(1)
    return out

def run_video(model: YoloV5ONNX, path: str, save: bool, view: bool) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    writer = None
    results = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        dets = model.infer(frame)
        elapsed = (time.time() - t0) * 1000.0
        results.append({"source": path, "elapsed_ms": round(elapsed, 3), "detections": [det_to_dict(d) for d in dets]})
        if save or view:
            drawn = draw(frame, dets)
            if view:
                cv2.imshow("vehicle-detect", drawn)
                if cv2.waitKey(1) == 27:
                    break
            if save:
                if writer is None:
                    h, w = drawn.shape[:2]
                    out_path = os.path.splitext(path)[0] + "_veh.mp4"
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                writer.write(drawn)
    cap.release()
    if writer is not None:
        writer.release()
    if view:
        cv2.destroyAllWindows()
    return results

def main():
    args = parse_args()
    keep_ids, labels = build_keep_lists(args.classes, args.class_ids)
    model = YoloV5ONNX(args.model, size=args.size, conf_thr=args.conf, iou_thr=args.iou, keep_cls=keep_ids, labels=labels)

    src = args.source
    all_out = []
    if os.path.isdir(src):
        imgs = list_images(src, args.exts)
        for p in imgs:
            all_out.append(run_image(model, p, args.save, args.view))
        if args.view:
            cv2.destroyAllWindows()
    elif os.path.isfile(src) and any(src.lower().endswith(e) for e in [".mp4", ".avi", ".mov", ".mkv"]):
        all_out.extend(run_video(model, src, args.save, args.view))
    elif os.path.isfile(src):
        all_out.append(run_image(model, src, args.save, args.view))
        if args.view:
            cv2.destroyAllWindows()
    else:
        raise FileNotFoundError(f"Source not found: {src}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
