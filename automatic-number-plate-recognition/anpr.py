#!/usr/bin/env python3
"""
ANPR (Automatic Number Plate Recognition) - Efficient CLI
- YOLOv5 ONNX (OpenCV DNN) for plate detection
- Tesseract OCR for text extraction
- Supports single image, folder, or video
- Outputs annotated media + JSON results

Usage examples:
  python anpr.py detect --model best.onnx --source /path/to/image.jpg
  python anpr.py detect --model best.onnx --source /path/to/folder --exts .jpg .png --save-crops
  python anpr.py detect --model best.onnx --source /path/to/video.mp4 --out output.mp4 --ocr --view
  python anpr.py detect --model best.onnx --source /path/to/image.jpg --json out.json

Notes:
- Model must be an ONNX export of YOLOv5 trained on "license_plate" (class 0).
- Tesseract must be installed and available on PATH (or provide --tesseract-cmd).
"""
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any, Iterable

import cv2
import numpy as np
import pytesseract as pt

# ---------------------------- Utils ----------------------------
def letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize and pad image to square keeping aspect ratio (YOLOv5 style)."""
    h0, w0 = img.shape[:2]
    r = new_shape / max(h0, w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(img_resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (dw, dh)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of boxes (N,4) and (M,4) in xyxy."""
    # Intersection
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    # Union
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Pure NumPy NMS returning indices to keep."""
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

# ---------------------------- Data classes ----------------------------
@dataclass
class Plate:
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    conf: float
    text: Optional[str] = None

@dataclass
class DetectionResult:
    source: str
    plates: List[Plate]
    elapsed_ms: float

# ---------------------------- YOLOv5 ONNX Wrapper ----------------------------
class YoloONNX:
    def __init__(self, model_path: str, input_size: int = 640, conf_thr: float = 0.25, iou_thr: float = 0.45, class_id: int = 0):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.size = input_size
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.class_id = class_id

    def infer(self, img_bgr: np.ndarray) -> List[Plate]:
        img, r, (dw, dh) = letterbox(img_bgr, self.size)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()  # shape: (1, n, 85) typical YOLOv5
        preds = np.squeeze(preds, axis=0)

        # Filter by objectness/confidence
        obj = preds[:, 4]
        class_scores = preds[:, 5:]  # assume class 0 is license_plate
        cls = class_scores[:, self.class_id]
        conf = obj * cls
        mask = conf > self.conf_thr
        preds = preds[mask]
        conf = conf[mask]

        if preds.size == 0:
            return []

        # xywh to xyxy in padded image coords
        xywh = preds[:, :4]
        xyxy = np.zeros_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2

        # NMS
        keep = nms(xyxy, conf, self.iou_thr)
        xyxy = xyxy[keep]
        conf = conf[keep]

        # Map back to original image coords
        gain = max(img_bgr.shape[0], img_bgr.shape[1]) / self.size
        # inverse of letterbox: subtract pad and divide by scale
        boxes = []
        for (x1, y1, x2, y2), c in zip(xyxy, conf):
            x1 = int(max((x1 - dw) / (self.size / max(img_bgr.shape[:2])), 0))
            y1 = int(max((y1 - dh) / (self.size / max(img_bgr.shape[:2])), 0))
            x2 = int(min((x2 - dw) / (self.size / max(img_bgr.shape[:2])), img_bgr.shape[1] - 1))
            y2 = int(min((y2 - dh) / (self.size / max(img_bgr.shape[:2])), img_bgr.shape[0] - 1))
            if x2 > x1 and y2 > y1:
                boxes.append(Plate((x1, y1, x2, y2), float(c)))
        return boxes

# ---------------------------- OCR ----------------------------
def preprocess_roi_for_ocr(roi: np.ndarray) -> np.ndarray:
    """Efficient OCR preproc: grayscale -> bilateral -> adaptive threshold -> morph close."""
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed

def ocr_text(roi: np.ndarray, psm: int = 7) -> str:
    pre = preprocess_roi_for_ocr(roi)
    config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    txt = pt.image_to_string(pre, config=config)
    return txt.strip()

# ---------------------------- Drawing ----------------------------
def draw_plates(img: np.ndarray, plates: List[Plate], show_text: bool=True) -> np.ndarray:
    out = img.copy()
    for p in plates:
        x1, y1, x2, y2 = p.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
        label = f"{p.conf:.2f}"
        if show_text and p.text:
            label += f" | {p.text}"
        cv2.putText(out, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return out

# ---------------------------- Detection Pipeline ----------------------------
def detect_image(yolo: YoloONNX, image_path: str, do_ocr: bool = True, save_crops: bool = False, crops_dir: Optional[str] = None) -> DetectionResult:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    t0 = time.time()
    plates = yolo.infer(img)
    if do_ocr:
        for p in plates:
            x1, y1, x2, y2 = p.bbox
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                p.text = ""
                continue
            p.text = ocr_text(roi)
            if save_crops:
                ensure_dir(crops_dir or "crops")
                base = os.path.splitext(os.path.basename(image_path))[0]
                crop_path = os.path.join(crops_dir or "crops", f"{base}_{x1}_{y1}_{x2}_{y2}.png")
                cv2.imwrite(crop_path, roi)
    elapsed = (time.time() - t0) * 1000.0
    return DetectionResult(source=image_path, plates=plates, elapsed_ms=elapsed)

def list_images(folder: str, exts: Tuple[str, ...]) -> List[str]:
    exts = tuple(e.lower() for e in exts)
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def detections_to_json(dr: DetectionResult) -> Dict[str, Any]:
    return {
        "source": dr.source,
        "elapsed_ms": round(dr.elapsed_ms, 3),
        "plates": [
            {"bbox": list(p.bbox), "conf": round(p.conf, 4), "text": p.text} for p in dr.plates
        ]
    }

# ---------------------------- CLI ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ANPR CLI (YOLOv5 ONNX + Tesseract)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="Run detection on image/folder/video")
    d.add_argument("--model", required=True, help="Path to YOLOv5 ONNX model (best.onnx)")
    d.add_argument("--source", required=True, help="Image path, folder, or video path")
    d.add_argument("--size", type=int, default=640, help="Model input size")
    d.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    d.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    d.add_argument("--ocr", action="store_true", help="Enable OCR (Tesseract)")
    d.add_argument("--tesseract-cmd", default="", help="Path to tesseract binary if not on PATH")
    d.add_argument("--save", action="store_true", help="Save annotated output next to source")
    d.add_argument("--json", default="", help="Write JSON results to this file")
    d.add_argument("--view", action="store_true", help="Preview window")
    d.add_argument("--save-crops", action="store_true", help="Save plate crops when --ocr is used")
    d.add_argument("--crops-dir", default="crops", help="Crops output directory (default: crops)")
    d.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png"], help="Valid extensions for folder mode")
    return p.parse_args()

def main():
    args = parse_args()
    if args.tesseract_cmd:
        pt.pytesseract.tesseract_cmd = args.tesseract_cmd

    yolo = YoloONNX(args.model, input_size=args.size, conf_thr=args.conf, iou_thr=args.iou, class_id=0)

    src = args.source
    results: List[DetectionResult] = []

    if os.path.isdir(src):
        paths = list_images(src, tuple(args.exts))
        for pth in paths:
            dr = detect_image(yolo, pth, do_ocr=args.ocr, save_crops=args.save_crops, crops_dir=args.crops_dir)
            results.append(dr)
            if args.save or args.view:
                img = cv2.imread(pth)
                img_anno = draw_plates(img, dr.plates, show_text=args.ocr)
                if args.save:
                    out_path = os.path.splitext(pth)[0] + "_anpr.jpg"
                    cv2.imwrite(out_path, img_anno)
                if args.view:
                    cv2.imshow("ANPR", img_anno)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC
                        break
        if args.view:
            cv2.destroyAllWindows()

    elif os.path.isfile(src) and any(src.lower().endswith(e) for e in [".mp4", ".avi", ".mov", ".mkv"]):
        cap = cv2.VideoCapture(src)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dr = detect_image(yolo, image_path="", do_ocr=args.ocr)  # Not used; we need a direct frame path
            # modify to run directly on frame:
            t0 = time.time()
            plates = yolo.infer(frame)
            if args.ocr:
                for p in plates:
                    x1, y1, x2, y2 = p.bbox
                    roi = frame[y1:y2, x1:x2]
                    p.text = ocr_text(roi) if roi.size > 0 else ""
            elapsed = (time.time() - t0) * 1000.0
            results.append(DetectionResult(source=src, plates=plates, elapsed_ms=elapsed))
            annotated = draw_plates(frame, plates, show_text=args.ocr)
            if args.view:
                cv2.imshow("ANPR", annotated)
                if cv2.waitKey(1) == 27:
                    break
            if args.save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    out_path = args.json[:-5] + ".mp4" if args.json.endswith(".json") else (os.path.splitext(src)[0] + "_anpr.mp4")
                    writer = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25.0, (w, h))
                writer.write(annotated)
        cap.release()
        if args.view:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.release()

    elif os.path.isfile(src):
        dr = detect_image(yolo, src, do_ocr=args.ocr, save_crops=args.save_crops, crops_dir=args.crops_dir)
        results.append(dr)
        if args.save or args.view:
            img = cv2.imread(src)
            img_anno = draw_plates(img, dr.plates, show_text=args.ocr)
            if args.save:
                out_path = os.path.splitext(src)[0] + "_anpr.jpg"
                cv2.imwrite(out_path, img_anno)
            if args.view:
                cv2.imshow("ANPR", img_anno)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        raise FileNotFoundError(f"Source not found: {src}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([detections_to_json(r) for r in results], f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
