# YOLOv5 Vehicle Detection (Executable CLI)

Turn the **yolo-v5-car-object-detection.ipynb** into a fast command-line tool using **YOLOv5 ONNX + OpenCV DNN**.  
Detect cars (and other vehicles) from images, folders, or videos. Save annotated outputs and JSON logs.

## 1) Install

```bash
# Python 3.9+ recommended
pip install -r requirements.txt
```

> This CLI runs inference with **OpenCV DNN** (no PyTorch needed at runtime).

## 2) Get a YOLOv5 ONNX model

If you trained YOLOv5 on vehicles, export to ONNX:
```bash
# within ultralytics/yolov5 repo (classic)
python export.py --weights runs/train/exp/weights/best.pt --include onnx --opset 12
```
Put the resulting `*.onnx` wherever you like.

## 3) Run

Single image (only cars):
```bash
python car_detect.py detect --model yolov5s.onnx --source /path/to/img.jpg --classes car --save --json out.json
```

Folder of images (cars, trucks, buses, motorcycles):
```bash
python car_detect.py detect --model yolov5s.onnx --source /path/to/folder --save --json results.json
```

Video with live preview:
```bash
python car_detect.py detect --model yolov5s.onnx --source /path/to/video.mp4 --classes car motorcycle --save --view
```

### Common flags
- `--size 640` YOLO input size (must match training/export assumptions)
- `--conf 0.25` confidence threshold
- `--iou 0.45` NMS IoU
- `--classes` names to keep (default: `car motorcycle bus truck`)
- `--class-ids` override with explicit class IDs (skips name mapping)
- `--save` writes `*_veh.jpg` or `*_veh.mp4` next to source
- `--json results.json` writes structured detections
- `--view` shows a preview window
- `--exts .jpg .png` extensions for folder mode

## 4) Output
- Annotated images/videos next to the source
- JSON array with per-frame detections:
```json
[
  {
    "source": "img.jpg",
    "elapsed_ms": 11.2,
    "detections": [
      {"bbox":[x1,y1,x2,y2], "conf":0.92, "class_id":2, "label":"car"}
    ]
  }
]
```

## Why this is more efficient than a notebook
- Single **model init** reused across all inputs (no reloading).
- **OpenCV DNN + ONNX** avoids Python/Torch overhead in inference-only runs.
- **Vectorized NMS** (NumPy) and one-pass **letterbox** + coordinate mapping.
- Streaming **video writer** initialized lazily to avoid overhead.

## Class mapping (COCO default)
We map names to IDs: `bicycle=1, car=2, motorcycle=3, bus=5, train=6, truck=7`.  
If your model uses different indices, pass `--class-ids` explicitly.
