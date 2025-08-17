# ANPR (Automatic Number Plate Recognition)

This script turns the notebook logic into a fast, executable CLI using **YOLOv5 ONNX** + **OpenCV DNN** for detection and **Tesseract** for OCR.

## 1) Install dependencies

```bash
# Python 3.9+ recommended
pip install -r requirements.txt

# Install Tesseract OCR:
# - Ubuntu/Debian: sudo apt-get install tesseract-ocr
# - macOS (Homebrew): brew install tesseract
# - Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

## 2) Prepare the model

Export your trained YOLOv5 model to ONNX, e.g.:
```bash
# inside the yolov5 repo
python export.py --weights runs/train/Model/weights/best.pt --include onnx
```
Copy `best.onnx` next to `anpr.py` (or pass a path with `--model`).

## 3) Run detection

Single image:
```bash
python anpr.py detect --model best.onnx --source /path/to/image.jpg --ocr --save --json result.json
```

Folder of images:
```bash
python anpr.py detect --model best.onnx --source /path/to/folder --ocr --save --json results.json
```

Video:
```bash
python anpr.py detect --model best.onnx --source /path/to/video.mp4 --ocr --save --json results.json --view
```

Options:
- `--size` (default: 640) YOLO input size
- `--conf` (default: 0.25) confidence threshold
- `--iou` (default: 0.45) NMS IoU threshold
- `--ocr` enable OCR via Tesseract
- `--tesseract-cmd` path to tesseract if not on PATH (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`)
- `--save` save annotated images/video
- `--json` write detection results to JSON
- `--view` preview window
- `--save-crops` also save plate crops when OCR is enabled
- `--crops-dir` where to save crops (default: `crops`)
- `--exts` file extensions to scan in folder mode

## 4) Output

- Annotated images/videos named `*_anpr.jpg` or `*_anpr.mp4` next to the source
- JSON with per-image detections (bbox, confidence, recognized text)
- Optional cropped plate images in `crops/`

