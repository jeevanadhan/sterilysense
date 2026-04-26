# Bacteria Webcam Detection Share

This folder contains a minimal shareable package for the two webcam detection scripts:

- `webcam_bact_detection.py`
- `webcam_yolov8.py`
- `bacteria_db.json`
- `requirements.txt`

## Requirements

1. Python 3.11+ (or Python 3.10)
2. A working webcam or an accessible IP camera stream
3. `yolov8n.pt` model file for `webcam_yolov8.py`
4. `best.pt` weights for `webcam_bact_detection.py`

## Setup

```bash
cd sterilysense-main
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Files and model placement

- Place `yolov8n.pt` in the `share/` folder next to `webcam_yolov8.py`
- Place `best.pt` in `share/weights/`

If you have the trained checkpoint from the original project, copy it from:

`../runs/train_stage3/stage2_surface/weights/best.pt`

## Run the examples

### Run YOLOv8 OBJECT detection

```bash
python webcam_yolov8.py
```

Press `q` to quit.

### Run the bacteria detection Flask stream

```bash
python webcam_bact_detection.py
```

Then open http://localhost:5000 in your browser.

## Notes

- `webcam_bact_detection.py` will use a local webcam first, then fall back to the mobile camera URL if available.
- The Flask endpoint `/analysis` returns the latest detected bacteria results as JSON.
- This package is intentionally minimal for GitHub sharing.
