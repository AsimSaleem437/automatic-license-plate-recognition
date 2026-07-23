# ALPR — Automatic License Plate Recognition

YOLOv8 plate detector + EasyOCR text extraction + SQLite logging, with a FastAPI wrapper.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Download the dataset

Get a free API key at https://app.roboflow.com/settings/api, then:

```bash
export ROBOFLOW_API_KEY=your_key_here
python download_dataset.py --out-dir ./data
```

This produces `data/data.yaml` plus `train/valid/test` folders in YOLO format
(8,823 images, 1 class: `License_Plate`).

## 2. Train the detector

```bash
python train.py --data data/data.yaml --epochs 50
```

Nano (`yolov8n.pt`) is plenty for a single-class detector — trains in under
an hour on a free Colab T4, or a few hours on CPU. Best weights land at
`runs/plate_detector/weights/best.pt`.

Sanity-check training worked: `mAP50-95` above ~0.6 is solid for this task
(the pretrained community model on Roboflow hits mAP@50 of 98.5%, so you
have real headroom to compare against).

## 3. Run the full pipeline

Single image:
```bash
python pipeline.py --weights runs/plate_detector/weights/best.pt --image path/to/car.jpg
```

Webcam (live, with debounce so the same plate isn't logged every frame):
```bash
python pipeline.py --weights runs/plate_detector/weights/best.pt --webcam
```

Both write to `alpr.db` (SQLite) via `db.py`, and save cropped plate images
to `crops/`.

## 4. Streamlit UI (for quick testing)

```bash
streamlit run streamlit_app.py
```

Opens a browser UI where you can:
- Upload a car photo and see the detected plate box drawn live
- Adjust the detection confidence threshold with a slider
- See the cropped plate + OCR text + confidence scores side by side
- Manually correct a misread plate and save that correction to the DB
- Browse the last 15 logged reads in the sidebar

This is the fastest way to eyeball whether your trained weights are actually
good before wiring anything else up. Point "YOLO weights path" in the
sidebar at your `runs/plate_detector/weights/best.pt` after training.

## 5. (Optional) Serve it as an API

```bash
uvicorn api:app --reload
```

```bash
curl -X POST -F "file=@car.jpg" http://localhost:8000/detect
curl http://localhost:8000/reads
```

## Design notes / things worth knowing going in

- **Why detect + crop before OCR, instead of OCR on the whole image?**
  OCR engines are tuned for roughly frontal, high-contrast text. Running
  EasyOCR directly on a full car photo gives it a huge false-positive
  surface (badges, bumper stickers, background signs). Cropping tightly to
  the plate first is most of what makes this reliable.
- **Preprocessing matters more than model choice for OCR accuracy.** Plates
  are often small, tilted, or motion-blurred in source images. The
  `preprocess_plate()` step (upscale, bilateral filter, histogram
  equalization) will move your accuracy more than swapping EasyOCR for
  Tesseract will.
- **Tesseract vs EasyOCR**: EasyOCR (deep-learning based) generally handles
  varied fonts/angles/lighting better out of the box; Tesseract is faster
  and lighter if you can guarantee clean, frontal, high-res crops. Worth
  benchmarking both on your actual crops once you have some — don't assume.
- **Regional plate formats**: `clean_plate_text()` currently just strips
  non-alphanumerics. If you want to validate against a specific country's
  format (e.g. Pakistani plates), add a regex check post-OCR and flag/reject
  reads that don't match, rather than silently trusting every OCR output.
- **Postgres swap**: `db.py` is intentionally small and isolated so you can
  swap it for SQLModel + asyncpg later (matching your usual stack) without
  touching the detection/OCR logic.


