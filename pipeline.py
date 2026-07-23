"""
End-to-end ALPR pipeline:
    image -> YOLOv8 plate detection -> crop -> preprocess -> EasyOCR -> clean text -> SQLite

Usage:
    python pipeline.py --image path/to/car.jpg --weights runs/plate_detector/weights/best.pt
    python pipeline.py --webcam --weights runs/plate_detector/weights/best.pt
"""
import argparse
import re
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

from db import init_db, save_plate_read

CROPS_DIR = Path("crops")
CROPS_DIR.mkdir(exist_ok=True)

# EasyOCR reader is expensive to init — build once, reuse across frames/images.
_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)  # set gpu=True if you have CUDA
    return _ocr_reader


def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """Crop borders, upscale, denoise, and apply CLAHE to optimize OCR readability."""
    h, w = crop.shape[:2]
    
    # Asymmetric crop to strip out black frames, screws, country strips, and dealer text:
    # - Top/bottom margins: ~5% top, ~12% bottom (often has frame text)
    # - Left/right margins: ~6% left (removes blue TR/IND strip), ~4% right
    top = int(h * 0.05)
    bottom = int(h * 0.88)
    left = int(w * 0.06)
    right = int(w * 0.96)
    cropped = crop[top:bottom, left:right]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    h_gray, w_gray = gray.shape
    
    # Upscale small crops — OCR struggles below ~100px tall
    if h_gray < 100:
        scale = 100 / h_gray
        gray = cv2.resize(gray, (int(w_gray * scale), 100), interpolation=cv2.INTER_CUBIC)
        
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) instead of global equalization
    # to avoid boosting noise in shadows and highlights
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def clean_plate_text(raw: str) -> str:
    """Strip anything that isn't alphanumeric, uppercase it."""
    text = re.sub(r"[^A-Za-z0-9]", "", raw)
    return text.upper()


def run_ocr(crop: np.ndarray) -> tuple[str, float]:
    reader = get_ocr_reader()
    processed = preprocess_plate(crop)
    
    # Restrict OCR to alphanumeric characters to avoid symbol hallucinations
    results = reader.readtext(processed, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if not results:
        return "", 0.0
    
    # Concatenate all detected text fragments (plates sometimes split into 2 boxes),
    # keep the average confidence.
    text = "".join(r[1] for r in results)
    conf = float(np.mean([r[2] for r in results]))
    return clean_plate_text(text), conf


def process_image(model: YOLO, image_path: str, conf_thresh: float = 0.4, save_crops: bool = True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    results = model.predict(img, conf=conf_thresh, verbose=False)[0]

    detections = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_conf = float(box.conf[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plate_text, ocr_conf = run_ocr(crop)
        if not plate_text:
            continue

        crop_path = None
        if save_crops:
            crop_path = str(CROPS_DIR / f"{Path(image_path).stem}_{i}.jpg")
            cv2.imwrite(crop_path, crop)

        save_plate_read(
            plate_text=plate_text,
            ocr_confidence=ocr_conf,
            detector_confidence=det_conf,
            source_image=image_path,
            cropped_image_path=crop_path,
        )
        detections.append({"plate": plate_text, "det_conf": det_conf, "ocr_conf": ocr_conf})

    return detections


def run_webcam(model: YOLO, conf_thresh: float = 0.4, cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    last_seen = {}  # simple debounce so we don't spam the DB every frame
    debounce_seconds = 5

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=conf_thresh, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            plate_text, ocr_conf = run_ocr(crop)
            if not plate_text:
                continue

            now = time.time()
            if plate_text in last_seen and now - last_seen[plate_text] < debounce_seconds:
                continue
            last_seen[plate_text] = now

            save_plate_read(
                plate_text=plate_text,
                ocr_confidence=ocr_conf,
                detector_confidence=float(box.conf[0]),
                source_image="webcam",
            )
            print(f"[{time.strftime('%H:%M:%S')}] Plate: {plate_text} (ocr={ocr_conf:.2f})")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("ALPR", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights (best.pt)")
    parser.add_argument("--image", help="Single image to process")
    parser.add_argument("--webcam", action="store_true", help="Run live on webcam instead")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    init_db()
    model = YOLO(args.weights)

    if args.webcam:
        run_webcam(model, conf_thresh=args.conf)
    elif args.image:
        detections = process_image(model, args.image, conf_thresh=args.conf)
        print(detections if detections else "No plates read.")
    else:
        parser.error("Pass --image or --webcam")


if __name__ == "__main__":
    main()
