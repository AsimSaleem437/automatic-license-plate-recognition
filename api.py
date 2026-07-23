"""
FastAPI wrapper around the ALPR pipeline.

Run: uvicorn api:app --reload
Then POST an image to /detect
"""
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

from db import get_recent_reads, init_db
from pipeline import process_image

WEIGHTS_PATH = "runs/plate_detector/weights/best.pt"  # adjust after training
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ALPR API")
model: YOLO | None = None


@app.on_event("startup")
def startup():
    global model
    init_db()
    model = YOLO(WEIGHTS_PATH)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(await file.read())

    detections = process_image(model, str(dest))
    return {"filename": file.filename, "detections": detections}


@app.get("/reads")
def reads(limit: int = 20):
    return get_recent_reads(limit)
