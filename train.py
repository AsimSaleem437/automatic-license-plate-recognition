"""
Train a YOLOv8 nano model to detect license plates.

1. Download the dataset from Roboflow first (see download_dataset.py),
   which produces a folder containing data.yaml + train/valid/test splits.
2. Run: python train.py --data path/to/data.yaml
"""
import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml from the Roboflow export")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n.pt", help="Base checkpoint (nano = fastest, good enough for 1 class)")
    args = parser.parse_args()

    model = YOLO(args.model)
    
    # Resolve the data YAML path to an absolute path to avoid path resolution errors in YOLO
    data_path = os.path.abspath(args.data)
    
    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=15,       # early stop if val mAP plateaus
        project="runs",
        name="plate_detector",
    )

    # Best weights land at runs/plate_detector/weights/best.pt
    metrics = model.val()
    print("Validation mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
