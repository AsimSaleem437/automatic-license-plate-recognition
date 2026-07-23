"""
Downloads the Vehicle Registration Plates dataset from Roboflow in YOLOv8 format.

Get a free API key from: https://app.roboflow.com/settings/api
Then either export it: export ROBOFLOW_API_KEY=xxxx
or pass --api-key.
"""
import argparse
import os
from roboflow import Roboflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    parser.add_argument("--out-dir", default="./data")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY env var or pass --api-key")

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace("augmented-startups").project("vehicle-registration-plates-trudk")
    version = project.version(2)
    
    # Resolve the output directory to an absolute path so the downloaded data.yaml 
    # contains the correct path for either local training or Colab training.
    out_dir = os.path.abspath(args.out_dir)
    dataset = version.download("yolov8", location=out_dir)
    print("location:", dataset.location)
    print("exists:", os.path.exists(dataset.location))
    print("contents:", os.listdir(dataset.location) if os.path.exists(dataset.location) else "MISSING")


if __name__ == "__main__":
    main()
