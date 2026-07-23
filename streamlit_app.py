"""
Streamlit UI for testing the ALPR pipeline end-to-end.

Run:
    streamlit run streamlit_app.py

Lets you:
- Upload a car image and see the detected plate box + OCR text
- Adjust the detection confidence threshold live
- Browse the history of everything logged to alpr.db
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from db import get_recent_reads, init_db
from pipeline import clean_plate_text, run_ocr

st.set_page_config(page_title="ALPR Tester", page_icon="🚗", layout="wide")

DEFAULT_WEIGHTS = "runs/plate_detector/weights/best.pt"
UPLOAD_DIR = Path("uploads")
CROPS_DIR = Path("crops")
UPLOAD_DIR.mkdir(exist_ok=True)
CROPS_DIR.mkdir(exist_ok=True)


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def draw_box(img: np.ndarray, x1, y1, x2, y2, label: str):
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 3)
    cv2.putText(out, label, (x1, max(y1 - 12, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    return out


def main():
    init_db()

    st.title("🚗 ALPR — License Plate Detection & OCR Tester")
    st.caption("Upload a car photo to run it through the detector + OCR pipeline and log the result.")

    with st.sidebar:
        st.header("Settings")
        weights_path = st.text_input("YOLO weights path", value=DEFAULT_WEIGHTS)
        conf_thresh = st.slider("Detection confidence threshold", 0.05, 0.95, 0.4, 0.05)
        save_to_db = st.checkbox("Save results to database", value=True)

        st.divider()
        st.header("Recent DB reads")
        rows = get_recent_reads(limit=15)
        if rows:
            df = pd.DataFrame(rows)[["detected_at", "plate_text", "ocr_confidence", "detector_confidence"]]
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.write("No reads logged yet.")

    if not Path(weights_path).exists():
        st.warning(
            f"Weights not found at `{weights_path}`. Train the model first "
            "(`python train.py --data data/data.yaml`) or point this at your `best.pt`."
        )
        st.stop()

    model = load_model(weights_path)

    uploaded = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    pil_img = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    saved_path = UPLOAD_DIR / uploaded.name
    saved_path.write_bytes(uploaded.getvalue())

    with st.spinner("Running detection..."):
        results = model.predict(img_bgr, conf=conf_thresh, verbose=False)[0]

    col1, col2 = st.columns(2)

    if len(results.boxes) == 0:
        col1.image(pil_img, caption="No plates detected — try lowering the confidence threshold", width="stretch")
        return

    annotated = img_bgr.copy()
    detections = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_conf = float(box.conf[0])
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        with st.spinner(f"Running OCR on plate {i + 1}..."):
            plate_text, ocr_conf = run_ocr(crop)

        label = plate_text if plate_text else "?"
        annotated = draw_box(annotated, x1, y1, x2, y2, f"{label} ({det_conf:.2f})")

        crop_path = CROPS_DIR / f"{saved_path.stem}_{i}.jpg"
        cv2.imwrite(str(crop_path), crop)

        detections.append(
            {
                "crop": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                "plate_text": plate_text,
                "det_conf": det_conf,
                "ocr_conf": ocr_conf,
            }
        )

        if save_to_db and plate_text:
            from db import save_plate_read

            save_plate_read(
                plate_text=plate_text,
                ocr_confidence=ocr_conf,
                detector_confidence=det_conf,
                source_image=str(saved_path),
                cropped_image_path=str(crop_path),
            )

    col1.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections", width="stretch")

    with col2:
        st.subheader(f"Found {len(detections)} plate(s)")
        for i, d in enumerate(detections):
            st.image(d["crop"], caption=f"Plate {i + 1} crop", width=300)
            if d["plate_text"]:
                st.success(f"**{d['plate_text']}**  ·  detector={d['det_conf']:.2f}  ·  ocr={d['ocr_conf']:.2f}")
            else:
                st.error("OCR could not read this crop (try a higher-res image or adjust preprocessing).")

            manual = st.text_input(f"Correct plate {i + 1} manually (optional)", key=f"manual_{i}")
            if manual and save_to_db:
                if st.button(f"Save correction for plate {i + 1}", key=f"save_{i}"):
                    from db import save_plate_read

                    save_plate_read(
                        plate_text=clean_plate_text(manual),
                        ocr_confidence=None,
                        detector_confidence=d["det_conf"],
                        source_image=str(saved_path),
                        cropped_image_path=None,
                    )
                    st.success("Saved manual correction.")


if __name__ == "__main__":
    main()
