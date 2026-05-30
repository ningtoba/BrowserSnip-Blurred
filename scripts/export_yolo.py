#!/usr/bin/env python3
"""Export YOLOv26n to ONNX for browser inference.

Usage:
    pip install ultralytics onnx
    python scripts/export_yolo.py

This produces public/models/yolo26n-face.onnx (~4.8 MB FP16, end-to-end with NMS).

For a face-specific model, you need a YOLO checkpoint trained on a face
dataset (e.g. WIDER Face). Standard COCO models do not have a face class.

Options:
    - Use a community YOLO-face model from HuggingFace / Ultralytics Hub
    - Custom train: yolo detect train data=widerface.yaml model=yolo26n.yaml epochs=100
"""

from ultralytics import YOLO
import shutil
import os

MODEL_PATH = "yolo26n-face.pt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "models")
OUTPUT_NAME = "yolo26n-face.onnx"


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        print("Download a face-detection YOLO model or train one on WIDER Face.")
        print("Place the .pt file in the project root and re-run this script.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    model.export(
        format="onnx",
        simplify=True,
        half=True,
        nms=True,
        imgsz=640,
    )

    # Move to public/models
    src = MODEL_PATH.replace(".pt", ".onnx")
    if os.path.exists(src):
        dst = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        shutil.move(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"Exported: {dst} ({size_mb:.1f} MB)")
    else:
        print(f"Export may have produced a different filename. Check the working directory.")


if __name__ == "__main__":
    main()
