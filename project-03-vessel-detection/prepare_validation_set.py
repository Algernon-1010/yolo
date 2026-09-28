"""Extract an independent validation set and create YOLOv8 cargo-vessel prelabels."""

from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE = PROJECT_ROOT / "videos" / "cargo_ship_river_aerial_33s.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "val"
LABEL_DIR = PROJECT_ROOT / "dataset" / "labels" / "val"
COCO_BOAT_CLASS_ID = 8
PROJECT_CARGO_VESSEL_CLASS_ID = 1
CONFIDENCE = 0.30


def to_yolo_line(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    return (
        f"{PROJECT_CARGO_VESSEL_CLASS_ID} {((x1 + x2) / 2) / width:.6f} "
        f"{((y1 + y2) / 2) / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def extract_frames() -> list[Path]:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Video not found: {SOURCE}")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(IMAGE_DIR.glob("val_*.jpg"))
    if existing:
        raise FileExistsError(f"Found {len(existing)} existing validation images. No files were overwritten.")

    capture = cv2.VideoCapture(str(SOURCE))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Could not read the video frame rate.")

    interval = max(1, round(fps))
    frame_index = 0
    outputs: list[Path] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % interval == 0:
            output = IMAGE_DIR / f"val_{len(outputs) + 1:04d}.jpg"
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"Could not write frame: {output}")
            outputs.append(output)
        frame_index += 1
    capture.release()
    if not outputs:
        raise RuntimeError("No validation frames were extracted.")
    return outputs


def create_prelabels(images: list[Path]) -> tuple[int, int]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(LABEL_DIR.glob("val_*.txt"))
    if existing:
        raise FileExistsError(f"Found {len(existing)} existing validation labels. No files were overwritten.")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=[str(image) for image in images], conf=CONFIDENCE, device="cpu", verbose=False)
    detected_images = 0
    total_boxes = 0
    for image, result in zip(images, results):
        height, width = result.orig_shape
        lines: list[str] = []
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls.item()) != COCO_BOAT_CLASS_ID:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                lines.append(to_yolo_line(x1, y1, x2, y2, width, height))
        (LABEL_DIR / f"{image.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        if lines:
            detected_images += 1
            total_boxes += len(lines)
    return detected_images, total_boxes


def main() -> None:
    images = extract_frames()
    detected_images, total_boxes = create_prelabels(images)
    print(f"Validation images: {len(images)}")
    print(f"Images with vessel detections: {detected_images}")
    print(f"Cargo-vessel boxes: {total_boxes}")


if __name__ == "__main__":
    main()
