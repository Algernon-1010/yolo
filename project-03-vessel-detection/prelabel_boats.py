"""Create YOLO-format cargo-vessel prelabels from extracted training images."""

from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "dataset" / "labels" / "train"
COCO_BOAT_CLASS_ID = 8
PROJECT_CARGO_VESSEL_CLASS_ID = 1
CONFIDENCE = 0.30


def to_yolo_line(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return (
        f"{PROJECT_CARGO_VESSEL_CLASS_ID} {x_center:.6f} {y_center:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    images = sorted(IMAGE_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No JPG images found: {IMAGE_DIR}")

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(LABEL_DIR.glob("*.txt"))
    if existing:
        raise FileExistsError(
            f"Found {len(existing)} existing label files in {LABEL_DIR}. No files were overwritten."
        )

    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=[str(image) for image in images], conf=CONFIDENCE, device="cpu", verbose=False)

    detected_images = 0
    total_boxes = 0
    for image, result in zip(images, results):
        height, width = result.orig_shape
        lines: list[str] = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id != COCO_BOAT_CLASS_ID:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                lines.append(to_yolo_line(x1, y1, x2, y2, width, height))

        label_path = LABEL_DIR / f"{image.stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        if lines:
            detected_images += 1
            total_boxes += len(lines)

    print(f"Processed images: {len(images)}")
    print(f"Images with vessel detections: {detected_images}")
    print(f"Cargo-vessel boxes written: {total_boxes}")
    print(f"Labels: {LABEL_DIR}")


if __name__ == "__main__":
    main()
