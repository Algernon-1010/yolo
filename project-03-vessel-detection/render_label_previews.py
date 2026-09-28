"""Render three representative YOLO label previews for visual review."""

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "dataset" / "labels" / "train"
OUTPUT_DIR = PROJECT_ROOT / "previews"
SAMPLES = ("ship_0001", "ship_0017", "ship_0034")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem in SAMPLES:
        image = cv2.imread(str(IMAGE_DIR / f"{stem}.jpg"))
        if image is None:
            raise FileNotFoundError(f"Missing image: {stem}.jpg")
        height, width = image.shape[:2]
        label_file = LABEL_DIR / f"{stem}.txt"
        for line in label_file.read_text(encoding="utf-8").splitlines():
            class_id, x, y, box_w, box_h = map(float, line.split())
            x1 = int((x - box_w / 2) * width)
            y1 = int((y - box_h / 2) * height)
            x2 = int((x + box_w / 2) * width)
            y2 = int((y + box_h / 2) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(
                image,
                "cargo_vessel",
                (x1, max(35, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
        output = OUTPUT_DIR / f"{stem}_preview.jpg"
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"Could not write preview: {output}")
        print(output)


if __name__ == "__main__":
    main()
