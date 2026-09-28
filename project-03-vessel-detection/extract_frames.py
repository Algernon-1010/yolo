"""Extract one training frame per second from the sample vessel video."""

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE = PROJECT_ROOT / "videos" / "cargo_ship_river_bridge_34s.mp4"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "images" / "train"
PREFIX = "ship_"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Video not found: {SOURCE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(OUTPUT_DIR.glob(f"{PREFIX}*.jpg"))
    if existing:
        raise FileExistsError(
            f"Found {len(existing)} existing '{PREFIX}*.jpg' images in {OUTPUT_DIR}. "
            "No files were overwritten."
        )

    capture = cv2.VideoCapture(str(SOURCE))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Could not read the video frame rate.")

    frame_interval = max(1, round(fps))
    frame_index = 0
    written = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % frame_interval == 0:
            output_file = OUTPUT_DIR / f"{PREFIX}{written + 1:04d}.jpg"
            if not cv2.imwrite(str(output_file), frame):
                raise RuntimeError(f"Could not write frame: {output_file}")
            written += 1

        frame_index += 1

    capture.release()

    if written == 0:
        raise RuntimeError("No frames were extracted.")

    print(f"Extracted {written} images at 1 frame per second.")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
