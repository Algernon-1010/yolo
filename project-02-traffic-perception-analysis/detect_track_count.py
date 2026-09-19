"""CPU-only traffic perception: YOLO detection, ByteTrack tracking and line counting."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = r"D:\ultralytics-main\ultralytics-main\models\yolov8n.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack traffic perception on CPU")
    parser.add_argument("--source", default=str(PROJECT_DIR / "data" / "input.mp4"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--line-ratio", type=float, default=0.55, help="Horizontal counting-line position, 0-1")
    parser.add_argument("--show", action="store_true", help="Show an OpenCV preview window")
    return parser.parse_args()


def crossed_line(previous_y, current_y, line_y):
    """Return True only when a tracked target crosses the counting line."""
    return (previous_y < line_y <= current_y) or (previous_y > line_y >= current_y)


def main():
    args = parse_args()
    source = Path(args.source)
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * args.line_ratio)
    output_path = output_dir / "tracked_video.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    model = YOLO(args.model)
    last_center_y = {}
    counted_track_ids = set()
    class_counts = Counter()
    class_confidences = defaultdict(list)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            device="cpu",
            verbose=False,
        )[0]
        annotated = result.plot()
        cv2.line(annotated, (0, line_y), (width, line_y), (0, 255, 255), 2)

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.int().cpu().tolist()

            for track_id, class_id, confidence, (x1, y1, x2, y2) in zip(
                track_ids, class_ids, confidences, coordinates
            ):
                center_y = (y1 + y2) // 2
                class_name = model.names[class_id]
                previous_y = last_center_y.get(track_id)
                if (
                    previous_y is not None
                    and track_id not in counted_track_ids
                    and crossed_line(previous_y, center_y, line_y)
                ):
                    counted_track_ids.add(track_id)
                    class_counts[class_name] += 1
                    class_confidences[class_name].append(confidence)
                last_center_y[track_id] = center_y

        summary = "  ".join(f"{name}: {count}" for name, count in class_counts.items()) or "No crossings"
        cv2.putText(
            annotated, summary, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        writer.write(annotated)

        if args.show:
            cv2.imshow("Traffic perception (press q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    statistics_path = output_dir / "statistics.csv"
    with statistics_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["class_name", "crossing_count", "mean_confidence"])
        for class_name, count in class_counts.most_common():
            mean_confidence = sum(class_confidences[class_name]) / len(class_confidences[class_name])
            csv_writer.writerow([class_name, count, f"{mean_confidence:.3f}"])

    print(f"Saved video: {output_path}")
    print(f"Saved statistics: {statistics_path}")


if __name__ == "__main__":
    main()
