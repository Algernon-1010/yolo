# 项目 02：交通场景目标感知与行为统计

## 目标

在 CPU 环境下，使用 YOLOv8 和 ByteTrack 对交通视频进行目标检测、多目标跟踪和越线计数，并导出类别统计与平均置信度。

## 功能流程

```text
输入视频 → YOLO 目标检测 → ByteTrack 目标 ID 跟踪 → 虚拟线越线判断 → 视频与 CSV 输出
```

## 环境

- Python 3.12
- Ultralytics 8.4.137
- OpenCV 5.0.0
- CPU 推理（不需要 GPU）

## 使用方法

1. 将视频放入 `data/input.mp4`。
2. 在本目录运行：

```bash
python detect_track_count.py --conf 0.45 --show
```

3. 输出文件：

```text
outputs/tracked_video.mp4
outputs/statistics.csv
```

## 实验建议

分别使用 `--conf 0.25`、`--conf 0.45` 和 `--conf 0.65` 运行，比较检测数量、漏检情况和平均置信度变化。

## 说明

计数线默认位于视频高度的 55%，可通过 `--line-ratio` 调整。每个跟踪 ID 只会计数一次，因此能够避免同一目标在多帧中被重复统计。
