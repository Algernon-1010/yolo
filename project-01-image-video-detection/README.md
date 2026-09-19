# 项目1：YOLO 图片与视频目标检测

## 目标

使用 YOLOv8n 预训练模型和 OpenCV 完成图片、视频目标检测，并输出带类别与置信度的检测结果。

## 项目文件

- `check_env.py`：检查 OpenCV 和 Ultralytics 是否可用。
- `run_detection.py`：逐帧读取视频并输出检测视频。
- `result.jpg`：图片检测结果示例。
- `data/input.mp4`：本地测试视频（不提交到 Git）。
- `detected.mp4`：生成的检测结果视频（不提交到 Git）。

## 环境

```text
Python 3.12
Ultralytics 8.4.137
OpenCV 5.0.0
```

## 运行

将测试视频放入 `data/input.mp4` 后，运行：

```bash
python run_detection.py
```

按 `q` 可提前停止。完成后会在本目录生成 `detected.mp4`。

## 结果

使用 YOLOv8n 对交通场景进行检测，可识别 person、car、bus、bicycle 等类别。
