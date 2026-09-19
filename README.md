# YOLO + OpenCV 目标检测

## 项目简介

使用 YOLOv8 和 OpenCV 实现图片与视频目标检测。

## 环境

- Python 3.12
- Ultralytics 8.4.137
- OpenCV 5.0.0

## 功能

- 对图片进行目标检测
- 对视频逐帧进行目标检测
- 输出带检测框的视频结果

## 运行方式

```bash
python run_detection.py
```

## 实验结果

使用 YOLOv8n 预训练模型对交通场景进行检测，可识别 person、car、bus、bicycle 等类别。
