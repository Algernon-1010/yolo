# 智能船舶 YOLO 船舶检测

本项目用于从船岸视频中抽帧、标注、训练并验证船舶检测模型。

## 目录

```text
yolo/
├─ dataset/
│  └─ data.yaml                 # YOLO 数据集配置
├─ models/                      # 本地 YOLO 权重，不提交 Git
├─ videos/                      # 原始视频，不提交 Git
├─ previews/                    # 标注预览图，不提交 Git
├─ extract_frames.py            # 训练集抽帧
├─ prelabel_boats.py            # 训练集 YOLOv8 预标注
├─ prepare_validation_set.py    # 验证集抽帧与预标注
├─ render_label_previews.py     # 训练集标注预览
├─ predefined_classes.txt       # LabelImg 类别顺序
├─ requirements.txt
└─ .gitignore
```

## 类别定义

| ID | 类别 |
|---:|---|
| 0 | construction_vessel |
| 1 | cargo_vessel |
| 2 | passenger_vessel |
| 3 | small_vessel |
| 4 | other_vessel |

## 当前样本

- 训练集：34 张图片，34 个自动预标注框。
- 验证集：34 张图片，57 个自动预标注框。
- 当前样本仅用于验证训练流程。正式训练前须补充岸基摄像头、夜间、雨雾、施工船和小型船数据，并人工复核自动标注。

## 环境与运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

将 `yolov8n.pt` 放入 `models/` 后，可运行：

```powershell
python extract_frames.py
python prelabel_boats.py
python prepare_validation_set.py
```

## Git 约定

Git 仅提交代码、数据集配置、类别定义和文档。原始视频、抽帧图片、标签、预览图、权重和训练输出均保留在本地或上传至专用数据存储。

## 数据来源

当前测试视频来自 Pexels 的免费可用素材。提交或发布时应保留原始来源和许可说明。
