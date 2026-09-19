from pathlib import Path

import cv2
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO(r"D:\ultralytics-main\ultralytics-main\models\yolov8n.pt")

# 打开输入视频
video_path = "data/input.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"无法打开视频：{video_path}")

# 获取视频参数，创建输出视频
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "detected.mp4"
writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# 逐帧检测
while True:
    success, frame = cap.read()

    if not success:
        break

    result = model(frame, conf=0.45, verbose=False)[0]
    annotated_frame = result.plot()

    writer.write(annotated_frame)

    cv2.imshow("YOLO Video Detection", annotated_frame)

    # 按 q 可提前退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#释放资源
cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"视频检测完成，结果已保存为：{Path(output_path).resolve()}")