import os
from ultralytics import YOLO

# 模型与数据路径
model_path = "/home/zgm2024/CVpj/yolo/code/yolo11m.pt"
data_yaml = "/home/zgm2024/CVpj/yolo/datasets/coco.yaml"
save_dir = "/home/zgm2024/CVpj/yolo/runs/train11m增强1"

# 创建保存路径
os.makedirs(save_dir, exist_ok=True)

# 初始化模型
model = YOLO(model_path)

# 开始训练
model.train(
    data=data_yaml,
    epochs=200,
    imgsz=800,
    batch=256,
    save_period=20,
    device='0,',  # 使用第一个 GPU
    project=save_dir,
    name="tampering_detection",
)
