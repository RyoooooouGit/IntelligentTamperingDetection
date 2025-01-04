import os
import json
from ultralytics import YOLO
from PIL import Image

# 模型路径和测试图片目录
model_path = "/home/zgm2024/CVpj/yolo/runs/train11m增强1/tampering_detection3/weights/best.pt"
test_images_dir = "/home/zgm2024/CVpj/val_images"
output_json_path = "/home/zgm2024/CVpj/yolo/datasets/json/eval_output_11m_epoch200_batch256_conf0.2_image_size800.json"

# 加载 YOLO 模型
model = YOLO(model_path)

# 定义结果列表
results_list = []

# 遍历测试图片目录中的所有图片
for img_name in sorted(os.listdir(test_images_dir)):
    img_path = os.path.join(test_images_dir, img_name)

    # 检查文件是否为图片
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    # 推理
    results = model(img_path, imgsz=800, conf=0.2, device=0)

    # 初始化检测区域
    region_list = []

    # 提取检测框信息
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(float, result.xyxy[0].tolist())
        region_list.append([x1, y1, x2, y2])

    # 构建 JSON 结果
    results_list.append({
        "id": img_name,
        "region": region_list
    })

# 保存结果到 JSON 文件
with open(output_json_path, 'w') as f:
    json.dump(results_list, f, indent=4)

print(f"Evaluation completed. Results saved to {output_json_path}")
