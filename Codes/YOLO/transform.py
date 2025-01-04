import os
import json
from PIL import Image

def convert_to_yolo_format(label_file, image_dir, output_dir, class_id=0):
    with open(label_file, "r") as f:
        labels = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    train_txt = open(os.path.join(output_dir, "train.txt"), "w")

    for data in labels:
        image_path = os.path.join(image_dir, data["id"])
        txt_path = os.path.join(output_dir, os.path.splitext(data["id"])[0] + ".txt")
        
        with open(txt_path, "w") as txt_file:
            img_width, img_height = Image.open(image_path).size
            for box in data["region"]:
                x_min, y_min, x_max, y_max = box
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        train_txt.write(f"{image_path}\n")
    
    train_txt.close()
    print(f"Labels converted to YOLO format and saved in {output_dir}")

# 使用路径
label_file = "/home/zgm2024/CVpj/train/label_train.json"
image_dir = "/home/zgm2024/CVpj/train/images"
output_dir = "/home/zgm2024/CVpj/yolo/labels"

convert_to_yolo_format(label_file, image_dir, output_dir)
