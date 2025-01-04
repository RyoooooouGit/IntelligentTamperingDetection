import os
import random
import shutil

# 设置图片和标签文件目录路径
image_dir = '/home/zgm2024/CVpj/yolo/datasets/image_change'
label_dir = '/home/zgm2024/CVpj/yolo/datasets/labels'

# 设置划分后的目标目录路径
train_image_dir = '/home/zgm2024/CVpj/yolo/datasets/train/images增强'
train_label_dir = '/home/zgm2024/CVpj/yolo/datasets/train/labels增强'
val_image_dir = '/home/zgm2024/CVpj/yolo/datasets/val/images增强'
val_label_dir = '/home/zgm2024/CVpj/yolo/datasets/val/labels增强'

# 创建目标文件夹
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取所有图片文件（假设是.jpg, .png格式）
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 打乱文件列表，确保随机划分
random.shuffle(image_files)

# 划分为80%训练集和20%验证集
split_index = int(len(image_files) * 0.8)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# 移动文件的函数
def move_files(files, source_image_dir, source_label_dir, target_image_dir, target_label_dir):
    for file in files:
        # 移动图片文件
        src_image = os.path.join(source_image_dir, file)
        dst_image = os.path.join(target_image_dir, file)
        shutil.copy(src_image, dst_image)

        # 移动对应的标签文件
        label_file = file.replace(file.split('.')[-1], 'txt')  # 替换扩展名为.txt
        src_label = os.path.join(source_label_dir, label_file)
        dst_label = os.path.join(target_label_dir, label_file)
        
        # 如果标签文件存在，才复制
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"标签文件未找到: {label_file}")

# 移动训练集文件
move_files(train_files, image_dir, label_dir, train_image_dir, train_label_dir)

# 移动验证集文件
move_files(val_files, image_dir, label_dir, val_image_dir, val_label_dir)

print(f"数据集划分完成: {len(train_files)}张图片用于训练，{len(val_files)}张图片用于验证。")
