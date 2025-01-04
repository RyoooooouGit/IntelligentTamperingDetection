import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
from tqdm import tqdm

# 1. 自定义数据集类
class TamperingDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.label_file = label_file
        self.transform = transform
        self.imgs = []
        self.regions = []
        
        # 加载标签
        with open(label_file, 'r') as f:
            self.labels = json.load(f)
        
        for label in self.labels:
            img_path = os.path.join(img_dir, label["id"])
            region = label["region"]
            self.imgs.append(img_path)
            self.regions.append(region)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        region = self.regions[idx]
        
        # 使用Pillow加载图像
        try:
            img = Image.open(img_path).convert("RGB")  # 转换为RGB模式，确保一致性
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None  # 返回None表示此图像加载失败
        
        if self.transform:
            img = self.transform(img)
        
        # 转换为框格式 (0: 非篡改, 1: 篡改区域)
        target = {}
        
        # 如果有篡改区域
        if region:
            valid_boxes = self._validate_boxes(region)  # 检查框的有效性
            target["boxes"] = torch.tensor(valid_boxes, dtype=torch.float32)
            target["labels"] = torch.tensor([1] * len(valid_boxes), dtype=torch.int64)  # 1代表篡改区域
        else:
            target["boxes"] = torch.tensor([], dtype=torch.float32)  # 空的bounding box
            target["labels"] = torch.tensor([], dtype=torch.int64)  # 空的labels
        
        return img, target
    
    def _validate_boxes(self, boxes):
        """验证边界框是否有效，确保它们的宽度和高度大于零"""
        valid_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            # 如果宽度和高度大于零，则认为框有效
            if x_min < x_max and y_min < y_max:
                valid_boxes.append(box)
            else:
                print(f"Invalid box: {box}")
        return valid_boxes

# 2. 生成label_val.json（支持多区域）
def generate_val_labels(model, val_img_dir, output_json_path, device, min_area=1024, score_threshold=0.5):
    """
    使用模型生成验证集的标签，支持多个可靠区域的输出。
    
    参数:
    - model: 训练好的检测模型
    - val_img_dir: 验证集图像目录
    - output_json_path: 输出JSON文件路径
    - device: 使用的设备（CPU或GPU）
    - min_area: 最小区域面积
    - score_threshold: 分数阈值
    """
    model.eval()
    result = []
    
    for img_name in tqdm(os.listdir(val_img_dir), desc="Processing Images"):
        img_path = os.path.join(val_img_dir, img_name)
        if os.path.isfile(img_path):
            try:
                original_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
            
            # 获取原始图像尺寸
            original_width, original_height = original_image.size
            
            # 定义图像变换
            transform = transforms.Compose([
                transforms.Resize((800, 800)),  # 统一变换大小
                transforms.ToTensor(),
            ])
            
            # 变换后的图像
            processed_image = transform(original_image).unsqueeze(0).to(device)
            processed_width, processed_height = 800, 800  # 变换后的图像尺寸
            
            # 坐标变换比例
            scale_x = original_width / processed_width
            scale_y = original_height / processed_height
            
            # 预测
            with torch.no_grad():
                outputs = model(processed_image)
            
            for output in outputs:
                pred_boxes = output['boxes'].cpu().numpy().tolist()
                pred_scores = output['scores'].cpu().numpy().tolist()
                
                # 过滤满足条件的框（根据分数和面积）
                valid_boxes = []
                for box, score in zip(pred_boxes, pred_scores):
                    x_min, y_min, x_max, y_max = box
                    area = (x_max - x_min) * (y_max - y_min)
                    if score >= score_threshold and area >= min_area:
                        # 将框从变换后的尺寸映射回原图尺寸
                        valid_boxes.append([
                            x_min * scale_x,
                            y_min * scale_y,
                            x_max * scale_x,
                            y_max * scale_y
                        ])
                
                # 保存所有满足条件的框
                result.append({
                    "id": img_name,
                    "region": valid_boxes,  # 包含多个区域
                })
    
    # 保存结果到JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Generated {output_json_path}")

# 3. 加载模型并生成评估结果
def main():
    # 模型路径
    model_path = '/home/zgm2024/CVpj/train/model/tampering_detection_model1.pth'
    
    # 验证集图片目录
    val_img_dir = '/home/zgm2024/CVpj/val_images'
    
    # 输出文件路径
    output_json_path = '/home/zgm2024/CVpj/train/json/label_val1.json'
    
    # 设置设备为 GPU 优先
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载预训练模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 生成验证集的label文件
    generate_val_labels(model, val_img_dir, output_json_path, device, min_area=32*32, score_threshold=0.5)

if __name__ == "__main__":
    main()
