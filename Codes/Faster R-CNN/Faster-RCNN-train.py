import os
import json
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm


# 自定义数据集类
class TamperingDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None, min_area=10):
        self.image_dir = image_dir
        self.transform = transform
        self.min_area = min_area
        with open(label_file, "r") as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.labels[idx]
        img_path = os.path.join(self.image_dir, data["id"])
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)

        # 处理空 region 的情况
        if len(data["region"]) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(data["region"], dtype=torch.float32)
            # 计算区域面积
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid_indices = areas >= self.min_area
            boxes = boxes[valid_indices]
            labels = torch.ones((len(boxes),), dtype=torch.int64)

            if len(boxes) == 0:
                print(f"Skipping image {data['id']} due to small regions.")
                return None  # 跳过此图片

        target = {"boxes": boxes, "labels": labels}
        if self.transform:
            image = self.transform(image)

        return image, target, original_size


# 定义 Faster R-CNN 模型
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# 自定义 collate_fn
def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # 过滤掉 None 数据
    images, targets, sizes = zip(*batch)
    return list(images), list(targets), sizes


# 训练模型函数
def train_model(model, dataloader, optimizer, num_epochs, device, scheduler=None):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

        # 更新学习率
        if scheduler:
            scheduler.step()


# 主函数
def main():
    # 参数设置
    train_label_file = "/home/zgm2024/CVpj/train/label_train.json"
    train_image_dir = "/home/zgm2024/CVpj/train/images"
    model_path = "/home/zgm2024/CVpj/train/model/tampering_detection_model10.pth"
    output_model_path = "/home/zgm2024/CVpj/train/model/tampering_detection_model_updated.pth"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 2  # 1 类篡改 + 1 背景

    # 数据加载与处理
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    train_dataset = TamperingDataset(train_label_file, train_image_dir, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 模型加载
    model = get_model(num_classes)
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No pre-trained model found. Starting training from scratch.")
    model = model.to(device)

    # 定义优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 模型训练
    num_epochs = 10
    train_model(model, train_dataloader, optimizer, num_epochs, device, scheduler=scheduler)

    # 保存模型
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")


if __name__ == "__main__":
    main()
