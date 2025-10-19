import argparse
import torch
from torch.utils.data import DataLoader
from moco.DMS import DMSEncoder
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
from tqdm import tqdm
import timm
import torch.nn as nn

class SwinEncoder(nn.Module):
    def __init__(self, base_model='swin_tiny_patch4_window7_224'):
        super(SwinEncoder, self).__init__()
        self.encoder = timm.create_model(base_model, pretrained=True, num_classes=0) 
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
# 自定义数据集
class JadeMoCoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集根目录，包含多个类别文件夹。
            transform (callable, optional): 可选的转换操作，例如数据增强等。
        """
        self.root_dir = root_dir
        self.transform = transform
        # 获取所有类别（文件夹名称作为类别）
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 获取所有图像路径和对应的标签
        self.imgs = []
        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):  # 只选择图片文件
                    img_path = os.path.join(cls_path, img_name)
                    label = self.class_to_idx[cls_name]
                    self.imgs.append((img_path, label))

    def __len__(self):
        """返回数据集的大小"""
        return len(self.imgs)

    def __getitem__(self, idx):
        """获取指定索引的图像和标签"""
        img_path, label = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')  # 加载图像，并确保是RGB格式

        if self.transform:
            img = self.transform(img)

        return img, label


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='恢复 MoCo 训练并评估检索准确率')
    parser.add_argument('--train_data', default=r"G:\CODE\data\data\train", help='训练集数据路径')
    parser.add_argument('--val_data', default=r"G:\CODE\data\data\val", help='验证集数据路径')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--resume', default='', help='MoCo 训练检查点路径')
    return parser.parse_args()


# 提取特征
def extract_features(model, dataloader, device):
    features = []
    labels = []
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = F.normalize(feats, dim=1)  # 特征归一化
            features.append(feats.cpu())
            labels.extend(targets)
    return torch.cat(features), torch.tensor(labels)


def compute_topk_accuracy(train_feats, train_labels, val_feats, val_labels, k=5):
    total = len(val_labels)
    # 对每个验证特征，计算与训练集中所有特征的相似度
    sims = torch.mm(val_feats, train_feats.t())  # shape: [val_num, train_num]
    
    # 计算Top-1准确率
    preds = torch.argmax(sims, dim=1)
    predicted_labels = train_labels[preds]
    top1_correct = (predicted_labels == val_labels).sum().item()
    top1_acc = top1_correct / total
    
    # 计算Top-5准确率
    _, top5_preds = torch.topk(sims, k=k, dim=1)  # 获取相似度最高的5个索引
    top5_predicted_labels = train_labels[top5_preds]  # 获取对应的标签
    # 检查真实标签是否在前5个预测中
    top5_correct = torch.any(top5_predicted_labels == val_labels.unsqueeze(1), dim=1).sum().item()
    top5_acc = top5_correct / total
    
    return top1_acc, top5_acc



# 主函数
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义图像预处理
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据
    train_dataset = JadeMoCoDataset(root_dir=args.train_data, transform=train_transforms)
    val_dataset = JadeMoCoDataset(root_dir=args.val_data, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    #model = DMSEncoder().to(device)
    model = SwinEncoder().to(device)

    # 恢复 MoCo checkpoint，只载入 encoder_q
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['model']
        encoder_q_state = {
            k[len('encoder_q.'):]: v
            for k, v in state_dict.items()
            if k.startswith('encoder_q.')
        }
        missing, unexpected = model.load_state_dict(encoder_q_state, strict=False)
        print(f"Loaded encoder_q weights from checkpoint, missing keys: {missing}, unexpected keys: {unexpected}")
        print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")

    model.eval()

    # 提取训练和验证集特征
    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)

    # 计算Top-1检索准确率
    top1_acc, top5_acc = compute_topk_accuracy(train_feats, train_labels, val_feats, val_labels)
    print(f"Top-1 Retrieval Accuracy: {top1_acc * 100:.2f}%")
    print(f"Top-5 Retrieval Accuracy: {top5_acc * 100:.2f}%")


if __name__ == '__main__':
    main()
