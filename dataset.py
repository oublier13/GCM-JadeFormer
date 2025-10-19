import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import glob


class JadeMoCoDataset(Dataset):
    """支持子文件夹结构的MoCo数据集"""
    def __init__(self, root_dir, bg_dir=None, img_size=224):
        self.root_dir = root_dir
        self.img_size = img_size
        self.samples = self._load_samples()  # 现在会递归查找子文件夹
        self.bg_images = self._load_bg_images(bg_dir) if bg_dir else []
        
        if len(self.samples) == 0:
            raise ValueError(f"未找到任何图像文件！请检查路径: {root_dir}")
            
        print(f"成功加载 {len(self.samples)} 张图片（来自子文件夹）")

        # 数据增强
        self.base_aug = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_samples(self):
        samples = []
        # 支持的图片扩展名（包括各种大小写组合）
        extensions = ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')
        
        # 递归搜索所有子文件夹
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    img_path = os.path.join(root, file)
                    samples.append((img_path, len(samples)))  # 使用索引作为伪标签
                    
        return samples

    def _load_bg_images(self, bg_dir):
        bg_paths = glob.glob(os.path.join(bg_dir, '*.[jp][pn]g'))
        return [Image.open(p).convert('RGB') for p in bg_paths]

    def _augment_with_bg(self, img):
        if not self.bg_images or random.random() > 0.2:
            return self.base_aug(img)
        
        bg = random.choice(self.bg_images).copy()
        bg_w, bg_h = bg.size
        
        angle = random.uniform(-30, 30)
        scale = random.uniform(0.5, 0.9)
        
        # 旋转和缩放图像
        img = img.rotate(angle, resample=Image.BILINEAR)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)

        img_w, img_h = img.size
        if img_w > bg_w or img_h > bg_h:
            img = img.resize((bg_w, bg_h), Image.BILINEAR)

        x = random.randint(0, max(0, bg_w - img_w))
        y = random.randint(0, max(0, bg_h - img_h))
        
        # 粘贴图像到背景上
        bg.paste(img, (x, y))
        return self.base_aug(bg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]  # MoCo不需要标签
        try:
            img = Image.open(img_path).convert('RGB')
            return [
                self.base_aug(img),    # 视图1：标准增强
                self._augment_with_bg(img)  # 视图2：背景增强
            ]
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy = torch.rand(3, self.img_size, self.img_size)
            return [dummy, dummy]