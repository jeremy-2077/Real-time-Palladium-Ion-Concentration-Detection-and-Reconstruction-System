import os
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import random
import torch

class CustomDataset(data.Dataset):
    """
    自定义数据集类，用于加载图片并进行数据增强
    """
    def __init__(self, data_path, transform=None, augment=True, aug_ratio=0.25):
        self.data_path = data_path
        self.transform = transform
        self.augment = augment
        self.aug_ratio = aug_ratio
        
        # 获取所有图片路径和标签
        self.images = []
        self.labels = []
        self.aug_indices = []
        
        for img_name in os.listdir(data_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(data_path, img_name)
                label = float(img_name.split('-')[0])
                self.images.append(img_path)
                self.labels.append(label)
        
        if self.augment:
            num_aug = int(len(self.images) * aug_ratio)
            self.aug_indices = random.sample(range(len(self.images)), num_aug)
        
        # 修改数据增强转换，移除 Lambda 函数
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)  # 使用内置的高斯模糊替代自定义噪声
            ], p=0.5)
        ])

    def __len__(self):
        return len(self.images) + len(self.aug_indices)

    def __getitem__(self, idx):
        is_augmented = self.augment and idx >= len(self.images)
        
        if is_augmented:
            orig_idx = self.aug_indices[idx - len(self.images)]
            img_path = self.images[orig_idx]
            label = self.labels[orig_idx]
            
            image = Image.open(img_path).convert('RGB')
            image = self.aug_transforms(image)
            
            # 如果仍然需要添加噪声，在这里直接添加
            if random.random() < 0.5:
                image = np.array(image)
                noise = np.random.normal(0, 25, image.shape)
                image = np.clip(image + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(image)
        else:
            img_path = self.images[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label, dtype=torch.float32)
            
        return image, label