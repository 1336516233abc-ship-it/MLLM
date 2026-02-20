"""
数据加载器 - 为训练和测试提供数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class MultiModalDataset(Dataset):
    """
    多模态数据集
    支持图像理解、生成、编辑任务
    """
    def __init__(self, config, split='train', transform=None):
        self.config = config
        self.split = split

        # 图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # 这里使用合成数据作为示例
        # 实际应用中应该从真实数据集加载
        self.data = self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """
        生成合成数据用于演示
        实际应用中应该替换为真实数据加载逻辑
        """
        num_samples = 1000 if self.split == 'train' else 200

        data = []
        for i in range(num_samples):
            # 生成随机图像和文本
            sample = {
                'image_id': i,
                'text': f"Sample text description {i}",
                'task_type': np.random.randint(0, 3),  # 0: understanding, 1: generation, 2: editing
            }
            data.append(sample)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 生成随机图像（实际应该加载真实图像）
        image = torch.randn(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        image = (image + 1) / 2  # 归一化到[0, 1]
        image = (image - 0.5) / 0.5  # 归一化到[-1, 1]

        # 文本处理（简化版本）
        text = sample['text']
        text_tokens = [ord(c) % self.config.TEXT_VOCAB_SIZE for c in text]

        # 截断或填充
        if len(text_tokens) > self.config.TEXT_MAX_LENGTH:
            text_tokens = text_tokens[:self.config.TEXT_MAX_LENGTH]
            text_mask = [1] * self.config.TEXT_MAX_LENGTH
        else:
            text_mask = [1] * len(text_tokens) + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))
            text_tokens = text_tokens + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_mask = torch.tensor(text_mask, dtype=torch.long)

        # 目标图像（用于生成任务）
        target_image = torch.randn(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        target_image = (target_image + 1) / 2
        target_image = (target_image - 0.5) / 0.5

        # 编辑任务的数据
        # 编辑后的图像（对源图像进行修改）
        edited_image = image.clone()
        # 简单模拟：在某个区域添加噪声作为"编辑"
        edit_region_h = np.random.randint(0, self.config.IMAGE_SIZE // 2)
        edit_region_w = np.random.randint(0, self.config.IMAGE_SIZE // 2)
        edit_h = np.random.randint(self.config.IMAGE_SIZE // 4, self.config.IMAGE_SIZE // 2)
        edit_w = np.random.randint(self.config.IMAGE_SIZE // 4, self.config.IMAGE_SIZE // 2)

        edited_image[:, edit_region_h:edit_region_h+edit_h, edit_region_w:edit_region_w+edit_w] = \
            torch.randn(3, edit_h, edit_w) * 0.5

        # 编辑掩码（1表示编辑区域，0表示保持区域）
        edit_mask = torch.zeros(1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        edit_mask[:, edit_region_h:edit_region_h+edit_h, edit_region_w:edit_region_w+edit_w] = 1.0

        # 生成理解任务的标签
        num_patches = (self.config.IMAGE_SIZE // self.config.PATCH_SIZE) ** 2 + 1

        targets = {
            'task_labels': torch.tensor(sample['task_type'], dtype=torch.long),
            'semantic_labels': torch.randint(0, 100, (num_patches,), dtype=torch.long),
            'bboxes': torch.rand(num_patches, 4),
            'relation_matrix': torch.randint(0, 10, (num_patches, num_patches), dtype=torch.long),
            'edit_mask': edit_mask  # 添加编辑掩码
        }

        return {
            'images': image,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'target_images': target_image,
            'edited_images': edited_image,  # 添加编辑后的图像
            'targets': targets
        }

def create_dataloaders(config, batch_size=None):
    """
    创建训练和验证数据加载器
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # 创建数据集
    train_dataset = MultiModalDataset(config, split='train')
    val_dataset = MultiModalDataset(config, split='val')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader

def collate_fn(batch):
    """
    自定义批次整理函数
    """
    result = {}

    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            # 对于张量，进行stack
            result[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], dict):
            # 对于字典（如targets），递归处理
            result[key] = {
                sub_key: torch.stack([item[key][sub_key] for item in batch])
                if isinstance(batch[0][key][sub_key], torch.Tensor)
                else [item[key][sub_key] for item in batch]
                for sub_key in batch[0][key].keys()
            }
        else:
            # 对于其他类型，保持为列表
            result[key] = [item[key] for item in batch]

    return result
