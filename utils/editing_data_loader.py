"""
图像编辑数据集加载器
支持多种编辑数据集格式：InstructPix2Pix, MagicBrush, RefCOCO等
"""

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms

class InstructPix2PixDataset(Dataset):
    """
    InstructPix2Pix格式的编辑数据集
    数据格式：
    - source_image: 原始图像
    - edited_image: 编辑后的图像
    - instruction: 编辑指令文本
    """
    def __init__(self, data_root, annotation_file, config, transform=None):
        self.data_root = data_root
        self.config = config

        # 加载标注
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # 图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # 加载源图像
        source_path = os.path.join(self.data_root, ann['source_image'])
        source_image = Image.open(source_path).convert('RGB')
        source_image = self.transform(source_image)

        # 加载编辑后的图像
        edited_path = os.path.join(self.data_root, ann['edited_image'])
        edited_image = Image.open(edited_path).convert('RGB')
        edited_image = self.transform(edited_image)

        # 编辑指令
        instruction = ann['instruction']

        # 文本分词（简化版本，实际应使用专业分词器）
        text_tokens = [ord(c) % self.config.TEXT_VOCAB_SIZE for c in instruction]
        if len(text_tokens) > self.config.TEXT_MAX_LENGTH:
            text_tokens = text_tokens[:self.config.TEXT_MAX_LENGTH]
            text_mask = [1] * self.config.TEXT_MAX_LENGTH
        else:
            text_mask = [1] * len(text_tokens) + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))
            text_tokens = text_tokens + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_mask = torch.tensor(text_mask, dtype=torch.long)

        # 如果有编辑掩码
        edit_mask = None
        if 'edit_mask' in ann:
            mask_path = os.path.join(self.data_root, ann['edit_mask'])
            edit_mask = Image.open(mask_path).convert('L')
            edit_mask = transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))(edit_mask)
            edit_mask = transforms.ToTensor()(edit_mask)
        else:
            # 如果没有掩码，使用全1（表示整个图像都可能被编辑）
            edit_mask = torch.ones(1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)

        return {
            'images': source_image,
            'edited_images': edited_image,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'targets': {
                'edit_mask': edit_mask,
                'task_labels': torch.tensor(2, dtype=torch.long)  # 2表示编辑任务
            }
        }


class RefCOCOEditDataset(Dataset):
    """
    RefCOCO格式的编辑数据集
    包含指代表达式和目标区域
    """
    def __init__(self, image_dir, annotation_file, config, transform=None):
        self.image_dir = image_dir
        self.config = config

        # 加载标注
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # 加载图像
        image_path = os.path.join(self.image_dir, ann['image_file'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # 指代表达式（如"左边的人"）
        referring_expression = ann['referring_expression']

        # 编辑指令（如"移除左边的人"）
        edit_instruction = ann.get('edit_instruction', f"Edit {referring_expression}")

        # 文本分词
        text_tokens = [ord(c) % self.config.TEXT_VOCAB_SIZE for c in edit_instruction]
        if len(text_tokens) > self.config.TEXT_MAX_LENGTH:
            text_tokens = text_tokens[:self.config.TEXT_MAX_LENGTH]
            text_mask = [1] * self.config.TEXT_MAX_LENGTH
        else:
            text_mask = [1] * len(text_tokens) + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))
            text_tokens = text_tokens + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_mask = torch.tensor(text_mask, dtype=torch.long)

        # 从边界框创建掩码
        bbox = ann['bbox']  # [x, y, w, h]
        edit_mask = torch.zeros(1, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)

        # 归一化边界框到图像尺寸
        x, y, w, h = bbox
        x_start = int(x * self.config.IMAGE_SIZE)
        y_start = int(y * self.config.IMAGE_SIZE)
        x_end = int((x + w) * self.config.IMAGE_SIZE)
        y_end = int((y + h) * self.config.IMAGE_SIZE)

        edit_mask[:, y_start:y_end, x_start:x_end] = 1.0

        # 编辑后的图像（实际应该加载真实的编辑结果）
        # 这里简化处理，实际数据集应该提供编辑后的图像
        edited_image = image.clone()

        return {
            'images': image,
            'edited_images': edited_image,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'targets': {
                'edit_mask': edit_mask,
                'task_labels': torch.tensor(2, dtype=torch.long),
                'bboxes': torch.tensor(bbox).unsqueeze(0)
            }
        }


class MagicBrushDataset(Dataset):
    """
    MagicBrush数据集
    包含源图像、编辑指令、编辑掩码和目标图像
    """
    def __init__(self, data_root, split='train', config=None, transform=None):
        self.data_root = data_root
        self.config = config
        self.split = split

        # 加载数据
        annotation_file = os.path.join(data_root, f'{split}.json')
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 加载源图像
        source_path = os.path.join(self.data_root, 'images', item['source_img'])
        source_image = Image.open(source_path).convert('RGB')
        source_image = self.transform(source_image)

        # 加载目标图像
        target_path = os.path.join(self.data_root, 'images', item['target_img'])
        target_image = Image.open(target_path).convert('RGB')
        target_image = self.transform(target_image)

        # 加载掩码
        mask_path = os.path.join(self.data_root, 'masks', item['mask_img'])
        mask = Image.open(mask_path).convert('L')
        mask = transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))(mask)
        mask = transforms.ToTensor()(mask)

        # 编辑指令
        instruction = item['instruction']

        # 文本分词
        text_tokens = [ord(c) % self.config.TEXT_VOCAB_SIZE for c in instruction]
        if len(text_tokens) > self.config.TEXT_MAX_LENGTH:
            text_tokens = text_tokens[:self.config.TEXT_MAX_LENGTH]
            text_mask = [1] * self.config.TEXT_MAX_LENGTH
        else:
            text_mask = [1] * len(text_tokens) + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))
            text_tokens = text_tokens + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_mask = torch.tensor(text_mask, dtype=torch.long)

        return {
            'images': source_image,
            'edited_images': target_image,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'targets': {
                'edit_mask': mask,
                'task_labels': torch.tensor(2, dtype=torch.long)
            }
        }


def create_editing_dataloader(dataset_type, data_root, config, batch_size=8, split='train'):
    """
    创建编辑任务的数据加载器

    Args:
        dataset_type: 'instructpix2pix' | 'refcoco' | 'magicbrush'
        data_root: 数据集根目录
        config: 配置对象
        batch_size: 批次大小
        split: 'train' | 'val' | 'test'
    """
    from torch.utils.data import DataLoader

    if dataset_type == 'instructpix2pix':
        annotation_file = os.path.join(data_root, f'{split}.json')
        dataset = InstructPix2PixDataset(data_root, annotation_file, config)

    elif dataset_type == 'refcoco':
        annotation_file = os.path.join(data_root, f'refcoco_{split}.json')
        dataset = RefCOCOEditDataset(data_root, annotation_file, config)

    elif dataset_type == 'magicbrush':
        dataset = MagicBrushDataset(data_root, split, config)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return dataloader


# 使用示例
"""
from MLLM.utils.config import Config
from MLLM.utils.editing_data_loader import create_editing_dataloader

config = Config()

# 加载InstructPix2Pix数据集
train_loader = create_editing_dataloader(
    dataset_type='instructpix2pix',
    data_root='/path/to/instructpix2pix',
    config=config,
    batch_size=8,
    split='train'
)

# 加载MagicBrush数据集
train_loader = create_editing_dataloader(
    dataset_type='magicbrush',
    data_root='/path/to/magicbrush',
    config=config,
    batch_size=8,
    split='train'
)

# 训练
for batch in train_loader:
    source_images = batch['images']
    edited_images = batch['edited_images']
    instructions = batch['text_tokens']
    edit_masks = batch['targets']['edit_mask']

    # 训练模型...
"""
