"""
图像生成任务数据集加载器
支持多种生成数据集：LAION, Conceptual Captions, COCO Captions等
"""

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class LAIONDataset(Dataset):
    """
    LAION数据集 - 大规模图像-文本对
    用于图像生成任务
    """
    def __init__(self, parquet_file, image_dir, config, transform=None):
        self.image_dir = image_dir
        self.config = config

        # 加载parquet文件
        self.data = pd.read_parquet(parquet_file)

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
        row = self.data.iloc[idx]

        # 获取图像路径和文本
        image_file = row.get('image_path', f"{idx}.jpg")
        caption = row.get('caption', row.get('text', ''))

        # 加载图像
        img_path = os.path.join(self.image_dir, image_file)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果图像加载失败，创建空白图像
            image = Image.new('RGB', (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), (128, 128, 128))

        image = self.transform(image)

        # 文本分词
        text_tokens = [ord(c) % self.config.TEXT_VOCAB_SIZE for c in caption]
        if len(text_tokens) > self.config.TEXT_MAX_LENGTH:
            text_tokens = text_tokens[:self.config.TEXT_MAX_LENGTH]
            text_mask = [1] * self.config.TEXT_MAX_LENGTH
        else:
            text_mask = [1] * len(text_tokens) + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))
            text_tokens = text_tokens + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_mask = torch.tensor(text_mask, dtype=torch.long)

        return {
            'images': torch.zeros_like(image),  # 生成任务不需要输入图像（或使用噪声）
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'target_images': image,  # 目标图像
            'targets': {
                'task_labels': torch.tensor(1, dtype=torch.long)  # 1表示生成任务
            }
        }


class ConceptualCaptionsDataset(Dataset):
    """
    Conceptual Captions (CC3M/CC12M) 数据集
    高质量的图像-文本对，适合图像生成
    """
    def __init__(self, tsv_file, image_dir, config, transform=None):
        self.image_dir = image_dir
        self.config = config

        # 加载TSV文件
        self.data = pd.read_csv(tsv_file, sep='\t', names=['caption', 'url'])

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
        row = self.data.iloc[idx]

        caption = row['caption']
        # 图像文件名通常是索引
        image_file = f"{idx:08d}.jpg"

        # 加载图像
        img_path = os.path.join(self.image_dir, image_file)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果图像加载失败，创建空白图像
            image = Image.new('RGB', (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), (128, 128, 128))

        image = self.transform(image)

        # 文本分词
        text_tokens = [ord(c) % self.config.TEXT_VOCAB_SIZE for c in caption]
        if len(text_tokens) > self.config.TEXT_MAX_LENGTH:
            text_tokens = text_tokens[:self.config.TEXT_MAX_LENGTH]
            text_mask = [1] * self.config.TEXT_MAX_LENGTH
        else:
            text_mask = [1] * len(text_tokens) + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))
            text_tokens = text_tokens + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_mask = torch.tensor(text_mask, dtype=torch.long)

        return {
            'images': torch.zeros_like(image),  # 生成任务不需要输入图像
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'target_images': image,
            'targets': {
                'task_labels': torch.tensor(1, dtype=torch.long)
            }
        }


class COCOCaptionsDataset(Dataset):
    """
    COCO Captions数据集
    每张图像有5个描述，适合图像生成
    """
    def __init__(self, image_dir, annotation_file, config, transform=None):
        self.image_dir = image_dir
        self.config = config

        # 加载COCO标注
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # 构建图像ID到描述的映射
        self.image_to_captions = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_captions:
                self.image_to_captions[img_id] = []
            self.image_to_captions[img_id].append(ann['caption'])

        # 构建图像ID到文件名的映射
        self.image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}

        # 创建样本列表（每个图像-描述对是一个样本）
        self.samples = []
        for img_id, captions in self.image_to_captions.items():
            for caption in captions:
                self.samples.append({
                    'image_id': img_id,
                    'caption': caption
                })

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_id = sample['image_id']
        caption = sample['caption']

        # 加载图像
        img_file = self.image_id_to_file[img_id]
        img_path = os.path.join(self.image_dir, img_file)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), (128, 128, 128))

        image = self.transform(image)

        # 文本分词
        text_tokens = [ord(c) % self.config.TEXT_VOCAB_SIZE for c in caption]
        if len(text_tokens) > self.config.TEXT_MAX_LENGTH:
            text_tokens = text_tokens[:self.config.TEXT_MAX_LENGTH]
            text_mask = [1] * self.config.TEXT_MAX_LENGTH
        else:
            text_mask = [1] * len(text_tokens) + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))
            text_tokens = text_tokens + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_mask = torch.tensor(text_mask, dtype=torch.long)

        return {
            'images': torch.zeros_like(image),  # 生成任务不需要输入图像
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'target_images': image,
            'targets': {
                'task_labels': torch.tensor(1, dtype=torch.long)
            }
        }


class TextToImageDataset(Dataset):
    """
    通用的文本到图像数据集
    支持自定义JSON格式
    """
    def __init__(self, json_file, image_dir, config, transform=None):
        self.image_dir = image_dir
        self.config = config

        # 加载JSON文件
        with open(json_file, 'r') as f:
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

        # 获取图像和文本
        image_file = item.get('image', item.get('image_path', f"{idx}.jpg"))
        caption = item.get('caption', item.get('text', item.get('prompt', '')))

        # 加载图像
        img_path = os.path.join(self.image_dir, image_file)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), (128, 128, 128))

        image = self.transform(image)

        # 文本分词
        text_tokens = [ord(c) % self.config.TEXT_VOCAB_SIZE for c in caption]
        if len(text_tokens) > self.config.TEXT_MAX_LENGTH:
            text_tokens = text_tokens[:self.config.TEXT_MAX_LENGTH]
            text_mask = [1] * self.config.TEXT_MAX_LENGTH
        else:
            text_mask = [1] * len(text_tokens) + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))
            text_tokens = text_tokens + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_mask = torch.tensor(text_mask, dtype=torch.long)

        # 支持条件生成（可选的参考图像）
        reference_image = torch.zeros_like(image)
        if 'reference_image' in item:
            ref_path = os.path.join(self.image_dir, item['reference_image'])
            try:
                reference_image = Image.open(ref_path).convert('RGB')
                reference_image = self.transform(reference_image)
            except:
                pass

        return {
            'images': reference_image,  # 参考图像（可以是零图像）
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'target_images': image,
            'targets': {
                'task_labels': torch.tensor(1, dtype=torch.long)
            }
        }


class DiffusionDBDataset(Dataset):
    """
    DiffusionDB数据集
    包含Stable Diffusion生成的图像和提示词
    """
    def __init__(self, data_dir, config, split='train', transform=None):
        self.data_dir = data_dir
        self.config = config
        self.split = split

        # 加载元数据
        metadata_file = os.path.join(data_dir, f'{split}_metadata.json')
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # 获取图像和提示词
        image_file = item['image_path']
        prompt = item['prompt']

        # 加载图像
        img_path = os.path.join(self.data_dir, 'images', image_file)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), (128, 128, 128))

        image = self.transform(image)

        # 文本分词
        text_tokens = [ord(c) % self.config.TEXT_VOCAB_SIZE for c in prompt]
        if len(text_tokens) > self.config.TEXT_MAX_LENGTH:
            text_tokens = text_tokens[:self.config.TEXT_MAX_LENGTH]
            text_mask = [1] * self.config.TEXT_MAX_LENGTH
        else:
            text_mask = [1] * len(text_tokens) + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))
            text_tokens = text_tokens + [0] * (self.config.TEXT_MAX_LENGTH - len(text_tokens))

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_mask = torch.tensor(text_mask, dtype=torch.long)

        return {
            'images': torch.zeros_like(image),
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'target_images': image,
            'targets': {
                'task_labels': torch.tensor(1, dtype=torch.long)
            }
        }


def create_generation_dataloader(dataset_type, data_root, config, batch_size=8, split='train'):
    """
    创建生成任务的数据加载器

    Args:
        dataset_type: 'laion' | 'conceptual_captions' | 'coco_captions' | 'text2image' | 'diffusiondb'
        data_root: 数据集根目录
        config: 配置对象
        batch_size: 批次大小
        split: 'train' | 'val' | 'test'
    """
    from torch.utils.data import DataLoader

    if dataset_type == 'laion':
        parquet_file = os.path.join(data_root, f'{split}.parquet')
        image_dir = os.path.join(data_root, 'images')
        dataset = LAIONDataset(parquet_file, image_dir, config)

    elif dataset_type == 'conceptual_captions':
        tsv_file = os.path.join(data_root, f'{split}.tsv')
        image_dir = os.path.join(data_root, 'images')
        dataset = ConceptualCaptionsDataset(tsv_file, image_dir, config)

    elif dataset_type == 'coco_captions':
        image_dir = os.path.join(data_root, f'{split}2017')
        annotation_file = os.path.join(data_root, 'annotations', f'captions_{split}2017.json')
        dataset = COCOCaptionsDataset(image_dir, annotation_file, config)

    elif dataset_type == 'text2image':
        json_file = os.path.join(data_root, f'{split}.json')
        image_dir = os.path.join(data_root, 'images')
        dataset = TextToImageDataset(json_file, image_dir, config)

    elif dataset_type == 'diffusiondb':
        dataset = DiffusionDBDataset(data_root, config, split)

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
from MLLM.utils.generation_data_loader import create_generation_dataloader

config = Config()

# 加载LAION数据集
train_loader = create_generation_dataloader(
    dataset_type='laion',
    data_root='/path/to/laion',
    config=config,
    batch_size=8,
    split='train'
)

# 加载Conceptual Captions数据集
train_loader = create_generation_dataloader(
    dataset_type='conceptual_captions',
    data_root='/path/to/cc3m',
    config=config,
    batch_size=8,
    split='train'
)

# 加载COCO Captions数据集
train_loader = create_generation_dataloader(
    dataset_type='coco_captions',
    data_root='/path/to/coco',
    config=config,
    batch_size=8,
    split='train'
)

# 训练
for batch in train_loader:
    text_tokens = batch['text_tokens']
    target_images = batch['target_images']

    # 训练模型...
"""
