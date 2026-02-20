"""
图像理解任务数据集加载器
支持多种理解数据集：COCO, Visual Genome, ADE20K等
"""

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class COCOUnderstandingDataset(Dataset):
    """
    COCO数据集 - 用于图像理解任务
    包含：目标检测、实例分割、图像描述
    """
    def __init__(self, image_dir, annotation_file, config, transform=None):
        self.image_dir = image_dir
        self.config = config

        # 加载COCO标注
        from pycocotools.coco import COCO
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # 类别映射（COCO有80个类别）
        self.category_mapping = {cat['id']: idx for idx, cat in enumerate(self.coco.loadCats(self.coco.getCatIds()))}

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # 加载图像
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (W, H)

        # 应用变换
        image = self.transform(image)

        # 加载标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 获取图像描述
        cap_ids = self.coco.getAnnIds(imgIds=img_id)
        caps = self.coco.loadAnns(cap_ids)
        caption = caps[0]['caption'] if caps else "Describe this image"

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

        # 处理边界框和类别
        num_patches = (self.config.IMAGE_SIZE // self.config.PATCH_SIZE) ** 2 + 1
        bboxes = torch.zeros(num_patches, 4)
        semantic_labels = torch.full((num_patches,), -1, dtype=torch.long)  # -1表示背景

        for i, ann in enumerate(anns[:num_patches]):  # 限制数量
            # 边界框 [x, y, w, h] -> 归一化到[0, 1]
            bbox = ann['bbox']
            x, y, w, h = bbox
            x_norm = x / original_size[0]
            y_norm = y / original_size[1]
            w_norm = w / original_size[0]
            h_norm = h / original_size[1]

            bboxes[i] = torch.tensor([x_norm, y_norm, w_norm, h_norm])

            # 类别标签
            cat_id = ann['category_id']
            semantic_labels[i] = self.category_mapping.get(cat_id, 0)

        # 构建关系矩阵（简化版本：基于空间关系）
        relation_matrix = self._build_relation_matrix(bboxes, num_patches)

        targets = {
            'task_labels': torch.tensor(0, dtype=torch.long),  # 0表示理解任务
            'semantic_labels': semantic_labels,
            'bboxes': bboxes,
            'relation_matrix': relation_matrix
        }

        return {
            'images': image,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'targets': targets
        }

    def _build_relation_matrix(self, bboxes, num_patches):
        """
        构建对象间的空间关系矩阵
        关系类型：
        0: 无关系, 1: 左边, 2: 右边, 3: 上方, 4: 下方,
        5: 包含, 6: 被包含, 7: 重叠, 8: 相邻, 9: 远离
        """
        relation_matrix = torch.zeros(num_patches, num_patches, dtype=torch.long)

        for i in range(num_patches):
            for j in range(num_patches):
                if i == j:
                    continue

                bbox_i = bboxes[i]
                bbox_j = bboxes[j]

                # 如果任一边界框为空，跳过
                if bbox_i.sum() == 0 or bbox_j.sum() == 0:
                    continue

                # 计算中心点
                center_i = bbox_i[:2] + bbox_i[2:] / 2
                center_j = bbox_j[:2] + bbox_j[2:] / 2

                # 判断空间关系
                if center_i[0] < center_j[0] - 0.1:
                    relation_matrix[i, j] = 1  # i在j左边
                elif center_i[0] > center_j[0] + 0.1:
                    relation_matrix[i, j] = 2  # i在j右边
                elif center_i[1] < center_j[1] - 0.1:
                    relation_matrix[i, j] = 3  # i在j上方
                elif center_i[1] > center_j[1] + 0.1:
                    relation_matrix[i, j] = 4  # i在j下方
                else:
                    relation_matrix[i, j] = 7  # 重叠

        return relation_matrix


class VisualGenomeUnderstandingDataset(Dataset):
    """
    Visual Genome数据集 - 用于关系推理
    包含：场景图、对象关系、属性
    """
    def __init__(self, image_dir, scene_graphs_file, config, transform=None):
        self.image_dir = image_dir
        self.config = config

        # 加载场景图
        with open(scene_graphs_file, 'r') as f:
            self.scene_graphs = json.load(f)

        # 构建关系词汇表
        self.predicate_to_id = self._build_predicate_vocab()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def _build_predicate_vocab(self):
        """构建关系词汇表"""
        common_predicates = [
            'on', 'has', 'in', 'wearing', 'of', 'near', 'with', 'above', 'holding', 'behind'
        ]
        return {pred: idx for idx, pred in enumerate(common_predicates)}

    def __len__(self):
        return len(self.scene_graphs)

    def __getitem__(self, idx):
        sg = self.scene_graphs[idx]

        # 加载图像
        img_path = os.path.join(self.image_dir, f"{sg['image_id']}.jpg")
        image = Image.open(img_path).convert('RGB')
        original_size = image.size

        image = self.transform(image)

        # 提取对象和关系
        objects = sg.get('objects', [])
        relationships = sg.get('relationships', [])

        # 构建描述
        caption = f"Image with {len(objects)} objects"
        if relationships:
            caption += f" and {len(relationships)} relationships"

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

        # 处理对象边界框
        num_patches = (self.config.IMAGE_SIZE // self.config.PATCH_SIZE) ** 2 + 1
        bboxes = torch.zeros(num_patches, 4)
        semantic_labels = torch.zeros(num_patches, dtype=torch.long)

        for i, obj in enumerate(objects[:num_patches]):
            if 'x' in obj and 'y' in obj and 'w' in obj and 'h' in obj:
                x_norm = obj['x'] / original_size[0]
                y_norm = obj['y'] / original_size[1]
                w_norm = obj['w'] / original_size[0]
                h_norm = obj['h'] / original_size[1]

                bboxes[i] = torch.tensor([x_norm, y_norm, w_norm, h_norm])
                semantic_labels[i] = hash(obj.get('name', 'object')) % 100

        # 构建关系矩阵
        relation_matrix = torch.zeros(num_patches, num_patches, dtype=torch.long)

        for rel in relationships:
            subj_idx = rel.get('subject', 0)
            obj_idx = rel.get('object', 0)

            if subj_idx < num_patches and obj_idx < num_patches:
                pred = rel.get('predicate', 'unknown')
                pred_id = self.predicate_to_id.get(pred, 0)
                relation_matrix[subj_idx, obj_idx] = pred_id

        targets = {
            'task_labels': torch.tensor(0, dtype=torch.long),
            'semantic_labels': semantic_labels,
            'bboxes': bboxes,
            'relation_matrix': relation_matrix
        }

        return {
            'images': image,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'targets': targets
        }


class ADE20KSegmentationDataset(Dataset):
    """
    ADE20K数据集 - 用于语义分割
    包含：150个类别的像素级标注
    """
    def __init__(self, image_dir, annotation_dir, config, split='train', transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.config = config
        self.split = split

        # 获取图像列表
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        self.seg_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]

        # 加载图像
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # 加载分割标注
        seg_file = img_file.replace('.jpg', '.png')
        seg_path = os.path.join(self.annotation_dir, seg_file)

        if os.path.exists(seg_path):
            segmentation = Image.open(seg_path)
            segmentation = self.seg_transform(segmentation)
            segmentation = (segmentation * 255).long().squeeze(0)
        else:
            # 如果没有标注，创建空标注
            segmentation = torch.zeros(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, dtype=torch.long)

        # 转换为patch级别的标签
        num_patches = (self.config.IMAGE_SIZE // self.config.PATCH_SIZE) ** 2 + 1
        patch_size = self.config.PATCH_SIZE

        semantic_labels = torch.zeros(num_patches, dtype=torch.long)

        # 为每个patch分配主要类别
        patch_idx = 1  # 0是CLS token
        for i in range(0, self.config.IMAGE_SIZE, patch_size):
            for j in range(0, self.config.IMAGE_SIZE, patch_size):
                if patch_idx >= num_patches:
                    break

                patch = segmentation[i:i+patch_size, j:j+patch_size]
                if patch.numel() > 0:
                    # 使用众数作为patch的标签
                    semantic_labels[patch_idx] = torch.mode(patch.flatten())[0]

                patch_idx += 1

        # 生成描述
        caption = f"Scene segmentation with {len(torch.unique(segmentation))} categories"

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

        # 边界框和关系矩阵（对于分割任务不太重要，使用默认值）
        bboxes = torch.zeros(num_patches, 4)
        relation_matrix = torch.zeros(num_patches, num_patches, dtype=torch.long)

        targets = {
            'task_labels': torch.tensor(0, dtype=torch.long),
            'semantic_labels': semantic_labels,
            'bboxes': bboxes,
            'relation_matrix': relation_matrix
        }

        return {
            'images': image,
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'targets': targets
        }


def create_understanding_dataloader(dataset_type, data_root, config, batch_size=8, split='train'):
    """
    创建理解任务的数据加载器

    Args:
        dataset_type: 'coco' | 'visual_genome' | 'ade20k'
        data_root: 数据集根目录
        config: 配置对象
        batch_size: 批次大小
        split: 'train' | 'val' | 'test'
    """
    from torch.utils.data import DataLoader

    if dataset_type == 'coco':
        image_dir = os.path.join(data_root, f'{split}2017')
        annotation_file = os.path.join(data_root, 'annotations', f'instances_{split}2017.json')
        dataset = COCOUnderstandingDataset(image_dir, annotation_file, config)

    elif dataset_type == 'visual_genome':
        image_dir = os.path.join(data_root, 'images')
        scene_graphs_file = os.path.join(data_root, f'scene_graphs_{split}.json')
        dataset = VisualGenomeUnderstandingDataset(image_dir, scene_graphs_file, config)

    elif dataset_type == 'ade20k':
        image_dir = os.path.join(data_root, 'images', split)
        annotation_dir = os.path.join(data_root, 'annotations', split)
        dataset = ADE20KSegmentationDataset(image_dir, annotation_dir, config, split)

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
from MLLM.utils.understanding_data_loader import create_understanding_dataloader

config = Config()

# 加载COCO数据集
train_loader = create_understanding_dataloader(
    dataset_type='coco',
    data_root='/path/to/coco',
    config=config,
    batch_size=8,
    split='train'
)

# 加载Visual Genome数据集
train_loader = create_understanding_dataloader(
    dataset_type='visual_genome',
    data_root='/path/to/visual_genome',
    config=config,
    batch_size=8,
    split='train'
)

# 训练
for batch in train_loader:
    images = batch['images']
    text_tokens = batch['text_tokens']
    targets = batch['targets']

    # 训练模型...
"""
