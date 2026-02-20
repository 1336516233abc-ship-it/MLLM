"""
Mixed1模式数据加载器 - 理解+生成混合训练
同时提供理解标注和生成标注的数据集
"""

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pycocotools.coco import COCO
import cv2


class COCOMixed1Dataset(Dataset):
    """
    COCO数据集用于mixed1训练（理解+生成）

    提供：
    - 理解任务：检测框、语义分割、关系矩阵、任务标签
    - 生成任务：文本描述、目标图像

    这是mixed1模式的首选数据集！
    """
    def __init__(self, image_dir, instances_file, captions_file, config, transform=None):
        """
        Args:
            image_dir: 图像目录（如 train2017/）
            instances_file: 检测标注文件（instances_train2017.json）
            captions_file: 文本描述文件（captions_train2017.json）
            config: 配置对象
        """
        self.image_dir = image_dir
        self.config = config

        # 加载COCO标注
        self.coco_instances = COCO(instances_file)

        with open(captions_file, 'r') as f:
            captions_data = json.load(f)

        # 构建图像ID到描述的映射
        self.image_to_captions = {}
        for ann in captions_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_captions:
                self.image_to_captions[img_id] = []
            self.image_to_captions[img_id].append(ann['caption'])

        # 只保留同时有检测标注和描述的图像
        all_img_ids = set(self.coco_instances.getImgIds())
        caption_img_ids = set(self.image_to_captions.keys())
        self.image_ids = list(all_img_ids & caption_img_ids)

        # COCO类别映射
        self.cat_ids = self.coco_instances.getCatIds()
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}

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

    def _compute_spatial_relation(self, bbox1, bbox2):
        """
        计算两个bbox之间的空间关系
        返回：0-9的关系类型
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        cx1, cy1 = x1 + w1/2, y1 + h1/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2

        # 计算IoU
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)

        if ix2 > ix1 and iy2 > iy1:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            union = w1 * h1 + w2 * h2 - intersection
            iou = intersection / union

            if iou > 0.5:
                return 0  # 重叠

        # 空间关系
        dx = cx2 - cx1
        dy = cy2 - cy1

        if abs(dx) < 10 and abs(dy) < 10:
            return 1  # 相邻
        elif dx > 0 and abs(dy) < abs(dx):
            return 2  # 右边
        elif dx < 0 and abs(dy) < abs(dx):
            return 3  # 左边
        elif dy > 0 and abs(dx) < abs(dy):
            return 4  # 下方
        elif dy < 0 and abs(dx) < abs(dy):
            return 5  # 上方
        else:
            return 6  # 其他

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # 加载图像
        img_info = self.coco_instances.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        try:
            image = Image.open(img_path).convert('RGB')
            original_w, original_h = image.size
        except:
            image = Image.new('RGB', (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), (128, 128, 128))
            original_w, original_h = self.config.IMAGE_SIZE, self.config.IMAGE_SIZE

        image_tensor = self.transform(image)

        # ========== 生成任务数据 ==========
        # 随机选择一个文本描述
        captions = self.image_to_captions[img_id]
        caption = np.random.choice(captions)

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

        # ========== 理解任务数据 ==========
        # 获取标注
        ann_ids = self.coco_instances.getAnnIds(imgIds=img_id)
        anns = self.coco_instances.loadAnns(ann_ids)

        # 边界框和类别
        bboxes = []
        categories = []
        segmentations = []

        for ann in anns:
            if 'bbox' in ann and ann['area'] > 0:
                bbox = ann['bbox']  # [x, y, w, h]
                # 归一化到[0,1]
                norm_bbox = [
                    bbox[0] / original_w,
                    bbox[1] / original_h,
                    bbox[2] / original_w,
                    bbox[3] / original_h
                ]
                bboxes.append(norm_bbox)
                categories.append(self.cat_id_to_idx[ann['category_id']])

        # 最多100个目标
        max_objects = 100
        num_objects = min(len(bboxes), max_objects)

        # 填充边界框
        padded_bboxes = torch.zeros((max_objects, 4))
        if num_objects > 0:
            padded_bboxes[:num_objects] = torch.tensor(bboxes[:num_objects], dtype=torch.float32)

        # 构建关系矩阵
        relation_matrix = torch.zeros((max_objects, max_objects), dtype=torch.long)
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    relation = self._compute_spatial_relation(bboxes[i], bboxes[j])
                    relation_matrix[i, j] = relation

        # 语义分割标签（转换为patch-level）
        num_patches = (self.config.IMAGE_SIZE // 16) ** 2  # ViT patch数量
        semantic_labels = torch.zeros(num_patches, dtype=torch.long)

        # 简化：使用最大的几个目标的类别填充对应patch
        if num_objects > 0:
            patch_size = 16
            for i in range(min(num_objects, 10)):  # 只处理前10个最大目标
                bbox = bboxes[i]
                cat = categories[i]

                # 计算bbox对应的patch范围
                x_start = int(bbox[0] * self.config.IMAGE_SIZE / patch_size)
                y_start = int(bbox[1] * self.config.IMAGE_SIZE / patch_size)
                x_end = int((bbox[0] + bbox[2]) * self.config.IMAGE_SIZE / patch_size)
                y_end = int((bbox[1] + bbox[3]) * self.config.IMAGE_SIZE / patch_size)

                patches_per_side = self.config.IMAGE_SIZE // patch_size
                for py in range(max(0, y_start), min(patches_per_side, y_end + 1)):
                    for px in range(max(0, x_start), min(patches_per_side, x_end + 1)):
                        patch_idx = py * patches_per_side + px
                        if patch_idx < num_patches:
                            semantic_labels[patch_idx] = cat + 1  # 0保留给背景

        # 任务标签（0=理解，1=生成，这里是混合任务）
        task_labels = torch.tensor(2, dtype=torch.long)  # 2表示混合任务

        return {
            'images': torch.zeros_like(image_tensor),  # 生成任务的输入（噪声/零图像）
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'target_images': image_tensor,  # 生成任务的目标
            'targets': {
                'task_labels': task_labels,
                'semantic_labels': semantic_labels,
                'bboxes': padded_bboxes,
                'relation_matrix': relation_matrix
            }
        }


class VisualGenomeMixed1Dataset(Dataset):
    """
    Visual Genome数据集用于mixed1训练（理解+生成）

    优势：
    - 明确的关系标注（relationships.json）
    - 丰富的区域描述（用于生成）
    - 完整的场景图（用于理解）
    """
    def __init__(self, image_dir, region_desc_file, objects_file, relationships_file, config, transform=None):
        """
        Args:
            image_dir: 图像目录
            region_desc_file: 区域描述文件（region_descriptions.json）
            objects_file: 目标文件（objects.json）
            relationships_file: 关系文件（relationships.json）
            config: 配置对象
        """
        self.image_dir = image_dir
        self.config = config

        # 加载数据
        with open(region_desc_file, 'r') as f:
            self.region_descriptions = json.load(f)

        with open(objects_file, 'r') as f:
            self.objects_data = json.load(f)

        with open(relationships_file, 'r') as f:
            self.relationships_data = json.load(f)

        # 构建索引
        self.image_id_to_objects = {item['image_id']: item['objects'] for item in self.objects_data}
        self.image_id_to_relationships = {item['image_id']: item['relationships'] for item in self.relationships_data}

        # 只保留同时有描述、目标和关系的图像
        valid_image_ids = set()
        for item in self.region_descriptions:
            img_id = item['id']
            if img_id in self.image_id_to_objects and img_id in self.image_id_to_relationships:
                if len(item['regions']) > 0:
                    valid_image_ids.add(img_id)

        self.image_ids = list(valid_image_ids)
        self.id_to_regions = {item['id']: item['regions'] for item in self.region_descriptions}

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # 关系类型映射
        self.relation_types = ['on', 'has', 'in', 'wearing', 'near', 'above', 'below', 'next to', 'behind', 'other']
        self.relation_to_idx = {rel: idx for idx, rel in enumerate(self.relation_types)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # 加载图像
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert('RGB')
            original_w, original_h = image.size
        except:
            image = Image.new('RGB', (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), (128, 128, 128))
            original_w, original_h = self.config.IMAGE_SIZE, self.config.IMAGE_SIZE

        image_tensor = self.transform(image)

        # ========== 生成任务数据 ==========
        # 随机选择一个区域描述
        regions = self.id_to_regions[img_id]
        region = np.random.choice(regions)
        caption = region['phrase']

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

        # ========== 理解任务数据 ==========
        # 获取目标
        objects = self.image_id_to_objects.get(img_id, [])
        relationships = self.image_id_to_relationships.get(img_id, [])

        # 边界框
        max_objects = 100
        bboxes = []
        object_id_to_idx = {}

        for i, obj in enumerate(objects[:max_objects]):
            if 'x' in obj and 'y' in obj and 'w' in obj and 'h' in obj:
                # 归一化
                norm_bbox = [
                    obj['x'] / original_w,
                    obj['y'] / original_h,
                    obj['w'] / original_w,
                    obj['h'] / original_h
                ]
                bboxes.append(norm_bbox)
                object_id_to_idx[obj['object_id']] = i

        num_objects = len(bboxes)
        padded_bboxes = torch.zeros((max_objects, 4))
        if num_objects > 0:
            padded_bboxes[:num_objects] = torch.tensor(bboxes, dtype=torch.float32)

        # 构建关系矩阵（使用Visual Genome的明确关系标注）
        relation_matrix = torch.zeros((max_objects, max_objects), dtype=torch.long)
        for rel in relationships:
            subj_id = rel['subject']['object_id']
            obj_id = rel['object']['object_id']
            predicate = rel['predicate'].lower()

            if subj_id in object_id_to_idx and obj_id in object_id_to_idx:
                i = object_id_to_idx[subj_id]
                j = object_id_to_idx[obj_id]

                # 映射关系类型
                rel_idx = 9  # 默认为other
                for rel_type, idx in self.relation_to_idx.items():
                    if rel_type in predicate:
                        rel_idx = idx
                        break

                relation_matrix[i, j] = rel_idx

        # 语义分割（简化版）
        num_patches = (self.config.IMAGE_SIZE // 16) ** 2
        semantic_labels = torch.zeros(num_patches, dtype=torch.long)

        # 任务标签
        task_labels = torch.tensor(2, dtype=torch.long)  # 混合任务

        return {
            'images': torch.zeros_like(image_tensor),
            'text_tokens': text_tokens,
            'text_mask': text_mask,
            'target_images': image_tensor,
            'targets': {
                'task_labels': task_labels,
                'semantic_labels': semantic_labels,
                'bboxes': padded_bboxes,
                'relation_matrix': relation_matrix
            }
        }


def create_mixed1_dataloader(dataset_type, data_root, config, batch_size=8, split='train'):
    """
    创建mixed1模式的数据加载器

    Args:
        dataset_type: 'coco' | 'visual_genome'
        data_root: 数据集根目录
        config: 配置对象
        batch_size: 批次大小
        split: 'train' | 'val'

    Returns:
        DataLoader: 适用于mixed1训练的数据加载器
    """
    from torch.utils.data import DataLoader

    if dataset_type == 'coco':
        image_dir = os.path.join(data_root, f'{split}2017')
        instances_file = os.path.join(data_root, 'annotations', f'instances_{split}2017.json')
        captions_file = os.path.join(data_root, 'annotations', f'captions_{split}2017.json')

        dataset = COCOMixed1Dataset(image_dir, instances_file, captions_file, config)

    elif dataset_type == 'visual_genome':
        image_dir = os.path.join(data_root, 'images')
        region_desc_file = os.path.join(data_root, 'region_descriptions.json')
        objects_file = os.path.join(data_root, 'objects.json')
        relationships_file = os.path.join(data_root, 'relationships.json')

        dataset = VisualGenomeMixed1Dataset(
            image_dir, region_desc_file, objects_file, relationships_file, config
        )

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


# ============ 使用示例 ============
"""
from MLLM.utils.config import Config
from MLLM.utils.mixed1_data_loader import create_mixed1_dataloader
from MLLM import MLLMModel
from MLLM.training import PreTrainer

config = Config()

# 方案1：使用COCO数据集（推荐！）
train_loader = create_mixed1_dataloader(
    dataset_type='coco',
    data_root='/path/to/coco',  # COCO数据集根目录
    config=config,
    batch_size=8,
    split='train'
)

# 方案2：使用Visual Genome数据集（关系标注更丰富）
train_loader = create_mixed1_dataloader(
    dataset_type='visual_genome',
    data_root='/path/to/visual_genome',
    config=config,
    batch_size=8,
    split='train'
)

# 训练
model = MLLMModel(config).to('cuda')
model.freeze_encoders()
trainer = PreTrainer(model, config)

for epoch in range(50):
    # mixed1模式：同一批数据同时训练理解和生成
    metrics = trainer.train_epoch(train_loader, epoch, mode='mixed1')
    print(f"Epoch {epoch}: {metrics}")

    # 输出示例：
    # {
    #     'task_loss': 0.234,         # 理解任务
    #     'semantic_loss': 0.456,     # 理解任务
    #     'bbox_loss': 0.123,         # 理解任务
    #     'relation_loss': 0.345,     # 理解任务
    #     'diffusion_loss': 0.567,    # 生成任务
    #     'total_loss': 1.725         # 总损失
    # }
"""
