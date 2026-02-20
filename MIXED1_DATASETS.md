# Mixed1训练模式数据集推荐

## 📋 概述

**mixed1模式**（理解+生成混合）要求数据集**同时包含**：

### 理解任务所需标注：
- ✅ **边界框** (bboxes) - 目标检测
- ✅ **语义分割** (semantic_labels) - 像素级/patch级分类
- ✅ **关系矩阵** (relation_matrix) - 目标间空间/语义关系
- ✅ **任务标签** (task_labels) - 任务类型标识

### 生成任务所需标注：
- ✅ **文本描述** (captions) - 自然语言描述
- ✅ **目标图像** (target_images) - 用于生成的目标

---

## 🎯 推荐数据集

### 1. COCO数据集 ⭐⭐⭐⭐⭐（首选）

**为什么选择COCO？**
- ✅ **同时包含所有需要的标注**
- ✅ 数据量大（330K训练图像）
- ✅ 标注质量高
- ✅ 广泛使用，易于获取

#### 包含的标注：

| 标注类型 | 文件 | 用途 | 数量 |
|---------|------|------|------|
| 文本描述 | `captions_train2017.json` | 生成任务 | 每张图5个描述 |
| 目标检测 | `instances_train2017.json` | 理解任务（bboxes） | 80类，886K实例 |
| 全景分割 | `panoptic_train2017.json` | 理解任务（semantic） | 133类 |
| 空间关系 | 从bbox计算 | 理解任务（relation） | 自动构建 |

#### 数据结构：
```
COCO/
├── train2017/                    # 118K训练图像
├── val2017/                      # 5K验证图像
└── annotations/
    ├── captions_train2017.json       # 文本描述 ✓
    ├── captions_val2017.json
    ├── instances_train2017.json      # 目标检测+实例分割 ✓
    ├── instances_val2017.json
    ├── panoptic_train2017.json       # 全景分割 ✓
    └── panoptic_val2017.json
```

#### 下载地址：
```bash
# 官方网站
https://cocodataset.org/#download

# 下载命令
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 解压
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

#### 数据示例：
```json
// captions_train2017.json
{
    "image_id": 179765,
    "caption": "A black Honda motorcycle parked in front of a garage."
}

// instances_train2017.json
{
    "image_id": 179765,
    "category_id": 4,  // motorcycle
    "bbox": [77.0, 64.0, 300.0, 200.0],  // [x, y, width, height]
    "area": 60000,
    "segmentation": [...]
}
```

---

### 2. Visual Genome ⭐⭐⭐⭐⭐（关系标注最丰富）

**为什么选择Visual Genome？**
- ✅ **明确的关系标注** - 独特优势！
- ✅ 丰富的区域描述（每张图平均35个）
- ✅ 完整的场景图
- ✅ 属性标注

#### 包含的标注：

| 标注类型 | 文件 | 用途 | 数量 |
|---------|------|------|------|
| 区域描述 | `region_descriptions.json` | 生成任务 | 平均35个/图 |
| 目标检测 | `objects.json` | 理解任务（bboxes） | 平均35个/图 |
| **关系标注** | `relationships.json` | **理解任务（relation）** | **平均21个/图** |
| 属性标注 | `attributes.json` | 理解任务增强 | 平均26个/图 |
| 场景图 | `scene_graphs.json` | 完整场景理解 | 108K图 |

#### 数据结构：
```
VisualGenome/
├── images/                           # 108K图像
│   ├── VG_100K/
│   └── VG_100K_2/
├── region_descriptions.json          # 区域描述 ✓
├── objects.json                      # 目标检测 ✓
├── relationships.json                # 关系标注 ✓ (独特优势)
├── attributes.json                   # 属性标注
└── scene_graphs.json                 # 场景图
```

#### 下载地址：
```bash
# 官方网站
https://visualgenome.org/api/v0/api_home.html

# 下载图像
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

# 下载标注
wget https://visualgenome.org/static/data/dataset/region_descriptions.json.zip
wget https://visualgenome.org/static/data/dataset/objects.json.zip
wget https://visualgenome.org/static/data/dataset/relationships.json.zip
```

#### 数据示例：
```json
// region_descriptions.json
{
    "id": 1,
    "regions": [
        {
            "region_id": 1234,
            "phrase": "A young girl standing next to a red bicycle",
            "x": 10, "y": 20, "width": 100, "height": 150
        }
    ]
}

// relationships.json (独特优势！)
{
    "image_id": 1,
    "relationships": [
        {
            "subject": {"object_id": 123, "name": "girl"},
            "predicate": "next to",  // 明确的关系类型
            "object": {"object_id": 456, "name": "bicycle"}
        }
    ]
}
```

---

### 3. Flickr30K Entities ⭐⭐⭐⭐

**为什么选择Flickr30K？**
- ✅ 高质量的文本描述
- ✅ 实体-边界框对齐
- ✅ 文本-图像共指关系

#### 包含的标注：

| 标注类型 | 用途 | 数量 |
|---------|------|------|
| 文本描述 | 生成任务 | 每张图5个 |
| 实体bbox | 理解任务 | 244K实体 |
| 共指关系 | 理解任务 | 完整标注 |

#### 数据结构：
```
Flickr30K/
├── flickr30k_images/          # 31K图像
├── Annotations/               # 实体标注
└── Sentences/                 # 文本描述
```

---

## 💻 使用方法

### 方案A：使用COCO（推荐）

```python
from MLLM.utils.config import Config
from MLLM.utils.mixed1_data_loader import create_mixed1_dataloader
from MLLM import MLLMModel
from MLLM.training import PreTrainer

config = Config()

# 创建数据加载器
train_loader = create_mixed1_dataloader(
    dataset_type='coco',
    data_root='/path/to/coco',  # COCO数据集根目录
    config=config,
    batch_size=8,
    split='train'
)

val_loader = create_mixed1_dataloader(
    dataset_type='coco',
    data_root='/path/to/coco',
    config=config,
    batch_size=8,
    split='val'
)

# 初始化模型
model = MLLMModel(config).to('cuda')
model.freeze_encoders()
trainer = PreTrainer(model, config)

# 训练
for epoch in range(50):
    # mixed1模式：同一批数据同时训练理解和生成
    metrics = trainer.train_epoch(train_loader, epoch, mode='mixed1')

    print(f"Epoch {epoch} 训练指标:")
    print(f"  理解任务:")
    print(f"    - task_loss: {metrics.get('task_loss', 0):.4f}")
    print(f"    - semantic_loss: {metrics.get('semantic_loss', 0):.4f}")
    print(f"    - bbox_loss: {metrics.get('bbox_loss', 0):.4f}")
    print(f"    - relation_loss: {metrics.get('relation_loss', 0):.4f}")
    print(f"  生成任务:")
    print(f"    - diffusion_loss: {metrics.get('diffusion_loss', 0):.4f}")
    print(f"  总损失: {metrics.get('total_loss', 0):.4f}")

    # 验证
    if epoch % 5 == 0:
        val_metrics = trainer.validate(val_loader, mode='mixed1')
        print(f"Epoch {epoch} 验证指标: {val_metrics}")

    # 保存
    if epoch % 10 == 0:
        trainer.save_checkpoint(
            f'checkpoints/mixed1_coco_epoch_{epoch}.pth',
            epoch,
            metrics
        )
```

### 方案B：使用Visual Genome（关系标注更丰富）

```python
# 创建数据加载器
train_loader = create_mixed1_dataloader(
    dataset_type='visual_genome',
    data_root='/path/to/visual_genome',
    config=config,
    batch_size=8,
    split='train'
)

# 训练（其他代码相同）
for epoch in range(50):
    metrics = trainer.train_epoch(train_loader, epoch, mode='mixed1')
    print(f"Epoch {epoch}: {metrics}")
```

---

## 📊 数据集对比

| 数据集 | 图像数量 | 文本描述 | 检测标注 | 关系标注 | 分割标注 | 推荐度 |
|--------|---------|---------|---------|---------|---------|--------|
| **COCO** | 330K | 5个/图 | ✅ 80类 | 自动构建 | ✅ 全景分割 | ⭐⭐⭐⭐⭐ |
| **Visual Genome** | 108K | 35个/图 | ✅ 丰富 | ✅ **明确标注** | 部分 | ⭐⭐⭐⭐⭐ |
| **Flickr30K** | 31K | 5个/图 | ✅ 实体 | 部分 | ❌ | ⭐⭐⭐⭐ |

---

## 🎯 选择建议

### 场景1：通用训练（选COCO）
- ✅ 数据量最大
- ✅ 标注最全面
- ✅ 最易获取
- ✅ 适合从头训练

**推荐配置**：
```python
train_loader = create_mixed1_dataloader('coco', '/path/to/coco', config, batch_size=8)
```

---

### 场景2：强化关系理解（选Visual Genome）
- ✅ **明确的关系标注**（on, has, in, wearing等）
- ✅ 更丰富的场景描述
- ✅ 适合关系推理重点优化

**推荐配置**：
```python
train_loader = create_mixed1_dataloader('visual_genome', '/path/to/vg', config, batch_size=8)
```

---

### 场景3：最佳实践（组合使用）
先用COCO打基础，再用Visual Genome强化关系理解：

```python
# 阶段1：COCO预训练（30 epochs）
coco_loader = create_mixed1_dataloader('coco', '/path/to/coco', config, batch_size=8)
for epoch in range(30):
    trainer.train_epoch(coco_loader, epoch, mode='mixed1')

# 阶段2：Visual Genome微调（20 epochs）
vg_loader = create_mixed1_dataloader('visual_genome', '/path/to/vg', config, batch_size=8)
for epoch in range(30, 50):
    trainer.train_epoch(vg_loader, epoch, mode='mixed1')
```

---

## 🔧 数据处理细节

### COCO数据加载器特性：

1. **自动关系构建**：从边界框自动计算空间关系
   - 重叠 (overlap)
   - 相邻 (adjacent)
   - 左/右/上/下 (left/right/above/below)

2. **语义分割转换**：全景分割 → patch级标签
   - ViT patch size: 16×16
   - 每张图像: (224/16)² = 196 patches

3. **文本处理**：随机选择5个描述中的一个
   - 增加训练多样性
   - 提升生成鲁棒性

### Visual Genome数据加载器特性：

1. **明确关系映射**：
   ```python
   relation_types = [
       'on', 'has', 'in', 'wearing', 'near',
       'above', 'below', 'next to', 'behind', 'other'
   ]
   ```

2. **区域描述采样**：从平均35个描述中随机选择
   - 更丰富的文本多样性

3. **场景图集成**：完整的目标-关系-属性三元组

---

## 💡 最佳实践

### 1. 批次大小设置
```python
# 根据GPU显存调整
batch_size_config = {
    '12GB GPU': 4,   # RTX 3060
    '24GB GPU': 8,   # RTX 3090/4090
    '40GB GPU': 16,  # A100
}
```

### 2. 数据增强
```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### 3. 学习率策略
```python
# mixed1模式推荐学习率
optimizer = AdamW([
    {'params': model.lot_module.parameters(), 'lr': 8e-5},
    {'params': model.integration_module.parameters(), 'lr': 8e-5},
    {'params': model.diffusion_module.parameters(), 'lr': 8e-5},
])
```

---

## 📈 预期效果

使用COCO数据集训练50个epoch后：

| 指标 | 预期值 |
|------|--------|
| task_loss | < 0.3 |
| semantic_loss | < 0.5 |
| bbox_loss | < 0.15 |
| relation_loss | < 0.4 |
| diffusion_loss | < 0.6 |
| **total_loss** | **< 2.0** |

---

## 🚀 快速开始

### 完整训练脚本：

```bash
# 1. 下载COCO数据集
cd /path/to/datasets
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip annotations_trainval2017.zip

# 2. 运行训练
python -c "
from MLLM.utils.config import Config
from MLLM.utils.mixed1_data_loader import create_mixed1_dataloader
from MLLM import MLLMModel
from MLLM.training import PreTrainer

config = Config()
train_loader = create_mixed1_dataloader('coco', '/path/to/coco', config, batch_size=8)

model = MLLMModel(config).to('cuda')
model.freeze_encoders()
trainer = PreTrainer(model, config)

for epoch in range(50):
    metrics = trainer.train_epoch(train_loader, epoch, mode='mixed1')
    print(f'Epoch {epoch}: {metrics}')
    if epoch % 10 == 0:
        trainer.save_checkpoint(f'checkpoints/epoch_{epoch}.pth', epoch, metrics)
"
```

现在您可以开始使用COCO或Visual Genome数据集进行mixed1训练了！🎉
