# 完整数据集使用指南

## 📋 概述

本指南详细说明如何使用真实数据集训练MLLM模型的三大任务：
1. **图像理解** - COCO, Visual Genome, ADE20K
2. **图像生成** - LAION, Conceptual Captions, COCO Captions
3. **图像编辑** - InstructPix2Pix, MagicBrush, RefCOCO

---

## 🎯 图像理解任务数据集

### 1. COCO数据集 ⭐⭐⭐⭐⭐

**用途**：目标检测、实例分割、图像描述

**数据规模**：
- 训练集：118K 图像
- 验证集：5K 图像
- 80个类别，5个描述/图像

**下载**：
```bash
# 创建目录
mkdir -p /path/to/coco

# 下载图像
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# 下载标注
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 解压
unzip train2017.zip -d /path/to/coco/
unzip val2017.zip -d /path/to/coco/
unzip annotations_trainval2017.zip -d /path/to/coco/
```

**目录结构**：
```
/path/to/coco/
├── train2017/
│   ├── 000000000001.jpg
│   └── ...
├── val2017/
│   ├── 000000000001.jpg
│   └── ...
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    ├── captions_train2017.json
    └── captions_val2017.json
```

**使用代码**：
```python
from MLLM.utils.understanding_data_loader import create_understanding_dataloader

train_loader = create_understanding_dataloader(
    dataset_type='coco',
    data_root='/path/to/coco',
    config=config,
    batch_size=8,
    split='train'
)

# 训练
for batch in train_loader:
    images = batch['images']  # (B, 3, 224, 224)
    targets = batch['targets']
    # targets包含：
    # - task_labels: 任务类型
    # - semantic_labels: 语义标签
    # - bboxes: 边界框
    # - relation_matrix: 关系矩阵
```

**特点**：
- ✅ 完美匹配LoT低层（元素识别）
- ✅ 完美匹配LoT中层（语义+空间定位）
- ✅ 边界框精确，类别丰富

---

### 2. Visual Genome数据集 ⭐⭐⭐⭐⭐

**用途**：场景图、关系推理、属性识别

**数据规模**：
- 108K 图像
- 3.8M 对象实例
- 2.3M 关系标注

**下载**：
```bash
# 下载图像
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

# 下载场景图
wget https://visualgenome.org/static/data/dataset/scene_graphs.json.zip

# 解压
unzip images.zip -d /path/to/visual_genome/
unzip images2.zip -d /path/to/visual_genome/
unzip scene_graphs.json.zip -d /path/to/visual_genome/
```

**目录结构**：
```
/path/to/visual_genome/
├── images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── scene_graphs_train.json
```

**使用代码**：
```python
train_loader = create_understanding_dataloader(
    dataset_type='visual_genome',
    data_root='/path/to/visual_genome',
    config=config,
    batch_size=8,
    split='train'
)
```

**特点**：
- ✅ **完美匹配LoT高层**（关系矩阵）
- ✅ 丰富的关系标注（on, near, holding等）
- ✅ 详细的属性标注

---

### 3. ADE20K数据集 ⭐⭐⭐⭐

**用途**：语义分割

**数据规模**：
- 20K 训练图像
- 2K 验证图像
- 150个类别

**下载**：
```bash
# 官方网站
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

# 解压
unzip ADEChallengeData2016.zip -d /path/to/ade20k/
```

**目录结构**：
```
/path/to/ade20k/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

**使用代码**：
```python
train_loader = create_understanding_dataloader(
    dataset_type='ade20k',
    data_root='/path/to/ade20k',
    config=config,
    batch_size=8,
    split='train'
)
```

**特点**：
- ✅ 像素级语义分割
- ✅ 150个细粒度类别
- ✅ 场景多样化

---

## 🎨 图像生成任务数据集

### 1. LAION-400M数据集 ⭐⭐⭐⭐⭐

**用途**：大规模文本到图像生成

**数据规模**：
- 400M 图像-文本对
- 多语言支持

**下载**：
```bash
# 安装img2dataset工具
pip install img2dataset

# 下载LAION-400M子集
img2dataset --url_list laion400m.parquet \
            --output_folder /path/to/laion400m \
            --thread_count 16 \
            --image_size 256
```

**目录结构**：
```
/path/to/laion400m/
├── train.parquet
└── images/
    ├── 00000/
    │   ├── 000000000.jpg
    │   └── ...
    └── ...
```

**使用代码**：
```python
from MLLM.utils.generation_data_loader import create_generation_dataloader

train_loader = create_generation_dataloader(
    dataset_type='laion',
    data_root='/path/to/laion400m',
    config=config,
    batch_size=8,
    split='train'
)
```

**特点**：
- ✅ **海量数据**，适合扩散模型
- ✅ 多样化的图像风格
- ✅ 已被Stable Diffusion使用

---

### 2. Conceptual Captions (CC3M/CC12M) ⭐⭐⭐⭐⭐

**用途**：高质量文本到图像生成

**数据规模**：
- CC3M：3.3M 图像-文本对
- CC12M：12M 图像-文本对

**下载**：
```bash
# 克隆下载工具
git clone https://github.com/igorbrigadir/DownloadConceptualCaptions
cd DownloadConceptualCaptions

# 下载CC3M
python download_data.py --dataset cc3m --output_dir /path/to/cc3m

# 下载CC12M
python download_data.py --dataset cc12m --output_dir /path/to/cc12m
```

**目录结构**：
```
/path/to/cc3m/
├── train.tsv
└── images/
    ├── 00000000.jpg
    ├── 00000001.jpg
    └── ...
```

**使用代码**：
```python
train_loader = create_generation_dataloader(
    dataset_type='conceptual_captions',
    data_root='/path/to/cc3m',
    config=config,
    batch_size=8,
    split='train'
)
```

**特点**：
- ✅ 质量高于LAION（经过过滤）
- ✅ 数据量适中，易于处理
- ✅ 描述质量高

---

### 3. COCO Captions数据集 ⭐⭐⭐⭐

**用途**：图像描述生成

**数据规模**：
- 118K 图像
- 每张图像5个描述

**下载**：
```bash
# 使用COCO数据集（已在理解任务中下载）
# 只需要captions标注文件
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

**使用代码**：
```python
train_loader = create_generation_dataloader(
    dataset_type='coco_captions',
    data_root='/path/to/coco',
    config=config,
    batch_size=8,
    split='train'
)
```

**特点**：
- ✅ 每张图像5个不同描述
- ✅ 描述质量高
- ✅ 与理解任务共享数据

---

### 4. DiffusionDB数据集 ⭐⭐⭐⭐

**用途**：Stable Diffusion生成的图像和提示词

**数据规模**：
- 2M 图像-提示词对
- 高质量生成结果

**下载**：
```bash
# 从Hugging Face下载
git lfs install
git clone https://huggingface.co/datasets/poloclub/diffusiondb
```

**使用代码**：
```python
train_loader = create_generation_dataloader(
    dataset_type='diffusiondb',
    data_root='/path/to/diffusiondb',
    config=config,
    batch_size=8,
    split='train'
)
```

**特点**：
- ✅ 高质量生成结果
- ✅ 多样化的提示词
- ✅ 适合学习提示词工程

---

## ✂️ 图像编辑任务数据集

### 1. InstructPix2Pix数据集 ⭐⭐⭐⭐⭐

**用途**：基于指令的图像编辑

**数据规模**：
- 450K 编辑样本
- 自动生成

**下载**：
```bash
# 官方仓库
git clone https://github.com/timothybrooks/instruct-pix2pix
cd instruct-pix2pix

# 下载数据
wget https://instruct-pix2pix.eecs.berkeley.edu/clip-filtered-dataset/data.tar
tar -xf data.tar -C /path/to/instructpix2pix/
```

**使用代码**：
```python
from MLLM.utils.editing_data_loader import create_editing_dataloader

train_loader = create_editing_dataloader(
    dataset_type='instructpix2pix',
    data_root='/path/to/instructpix2pix',
    config=config,
    batch_size=8,
    split='train'
)
```

**特点**：
- ✅ 大规模数据
- ✅ 多样化的编辑类型
- ✅ 已被广泛使用

---

### 2. MagicBrush数据集 ⭐⭐⭐⭐⭐

**用途**：高质量人工标注的编辑数据

**数据规模**：
- 10K 编辑样本
- 人工标注

**下载**：
```bash
git clone https://github.com/OSU-NLP-Group/MagicBrush
cd MagicBrush
python download_data.py --output_dir /path/to/magicbrush
```

**使用代码**：
```python
train_loader = create_editing_dataloader(
    dataset_type='magicbrush',
    data_root='/path/to/magicbrush',
    config=config,
    batch_size=8,
    split='train'
)
```

**特点**：
- ✅ **最高质量**的编辑数据
- ✅ 人工标注的掩码
- ✅ 精确的编辑指令

---

### 3. RefCOCO数据集 ⭐⭐⭐⭐

**用途**：基于指代表达式的编辑

**数据规模**：
- 142K 表达式
- 精确区域定位

**下载**：
```bash
git clone https://github.com/lichengunc/refer
cd refer
python setup.py install
```

**使用代码**：
```python
train_loader = create_editing_dataloader(
    dataset_type='refcoco',
    data_root='/path/to/refcoco',
    config=config,
    batch_size=8,
    split='train'
)
```

**特点**：
- ✅ 精确的区域定位
- ✅ 自然语言指代
- ✅ 与COCO兼容

---

## 🚀 完整训练流程

### 方案A：分任务训练（推荐）

#### 阶段1：理解任务预训练
```python
from MLLM import MLLMModel, Config
from MLLM.training import PreTrainer
from MLLM.utils.understanding_data_loader import create_understanding_dataloader

config = Config()
model = MLLMModel(config).to('cuda')
model.freeze_encoders()

# 加载COCO数据
coco_loader = create_understanding_dataloader(
    dataset_type='coco',
    data_root='/path/to/coco',
    config=config,
    batch_size=8,
    split='train'
)

# 训练理解任务
trainer = PreTrainer(model, config)
for epoch in range(50):
    metrics = trainer.train_epoch(coco_loader, epoch, mode='understanding')
    print(f"Epoch {epoch}: {metrics}")
```

#### 阶段2：生成任务预训练
```python
from MLLM.utils.generation_data_loader import create_generation_dataloader

# 加载CC12M数据
cc12m_loader = create_generation_dataloader(
    dataset_type='conceptual_captions',
    data_root='/path/to/cc12m',
    config=config,
    batch_size=8,
    split='train'
)

# 训练生成任务
for epoch in range(50):
    metrics = trainer.train_epoch(cc12m_loader, epoch, mode='generation')
    print(f"Epoch {epoch}: {metrics}")
```

#### 阶段3：编辑任务预训练
```python
from MLLM.utils.editing_data_loader import create_editing_dataloader

# 加载InstructPix2Pix数据
edit_loader = create_editing_dataloader(
    dataset_type='instructpix2pix',
    data_root='/path/to/instructpix2pix',
    config=config,
    batch_size=8,
    split='train'
)

# 训练编辑任务
for epoch in range(50):
    metrics = trainer.train_epoch(edit_loader, epoch, mode='editing')
    print(f"Epoch {epoch}: {metrics}")
```

#### 阶段4：CGPO强化学习
```python
from MLLM.training import CGPOTrainer, RewardModel

# 创建奖励模型
reward_model = RewardModel(config).to('cuda')

# 创建CGPO训练器
cgpo_trainer = CGPOTrainer(model, reward_model, config)

# 混合所有数据进行CGPO训练
for epoch in range(50):
    # 可以轮流使用不同数据集
    metrics = cgpo_trainer.train_epoch(coco_loader, epoch)
    print(f"CGPO Epoch {epoch}: {metrics}")
```

---

### 方案B：混合训练

```python
from torch.utils.data import ConcatDataset, DataLoader

# 创建所有数据集
coco_dataset = create_understanding_dataloader(...).dataset
cc12m_dataset = create_generation_dataloader(...).dataset
edit_dataset = create_editing_dataloader(...).dataset

# 合并数据集
mixed_dataset = ConcatDataset([coco_dataset, cc12m_dataset, edit_dataset])

# 创建混合数据加载器
mixed_loader = DataLoader(
    mixed_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# 训练
for epoch in range(100):
    metrics = trainer.train_epoch(mixed_loader, epoch, mode='mixed')
    print(f"Epoch {epoch}: {metrics}")
```

---

## 📊 数据集对比表

### 理解任务

| 数据集 | 规模 | 任务 | LoT低层 | LoT中层 | LoT高层 | 推荐度 |
|--------|------|------|---------|---------|---------|--------|
| COCO | 118K | 检测+分割 | ✅ | ✅ | ⚠️ | ⭐⭐⭐⭐⭐ |
| Visual Genome | 108K | 场景图 | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| ADE20K | 20K | 分割 | ✅ | ✅ | ❌ | ⭐⭐⭐⭐ |

### 生成任务

| 数据集 | 规模 | 质量 | 多样性 | 推荐度 |
|--------|------|------|--------|--------|
| LAION-400M | 400M | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CC12M | 12M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| COCO Captions | 118K | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| DiffusionDB | 2M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 编辑任务

| 数据集 | 规模 | 质量 | 掩码 | 推荐度 |
|--------|------|------|------|--------|
| InstructPix2Pix | 450K | ⭐⭐⭐ | ❌ | ⭐⭐⭐⭐⭐ |
| MagicBrush | 10K | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ |
| RefCOCO | 142K | ⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ |

---

## 💡 最佳实践

### 1. 数据集选择建议

**快速原型（1-2天）**：
- 理解：COCO (118K)
- 生成：COCO Captions (118K)
- 编辑：MagicBrush (10K)

**平衡训练（1-2周）**：
- 理解：COCO + Visual Genome
- 生成：CC12M
- 编辑：InstructPix2Pix

**完整训练（1-2月）**：
- 理解：COCO + Visual Genome + ADE20K
- 生成：LAION-400M
- 编辑：InstructPix2Pix + MagicBrush

### 2. 数据预处理

```python
# 统一的预处理流程
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 数据增强
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### 3. 批次大小建议

| GPU内存 | 批次大小 | 建议 |
|---------|---------|------|
| 8GB | 2-4 | 使用梯度累积 |
| 16GB | 4-8 | 标准训练 |
| 24GB | 8-16 | 推荐配置 |
| 40GB+ | 16-32 | 大批次训练 |

### 4. 学习率调度

```python
# 预训练阶段
lr_schedule = {
    'understanding': 1e-4,
    'generation': 1e-4,
    'editing': 5e-5
}

# CGPO阶段
cgpo_lr = 5e-5
```

---

## 📚 相关文档

- `understanding_data_loader.py` - 理解任务数据加载器
- `generation_data_loader.py` - 生成任务数据加载器
- `editing_data_loader.py` - 编辑任务数据加载器
- `EDITING_TRAINING_GUIDE.md` - 编辑任务详细指南

---

## 🎉 总结

现在您拥有完整的数据集支持：

✅ **理解任务**：3个数据集（COCO, Visual Genome, ADE20K）
✅ **生成任务**：4个数据集（LAION, CC12M, COCO Captions, DiffusionDB）
✅ **编辑任务**：3个数据集（InstructPix2Pix, MagicBrush, RefCOCO）

**总计10个专业数据集，覆盖所有任务！**

开始训练您的多模态大模型吧！🚀
