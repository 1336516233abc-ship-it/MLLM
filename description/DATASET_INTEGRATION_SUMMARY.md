# 数据集集成更新总结

## 🎉 新增内容

为图像理解、生成、编辑三大任务添加了完整的真实数据集支持！

---

## 📁 新增文件

### 1. `MLLM/utils/understanding_data_loader.py`
**图像理解任务数据加载器**

包含3个数据集类：
- ✅ `COCOUnderstandingDataset` - COCO目标检测和分割
- ✅ `VisualGenomeUnderstandingDataset` - Visual Genome场景图
- ✅ `ADE20KSegmentationDataset` - ADE20K语义分割

**核心功能**：
- 自动构建关系矩阵（基于空间关系）
- 边界框归一化处理
- Patch级别的语义标签
- 支持COCO的80个类别
- 支持Visual Genome的关系词汇表

---

### 2. `MLLM/utils/generation_data_loader.py`
**图像生成任务数据加载器**

包含5个数据集类：
- ✅ `LAIONDataset` - LAION大规模数据集
- ✅ `ConceptualCaptionsDataset` - CC3M/CC12M
- ✅ `COCOCaptionsDataset` - COCO图像描述
- ✅ `TextToImageDataset` - 通用JSON格式
- ✅ `DiffusionDBDataset` - Stable Diffusion生成数据

**核心功能**：
- 支持Parquet、TSV、JSON多种格式
- 自动处理图像加载失败
- 支持条件生成（参考图像）
- 每张COCO图像5个不同描述

---

### 3. `MLLM/utils/editing_data_loader.py`（已有）
**图像编辑任务数据加载器**

包含3个数据集类：
- ✅ `InstructPix2PixDataset` - 450K编辑样本
- ✅ `RefCOCOEditDataset` - 142K指代表达式
- ✅ `MagicBrushDataset` - 10K高质量样本

---

### 4. `MLLM/DATASET_GUIDE.md`
**完整数据集使用指南**

包含内容：
- 📊 10个数据集的详细介绍
- 📥 下载链接和命令
- 💻 使用代码示例
- 🚀 完整训练流程
- 📈 数据集对比表
- 💡 最佳实践建议

---

## 📊 支持的数据集总览

### 图像理解（3个）

| 数据集 | 规模 | 用途 | 特点 |
|--------|------|------|------|
| **COCO** | 118K | 检测+分割 | 80类别，精确标注 |
| **Visual Genome** | 108K | 场景图 | 2.3M关系，完美匹配LoT高层 |
| **ADE20K** | 20K | 语义分割 | 150类别，像素级标注 |

### 图像生成（4个）

| 数据集 | 规模 | 用途 | 特点 |
|--------|------|------|------|
| **LAION-400M** | 400M | 大规模生成 | 海量数据，多样化 |
| **CC12M** | 12M | 高质量生成 | 过滤后的高质量数据 |
| **COCO Captions** | 118K | 图像描述 | 每图5个描述 |
| **DiffusionDB** | 2M | 提示词学习 | SD生成结果 |

### 图像编辑（3个）

| 数据集 | 规模 | 用途 | 特点 |
|--------|------|------|------|
| **InstructPix2Pix** | 450K | 指令编辑 | 大规模，多样化 |
| **MagicBrush** | 10K | 精确编辑 | 人工标注，最高质量 |
| **RefCOCO** | 142K | 区域编辑 | 精确定位 |

**总计：10个专业数据集！**

---

## 🔧 核心功能实现

### 1. 统一的数据加载接口

```python
# 理解任务
from MLLM.utils.understanding_data_loader import create_understanding_dataloader

loader = create_understanding_dataloader(
    dataset_type='coco',  # 或 'visual_genome', 'ade20k'
    data_root='/path/to/data',
    config=config,
    batch_size=8,
    split='train'
)

# 生成任务
from MLLM.utils.generation_data_loader import create_generation_dataloader

loader = create_generation_dataloader(
    dataset_type='laion',  # 或 'conceptual_captions', 'coco_captions', 'diffusiondb'
    data_root='/path/to/data',
    config=config,
    batch_size=8,
    split='train'
)

# 编辑任务
from MLLM.utils.editing_data_loader import create_editing_dataloader

loader = create_editing_dataloader(
    dataset_type='instructpix2pix',  # 或 'magicbrush', 'refcoco'
    data_root='/path/to/data',
    config=config,
    batch_size=8,
    split='train'
)
```

### 2. 智能的数据处理

**COCO数据集**：
- ✅ 自动构建空间关系矩阵（9种关系类型）
- ✅ 边界框归一化到[0,1]
- ✅ 类别ID映射
- ✅ 支持多个描述

**Visual Genome**：
- ✅ 场景图解析
- ✅ 关系词汇表构建（10种常见关系）
- ✅ 对象属性提取
- ✅ 关系矩阵生成

**ADE20K**：
- ✅ 像素级分割转Patch级标签
- ✅ 使用众数作为Patch标签
- ✅ 150个类别支持

**LAION/CC**：
- ✅ 支持Parquet和TSV格式
- ✅ 自动处理图像加载失败
- ✅ 大规模数据流式处理

**编辑数据集**：
- ✅ 源图像和编辑图像配对
- ✅ 编辑掩码处理
- ✅ 编辑指令解析

### 3. 关系矩阵构建算法

```python
def _build_relation_matrix(self, bboxes, num_patches):
    """
    构建对象间的空间关系矩阵
    关系类型：
    0: 无关系, 1: 左边, 2: 右边, 3: 上方, 4: 下方,
    5: 包含, 6: 被包含, 7: 重叠, 8: 相邻, 9: 远离
    """
    # 基于边界框中心点计算空间关系
    # 完美匹配LoT高层的关系推理需求
```

---

## 🚀 使用示例

### 完整训练流程

```python
from MLLM import MLLMModel, Config
from MLLM.training import PreTrainer
from MLLM.utils.understanding_data_loader import create_understanding_dataloader
from MLLM.utils.generation_data_loader import create_generation_dataloader
from MLLM.utils.editing_data_loader import create_editing_dataloader

# 配置
config = Config()
model = MLLMModel(config).to('cuda')
model.freeze_encoders()
trainer = PreTrainer(model, config)

# 阶段1：理解任务（COCO）
print("训练理解任务...")
coco_loader = create_understanding_dataloader(
    dataset_type='coco',
    data_root='/path/to/coco',
    config=config,
    batch_size=8,
    split='train'
)

for epoch in range(50):
    metrics = trainer.train_epoch(coco_loader, epoch, mode='understanding')
    print(f"Epoch {epoch}: {metrics}")

# 阶段2：生成任务（CC12M）
print("训练生成任务...")
cc12m_loader = create_generation_dataloader(
    dataset_type='conceptual_captions',
    data_root='/path/to/cc12m',
    config=config,
    batch_size=8,
    split='train'
)

for epoch in range(50):
    metrics = trainer.train_epoch(cc12m_loader, epoch, mode='generation')
    print(f"Epoch {epoch}: {metrics}")

# 阶段3：编辑任务（InstructPix2Pix）
print("训练编辑任务...")
edit_loader = create_editing_dataloader(
    dataset_type='instructpix2pix',
    data_root='/path/to/instructpix2pix',
    config=config,
    batch_size=8,
    split='train'
)

for epoch in range(50):
    metrics = trainer.train_epoch(edit_loader, epoch, mode='editing')
    print(f"Epoch {epoch}: {metrics}")

print("预训练完成！")
```

---

## 📈 数据集下载指南

### 快速开始（小规模测试）

```bash
# 1. COCO（理解）- 约25GB
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 2. COCO Captions（生成）- 使用相同的COCO数据

# 3. MagicBrush（编辑）- 约2GB
git clone https://github.com/OSU-NLP-Group/MagicBrush
cd MagicBrush && python download_data.py
```

### 完整训练（大规模）

```bash
# 1. COCO + Visual Genome（理解）- 约50GB
# COCO
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Visual Genome
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip

# 2. CC12M（生成）- 约100GB
git clone https://github.com/igorbrigadir/DownloadConceptualCaptions
python download_data.py --dataset cc12m

# 3. InstructPix2Pix（编辑）- 约50GB
wget https://instruct-pix2pix.eecs.berkeley.edu/clip-filtered-dataset/data.tar
```

---

## 💡 最佳实践

### 1. 推荐的训练方案

**方案A：快速原型（1-2天）**
```
理解：COCO val (5K)
生成：COCO Captions (118K)
编辑：MagicBrush (10K)
总计：~133K 样本
```

**方案B：平衡训练（1-2周）⭐ 推荐**
```
理解：COCO (118K) + Visual Genome (108K)
生成：CC12M (12M)
编辑：InstructPix2Pix (450K)
总计：~12.7M 样本
```

**方案C：完整训练（1-2月）**
```
理解：COCO + Visual Genome + ADE20K
生成：LAION-400M (400M)
编辑：InstructPix2Pix + MagicBrush
总计：~400M+ 样本
```

### 2. 数据集组合建议

| 任务 | 主数据集 | 辅助数据集 | 目的 |
|------|---------|-----------|------|
| 理解 | COCO | Visual Genome | 检测+关系 |
| 生成 | CC12M | COCO Captions | 质量+多样性 |
| 编辑 | InstructPix2Pix | MagicBrush | 规模+质量 |

### 3. 训练顺序建议

```
1. 理解任务（50 epochs）
   ↓
2. 生成任务（50 epochs）
   ↓
3. 编辑任务（50 epochs）
   ↓
4. 混合微调（20 epochs）
   ↓
5. CGPO强化学习（50 epochs）
```

---

## 📚 文档导航

1. **数据集详细指南** → `DATASET_GUIDE.md`
2. **理解任务数据加载器** → `understanding_data_loader.py`
3. **生成任务数据加载器** → `generation_data_loader.py`
4. **编辑任务数据加载器** → `editing_data_loader.py`
5. **编辑任务训练指南** → `EDITING_TRAINING_GUIDE.md`

---

## ✅ 功能清单

### 理解任务
- ✅ COCO数据集支持（目标检测、分割）
- ✅ Visual Genome支持（场景图、关系）
- ✅ ADE20K支持（语义分割）
- ✅ 自动关系矩阵构建
- ✅ 边界框归一化
- ✅ Patch级语义标签

### 生成任务
- ✅ LAION数据集支持（大规模）
- ✅ Conceptual Captions支持（高质量）
- ✅ COCO Captions支持（多描述）
- ✅ DiffusionDB支持（提示词）
- ✅ 通用JSON格式支持
- ✅ 条件生成支持

### 编辑任务
- ✅ InstructPix2Pix支持（大规模）
- ✅ MagicBrush支持（高质量）
- ✅ RefCOCO支持（精确定位）
- ✅ 编辑掩码处理
- ✅ 区域保持约束

---

## 🎉 总结

现在您的MLLM模型拥有：

✅ **10个专业数据集**的完整支持
✅ **3个任务**的统一数据加载接口
✅ **智能的数据处理**（关系矩阵、掩码、归一化）
✅ **完整的训练流程**文档
✅ **最佳实践**建议

**从合成数据到真实数据集，完整的训练pipeline！**

开始使用真实数据训练您的多模态大模型吧！🚀
