# 图像编辑任务训练指南

## 📋 概述

图像编辑任务已经完全集成到预训练流程中。编辑任务的特点是：
- **输入**：源图像 + 编辑指令（文本）
- **输出**：编辑后的图像
- **约束**：未编辑区域应保持不变

---

## 🔧 代码修改说明

### 1. PreTrainer类的更新

#### 新增：`compute_editing_losses()` 方法
```python
def compute_editing_losses(self, outputs, targets, source_images, edited_images):
    """
    计算编辑任务的4种损失：
    1. 扩散重构损失：生成图像匹配目标
    2. 区域保持损失：未编辑区域保持不变
    3. 编辑区域一致性损失：编辑区域匹配目标
    4. 感知损失：特征空间的相似性
    """
```

**损失函数详解**：

| 损失类型 | 权重 | 作用 |
|---------|------|------|
| editing_diffusion_loss | 1.0 | 主要重构损失 |
| preserve_loss | 0.5 | 保持未编辑区域 |
| edit_region_loss | 1.0 | 编辑区域准确性 |
| perceptual_loss | 0.1 | 高层特征匹配 |

#### 更新：`train_step()` 方法
```python
# 新增编辑任务训练分支
if mode in ['editing', 'mixed'] and 'edited_images' in batch:
    source_images = batch['images']
    edited_images = batch['edited_images']
    edit_instruction = batch['text_tokens']

    # 前向传播
    editing_outputs = self.model(
        source_images, edit_instruction, text_mask,
        target_images=edited_images,
        mode='editing'
    )

    # 计算编辑损失
    editing_losses = self.compute_editing_losses(...)
```

#### 更新：`validate()` 方法
添加了编辑任务的验证逻辑。

---

## 📊 数据格式要求

### 基本数据格式
```python
{
    'images': torch.Tensor,          # (B, 3, H, W) 源图像
    'edited_images': torch.Tensor,   # (B, 3, H, W) 编辑后的图像
    'text_tokens': torch.Tensor,     # (B, seq_len) 编辑指令
    'text_mask': torch.Tensor,       # (B, seq_len) 注意力掩码
    'targets': {
        'edit_mask': torch.Tensor,   # (B, 1, H, W) 编辑区域掩码
        'task_labels': torch.Tensor  # (B,) 任务类型=2
    }
}
```

### 编辑掩码说明
- **值为1**：表示该区域被编辑
- **值为0**：表示该区域应保持不变
- **形状**：(B, 1, H, W)

---

## 🎯 支持的编辑数据集

### 1. InstructPix2Pix ⭐⭐⭐⭐⭐
**最推荐！**

**数据规模**：
- 训练集：~450K 编辑样本
- 基于Stable Diffusion生成

**数据格式**：
```json
{
    "source_image": "images/000001.jpg",
    "edited_image": "images/000001_edited.jpg",
    "instruction": "Make the sky more blue"
}
```

**下载**：
```bash
# 官方仓库
git clone https://github.com/timothybrooks/instruct-pix2pix
cd instruct-pix2pix

# 下载数据
wget https://instruct-pix2pix.eecs.berkeley.edu/clip-filtered-dataset/data.tar
tar -xf data.tar
```

**使用**：
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

---

### 2. MagicBrush ⭐⭐⭐⭐⭐
**高质量人工标注！**

**数据规模**：
- 训练集：~10K 编辑样本
- 人工标注的编辑指令和掩码

**数据格式**：
```json
{
    "source_img": "source/img_001.jpg",
    "target_img": "target/img_001.jpg",
    "mask_img": "mask/img_001.png",
    "instruction": "Change the color of the car to red"
}
```

**下载**：
```bash
# 官方仓库
git clone https://github.com/OSU-NLP-Group/MagicBrush
cd MagicBrush

# 下载数据
python download_data.py
```

**使用**：
```python
train_loader = create_editing_dataloader(
    dataset_type='magicbrush',
    data_root='/path/to/magicbrush',
    config=config,
    batch_size=8,
    split='train'
)
```

---

### 3. RefCOCO/RefCOCO+/RefCOCOg ⭐⭐⭐⭐
**精确的区域定位！**

**数据规模**：
- RefCOCO：142K 表达式
- RefCOCO+：142K 表达式
- RefCOCOg：95K 表达式

**数据格式**：
```json
{
    "image_file": "COCO_train2014_000000123456.jpg",
    "referring_expression": "the person on the left",
    "edit_instruction": "Remove the person on the left",
    "bbox": [0.2, 0.3, 0.15, 0.4]  // [x, y, w, h] 归一化
}
```

**下载**：
```bash
git clone https://github.com/lichengunc/refer
cd refer
python setup.py install
```

**使用**：
```python
train_loader = create_editing_dataloader(
    dataset_type='refcoco',
    data_root='/path/to/refcoco',
    config=config,
    batch_size=8,
    split='train'
)
```

---

## 🚀 训练流程

### 方案A：分阶段训练（推荐）

#### 阶段1：理解+生成预训练
```bash
python MLLM/train.py \
    --device cuda \
    --batch-size 8 \
    --skip-cgpo
```

在 `train.py` 中，预训练默认使用 `mode='mixed'`，包含理解和生成任务。

#### 阶段2：添加编辑任务
修改训练脚本，使用编辑数据：

```python
# 在train.py中修改
from MLLM.utils.editing_data_loader import create_editing_dataloader

# 创建编辑数据加载器
edit_train_loader = create_editing_dataloader(
    dataset_type='instructpix2pix',
    data_root='/path/to/instructpix2pix',
    config=config,
    batch_size=8,
    split='train'
)

# 训练编辑任务
for epoch in range(config.PRETRAIN_EPOCHS):
    # 混合训练：理解+生成+编辑
    for batch in edit_train_loader:
        metrics = trainer.train_step(batch, mode='editing')
```

#### 阶段3：CGPO强化学习
```bash
python MLLM/train.py \
    --device cuda \
    --batch-size 8 \
    --skip-pretrain
```

---

### 方案B：统一训练

创建混合数据加载器：

```python
from torch.utils.data import ConcatDataset, DataLoader

# 创建多个数据集
understanding_dataset = MultiModalDataset(config, split='train')
editing_dataset = InstructPix2PixDataset(...)

# 合并数据集
mixed_dataset = ConcatDataset([understanding_dataset, editing_dataset])

# 创建数据加载器
mixed_loader = DataLoader(mixed_dataset, batch_size=8, shuffle=True)

# 训练
for batch in mixed_loader:
    # 根据batch中的task_labels自动选择模式
    task_type = batch['targets']['task_labels'][0].item()
    if task_type == 0:
        mode = 'understanding'
    elif task_type == 1:
        mode = 'generation'
    else:  # task_type == 2
        mode = 'editing'

    metrics = trainer.train_step(batch, mode=mode)
```

---

## 📈 训练监控

### 关键指标

**编辑任务指标**：
```python
{
    'editing_diffusion_loss': 0.234,    # 主要重构损失
    'preserve_loss': 0.045,             # 保持区域损失
    'edit_region_loss': 0.156,          # 编辑区域损失
    'perceptual_loss': 0.023,           # 感知损失
    'total_loss': 0.458                 # 总损失
}
```

### 评估指标

1. **PSNR** (Peak Signal-to-Noise Ratio)
   - 评估生成图像质量
   - 越高越好（通常>25dB）

2. **LPIPS** (Learned Perceptual Image Patch Similarity)
   - 评估感知相似度
   - 越低越好（通常<0.2）

3. **CLIP Score**
   - 评估编辑与指令的一致性
   - 越高越好（通常>0.7）

4. **区域保持率**
   - 未编辑区域的相似度
   - 越高越好（通常>0.95）

---

## 💻 使用示例

### 训练编辑模型

```python
from MLLM import MLLMModel, Config
from MLLM.training import PreTrainer
from MLLM.utils.editing_data_loader import create_editing_dataloader

# 配置
config = Config()
config.DEVICE = 'cuda'

# 创建模型
model = MLLMModel(config).to(config.DEVICE)
model.freeze_encoders()

# 创建训练器
trainer = PreTrainer(model, config)

# 加载编辑数据
train_loader = create_editing_dataloader(
    dataset_type='instructpix2pix',
    data_root='/path/to/data',
    config=config,
    batch_size=8,
    split='train'
)

# 训练
for epoch in range(50):
    metrics = trainer.train_epoch(train_loader, epoch, mode='editing')
    print(f"Epoch {epoch}: {metrics}")
```

### 使用编辑功能

```python
# 加载训练好的模型
model = MLLMModel(config).to('cuda')
checkpoint = torch.load('checkpoints/final_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 编辑图像
from PIL import Image
import torchvision.transforms as transforms

# 加载源图像
source_image = Image.open('source.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
source_tensor = transform(source_image).unsqueeze(0).to('cuda')

# 编辑指令
instruction = "Make the sky more blue"
text_tokens, text_mask = model.text_tokenizer.tokenize([instruction])
text_tokens = text_tokens.to('cuda')
text_mask = text_mask.to('cuda')

# 执行编辑
with torch.no_grad():
    outputs = model(
        source_tensor,
        text_tokens,
        text_mask,
        mode='editing'
    )

    # 生成编辑后的图像
    edited_image = model.diffusion_module.sample(
        outputs['generation_condition'],
        image_size=(224, 224)
    )

# 保存结果
from torchvision.utils import save_image
save_image((edited_image + 1) / 2, 'edited.jpg')
```

---

## 🎯 最佳实践

### 1. 数据准备
- 使用高质量的编辑数据集（MagicBrush）
- 确保编辑掩码准确
- 平衡不同类型的编辑（颜色、形状、添加、删除等）

### 2. 训练策略
- 先训练理解+生成，再加入编辑
- 使用较小的学习率（5e-5）
- 监控preserve_loss，确保未编辑区域保持不变

### 3. 损失权重调整
```python
# 如果编辑区域不够准确
edit_region_loss_weight = 1.5  # 增加权重

# 如果未编辑区域变化太大
preserve_loss_weight = 1.0  # 增加权重

# 如果整体质量不够好
perceptual_loss_weight = 0.2  # 增加权重
```

### 4. 数据增强
```python
# 对源图像和编辑图像应用相同的增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

---

## 📝 总结

编辑任务已完全集成到预训练流程中：

✅ **新增功能**：
- `compute_editing_losses()` - 4种编辑损失
- `train_step()` 支持 `mode='editing'`
- `validate()` 支持编辑任务验证
- 专门的编辑数据加载器

✅ **支持的数据集**：
- InstructPix2Pix（450K样本）
- MagicBrush（10K高质量样本）
- RefCOCO系列（精确区域定位）

✅ **训练模式**：
- `mode='understanding'` - 只训练理解
- `mode='generation'` - 只训练生成
- `mode='editing'` - 只训练编辑
- `mode='mixed'` - 混合训练（推荐）

现在您可以使用完整的理解+生成+编辑统一架构了！🚀
