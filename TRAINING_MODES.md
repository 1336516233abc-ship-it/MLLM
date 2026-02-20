# 预训练模式详细说明

## 📋 概述

MLLM预训练支持6种不同的训练模式，灵活控制训练过程：

1. **understanding** - 只训练理解任务
2. **generation** - 只训练生成任务（无参考图像）
3. **editing** - 只训练编辑任务（有参考图像的生成）
4. **mixed1** - 理解+生成混合（同一批数据同时训练）⭐
5. **mixed2** - 理解+编辑混合（同一批数据同时训练）⭐
6. **mixed** - 理解+生成+编辑全部混合

---

## 🔑 关键概念

### 1. 编辑 = 有参考图像的条件生成

**编辑任务的本质**：
```
编辑 = 条件图像生成 + 源图像参考 + 区域约束

输入：源图像 + 编辑指令
输出：编辑后的图像
约束：未编辑区域保持不变
```

**与生成任务的区别**：
- **生成任务**：从噪声/零图像开始，根据文本生成全新图像
- **编辑任务**：从源图像开始，根据文本指令修改特定区域

### 2. 混合模式的训练方式

**重要**：mixed1和mixed2都是**用同一批数据同时训练两个任务**！

```python
# mixed1模式（理解+生成）
for batch in dataloader:
    # 同一个batch同时计算两个任务的损失
    understanding_loss = compute_understanding_loss(batch)
    generation_loss = compute_generation_loss(batch)
    total_loss = understanding_loss + generation_loss
    total_loss.backward()

# mixed2模式（理解+编辑）
for batch in dataloader:
    # 同一个batch同时计算两个任务的损失
    understanding_loss = compute_understanding_loss(batch)
    editing_loss = compute_editing_loss(batch)
    total_loss = understanding_loss + editing_loss
    total_loss.backward()
```

**不是**这样的：
```python
# ❌ 错误理解：分别用不同数据
for understanding_batch in understanding_loader:
    understanding_loss.backward()
for generation_batch in generation_loader:
    generation_loss.backward()
```

---

## 🎯 训练模式详解

### 1. understanding模式

**训练内容**：
- ✅ LoT低层（元素识别、任务理解）
- ✅ LoT中层（语义理解、空间定位）
- ✅ LoT高层（关系推理、场景理解）
- ✅ 整合模块（连接LoT各层）
- ❌ 扩散模块（不更新参数）

**损失函数**：
```python
{
    'task_loss': 任务分类损失,
    'semantic_loss': 语义分割损失,
    'bbox_loss': 边界框回归损失,
    'relation_loss': 关系矩阵损失
}
```

**适用场景**：
- 提升图像理解能力
- 训练LoT分层推理
- 使用：COCO、Visual Genome、ADE20K

**代码示例**：
```python
trainer.train_epoch(dataloader, epoch, mode='understanding')
```

---

### 2. generation模式

**训练内容**：
- ✅ 扩散模块（图像生成）
- ✅ 整合模块（生成条件）
- ✅ LoT模块（用于生成条件，但不主要优化理解）
- ❌ 理解任务头（不更新）

**损失函数**：
```python
{
    'diffusion_loss': 扩散重构损失
}
```

**生成方式**：
```
输入：文本描述 + 噪声/零图像
过程：LoT提取语义 → 整合模块 → 扩散模块生成
输出：全新生成的图像
```

**适用场景**：
- 提升图像生成能力
- 文本到图像生成
- 使用：LAION、CC12M、COCO Captions

**代码示例**：
```python
trainer.train_epoch(dataloader, epoch, mode='generation')
```

---

### 3. editing模式

**训练内容**：
- ✅ 扩散模块（条件图像编辑）
- ✅ 整合模块（编辑条件）
- ✅ 区域约束（保持+编辑）
- ✅ LoT模块（用于理解编辑指令）

**损失函数**：
```python
{
    'editing_diffusion_loss': 编辑重构损失（权重1.0）,
    'preserve_loss': 保持区域损失（权重0.5）,
    'edit_region_loss': 编辑区域损失（权重1.0）,
    'perceptual_loss': 感知损失（权重0.1）
}
```

**编辑方式**：
```
输入：源图像 + 编辑指令 + 编辑掩码（可选）
过程：LoT理解指令 → 整合模块 → 扩散模块编辑
输出：编辑后的图像
约束：未编辑区域≈源图像
```

**适用场景**：
- 提升图像编辑能力
- 指令驱动的图像修改
- 使用：InstructPix2Pix、MagicBrush、RefCOCO

**代码示例**：
```python
trainer.train_epoch(dataloader, epoch, mode='editing')
```

---

### 4. mixed1模式（理解+生成）⭐⭐⭐

**核心特点**：同一批数据同时训练两个任务

**训练内容**：
- ✅ LoT所有层（理解能力）
- ✅ 扩散模块（生成能力）
- ✅ 整合模块（连接理解和生成）
- ✅ 理解和生成相互增强

**损失函数**：
```python
{
    # 理解损失
    'task_loss': 任务分类,
    'semantic_loss': 语义分割,
    'bbox_loss': 边界框,
    'relation_loss': 关系矩阵,

    # 生成损失
    'diffusion_loss': 扩散重构,

    # 总损失 = 理解损失 + 生成损失
    'total_loss': sum(all losses)
}
```

**训练流程**：
```python
for batch in dataloader:
    # 步骤1：理解任务前向传播
    understanding_outputs = model(batch, mode='understanding')
    understanding_loss = compute_understanding_losses(understanding_outputs)

    # 步骤2：生成任务前向传播（使用同一个batch）
    generation_outputs = model(batch, mode='generation')
    generation_loss = compute_generation_losses(generation_outputs)

    # 步骤3：合并损失，一次反向传播
    total_loss = understanding_loss + generation_loss
    total_loss.backward()
    optimizer.step()
```

**为什么有效**：
1. **理解帮助生成**：更好的图像理解 → 更准确的生成条件 → 更高质量的生成结果
2. **生成帮助理解**：生成任务需要深入理解 → 促进LoT学习更好的特征表示
3. **统一表示**：两个任务共享LoT和整合模块，学习统一的多模态表示

**数据要求**：
```python
{
    'images': 输入图像,
    'text_tokens': 文本描述,
    'text_mask': 注意力掩码,
    'target_images': 目标图像（用于生成）,
    'targets': {
        'task_labels': 任务标签,
        'semantic_labels': 语义标签,
        'bboxes': 边界框,
        'relation_matrix': 关系矩阵
    }
}
```

**适用场景**：
- 训练统一的理解-生成模型
- 两个任务互相促进
- 最常用的混合模式

**代码示例**：
```python
# 数据集应该同时包含理解和生成所需的标注
trainer.train_epoch(dataloader, epoch, mode='mixed1')
```

---

### 5. mixed2模式（理解+编辑）⭐⭐⭐

**核心特点**：同一批数据同时训练两个任务

**训练内容**：
- ✅ LoT所有层（理解能力）
- ✅ 扩散模块（编辑能力）
- ✅ 整合模块（连接理解和编辑）
- ✅ 区域约束（编辑精确性）
- ✅ 理解和编辑相互增强

**损失函数**：
```python
{
    # 理解损失
    'task_loss': 任务分类,
    'semantic_loss': 语义分割,
    'bbox_loss': 边界框,
    'relation_loss': 关系矩阵,

    # 编辑损失
    'editing_diffusion_loss': 编辑重构,
    'preserve_loss': 保持区域,
    'edit_region_loss': 编辑区域,
    'perceptual_loss': 感知损失,

    # 总损失 = 理解损失 + 编辑损失
    'total_loss': sum(all losses)
}
```

**训练流程**：
```python
for batch in dataloader:
    # 步骤1：理解任务前向传播
    understanding_outputs = model(batch, mode='understanding')
    understanding_loss = compute_understanding_losses(understanding_outputs)

    # 步骤2：编辑任务前向传播（使用同一个batch）
    editing_outputs = model(batch, mode='editing')
    editing_loss = compute_editing_losses(editing_outputs)

    # 步骤3：合并损失，一次反向传播
    total_loss = understanding_loss + editing_loss
    total_loss.backward()
    optimizer.step()
```

**为什么有效**：
1. **理解定位编辑区域**：语义分割和目标检测 → 精确定位需要编辑的区域
2. **编辑需要深入理解**：准确编辑需要理解对象、关系、空间布局
3. **天然匹配**：编辑本质上需要理解图像内容

**数据要求**：
```python
{
    'images': 源图像,
    'text_tokens': 编辑指令,
    'text_mask': 注意力掩码,
    'edited_images': 编辑后的图像,
    'targets': {
        'task_labels': 任务标签,
        'semantic_labels': 语义标签,
        'bboxes': 边界框,
        'relation_matrix': 关系矩阵,
        'edit_mask': 编辑区域掩码
    }
}
```

**适用场景**：
- 训练精确的指令编辑模型
- 理解和编辑深度结合
- 需要精确控制编辑区域

**代码示例**：
```python
# 数据集应该同时包含理解和编辑所需的标注
trainer.train_epoch(dataloader, epoch, mode='mixed2')
```

---

### 6. mixed模式（全部混合）

**训练内容**：
- ✅ 所有模块全面训练
- ✅ 三个任务同时训练

**损失函数**：
```python
{
    # 所有理解损失
    'task_loss', 'semantic_loss', 'bbox_loss', 'relation_loss',

    # 所有生成损失
    'diffusion_loss',

    # 所有编辑损失
    'editing_diffusion_loss', 'preserve_loss', 'edit_region_loss', 'perceptual_loss',

    # 总损失
    'total_loss': sum(all)
}
```

**适用场景**：
- 训练最强大的统一模型
- 数据丰富，计算资源充足

---

## 📊 训练模式对比表

| 模式 | 理解 | 生成 | 编辑 | 同批训练 | 推荐度 | 计算量 |
|------|------|------|------|---------|--------|--------|
| understanding | ✅ | ❌ | ❌ | - | ⭐⭐⭐ | ⚡ |
| generation | ❌ | ✅ | ❌ | - | ⭐⭐⭐ | ⚡⚡ |
| editing | ❌ | ❌ | ✅ | - | ⭐⭐⭐ | ⚡⚡ |
| **mixed1** | ✅ | ✅ | ❌ | ✅ | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ |
| **mixed2** | ✅ | ❌ | ✅ | ✅ | ⭐⭐⭐⭐⭐ | ⭐⚡⚡ |
| mixed | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ |

---

## 🚀 推荐的训练策略

### 策略A：阶段性训练（最推荐）⭐⭐⭐⭐⭐

```python
# 阶段1：专项基础训练（各30 epochs）
trainer.train_epoch(dataloader, epoch, mode='understanding')  # 30 epochs
trainer.train_epoch(dataloader, epoch, mode='generation')     # 30 epochs
trainer.train_epoch(dataloader, epoch, mode='editing')        # 30 epochs

# 阶段2：混合协同训练（各50 epochs）
trainer.train_epoch(dataloader, epoch, mode='mixed1')  # 理解+生成，50 epochs
trainer.train_epoch(dataloader, epoch, mode='mixed2')  # 理解+编辑，50 epochs

# 阶段3：全面整合（30 epochs）
trainer.train_epoch(dataloader, epoch, mode='mixed')   # 全部混合，30 epochs
```

**优点**：
- 每个模块先独立学习基础能力
- 混合模式促进任务协同
- 最后全面整合
- **最稳定的训练方式**

---

### 策略B：快速混合训练

```python
# 直接从mixed1开始（70 epochs）
trainer.train_epoch(dataloader, epoch, mode='mixed1')

# 然后mixed2（70 epochs）
trainer.train_epoch(dataloader, epoch, mode='mixed2')

# 最后全面混合（30 epochs）
trainer.train_epoch(dataloader, epoch, mode='mixed')
```

**优点**：
- 节省时间
- 任务早期互相促进
- 适合数据量充足

---

### 策略C：专注特定组合

```python
# 如果只关心理解+编辑
trainer.train_epoch(dataloader, epoch, mode='understanding')  # 30 epochs
trainer.train_epoch(dataloader, epoch, mode='editing')        # 30 epochs
trainer.train_epoch(dataloader, epoch, mode='mixed2')         # 100 epochs
```

---

## 💻 完整代码示例

### 示例1：使用mixed1模式

```python
from MLLM import MLLMModel, Config
from MLLM.training import PreTrainer
from torch.utils.data import DataLoader

# 初始化
config = Config()
model = MLLMModel(config).to('cuda')
model.freeze_encoders()
trainer = PreTrainer(model, config)

# 准备数据集（需要包含理解和生成所需的所有标注）
class Mixed1Dataset(Dataset):
    def __getitem__(self, idx):
        return {
            'images': ...,           # 输入图像
            'text_tokens': ...,      # 文本描述
            'text_mask': ...,        # 掩码
            'target_images': ...,    # 目标图像（生成用）
            'targets': {
                'task_labels': ...,      # 理解用
                'semantic_labels': ...,  # 理解用
                'bboxes': ...,           # 理解用
                'relation_matrix': ...   # 理解用
            }
        }

dataloader = DataLoader(Mixed1Dataset(), batch_size=8)

# 训练
for epoch in range(50):
    print(f"\nEpoch {epoch}")

    # mixed1模式：同一批数据同时训练理解和生成
    metrics = trainer.train_epoch(dataloader, epoch, mode='mixed1')

    print(f"训练指标: {metrics}")
    # 输出示例：
    # {
    #     'task_loss': 0.234,
    #     'semantic_loss': 0.456,
    #     'bbox_loss': 0.123,
    #     'relation_loss': 0.345,
    #     'diffusion_loss': 0.567,
    #     'total_loss': 1.725  # = 所有损失之和
    # }

    # 验证
    if epoch % 10 == 0:
        val_metrics = trainer.validate(val_loader, mode='mixed1')
        print(f"验证指标: {val_metrics}")

    # 保存检查点
    if epoch % 20 == 0:
        trainer.save_checkpoint(
            f'checkpoints/mixed1_epoch_{epoch}.pth',
            epoch,
            metrics
        )
```

---

### 示例2：使用mixed2模式

```python
# 准备数据集（需要包含理解和编辑所需的所有标注）
class Mixed2Dataset(Dataset):
    def __getitem__(self, idx):
        return {
            'images': ...,           # 源图像
            'text_tokens': ...,      # 编辑指令
            'text_mask': ...,        # 掩码
            'edited_images': ...,    # 编辑后的图像
            'targets': {
                'task_labels': ...,      # 理解用
                'semantic_labels': ...,  # 理解用
                'bboxes': ...,           # 理解用
                'relation_matrix': ...,  # 理解用
                'edit_mask': ...         # 编辑用
            }
        }

dataloader = DataLoader(Mixed2Dataset(), batch_size=8)

# 训练
for epoch in range(50):
    # mixed2模式：同一批数据同时训练理解和编辑
    metrics = trainer.train_epoch(dataloader, epoch, mode='mixed2')

    print(f"Epoch {epoch}: {metrics}")
    # 输出示例：
    # {
    #     'task_loss': 0.234,
    #     'semantic_loss': 0.456,
    #     'bbox_loss': 0.123,
    #     'relation_loss': 0.345,
    #     'editing_diffusion_loss': 0.234,
    #     'preserve_loss': 0.045,
    #     'edit_region_loss': 0.156,
    #     'perceptual_loss': 0.023,
    #     'total_loss': 1.616  # = 所有损失之和
    # }
```

---

## 📈 训练监控

### 不同模式的关键指标

**understanding模式**：
```
PreTrain [理解] Epoch 10: 100%|████| 1250/1250 [05:23<00:00, loss=1.158]
训练指标: {'task_loss': 0.234, 'semantic_loss': 0.456, ...}
```

**mixed1模式**：
```
PreTrain [理解+生成] Epoch 10: 100%|████| 1250/1250 [08:45<00:00, loss=1.725]
训练指标: {'task_loss': 0.234, 'diffusion_loss': 0.567, ...}
```

**mixed2模式**：
```
PreTrain [理解+编辑] Epoch 10: 100%|████| 1250/1250 [09:12<00:00, loss=1.616]
训练指标: {'task_loss': 0.234, 'editing_diffusion_loss': 0.234, ...}
```

---

## 💡 最佳实践

### 1. 数据准备

**mixed1模式数据**：
- 必须包含：图像、文本、目标图像
- 必须包含：理解标注（语义、边界框、关系）
- 推荐来源：COCO（含检测+描述）

**mixed2模式数据**：
- 必须包含：源图像、编辑指令、编辑后图像
- 必须包含：理解标注 + 编辑掩码
- 推荐来源：COCO + InstructPix2Pix的组合

### 2. 学习率设置

```python
lr_config = {
    'understanding': 1e-4,
    'generation': 1e-4,
    'editing': 5e-5,
    'mixed1': 8e-5,      # 理解+生成
    'mixed2': 8e-5,      # 理解+编辑
    'mixed': 5e-5        # 全部混合
}
```

### 3. 批次大小

```python
batch_size_config = {
    'understanding': 16,
    'generation': 8,
    'editing': 8,
    'mixed1': 8,         # 需要同时计算两个任务
    'mixed2': 8,         # 需要同时计算两个任务
    'mixed': 4           # 需要同时计算三个任务
}
```

---

## 🎯 总结

### 模式选择建议

**如果您想要**：
- ✅ **最强大的统一模型** → 策略A（阶段性训练）
- ✅ **理解+生成能力** → 使用 `mixed1` 模式
- ✅ **理解+编辑能力** → 使用 `mixed2` 模式
- ✅ **快速原型验证** → 单一模式 + mixed1
- ✅ **专项能力提升** → 对应的单一模式

### 关键要点

1. **编辑 = 有参考图像的条件生成**
2. **mixed1和mixed2用同一批数据同时训练两个任务**
3. **混合模式促进任务协同，效果更好**
4. **阶段性训练最稳定**

现在您可以根据需求灵活选择训练模式了！🚀
