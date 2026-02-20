# 多模态大模型 (MLLM) - 图像理解+生成+编辑统一架构

这是一个创新的多模态大模型架构，实现了图像理解、生成和编辑的统一框架。

## 核心特性

### 1. LoT (Layers of Thought) 分层推理
- **低层**: 任务理解和元素识别
- **中层**: 元素语义理解和空间定位
- **高层**: 高级语义关系和空间关系推理

### 2. 模块化架构
- **ViT Encoder**: 视觉Transformer编码器（预训练冻结）
- **Text Tokenizer**: 文本编码器（预训练冻结）
- **LoT Module**: 分层推理核心模块
- **Integration Module**: 自适应整合模块
- **Diffusion Module**: 扩散模型用于图像生成/编辑

### 3. 两阶段训练策略
- **Pre-train**: 预训练微调LoT、整合模块、扩散模块
- **CGPO**: 约束生成策略优化（强化学习）

### 4. 分层奖励模型
- **理解部分**: 完整性、语义逻辑、空间合理性、一致性
- **生成部分**: 颜色质量、语义准确性、视觉效果
- **自适应权重**: 着重优化分数低的方面

## 项目结构

```
MLLM/
├── models/
│   ├── vit_encoder.py          # ViT图像编码器
│   ├── text_tokenizer.py       # 文本分词器
│   ├── lot_layers.py           # LoT分层推理（核心创新）
│   ├── integration_module.py   # 整合模块
│   ├── diffusion_module.py     # 扩散模块
│   └── mllm_model.py           # 主模型
├── training/
│   ├── pretrain.py             # 预训练
│   ├── cgpo.py                 # CGPO强化学习
│   └── reward_model.py         # 奖励模型
├── utils/
│   ├── config.py               # 配置
│   └── data_loader.py          # 数据加载
├── train.py                    # 训练脚本
└── test.py                     # 测试脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

完整训练（预训练 + CGPO）:
```bash
python MLLM/train.py --device cuda --batch-size 8
```

只进行预训练:
```bash
python MLLM/train.py --device cuda --batch-size 8 --skip-cgpo
```

只进行CGPO（需要先完成预训练）:
```bash
python MLLM/train.py --device cuda --batch-size 8 --skip-pretrain
```

### 2. 测试模型

运行所有测试:
```bash
python MLLM/test.py --test-all
```

测试图像生成:
```bash
python MLLM/test.py --test-generation
```

测试图像理解:
```bash
python MLLM/test.py --test-understanding
```

测试LoT分层推理:
```bash
python MLLM/test.py --test-lot
```

测试奖励模型:
```bash
python MLLM/test.py --test-reward
```

指定检查点:
```bash
python MLLM/test.py --checkpoint checkpoints/final_model.pth --test-all
```

### 3. 使用模型API

```python
from MLLM import MLLMModel, Config
import torch

# 创建配置和模型
config = Config()
model = MLLMModel(config).to('cuda')

# 加载训练好的权重
checkpoint = torch.load('checkpoints/final_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 图像生成
generated_image = model.generate_image("A beautiful sunset")

# 图像理解
understanding = model.understand_image(image, "What is in this image?")
print(understanding['task_type'])
print(understanding['semantic_segmentation'])
print(understanding['bounding_boxes'])
```

## 架构详解

### LoT分层推理

**低层推理** (low_layer.py):
- 任务理解分支: 识别用户意图（生成/编辑/理解）
- 元素识别分支: 检测图像中的基本元素

**中层推理** (mid_layer.py):
- 语义理解: 为识别的元素赋予语义含义
- 空间定位: 确定元素的位置和边界框

**高层推理** (high_layer.py):
- 语义关系: 理解元素间的语义联系
- 空间关系: 分析元素间的相对空间关系
- 全局场景理解

### CGPO奖励策略

**理解部分奖励**:
1. 完整性 (25%): 评估元素识别的完整性
2. 语义逻辑 (25%): 评估语义标注的逻辑性
3. 空间合理性 (25%): 评估空间定位的合理性
4. 一致性 (25%): 评估推理与输入的一致性

**生成部分奖励**:
1. 颜色质量 (30%): 评估颜色分布和饱和度
2. 语义准确性 (40%): 评估生成内容与条件的匹配度
3. 视觉效果 (30%): 评估整体视觉质量

**自适应权重机制**:
- 分数低的维度自动获得更高权重
- 优化时着重改进薄弱环节

## 配置说明

主要配置参数在 `utils/config.py`:

```python
# 模型维度
LOT_LOW_DIM = 512    # 低层维度
LOT_MID_DIM = 768    # 中层维度
LOT_HIGH_DIM = 1024  # 高层维度

# 训练参数
PRETRAIN_EPOCHS = 100
CGPO_EPOCHS = 50
BATCH_SIZE = 8

# 奖励权重
REWARD_UNDERSTANDING_TOTAL = 0.5
REWARD_GENERATION_TOTAL = 0.5
```

## 训练监控

训练过程会输出详细指标:
- 总损失
- 策略损失
- KL散度
- 扩散损失
- 平均奖励
- 各维度详细分数

检查点保存在 `checkpoints/` 目录:
- `pretrain_best.pth`: 预训练最佳模型
- `cgpo_best.pth`: CGPO最佳模型
- `final_model.pth`: 最终模型

## 技术细节

### 冻结策略
- ViT Encoder: 全程冻结
- Text Tokenizer: 全程冻结
- LoT Module: 可训练
- Integration Module: 可训练
- Diffusion Module: 可训练

### 优化器配置
- 预训练: AdamW, lr=1e-4
- CGPO: AdamW, lr=5e-5 (扩散模块 lr=2.5e-5)

### KL散度约束
- Beta=0.1
- 防止模型在强化学习中偏离太远

## 注意事项

1. **数据集**: 当前使用合成数据演示。实际应用需要替换为真实数据集。
2. **计算资源**: 建议使用GPU训练，内存至少16GB。
3. **训练时间**: 完整训练可能需要数天时间，取决于硬件。
4. **检查点**: 定期保存检查点，可以随时恢复训练。

## 扩展方向

1. 集成真实数据集（COCO, ImageNet等）
2. 添加更多任务类型
3. 优化扩散采样速度
4. 实现条件编辑功能
5. 增加多模态输入支持

## 引用

如果使用本代码，请引用:
```
@article{mllm2024,
  title={Unified Multimodal Large Model with Layers of Thought},
  author={Your Name},
  year={2024}
}
```

## 许可证

MIT License
