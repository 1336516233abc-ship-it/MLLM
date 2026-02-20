"""
快速演示脚本 - 展示如何使用MLLM模型
"""

import torch
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from MLLM import MLLMModel, Config

def demo_model_creation():
    """演示模型创建"""
    print("="*60)
    print("演示1: 创建MLLM模型")
    print("="*60)

    config = Config()
    config.DEVICE = 'cpu'  # 使用CPU进行演示

    print("\n创建模型...")
    model = MLLMModel(config).to(config.DEVICE)

    print("冻结编码器...")
    model.freeze_encoders()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  冻结参数量: {total_params - trainable_params:,}")

    return model, config

def demo_image_generation(model, config):
    """演示图像生成"""
    print("\n" + "="*60)
    print("演示2: 图像生成")
    print("="*60)

    model.eval()

    prompt = "A beautiful landscape"
    print(f"\n文本提示: {prompt}")

    print("\n生成图像（使用小尺寸加速演示）...")
    with torch.no_grad():
        generated_image = model.generate_image(prompt)

    print(f"生成的图像形状: {generated_image.shape}")
    print(f"图像值范围: [{generated_image.min():.2f}, {generated_image.max():.2f}]")

def demo_image_understanding(model, config):
    """演示图像理解"""
    print("\n" + "="*60)
    print("演示3: 图像理解")
    print("="*60)

    model.eval()

    # 创建测试图像
    test_image = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    test_image = (test_image - 0.5) / 0.5

    question = "What is in this image?"
    print(f"\n问题: {question}")

    print("\n分析图像...")
    with torch.no_grad():
        understanding = model.understand_image(test_image, question)

    print(f"\n理解结果:")
    print(f"  任务类型: {understanding['task_type']}")
    print(f"  语义分割形状: {understanding['semantic_segmentation'].shape}")
    print(f"  边界框形状: {understanding['bounding_boxes'].shape}")
    print(f"  关系矩阵形状: {understanding['relations'].shape}")

def demo_lot_reasoning(model, config):
    """演示LoT分层推理"""
    print("\n" + "="*60)
    print("演示4: LoT分层推理详解")
    print("="*60)

    model.eval()

    # 准备输入
    test_image = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    test_image = (test_image - 0.5) / 0.5

    text = "Analyze this image"
    text_tokens, text_mask = model.text_tokenizer.tokenize([text])

    print("\n输入:")
    print(f"  图像形状: {test_image.shape}")
    print(f"  文本: {text}")

    print("\n执行分层推理...")
    with torch.no_grad():
        outputs = model(test_image, text_tokens, text_mask, mode='understanding')

    lot_outputs = outputs['lot_outputs']

    print("\n低层推理结果:")
    print(f"  任务分类: {lot_outputs['low']['task_logits'].shape}")
    task_pred = torch.argmax(lot_outputs['low']['task_logits'], dim=-1).item()
    task_names = ['理解', '生成', '编辑']
    print(f"  预测任务: {task_names[task_pred]}")
    print(f"  元素特征: {lot_outputs['low']['element_features'].shape}")
    print("  -> 识别了图像中的基本元素")

    print("\n中层推理结果:")
    print(f"  语义特征: {lot_outputs['mid']['semantic_features'].shape}")
    print(f"  空间特征: {lot_outputs['mid']['spatial_features'].shape}")
    print(f"  语义分类: {lot_outputs['mid']['semantic_logits'].shape}")
    print(f"  边界框: {lot_outputs['mid']['bboxes'].shape}")
    print("  -> 为元素赋予语义含义并定位位置")

    print("\n高层推理结果:")
    print(f"  语义关系特征: {lot_outputs['high']['semantic_relation_features'].shape}")
    print(f"  空间关系特征: {lot_outputs['high']['spatial_relation_features'].shape}")
    print(f"  关系矩阵: {lot_outputs['high']['relation_matrix'].shape}")
    print(f"  场景特征: {lot_outputs['high']['scene_features'].shape}")
    print("  -> 理解元素间的关系和整体场景")

    print("\n整合模块结果:")
    print(f"  整合特征: {outputs['integrated_features'].shape}")
    print(f"  生成条件: {outputs['generation_condition'].shape}")
    print(f"  层权重: {outputs['layer_weights']}")
    print(f"  -> 低层权重: {outputs['layer_weights'][0]:.3f}")
    print(f"  -> 中层权重: {outputs['layer_weights'][1]:.3f}")
    print(f"  -> 高层权重: {outputs['layer_weights'][2]:.3f}")

def demo_training_workflow():
    """演示训练工作流程"""
    print("\n" + "="*60)
    print("演示5: 训练工作流程说明")
    print("="*60)

    print("\n第一阶段: 预训练")
    print("  目标: 训练LoT、整合模块、扩散模块")
    print("  方法: 监督学习")
    print("  损失函数:")
    print("    - 任务分类损失")
    print("    - 语义分割损失")
    print("    - 边界框回归损失")
    print("    - 关系矩阵损失")
    print("    - 扩散重构损失")
    print("  命令: python MLLM/train.py --device cpu --skip-cgpo")

    print("\n第二阶段: CGPO强化学习")
    print("  目标: 优化整体生成质量")
    print("  方法: 约束生成策略优化")
    print("  奖励函数:")
    print("    理解部分 (50%):")
    print("      - 完整性 (25%)")
    print("      - 语义逻辑 (25%)")
    print("      - 空间合理性 (25%)")
    print("      - 一致性 (25%)")
    print("    生成部分 (50%):")
    print("      - 颜色质量 (30%)")
    print("      - 语义准确性 (40%)")
    print("      - 视觉效果 (30%)")
    print("  约束: KL散度（beta=0.1）")
    print("  命令: python MLLM/train.py --device cpu --skip-pretrain")

def demo_architecture_summary():
    """演示架构总结"""
    print("\n" + "="*60)
    print("架构总结")
    print("="*60)

    print("\n数据流:")
    print("  1. 输入图像 -> ViT Encoder -> 视觉特征 (冻结)")
    print("  2. 输入文本 -> Text Tokenizer -> 文本特征 (冻结)")
    print("  3. 特征 -> LoT低层 -> 任务理解 + 元素识别")
    print("  4. 低层输出 -> LoT中层 -> 语义理解 + 空间定位")
    print("  5. 中层输出 -> LoT高层 -> 关系推理 + 场景理解")
    print("  6. 三层输出 -> 整合模块 -> 自适应融合")
    print("  7. 整合特征 -> 扩散模块 -> 生成/编辑图像")

    print("\n关键创新:")
    print("  ✓ LoT分层推理: 从低到高逐层深入理解")
    print("  ✓ 自适应整合: 动态调整各层权重")
    print("  ✓ 分层奖励: 多维度评估，着重优化弱项")
    print("  ✓ CGPO优化: 强化学习提升生成质量")
    print("  ✓ 统一架构: 理解、生成、编辑一体化")

def main():
    """主函数"""
    print("\n" + "="*60)
    print("MLLM多模态大模型 - 快速演示")
    print("="*60)

    # 创建模型
    model, config = demo_model_creation()

    # 演示各功能
    demo_image_generation(model, config)
    demo_image_understanding(model, config)
    demo_lot_reasoning(model, config)
    demo_training_workflow()
    demo_architecture_summary()

    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)

    print("\n接下来你可以:")
    print("  1. 运行训练: python MLLM/train.py --device cuda")
    print("  2. 运行测试: python MLLM/test.py --test-all")
    print("  3. 查看README: cat MLLM/README.md")

if __name__ == '__main__':
    main()
