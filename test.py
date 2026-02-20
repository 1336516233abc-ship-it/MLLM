"""
测试脚本 - 测试训练好的模型
"""

import torch
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from MLLM import MLLMModel, Config

def load_model(checkpoint_path, config, device):
    """加载训练好的模型"""
    model = MLLMModel(config).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"模型加载成功: {checkpoint_path}")
    else:
        print(f"警告: 模型文件不存在: {checkpoint_path}")
        print("将使用随机初始化的模型进行测试")

    model.eval()
    return model

def tensor_to_image(tensor):
    """将张量转换为可显示的图像"""
    # tensor: (C, H, W) in range [-1, 1]
    image = tensor.cpu().detach()
    image = (image + 1) / 2  # [-1, 1] -> [0, 1]
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return image

def test_image_generation(model, config, device, save_dir='test_results'):
    """测试图像生成功能"""
    print("\n" + "="*50)
    print("测试图像生成")
    print("="*50)

    os.makedirs(save_dir, exist_ok=True)

    # 测试提示
    test_prompts = [
        "A beautiful sunset over the ocean",
        "A cat sitting on a window",
        "A modern city skyline at night",
        "A red car in a forest"
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n生成图像 {i+1}: {prompt}")

        # 生成图像
        with torch.no_grad():
            generated_image = model.generate_image(prompt)

        # 保存和显示
        img = tensor_to_image(generated_image[0])

        # 保存图像
        save_path = os.path.join(save_dir, f'generated_{i+1}.png')
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(prompt)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"图像已保存: {save_path}")

def test_image_understanding(model, config, device):
    """测试图像理解功能"""
    print("\n" + "="*50)
    print("测试图像理解")
    print("="*50)

    # 创建测试图像
    test_image = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    test_image = (test_image - 0.5) / 0.5

    # 测试问题
    questions = [
        "What is in this image?",
        "Describe the colors",
        "What objects can you see?"
    ]

    for i, question in enumerate(questions):
        print(f"\n问题 {i+1}: {question}")

        # 理解图像
        with torch.no_grad():
            understanding = model.understand_image(test_image, question)

        print(f"任务类型: {understanding['task_type']}")
        print(f"检测到的边界框数量: {understanding['bounding_boxes'].shape[1]}")
        print(f"语义分割形状: {understanding['semantic_segmentation'].shape}")

        # 分析LoT各层输出
        lot_analysis = understanding['lot_analysis']
        print("\nLoT分层推理分析:")
        print(f"  低层 - 任务理解特征维度: {lot_analysis['low']['task_features'].shape}")
        print(f"  中层 - 语义特征维度: {lot_analysis['mid']['semantic_features'].shape}")
        print(f"  高层 - 场景特征维度: {lot_analysis['high']['scene_features'].shape}")

def test_lot_reasoning(model, config, device):
    """详细测试LoT分层推理"""
    print("\n" + "="*50)
    print("测试LoT分层推理详细分析")
    print("="*50)

    # 创建测试输入
    test_image = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    test_image = (test_image - 0.5) / 0.5

    text = "Test prompt for reasoning"
    text_tokens, text_mask = model.text_tokenizer.tokenize([text, text])
    text_tokens = text_tokens.to(device)
    text_mask = text_mask.to(device)

    # 前向传播
    with torch.no_grad():
        outputs = model(test_image, text_tokens, text_mask, mode='understanding')

    lot_outputs = outputs['lot_outputs']

    print("\n低层推理结果:")
    print(f"  任务分类logits: {lot_outputs['low']['task_logits'].shape}")
    print(f"  任务预测: {torch.argmax(lot_outputs['low']['task_logits'], dim=-1).tolist()}")
    print(f"  元素特征: {lot_outputs['low']['element_features'].shape}")

    print("\n中层推理结果:")
    print(f"  语义特征: {lot_outputs['mid']['semantic_features'].shape}")
    print(f"  空间特征: {lot_outputs['mid']['spatial_features'].shape}")
    print(f"  语义logits: {lot_outputs['mid']['semantic_logits'].shape}")
    print(f"  边界框: {lot_outputs['mid']['bboxes'].shape}")

    print("\n高层推理结果:")
    print(f"  语义关系特征: {lot_outputs['high']['semantic_relation_features'].shape}")
    print(f"  空间关系特征: {lot_outputs['high']['spatial_relation_features'].shape}")
    print(f"  关系矩阵: {lot_outputs['high']['relation_matrix'].shape}")
    print(f"  场景特征: {lot_outputs['high']['scene_features'].shape}")

    print("\n整合模块结果:")
    print(f"  整合特征: {outputs['integrated_features'].shape}")
    print(f"  生成条件: {outputs['generation_condition'].shape}")
    print(f"  层权重: {outputs['layer_weights']}")

def test_reward_model(model, config, device):
    """测试奖励模型"""
    print("\n" + "="*50)
    print("测试奖励模型")
    print("="*50)

    from MLLM.training.reward_model import RewardModel

    # 创建奖励模型
    reward_model = RewardModel(config).to(device)
    reward_model.eval()

    # 创建测试数据
    test_image = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    test_image = (test_image - 0.5) / 0.5

    text = "Test prompt"
    text_tokens, text_mask = model.text_tokenizer.tokenize([text, text])
    text_tokens = text_tokens.to(device)
    text_mask = text_mask.to(device)

    # 获取模型输出
    with torch.no_grad():
        outputs = model(test_image, text_tokens, text_mask, mode='generation')

        # 生成图像
        generated_images = model.diffusion_module.sample(
            outputs['generation_condition'],
            image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)
        )

        # 获取视觉特征
        vision_features, vision_cls = model.vit_encoder(test_image)

        # 计算奖励
        rewards, details = reward_model(
            outputs['lot_outputs'],
            vision_cls,
            generated_images,
            vision_cls
        )

    print(f"\n总奖励分数: {rewards}")
    print(f"\n理解部分奖励: {details['understanding_reward']}")
    print("  详细分数:")
    for key, value in details['understanding'].items():
        print(f"    {key}: {value}")

    print(f"\n生成部分奖励: {details['generation_reward']}")
    print("  详细分数:")
    for key, value in details['generation'].items():
        print(f"    {key}: {value}")

    print(f"\n自适应权重:")
    print(f"  理解: {details['adaptive_weights']['understanding']}")
    print(f"  生成: {details['adaptive_weights']['generation']}")

    # 识别低分方面
    low_aspects = reward_model.get_low_score_aspects(details)
    print(f"\n需要改进的方面: {low_aspects}")

def main(args):
    """主测试函数"""
    print("="*50)
    print("多模态大模型测试")
    print("="*50)

    # 配置
    config = Config()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载模型
    checkpoint_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    model = load_model(checkpoint_path, config, device)

    # 运行测试
    if args.test_generation:
        test_image_generation(model, config, device, args.output_dir)

    if args.test_understanding:
        test_image_understanding(model, config, device)

    if args.test_lot:
        test_lot_reasoning(model, config, device)

    if args.test_reward:
        test_reward_model(model, config, device)

    print("\n" + "="*50)
    print("测试完成!")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试多模态大模型')

    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='测试设备 (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='输出目录')

    # 测试选项
    parser.add_argument('--test-generation', action='store_true',
                       help='测试图像生成')
    parser.add_argument('--test-understanding', action='store_true',
                       help='测试图像理解')
    parser.add_argument('--test-lot', action='store_true',
                       help='测试LoT分层推理')
    parser.add_argument('--test-reward', action='store_true',
                       help='测试奖励模型')
    parser.add_argument('--test-all', action='store_true',
                       help='运行所有测试')

    args = parser.parse_args()

    # 如果指定test-all，启用所有测试
    if args.test_all:
        args.test_generation = True
        args.test_understanding = True
        args.test_lot = True
        args.test_reward = True

    # 如果没有指定任何测试，默认运行所有测试
    if not any([args.test_generation, args.test_understanding,
                args.test_lot, args.test_reward]):
        args.test_all = True
        args.test_generation = True
        args.test_understanding = True
        args.test_lot = True
        args.test_reward = True

    main(args)
