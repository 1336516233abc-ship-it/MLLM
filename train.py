"""
主训练脚本 - 完整的训练流程
包括预训练和CGPO强化学习
"""

import torch
import os
import argparse
from datetime import datetime
from torch.utils.data import DataLoader

from MLLM import MLLMModel, Config, create_dataloaders, PreTrainer, CGPOTrainer, RewardModel
from MLLM.utils.mixed1_data_loader import COCOMixed1Dataset


def setup_directories(config):
    """创建必要的目录"""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)


def create_coco_mixed1_dataloaders(config, coco_root):
    """
    创建基于COCO数据集的mixed1数据加载器

    自动检测标注目录，兼容以下两种常见结构：
      coco_root/annotations/...
      coco_root/annotations_trainval2017/annotations/...

    Args:
        config: 配置对象
        coco_root: COCO数据集根目录
    Returns:
        (train_loader, val_loader)
    """
    # 自动检测标注目录
    ann_dir_candidates = [
        os.path.join(coco_root, 'annotations'),
        os.path.join(coco_root, 'annotations_trainval2017', 'annotations'),
    ]
    ann_dir = None
    for candidate in ann_dir_candidates:
        if os.path.isdir(candidate):
            ann_dir = candidate
            break
    if ann_dir is None:
        raise FileNotFoundError(
            f"未找到COCO标注目录，已尝试以下路径：\n"
            + "\n".join(f"  {p}" for p in ann_dir_candidates)
        )

    print(f"  标注目录: {ann_dir}")
    print(f"  训练图像: {os.path.join(coco_root, 'train2017')}")
    print(f"  验证图像: {os.path.join(coco_root, 'val2017')}")

    train_dataset = COCOMixed1Dataset(
        image_dir=os.path.join(coco_root, 'train2017'),
        instances_file=os.path.join(ann_dir, 'instances_train2017.json'),
        captions_file=os.path.join(ann_dir, 'captions_train2017.json'),
        config=config
    )

    val_dataset = COCOMixed1Dataset(
        image_dir=os.path.join(coco_root, 'val2017'),
        instances_file=os.path.join(ann_dir, 'instances_val2017.json'),
        captions_file=os.path.join(ann_dir, 'captions_val2017.json'),
        config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader


def pretrain_phase(model, config, train_loader, val_loader, mode='mixed1'):
    """
    预训练阶段
    训练LoT、整合模块、扩散模块

    Args:
        mode: 训练模式，'mixed1'表示理解+生成混合训练
    """
    print("\n" + "="*50)
    print(f"开始预训练阶段（模式: {mode}）")
    print("="*50)

    trainer = PreTrainer(model, config)

    best_val_loss = float('inf')

    for epoch in range(1, config.PRETRAIN_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.PRETRAIN_EPOCHS} ---")

        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch, mode=mode)

        # 输出训练指标
        if mode == 'mixed1':
            print(f"Epoch {epoch} 训练指标:")
            print(f"  理解任务:")
            print(f"    - task_loss:     {train_metrics.get('task_loss', 0):.4f}")
            print(f"    - semantic_loss: {train_metrics.get('semantic_loss', 0):.4f}")
            print(f"    - bbox_loss:     {train_metrics.get('bbox_loss', 0):.4f}")
            print(f"    - relation_loss: {train_metrics.get('relation_loss', 0):.4f}")
            print(f"  生成任务:")
            print(f"    - diffusion_loss:{train_metrics.get('diffusion_loss', 0):.4f}")
            print(f"  总损失: {train_metrics.get('total_loss', 0):.4f}")
        else:
            print(f"训练指标: {train_metrics}")

        # 每5个epoch验证一次
        if epoch % 5 == 0:
            val_metrics = trainer.validate(val_loader, mode=mode)
            print(f"Epoch {epoch} 验证指标: {val_metrics}")

            # 保存最佳模型
            val_loss = val_metrics.get('total_loss', val_metrics.get('diffusion_loss', 0))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'pretrain_best.pth')
                trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)
                print(f"保存最佳模型到 {checkpoint_path}")

        # 每10个epoch保存一次检查点
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'pretrain_epoch_{epoch}.pth'
            )
            trainer.save_checkpoint(checkpoint_path, epoch, train_metrics)
            print(f"保存检查点到 {checkpoint_path}")

    print("\n预训练阶段完成!")
    return model

def cgpo_phase(model, config, train_loader, val_loader):
    """
    CGPO强化学习阶段
    使用奖励模型优化MLLM
    """
    print("\n" + "="*50)
    print("开始CGPO强化学习阶段")
    print("="*50)

    # 创建奖励模型
    reward_model = RewardModel(config).to(config.DEVICE)

    # 创建CGPO训练器
    trainer = CGPOTrainer(model, reward_model, config)

    best_reward = float('-inf')

    for epoch in range(1, config.CGPO_EPOCHS + 1):
        print(f"\n--- CGPO Epoch {epoch}/{config.CGPO_EPOCHS} ---")

        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"训练指标: {train_metrics}")

        # 更新参考模型（每5个epoch）
        if epoch % 5 == 0:
            print("更新参考模型...")
            trainer.update_reference_model()

        # 保存最佳模型
        avg_reward = train_metrics.get('reward_mean', 0)
        if avg_reward > best_reward:
            best_reward = avg_reward
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                'cgpo_best.pth'
            )
            trainer.save_checkpoint(checkpoint_path, epoch, train_metrics)
            print(f"保存最佳模型到 {checkpoint_path}")

        # 定期保存检查点
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'cgpo_epoch_{epoch}.pth'
            )
            trainer.save_checkpoint(checkpoint_path, epoch, train_metrics)

    print("\nCGPO阶段完成!")
    return model

def main(args):
    """主训练函数"""
    print("="*50)
    print("多模态大模型训练")
    print("="*50)

    # 配置
    config = Config()
    config.DEVICE = args.device
    config.BATCH_SIZE = args.batch_size

    # 打印配置
    print(f"\n配置:")
    print(f"  设备: {config.DEVICE}")
    print(f"  批次大小: {config.BATCH_SIZE}")
    print(f"  训练模式: {args.mode}")
    print(f"  预训练轮数: {config.PRETRAIN_EPOCHS}")
    print(f"  CGPO轮数: {config.CGPO_EPOCHS}")

    # 创建目录
    setup_directories(config)

    # 创建数据加载器
    print("\n加载数据...")
    if args.mode == 'mixed1':
        print(f"使用COCO数据集进行mixed1训练，路径: {args.coco_path}")
        train_loader, val_loader = create_coco_mixed1_dataloaders(config, args.coco_path)
    else:
        train_loader, val_loader = create_dataloaders(config)
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")

    # 创建模型
    print("\n创建模型...")
    model = MLLMModel(config).to(config.DEVICE)

    # 冻结编码器
    print("冻结ViT和文本编码器...")
    model.freeze_encoders()

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 阶段1: 预训练
    if args.skip_pretrain:
        print("\n跳过预训练阶段，加载预训练模型...")
        pretrain_checkpoint = os.path.join(config.CHECKPOINT_DIR, 'pretrain_best.pth')
        if os.path.exists(pretrain_checkpoint):
            checkpoint = torch.load(pretrain_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("预训练模型加载成功!")
        else:
            print("警告: 未找到预训练模型，将从头开始训练...")
            model = pretrain_phase(model, config, train_loader, val_loader, mode=args.mode)
    else:
        model = pretrain_phase(model, config, train_loader, val_loader, mode=args.mode)

    # 阶段2: CGPO强化学习
    if not args.skip_cgpo:
        model = cgpo_phase(model, config, train_loader, val_loader)

    print("\n" + "="*50)
    print("训练完成!")
    print("="*50)

    # 保存最终模型
    final_checkpoint = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_checkpoint)
    print(f"\n最终模型保存到: {final_checkpoint}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练多模态大模型')

    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备 (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--mode', type=str, default='mixed1',
                       choices=['mixed', 'mixed1', 'mixed2', 'understanding', 'generation'],
                       help='预训练模式: mixed1=理解+生成(COCO), mixed=全混合(合成数据)')
    parser.add_argument('--coco-path', type=str,
                       default='D:/llm-model/MLLM/cocodataset',
                       help='COCO数据集根目录（mixed1模式使用）')
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='跳过预训练阶段')
    parser.add_argument('--skip-cgpo', action='store_true',
                       help='跳过CGPO阶段')

    args = parser.parse_args()

    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU训练")
        args.device = 'cpu'

    main(args)
