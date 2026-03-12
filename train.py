"""
主训练脚本 - 完整的训练流程
包括预训练和CGPO强化学习
支持单卡、多卡 DDP 和 DeepSpeed ZeRO-3 训练

改进内容：
1. DeepSpeed ZeRO-3 内存优化支持
2. Early Stopping 机制
3. 多任务学习策略（不确定性加权）
4. 完善的评估指标输出
5. 新loss组件（分层一致性、特征匹配）
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import json
import argparse
from datetime import datetime
from torch.utils.data import DataLoader

from MLLM import MLLMModel, Config, create_dataloaders, PreTrainer, CGPOTrainer, RewardModel
from MLLM.utils.mixed1_data_loader import COCOMixed1Dataset


def is_main_process():
    """判断是否为主进程（rank 0），单卡时始终返回 True"""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    """初始化 DDP 进程组，返回当前进程的 local_rank"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """销毁进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_directories(config):
    """创建必要的目录（只在主进程执行）"""
    if is_main_process():
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

    if is_main_process():
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

    # DDP 模式下使用 DistributedSampler，保证每张卡拿到不同的数据分片
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(train_sampler is None),  # 使用 sampler 时 shuffle 必须为 False
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader


def pretrain_phase(model, config, train_loader, val_loader, mode='mixed1'):
    """
    预训练阶段
    训练LoT、整合模块、扩散模块

    改进内容：
    - 改进方向1: 支持DeepSpeed ZeRO-3
    - 改进方向2: Early Stopping
    - 改进方向3: 多任务学习策略（不确定性加权）
    - 改进方向4: 完善的评估指标
    - 改进方向5: 新loss组件

    Args:
        mode: 训练模式，'mixed1'表示理解+生成混合训练
    """
    if is_main_process():
        print("\n" + "="*50)
        print(f"开始预训练阶段（模式: {mode}）")
        if config.MULTITASK_STRATEGY == 'uncertainty':
            print("多任务策略: 不确定性加权（Kendall et al.）")
        print(f"Early Stopping: patience={config.EARLY_STOP_PATIENCE}")
        print(f"验证频率: 每{config.EVAL_EVERY_N_EPOCHS}个epoch")
        print("="*50)

    trainer = PreTrainer(model, config)

    # 改进方向1: DeepSpeed ZeRO-3 初始化
    if config.USE_DEEPSPEED:
        try:
            import deepspeed
            raw_model = model.module if hasattr(model, 'module') else model

            # 加载DeepSpeed配置
            ds_config_path = config.DS_CONFIG_PATH
            with open(ds_config_path, 'r') as f:
                ds_config = json.load(f)

            # 设置batch size相关参数
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            ds_config['train_micro_batch_size_per_gpu'] = config.BATCH_SIZE
            ds_config['train_batch_size'] = config.BATCH_SIZE * world_size
            ds_config['gradient_accumulation_steps'] = 1

            # 添加optimizer配置到ds_config，让DeepSpeed完整管理optimizer
            ds_config['optimizer'] = {
                'type': 'AdamW',
                'params': {
                    'lr': config.PRETRAIN_LR,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8,
                    'weight_decay': 0.01
                }
            }

            ds_engine, ds_optimizer, _, _ = deepspeed.initialize(
                model=raw_model,
                config=ds_config
            )

            # 用DeepSpeed engine替换model
            model = ds_engine
            trainer.model = ds_engine
            trainer.use_deepspeed = True
            trainer.ds_engine = ds_engine
            trainer.optimizer = ds_optimizer

            if is_main_process():
                print("DeepSpeed ZeRO-3 初始化成功!")

        except ImportError:
            if is_main_process():
                print("警告: DeepSpeed未安装，回退到标准训练模式")
                print("  安装方法: pip install deepspeed")
            config.USE_DEEPSPEED = False

    best_val_loss = float('inf')
    eval_every = getattr(config, 'EVAL_EVERY_N_EPOCHS', 1)

    for epoch in range(1, config.PRETRAIN_EPOCHS + 1):
        # DDP 模式下每个 epoch 需要通知 sampler，保证各卡数据打乱不重复
        if dist.is_initialized() and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n--- Epoch {epoch}/{config.PRETRAIN_EPOCHS} ---")

        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch, mode=mode)

        # 输出训练指标（只在主进程打印）
        if is_main_process():
            if mode == 'mixed1':
                print(f"Epoch {epoch} 训练指标:")
                print(f"  理解任务:")
                print(f"    - task_loss:     {train_metrics.get('task_loss', 0):.4f}")
                print(f"    - semantic_loss: {train_metrics.get('semantic_loss', 0):.4f}")
                print(f"    - bbox_loss:     {train_metrics.get('bbox_loss', 0):.4f}")
                print(f"    - relation_loss: {train_metrics.get('relation_loss', 0):.4f}")
                # 改进方向5: 新loss
                if 'hierarchical_consistency_loss' in train_metrics:
                    print(f"    - hierarchical_consistency_loss: "
                          f"{train_metrics['hierarchical_consistency_loss']:.4f}")
                print(f"  生成任务:")
                print(f"    - diffusion_loss:{train_metrics.get('diffusion_loss', 0):.4f}")
                if 'feature_matching_loss' in train_metrics:
                    print(f"    - feature_matching_loss: "
                          f"{train_metrics['feature_matching_loss']:.4f}")
                print(f"  总损失: {train_metrics.get('total_loss', 0):.4f}")
                # 改进方向3: 多任务权重
                if 'understanding_weight' in train_metrics:
                    print(f"  多任务权重:")
                    print(f"    - 理解任务权重: {train_metrics['understanding_weight']:.4f}")
                    print(f"    - 生成任务权重: {train_metrics['generation_weight']:.4f}")
            else:
                print(f"训练指标: {train_metrics}")

        # 改进方向4: 按配置频率验证（默认每个epoch验证一次）
        if epoch % eval_every == 0:
            val_metrics = trainer.validate(val_loader, mode=mode)
            if is_main_process():
                # 输出验证loss
                print(f"\nEpoch {epoch} 验证结果:")
                print(f"  Loss指标:")
                for k, v in val_metrics.items():
                    if 'loss' in k:
                        print(f"    - {k}: {v:.4f}")

                # 改进方向4: 输出评估指标
                eval_keys = ['task_accuracy', 'semantic_miou', 'bbox_iou',
                             'relation_accuracy', 'psnr', 'ssim']
                has_eval = any(k in val_metrics for k in eval_keys)
                if has_eval:
                    print(f"  评估指标:")
                    if 'task_accuracy' in val_metrics:
                        print(f"    - 任务分类准确率: {val_metrics['task_accuracy']:.4f}")
                    if 'semantic_miou' in val_metrics:
                        print(f"    - 语义分割mIoU:  {val_metrics['semantic_miou']:.4f}")
                    if 'bbox_iou' in val_metrics:
                        print(f"    - 边界框平均IoU:  {val_metrics['bbox_iou']:.4f}")
                    if 'relation_accuracy' in val_metrics:
                        print(f"    - 关系分类准确率: {val_metrics['relation_accuracy']:.4f}")
                    if 'psnr' in val_metrics:
                        print(f"    - PSNR(dB):      {val_metrics['psnr']:.2f}")
                    if 'ssim' in val_metrics:
                        print(f"    - SSIM:          {val_metrics['ssim']:.4f}")

                # 计算验证总loss用于模型保存和early stopping
                val_loss_keys = [k for k in val_metrics if 'loss' in k]
                if val_loss_keys:
                    val_loss = sum(val_metrics[k] for k in val_loss_keys) / len(val_loss_keys)
                else:
                    val_loss = val_metrics.get('total_loss',
                                               val_metrics.get('diffusion_loss', float('inf')))

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'pretrain_best.pth')
                    trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)
                    print(f"  保存最佳模型到 {checkpoint_path}")

                # 改进方向2: Early Stopping 检查
                if trainer.check_early_stopping(val_loss):
                    print(f"\n[Early Stopping] 验证损失连续{config.EARLY_STOP_PATIENCE}个"
                          f"验证周期未改善，提前终止训练。")
                    print(f"  最佳验证损失: {best_val_loss:.4f}")
                    break

        # 每10个epoch保存一次检查点（只在主进程）
        if epoch % 10 == 0 and is_main_process():
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'pretrain_epoch_{epoch}.pth'
            )
            trainer.save_checkpoint(checkpoint_path, epoch, train_metrics)
            print(f"保存检查点到 {checkpoint_path}")

    if is_main_process():
        print("\n预训练阶段完成!")
    return model


def cgpo_phase(model, config, train_loader, val_loader):
    """
    CGPO强化学习阶段
    使用奖励模型优化MLLM
    """
    if is_main_process():
        print("\n" + "="*50)
        print("开始CGPO强化学习阶段")
        print("="*50)

    # 创建奖励模型
    reward_model = RewardModel(config).to(config.DEVICE)

    # 创建CGPO训练器
    trainer = CGPOTrainer(model, reward_model, config)

    best_reward = float('-inf')

    for epoch in range(1, config.CGPO_EPOCHS + 1):
        if dist.is_initialized() and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n--- CGPO Epoch {epoch}/{config.CGPO_EPOCHS} ---")

        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch)
        if is_main_process():
            print(f"训练指标: {train_metrics}")

        # 更新参考模型（每5个epoch）
        if epoch % 5 == 0:
            if is_main_process():
                print("更新参考模型...")
            trainer.update_reference_model()

        # 保存最佳模型（只在主进程）
        if is_main_process():
            avg_reward = train_metrics.get('reward_mean', 0)
            if avg_reward > best_reward:
                best_reward = avg_reward
                checkpoint_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    'cgpo_best.pth'
                )
                trainer.save_checkpoint(checkpoint_path, epoch, train_metrics)
                print(f"保存最佳模型到 {checkpoint_path}")

            if epoch % 5 == 0:
                checkpoint_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f'cgpo_epoch_{epoch}.pth'
                )
                trainer.save_checkpoint(checkpoint_path, epoch, train_metrics)

    if is_main_process():
        print("\nCGPO阶段完成!")
    return model


def main(args):
    """主训练函数"""
    # 初始化 DDP（torchrun 启动时环境变量 LOCAL_RANK 会自动设置）
    use_ddp = 'LOCAL_RANK' in os.environ
    if use_ddp:
        local_rank = setup_ddp()
        device = f'cuda:{local_rank}'
    else:
        device = args.device

    if is_main_process():
        print("="*50)
        print("多模态大模型训练")
        print("="*50)

    # 配置
    config = Config()
    config.DEVICE = device
    config.BATCH_SIZE = args.batch_size

    # 改进方向1: DeepSpeed 选项
    if args.use_deepspeed:
        config.USE_DEEPSPEED = True
        if args.ds_config:
            config.DS_CONFIG_PATH = args.ds_config

    if is_main_process():
        print(f"\n配置:")
        print(f"  设备: {config.DEVICE}")
        print(f"  每卡批次大小: {config.BATCH_SIZE}")
        if use_ddp:
            print(f"  GPU 数量: {dist.get_world_size()}")
            print(f"  等效总批次大小: {config.BATCH_SIZE * dist.get_world_size()}")
        print(f"  训练模式: {args.mode}")
        print(f"  预训练轮数: {config.PRETRAIN_EPOCHS}")
        print(f"  CGPO轮数: {config.CGPO_EPOCHS}")
        print(f"  DeepSpeed ZeRO-3: {'启用' if config.USE_DEEPSPEED else '禁用'}")
        print(f"  多任务策略: {config.MULTITASK_STRATEGY}")
        print(f"  Early Stopping patience: {config.EARLY_STOP_PATIENCE}")

    # 创建目录（主进程），其他进程等待
    setup_directories(config)
    if use_ddp:
        dist.barrier()

    # 创建数据加载器
    if is_main_process():
        print("\n加载数据...")
    if args.mode == 'mixed1':
        if is_main_process():
            print(f"使用COCO数据集进行mixed1训练，路径: {args.coco_path}")
        train_loader, val_loader = create_coco_mixed1_dataloaders(config, args.coco_path)
    else:
        train_loader, val_loader = create_dataloaders(config)
    if is_main_process():
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")

    # 创建模型
    if is_main_process():
        print("\n创建模型...")
    model = MLLMModel(config).to(device)

    # 冻结编码器
    model.freeze_encoders()

    # DDP 包装（DeepSpeed模式下由DeepSpeed管理，不用DDP）
    # mixed1 模式已在模型层合并为单次 forward，find_unused_parameters=True
    # 兼容 understanding/generation 等单任务模式下部分参数不参与 forward 的情况
    if use_ddp and not config.USE_DEEPSPEED:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    if is_main_process():
        raw_model = model.module if (use_ddp and not config.USE_DEEPSPEED) else model
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"冻结ViT和文本编码器")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

    # 阶段1: 预训练
    if args.skip_pretrain:
        if is_main_process():
            print("\n跳过预训练阶段，加载预训练模型...")
        pretrain_checkpoint = os.path.join(config.CHECKPOINT_DIR, 'pretrain_best.pth')
        if os.path.exists(pretrain_checkpoint):
            # map_location 确保各卡加载到正确设备
            checkpoint = torch.load(pretrain_checkpoint, map_location=device)
            raw_model = model.module if hasattr(model, 'module') else model
            raw_model.load_state_dict(checkpoint['model_state_dict'])
            if is_main_process():
                print("预训练模型加载成功!")
        else:
            if is_main_process():
                print("警告: 未找到预训练模型，将从头开始训练...")
            model = pretrain_phase(model, config, train_loader, val_loader, mode=args.mode)
    else:
        model = pretrain_phase(model, config, train_loader, val_loader, mode=args.mode)

    # 阶段2: CGPO强化学习
    if not args.skip_cgpo:
        model = cgpo_phase(model, config, train_loader, val_loader)

    # 保存最终模型（只在主进程）
    if is_main_process():
        print("\n" + "="*50)
        print("训练完成!")
        print("="*50)
        final_checkpoint = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
        raw_model = model.module if hasattr(model, 'module') else model
        torch.save({
            'model_state_dict': raw_model.state_dict(),
            'config': config
        }, final_checkpoint)
        print(f"\n最终模型保存到: {final_checkpoint}")

    if use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练多模态大模型')

    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备 (cuda/cpu)，多卡时由 torchrun 自动管理，无需手动指定')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='每张卡的批次大小')
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
    # 改进方向1: DeepSpeed 参数
    parser.add_argument('--use-deepspeed', action='store_true',
                       help='启用DeepSpeed ZeRO-3内存优化（需要安装deepspeed）')
    parser.add_argument('--ds-config', type=str, default=None,
                       help='DeepSpeed配置文件路径（默认使用ds_config.json）')

    args = parser.parse_args()

    # 单卡模式下检查 CUDA 可用性；多卡时由 torchrun 管理设备
    if 'LOCAL_RANK' not in os.environ:
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，使用CPU训练")
            args.device = 'cpu'

    main(args)
