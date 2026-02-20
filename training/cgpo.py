"""
CGPO (Constrained Generative Policy Optimization) 训练模块
使用强化学习优化MLLM和扩散模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

class CGPOTrainer:
    """
    CGPO训练器
    - 迭代优化MLLM
    - 将MLLM作为扩散模块的约束条件
    """
    def __init__(self, model, reward_model, config):
        self.model = model
        self.reward_model = reward_model
        self.config = config

        # 优化器
        self.optimizer = AdamW([
            {'params': model.lot_module.parameters(), 'lr': config.CGPO_LR},
            {'params': model.integration_module.parameters(), 'lr': config.CGPO_LR},
            {'params': model.diffusion_module.parameters(), 'lr': config.CGPO_LR * 0.5}
        ])

        # KL散度权重
        self.beta = config.CGPO_BETA

        # 保存参考模型（用于KL散度约束）
        self.reference_model = self._create_reference_model()

    def _create_reference_model(self):
        """创建参考模型的副本"""
        import copy
        ref_model = copy.deepcopy(self.model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    def compute_kl_divergence(self, current_outputs, reference_outputs):
        """
        计算当前模型和参考模型之间的KL散度
        防止模型偏离太远
        """
        kl_loss = 0.0
        count = 0

        # 对LoT各层的输出计算KL散度
        for layer in ['low', 'mid', 'high']:
            if layer in current_outputs['lot_outputs'] and layer in reference_outputs['lot_outputs']:
                # 使用场景特征或任务特征
                if layer == 'low':
                    curr_feat = current_outputs['lot_outputs'][layer]['task_features']
                    ref_feat = reference_outputs['lot_outputs'][layer]['task_features']
                elif layer == 'mid':
                    curr_feat = current_outputs['lot_outputs'][layer]['semantic_features'].mean(dim=1)
                    ref_feat = reference_outputs['lot_outputs'][layer]['semantic_features'].mean(dim=1)
                else:  # high
                    curr_feat = current_outputs['lot_outputs'][layer]['scene_features']
                    ref_feat = reference_outputs['lot_outputs'][layer]['scene_features']

                # L2距离作为KL的近似
                kl = F.mse_loss(curr_feat, ref_feat)
                kl_loss += kl
                count += 1

        return kl_loss / max(count, 1)

    def train_step(self, batch):
        """
        单步CGPO训练
        Args:
            batch: dict containing:
                - images: (B, C, H, W)
                - text_tokens: (B, seq_len)
                - text_mask: (B, seq_len)
                - target_images: (B, C, H, W)
        Returns:
            metrics: dict 训练指标
        """
        self.model.train()
        self.optimizer.zero_grad()

        images = batch['images']
        text_tokens = batch['text_tokens']
        text_mask = batch['text_mask']
        target_images = batch['target_images']

        device = images.device

        # 1. 当前模型前向传播（生成模式）
        current_outputs = self.model(
            images, text_tokens, text_mask,
            target_images=target_images,
            mode='generation'
        )

        # 2. 参考模型前向传播（用于KL约束）
        with torch.no_grad():
            reference_outputs = self.reference_model(
                images, text_tokens, text_mask,
                mode='generation'
            )

        # 3. 生成图像
        with torch.no_grad():
            generated_images = self.model.diffusion_module.sample(
                current_outputs['generation_condition'],
                image_size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
            )

        # 4. 计算奖励
        vision_features, vision_cls = self.model.vit_encoder(images)

        rewards, reward_details = self.reward_model(
            current_outputs['lot_outputs'],
            vision_cls,
            generated_images,
            vision_cls  # 使用原始图像特征作为条件
        )

        # 5. 计算KL散度（约束项）
        kl_divergence = self.compute_kl_divergence(current_outputs, reference_outputs)

        # 6. CGPO目标函数
        # 最大化奖励，同时最小化与参考模型的KL散度
        policy_loss = -rewards.mean()  # 负号因为要最大化奖励
        kl_loss = self.beta * kl_divergence

        # 7. 扩散损失（重构目标图像）
        diffusion_loss = current_outputs['diffusion_loss']

        # 8. 总损失
        # 结合策略优化和生成质量
        total_loss = policy_loss + kl_loss + diffusion_loss

        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # 返回指标
        metrics = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'understanding_reward': reward_details['understanding_reward'].mean().item(),
            'generation_reward': reward_details['generation_reward'].mean().item()
        }

        # 添加详细的奖励分数
        for category in ['understanding', 'generation']:
            for aspect, score in reward_details[category].items():
                metrics[f'{category}_{aspect}'] = score.mean().item()

        return metrics

    def train_epoch(self, dataloader, epoch):
        """
        训练一个epoch
        """
        self.model.train()

        total_metrics = {}
        progress_bar = tqdm(dataloader, desc=f"CGPO Epoch {epoch}")

        for batch in progress_bar:
            # 将数据移到设备
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(self.config.DEVICE)
                elif isinstance(v, dict):
                    batch_device[k] = {
                        sub_k: sub_v.to(self.config.DEVICE) if isinstance(sub_v, torch.Tensor) else sub_v
                        for sub_k, sub_v in v.items()
                    }
                else:
                    batch_device[k] = v

            # 训练步骤
            metrics = self.train_step(batch_device)

            # 累积指标
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = []
                total_metrics[key].append(value)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': metrics['total_loss'],
                'reward': metrics['reward_mean']
            })

        # 计算平均指标
        avg_metrics = {key: sum(values) / len(values)
                      for key, values in total_metrics.items()}

        return avg_metrics

    def update_reference_model(self):
        """
        更新参考模型（定期执行，如每N个epoch）
        """
        import copy
        self.reference_model = copy.deepcopy(self.model)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def save_checkpoint(self, path, epoch, metrics):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
