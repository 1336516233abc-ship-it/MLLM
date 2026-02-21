"""
预训练模块 - 训练LoT、整合模块和扩散模块
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

class PreTrainer:
    """
    预训练器
    目标：训练LoT分层推理、整合模块、扩散模块
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # 只优化需要训练的模块
        self.optimizer = AdamW([
            {'params': model.lot_module.parameters(), 'lr': config.PRETRAIN_LR},
            {'params': model.integration_module.parameters(), 'lr': config.PRETRAIN_LR},
            {'params': model.diffusion_module.parameters(), 'lr': config.PRETRAIN_LR},
            {'params': model.understanding_head.parameters(), 'lr': config.PRETRAIN_LR}
        ])

    def compute_understanding_losses(self, outputs, targets):
        """
        计算理解任务的损失
        """
        losses = {}

        # 1. 任务分类损失
        if 'task_labels' in targets:
            task_loss = nn.functional.cross_entropy(
                outputs['task_logits'],
                targets['task_labels']
            )
            losses['task_loss'] = task_loss

        # 2. 语义分割损失
        if 'semantic_labels' in targets:
            B, num_patches, num_classes = outputs['semantic_logits'].shape
            semantic_logits = outputs['semantic_logits'].reshape(-1, num_classes)
            semantic_labels = targets['semantic_labels'].reshape(-1)

            semantic_loss = nn.functional.cross_entropy(
                semantic_logits,
                semantic_labels,
                ignore_index=-1
            )
            losses['semantic_loss'] = semantic_loss

        # 3. 边界框回归损失
        if 'bboxes' in targets:
            num_target_boxes = targets['bboxes'].shape[1]  # 100
            pred_bboxes = outputs['bboxes'][:, :num_target_boxes, :]  # (B, 100, 4)
            bbox_loss = nn.functional.l1_loss(pred_bboxes, targets['bboxes'])
            losses['bbox_loss'] = bbox_loss

        # 4. 关系矩阵损失
        if 'relation_matrix' in targets:
            num_target_nodes = targets['relation_matrix'].shape[1]  # 100
            pred_relations = outputs['relation_matrix'][:, :num_target_nodes, :num_target_nodes, :]  # (B, 100, 100, 10)
            relation_loss = nn.functional.cross_entropy(
                pred_relations.reshape(-1, 10),
                targets['relation_matrix'].reshape(-1)
            )
            losses['relation_loss'] = relation_loss

        return losses

    def compute_editing_losses(self, outputs, targets, source_images, edited_images):
        """
        计算编辑任务的损失

        编辑任务是有参考图像的特殊图像生成模式
        包括：
        1. 扩散重构损失：生成的图像应该匹配目标编辑图像
        2. 区域保持损失：未编辑区域应该保持不变
        3. 编辑一致性损失：编辑应该符合文本指令
        """
        losses = {}

        # 1. 扩散重构损失（主要损失）
        if 'diffusion_loss' in outputs:
            losses['editing_diffusion_loss'] = outputs['diffusion_loss']

        # 2. 区域保持损失
        # 如果提供了编辑掩码，未编辑区域应该保持不变
        if 'edit_mask' in targets:
            # edit_mask: (B, 1, H, W), 1表示编辑区域，0表示保持区域
            edit_mask = targets['edit_mask']

            # 生成图像（需要采样）
            with torch.no_grad():
                generated_images = self.model.diffusion_module.sample(
                    outputs['generation_condition'],
                    image_size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
                )

            # 保持区域的损失（未编辑区域应该与源图像相同）
            preserve_mask = 1 - edit_mask
            preserve_loss = nn.functional.l1_loss(
                generated_images * preserve_mask,
                source_images * preserve_mask
            )
            losses['preserve_loss'] = preserve_loss * 0.5  # 权重0.5

            # 编辑区域的损失
            edit_loss = nn.functional.l1_loss(
                generated_images * edit_mask,
                edited_images * edit_mask
            )
            losses['edit_region_loss'] = edit_loss * 1.0  # 权重1.0

            # 感知损失（使用ViT特征）
            _, gen_features = self.model.vit_encoder(generated_images)
            _, target_features = self.model.vit_encoder(edited_images)
            perceptual_loss = nn.functional.mse_loss(gen_features, target_features)
            losses['perceptual_loss'] = perceptual_loss * 0.1  # 权重0.1

        return losses

    def train_step(self, batch, mode='mixed'):
        """
        单步训练
        Args:
            batch: dict containing training data
            mode: 训练模式
                - 'understanding': 只训练理解任务
                - 'generation': 只训练生成任务（无参考图像）
                - 'editing': 只训练编辑任务（有参考图像的生成）
                - 'mixed1': 理解+生成混合（同一批数据同时训练）
                - 'mixed2': 理解+编辑混合（同一批数据同时训练）
                - 'mixed': 理解+生成+编辑全部混合
        Returns:
            metrics: dict 训练指标
        """
        self.model.train()
        self.optimizer.zero_grad()

        images = batch['images']
        text_tokens = batch['text_tokens']
        text_mask = batch['text_mask']

        device = images.device
        total_loss = 0.0
        metrics = {}

        # 理解任务训练
        # 在 'understanding', 'mixed1', 'mixed2', 'mixed' 模式下训练
        if mode in ['understanding', 'mixed1', 'mixed2', 'mixed']:
            understanding_outputs = self.model(
                images, text_tokens, text_mask,
                mode='understanding'
            )

            # 计算理解损失
            if 'targets' in batch:
                understanding_losses = self.compute_understanding_losses(
                    understanding_outputs,
                    batch['targets']
                )

                for loss_name, loss_value in understanding_losses.items():
                    total_loss += loss_value
                    metrics[loss_name] = loss_value.item()

        # 生成任务训练（无参考图像的纯生成）
        # 在 'generation', 'mixed1', 'mixed' 模式下训练
        if mode in ['generation', 'mixed1', 'mixed'] and 'target_images' in batch:
            target_images = batch['target_images']

            generation_outputs = self.model(
                images, text_tokens, text_mask,
                target_images=target_images,
                mode='generation'
            )

            # 扩散损失
            diffusion_loss = generation_outputs['diffusion_loss']
            total_loss += diffusion_loss
            metrics['diffusion_loss'] = diffusion_loss.item()

        # 编辑任务训练（有参考图像的条件生成）
        # 在 'editing', 'mixed2', 'mixed' 模式下训练
        if mode in ['editing', 'mixed2', 'mixed'] and 'edited_images' in batch:
            source_images = batch['images']  # 源图像（参考图像）
            edited_images = batch['edited_images']  # 编辑后的目标图像
            edit_instruction = batch['text_tokens']  # 编辑指令

            # 前向传播（编辑模式 = 条件图像生成）
            editing_outputs = self.model(
                source_images, edit_instruction, text_mask,
                target_images=edited_images,
                mode='editing'
            )

            # 计算编辑损失
            editing_losses = self.compute_editing_losses(
                editing_outputs,
                batch.get('targets', {}),
                source_images,
                edited_images
            )

            for loss_name, loss_value in editing_losses.items():
                total_loss += loss_value
                metrics[loss_name] = loss_value.item()

        # 反向传播
        if total_loss > 0:
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            metrics['total_loss'] = total_loss.item()

        return metrics

    def train_epoch(self, dataloader, epoch, mode='mixed'):
        """
        训练一个epoch

        Args:
            dataloader: 数据加载器
            epoch: 当前epoch
            mode: 训练模式
                - 'understanding': 只训练理解
                - 'generation': 只训练生成（无参考）
                - 'editing': 只训练编辑（有参考）
                - 'mixed1': 理解+生成（同一批数据）
                - 'mixed2': 理解+编辑（同一批数据）
                - 'mixed': 全部混合
        """
        self.model.train()

        total_metrics = {}

        # 进度条描述
        mode_desc = {
            'understanding': '理解',
            'generation': '生成',
            'editing': '编辑',
            'mixed1': '理解+生成',
            'mixed2': '理解+编辑',
            'mixed': '全混合'
        }
        desc = f"PreTrain [{mode_desc.get(mode, mode)}] Epoch {epoch}"
        progress_bar = tqdm(dataloader, desc=desc)

        for batch in progress_bar:
            # 将数据移到设备（处理嵌套字典）
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
            metrics = self.train_step(batch_device, mode=mode)

            # 累积指标
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = []
                total_metrics[key].append(value)

            # 更新进度条
            if 'total_loss' in metrics:
                progress_bar.set_postfix({'loss': f"{metrics['total_loss']:.4f}"})

        # 计算平均指标
        avg_metrics = {key: sum(values) / len(values)
                      for key, values in total_metrics.items()}

        return avg_metrics

    def validate(self, dataloader, mode='mixed'):
        """
        验证模型

        Args:
            dataloader: 验证数据加载器
            mode: 验证模式（与训练模式相同）
                - 'understanding': 只验证理解
                - 'generation': 只验证生成
                - 'editing': 只验证编辑
                - 'mixed1': 理解+生成（同一批数据）
                - 'mixed2': 理解+编辑（同一批数据）
                - 'mixed': 全部混合
        """
        self.model.eval()

        total_metrics = {}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # 将数据移到设备（处理嵌套字典）
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

                images = batch_device['images']
                text_tokens = batch_device['text_tokens']
                text_mask = batch_device['text_mask']

                # 理解任务验证
                # 在 'understanding', 'mixed1', 'mixed2', 'mixed' 模式下验证
                if mode in ['understanding', 'mixed1', 'mixed2', 'mixed']:
                    understanding_outputs = self.model(
                        images, text_tokens, text_mask,
                        mode='understanding'
                    )

                    if 'targets' in batch_device:
                        understanding_losses = self.compute_understanding_losses(
                            understanding_outputs,
                            batch_device['targets']
                        )

                        for loss_name, loss_value in understanding_losses.items():
                            if loss_name not in total_metrics:
                                total_metrics[loss_name] = []
                            total_metrics[loss_name].append(loss_value.item())

                # 生成任务验证
                # 在 'generation', 'mixed1', 'mixed' 模式下验证
                if mode in ['generation', 'mixed1', 'mixed'] and 'target_images' in batch_device:
                    target_images = batch_device['target_images']

                    generation_outputs = self.model(
                        images, text_tokens, text_mask,
                        target_images=target_images,
                        mode='generation'
                    )

                    diffusion_loss = generation_outputs['diffusion_loss']

                    if 'diffusion_loss' not in total_metrics:
                        total_metrics['diffusion_loss'] = []
                    total_metrics['diffusion_loss'].append(diffusion_loss.item())

                # 编辑任务验证
                # 在 'editing', 'mixed2', 'mixed' 模式下验证
                if mode in ['editing', 'mixed2', 'mixed'] and 'edited_images' in batch_device:
                    source_images = batch_device['images']
                    edited_images = batch_device['edited_images']

                    editing_outputs = self.model(
                        source_images, text_tokens, text_mask,
                        target_images=edited_images,
                        mode='editing'
                    )

                    editing_losses = self.compute_editing_losses(
                        editing_outputs,
                        batch_device.get('targets', {}),
                        source_images,
                        edited_images
                    )

                    for loss_name, loss_value in editing_losses.items():
                        if loss_name not in total_metrics:
                            total_metrics[loss_name] = []
                        total_metrics[loss_name].append(loss_value.item())

        # 计算平均指标
        avg_metrics = {key: sum(values) / len(values)
                      for key, values in total_metrics.items()}

        return avg_metrics

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
