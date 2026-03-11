"""
预训练模块 - 训练LoT、整合模块和扩散模块
改进：
1. DeepSpeed ZeRO-3 优化器支持
2. Early Stopping 机制
3. 基于不确定性的多任务学习策略（Kendall et al., CVPR 2018）
4. 完善的评估指标（准确率、mIoU、bbox IoU、PSNR、SSIM等）
5. 新增loss：分层一致性loss、特征匹配loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import math


# ============================================================
# 改进方向2: Early Stopping
# ============================================================
class EarlyStopping:
    """
    Early Stopping 机制
    当验证指标连续 patience 个周期没有改善时，提前终止训练。
    用于解决 task_loss/bbox_loss 过早收敛、后续 epoch loss 反弹的问题。
    """
    def __init__(self, patience=5, min_delta=1e-4):
        """
        Args:
            patience: 允许验证指标不改善的连续epoch数
            min_delta: 最小改善幅度，小于此值视为无改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_loss):
        """
        Args:
            val_loss: 当前epoch的验证损失
        Returns:
            bool: 是否应该停止训练
        """
        score = -val_loss  # 损失越低越好，取负值用于比较

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop


# ============================================================
# 改进方向4: 评估指标
# ============================================================
class MetricsComputer:
    """
    评估指标计算器
    参考COCO数据集上图像理解和生成任务的标准benchmark指标：
    - 理解任务：任务分类准确率、语义分割mIoU、边界框IoU、关系分类准确率
    - 生成任务：PSNR、SSIM
    """

    @staticmethod
    def task_accuracy(pred_logits, target_labels):
        """
        任务分类准确率
        Args:
            pred_logits: (B, num_classes) 预测logits
            target_labels: (B,) 真实标签
        Returns:
            accuracy: float
        """
        pred = pred_logits.argmax(dim=-1)
        correct = (pred == target_labels).float().sum()
        total = target_labels.numel()
        return (correct / total).item() if total > 0 else 0.0

    @staticmethod
    def semantic_miou(pred_logits, target_labels, num_classes=100, ignore_index=-1):
        """
        语义分割 Mean Intersection over Union (mIoU)
        参考COCO-Stuff语义分割评估标准
        Args:
            pred_logits: (B, num_patches, num_classes)
            target_labels: (B, num_patches) 真实标签
            num_classes: 类别数
            ignore_index: 忽略的标签值
        Returns:
            miou: float
        """
        pred = pred_logits.argmax(dim=-1).reshape(-1)   # (B*num_patches,)
        target = target_labels.reshape(-1)               # (B*num_patches,)

        # 过滤ignore_index
        valid_mask = target != ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]

        if pred.numel() == 0:
            return 0.0

        iou_per_class = []
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            intersection = (pred_cls & target_cls).float().sum()
            union = (pred_cls | target_cls).float().sum()
            if union > 0:
                iou_per_class.append((intersection / union).item())

        return sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0.0

    @staticmethod
    def bbox_mean_iou(pred_bboxes, target_bboxes):
        """
        边界框平均IoU
        参考COCO目标检测评估中的IoU计算
        Args:
            pred_bboxes: (B, N, 4) 预测框 [x, y, w, h] 归一化坐标
            target_bboxes: (B, N, 4) 真实框
        Returns:
            mean_iou: float
        """
        # 将 (x, y, w, h) 转换为 (x1, y1, x2, y2)
        pred_x1 = pred_bboxes[..., 0] - pred_bboxes[..., 2] / 2
        pred_y1 = pred_bboxes[..., 1] - pred_bboxes[..., 3] / 2
        pred_x2 = pred_bboxes[..., 0] + pred_bboxes[..., 2] / 2
        pred_y2 = pred_bboxes[..., 1] + pred_bboxes[..., 3] / 2

        target_x1 = target_bboxes[..., 0] - target_bboxes[..., 2] / 2
        target_y1 = target_bboxes[..., 1] - target_bboxes[..., 3] / 2
        target_x2 = target_bboxes[..., 0] + target_bboxes[..., 2] / 2
        target_y2 = target_bboxes[..., 1] + target_bboxes[..., 3] / 2

        # 计算交集
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        # 计算并集
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + 1e-6)
        return iou.mean().item()

    @staticmethod
    def relation_accuracy(pred_logits, target_labels):
        """
        关系分类准确率
        Args:
            pred_logits: (B, N, N, num_relations) 预测关系logits
            target_labels: (B, N, N) 真实关系标签
        Returns:
            accuracy: float
        """
        pred = pred_logits.argmax(dim=-1).reshape(-1)
        target = target_labels.reshape(-1)
        correct = (pred == target).float().sum()
        total = target.numel()
        return (correct / total).item() if total > 0 else 0.0

    @staticmethod
    def psnr(pred_images, target_images, max_val=2.0):
        """
        Peak Signal-to-Noise Ratio (PSNR)
        参考图像生成质量评估标准
        图像范围假设为[-1, 1]，所以 max_val=2.0
        Args:
            pred_images: (B, C, H, W)
            target_images: (B, C, H, W)
            max_val: 像素值范围
        Returns:
            psnr: float (dB)
        """
        mse = F.mse_loss(pred_images, target_images)
        if mse == 0:
            return float('inf')
        return (10 * torch.log10(max_val ** 2 / mse)).item()

    @staticmethod
    def ssim(pred_images, target_images, window_size=11):
        """
        Structural Similarity Index (SSIM)
        参考图像质量评估标准（Wang et al., 2004）
        Args:
            pred_images: (B, C, H, W)
            target_images: (B, C, H, W)
            window_size: 高斯窗口大小
        Returns:
            ssim: float
        """
        C1 = (0.01 * 2.0) ** 2  # max_val = 2.0 for [-1, 1] range
        C2 = (0.03 * 2.0) ** 2

        # 创建高斯窗口
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)  # (window_size, window_size)
        window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)

        channels = pred_images.shape[1]
        window = window.expand(channels, -1, -1, -1).to(pred_images.device)

        pad = window_size // 2

        mu1 = F.conv2d(pred_images, window, padding=pad, groups=channels)
        mu2 = F.conv2d(target_images, window, padding=pad, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred_images ** 2, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target_images ** 2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred_images * target_images, window, padding=pad, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()


# ============================================================
# 改进方向3 & 5: 多任务学习策略 + 新loss
# ============================================================
class MultiTaskLossBalancer(nn.Module):
    """
    基于不确定性的多任务学习损失平衡器（Kendall et al., CVPR 2018）

    原理：
    每个任务有一个可学习的同方差不确定性参数 log(σ²)，
    损失公式：L_total = Σ (1/(2*σ²_i)) * L_i + (1/2) * log(σ²_i)
    等价于：L_total = Σ (1/2) * exp(-s_i) * L_i + (1/2) * s_i
    其中 s_i = log(σ²_i) 为可学习参数。

    当某个任务的loss较大时，对应的σ²会增大，自动降低该任务的权重，
    避免任务间的梯度冲突。
    """
    def __init__(self, num_tasks):
        super().__init__()
        # 初始化 log(σ²) = 0，即σ² = 1，初始权重相等
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        Args:
            losses: list of scalar tensors, 各任务的loss
        Returns:
            total_loss: 加权后的总loss
            weights: 各任务的有效权重（用于日志）
        """
        total_loss = 0.0
        weights = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])  # 1/σ²
            total_loss += 0.5 * precision * loss + 0.5 * self.log_vars[i]
            weights.append(precision.item())
        return total_loss, weights


class PreTrainer:
    """
    预训练器
    目标：训练LoT分层推理、整合模块、扩散模块
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.metrics_computer = MetricsComputer()

        # 兼容 DDP：DDP 将原始模型藏在 .module 下，需通过 raw_model 访问子模块
        raw_model = model.module if hasattr(model, 'module') else model

        # 改进方向3: 多任务损失平衡器
        # 任务0: 理解任务总loss, 任务1: 生成任务总loss
        self.multitask_strategy = getattr(config, 'MULTITASK_STRATEGY', 'fixed')
        if self.multitask_strategy == 'uncertainty':
            self.loss_balancer = MultiTaskLossBalancer(num_tasks=2).to(config.DEVICE)
            balancer_params = list(self.loss_balancer.parameters())
        else:
            self.loss_balancer = None
            balancer_params = []

        # 改进方向2: Early Stopping
        self.early_stopping = EarlyStopping(
            patience=getattr(config, 'EARLY_STOP_PATIENCE', 5),
            min_delta=getattr(config, 'EARLY_STOP_MIN_DELTA', 1e-4)
        )

        # 新loss权重
        self.hierarchical_consistency_weight = getattr(config, 'HIERARCHICAL_CONSISTENCY_WEIGHT', 0.1)
        self.feature_matching_weight = getattr(config, 'FEATURE_MATCHING_WEIGHT', 0.05)

        # 只优化需要训练的模块
        # DeepSpeed模式下，优化器由DeepSpeed管理，此处不创建
        self.use_deepspeed = getattr(config, 'USE_DEEPSPEED', False)
        if not self.use_deepspeed:
            self.optimizer = AdamW([
                {'params': raw_model.lot_module.parameters(), 'lr': config.PRETRAIN_LR},
                {'params': raw_model.integration_module.parameters(), 'lr': config.PRETRAIN_LR},
                {'params': raw_model.diffusion_module.parameters(), 'lr': config.PRETRAIN_LR},
                {'params': raw_model.understanding_head.parameters(), 'lr': config.PRETRAIN_LR},
            ] + ([{'params': balancer_params, 'lr': config.PRETRAIN_LR}] if balancer_params else []))
        else:
            self.optimizer = None  # DeepSpeed自行管理

    def set_deepspeed_engine(self, engine, optimizer):
        """DeepSpeed模式下，由外部设置engine和optimizer"""
        self.ds_engine = engine
        self.optimizer = optimizer

    def compute_understanding_losses(self, outputs, targets):
        """
        计算理解任务的损失
        """
        losses = {}

        # 1. 任务分类损失
        if 'task_labels' in targets:
            task_loss = F.cross_entropy(
                outputs['task_logits'],
                targets['task_labels']
            )
            losses['task_loss'] = task_loss

        # 2. 语义分割损失
        if 'semantic_labels' in targets:
            B, num_patches, num_classes = outputs['semantic_logits'].shape
            semantic_logits = outputs['semantic_logits'].reshape(-1, num_classes)
            semantic_labels = targets['semantic_labels'].reshape(-1)

            semantic_loss = F.cross_entropy(
                semantic_logits,
                semantic_labels,
                ignore_index=-1
            )
            losses['semantic_loss'] = semantic_loss

        # 3. 边界框回归损失
        if 'bboxes' in targets:
            num_target_boxes = targets['bboxes'].shape[1]  # 100
            pred_bboxes = outputs['bboxes'][:, :num_target_boxes, :]  # (B, 100, 4)
            bbox_loss = F.l1_loss(pred_bboxes, targets['bboxes'])
            losses['bbox_loss'] = bbox_loss

        # 4. 关系矩阵损失
        if 'relation_matrix' in targets:
            num_target_nodes = targets['relation_matrix'].shape[1]  # 100
            pred_relations = outputs['relation_matrix'][:, :num_target_nodes, :num_target_nodes, :]  # (B, 100, 100, 10)
            relation_loss = F.cross_entropy(
                pred_relations.reshape(-1, 10),
                targets['relation_matrix'].reshape(-1)
            )
            losses['relation_loss'] = relation_loss

        return losses

    # ============================================================
    # 改进方向5: 新loss组件
    # ============================================================
    def compute_hierarchical_consistency_loss(self, lot_outputs):
        """
        分层一致性损失（改进方向5）

        确保LoT模块不同层级的特征表示保持一致性，
        强制低层->中层->高层的特征在语义空间中对齐。
        原理：相邻层级的特征经过投影后应在一定程度上保持一致，
        防止不同层级的推理结果出现矛盾。

        具体实现：
        - 低层element_features投影到中层维度后，与中层semantic_features计算余弦相似性
        - 中层semantic_features投影到高层维度后，与高层features计算余弦相似性
        """
        low_feat = lot_outputs['low']['element_features']    # (B, N, LOT_LOW_DIM)
        mid_feat = lot_outputs['mid']['semantic_features']    # (B, N, LOT_MID_DIM)
        high_feat = lot_outputs['high']['semantic_relation_features']  # (B, N, LOT_HIGH_DIM)

        # 对齐维度：将低层特征插值到中层维度进行比较
        # 使用均值池化后的全局表示来计算一致性
        low_global = low_feat.mean(dim=1)    # (B, LOT_LOW_DIM)
        mid_global = mid_feat.mean(dim=1)    # (B, LOT_MID_DIM)
        high_global = high_feat.mean(dim=1)  # (B, LOT_HIGH_DIM)

        # 使用L2归一化后的余弦相似性损失
        # 目标：相邻层的全局表示方向应该一致（取前min_dim维）
        min_dim_lm = min(low_global.shape[-1], mid_global.shape[-1])
        low_norm = F.normalize(low_global[:, :min_dim_lm], dim=-1)
        mid_norm = F.normalize(mid_global[:, :min_dim_lm], dim=-1)
        loss_low_mid = 1.0 - (low_norm * mid_norm).sum(dim=-1).mean()

        min_dim_mh = min(mid_global.shape[-1], high_global.shape[-1])
        mid_norm2 = F.normalize(mid_global[:, :min_dim_mh], dim=-1)
        high_norm = F.normalize(high_global[:, :min_dim_mh], dim=-1)
        loss_mid_high = 1.0 - (mid_norm2 * high_norm).sum(dim=-1).mean()

        return (loss_low_mid + loss_mid_high) / 2.0

    def compute_feature_matching_loss(self, outputs, target_images):
        """
        特征匹配损失（改进方向5）

        使用ViT编码器的中间特征计算感知级别的生成质量损失。
        原理：直接的像素级MSE loss（diffusion loss）不足以捕捉高层语义信息，
        特征匹配损失通过比较生成图像和目标图像在预训练ViT特征空间中的距离，
        使生成结果在语义层面更接近目标。

        注意：此loss仅在训练时通过条件特征间接优化生成质量，
        不需要真正采样生成图像（采样开销太大）。
        改为比较条件特征与目标图像编码特征的距离。
        """
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        # 获取目标图像的ViT特征
        with torch.no_grad():
            _, target_cls = raw_model.vit_encoder(target_images)  # (B, VIT_DIM)

        # 获取生成条件特征（已经通过整合模块产生）
        gen_condition = outputs['generation_condition']  # (B, DIFFUSION_DIM)

        # 将目标特征投影到相同维度进行比较
        target_proj = target_cls[:, :gen_condition.shape[-1]]  # 截取到相同维度
        if target_proj.shape[-1] < gen_condition.shape[-1]:
            # 如果VIT_DIM < DIFFUSION_DIM，用零填充
            pad = torch.zeros(
                target_proj.shape[0],
                gen_condition.shape[-1] - target_proj.shape[-1],
                device=target_proj.device
            )
            target_proj = torch.cat([target_proj, pad], dim=-1)

        # 归一化后计算MSE
        gen_norm = F.normalize(gen_condition, dim=-1)
        target_norm = F.normalize(target_proj, dim=-1)

        return F.mse_loss(gen_norm, target_norm)

    def compute_editing_losses(self, outputs, targets, source_images, edited_images):
        """
        计算编辑任务的损失
        """
        losses = {}

        # 1. 扩散重构损失（主要损失）
        if 'diffusion_loss' in outputs:
            losses['editing_diffusion_loss'] = outputs['diffusion_loss']

        # 2. 区域保持损失
        if 'edit_mask' in targets:
            edit_mask = targets['edit_mask']

            with torch.no_grad():
                raw_model = self.model.module if hasattr(self.model, 'module') else self.model
                generated_images = raw_model.diffusion_module.sample(
                    outputs['generation_condition'],
                    image_size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
                )

            preserve_mask = 1 - edit_mask
            preserve_loss = F.l1_loss(
                generated_images * preserve_mask,
                source_images * preserve_mask
            )
            losses['preserve_loss'] = preserve_loss * 0.5

            edit_loss = F.l1_loss(
                generated_images * edit_mask,
                edited_images * edit_mask
            )
            losses['edit_region_loss'] = edit_loss * 1.0

            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            _, gen_features = raw_model.vit_encoder(generated_images)
            _, target_features = raw_model.vit_encoder(edited_images)
            perceptual_loss = F.mse_loss(gen_features, target_features)
            losses['perceptual_loss'] = perceptual_loss * 0.1

        return losses

    def compute_eval_metrics(self, outputs, batch, mode='mixed1'):
        """
        计算评估指标（改进方向4）

        理解任务指标（参考COCO benchmark）：
        - task_accuracy: 任务分类准确率
        - semantic_miou: 语义分割mIoU（参考COCO-Stuff评估）
        - bbox_iou: 边界框平均IoU（参考COCO检测评估）
        - relation_accuracy: 关系分类准确率

        生成任务指标：
        - psnr: 峰值信噪比
        - ssim: 结构相似性指数
        """
        metrics = {}
        targets = batch.get('targets', {})

        # 理解任务指标
        if mode in ['understanding', 'mixed1', 'mixed2', 'mixed']:
            if 'task_labels' in targets and 'task_logits' in outputs:
                metrics['task_accuracy'] = self.metrics_computer.task_accuracy(
                    outputs['task_logits'], targets['task_labels']
                )

            if 'semantic_labels' in targets and 'semantic_logits' in outputs:
                metrics['semantic_miou'] = self.metrics_computer.semantic_miou(
                    outputs['semantic_logits'], targets['semantic_labels']
                )

            if 'bboxes' in targets and 'bboxes' in outputs:
                num_target_boxes = targets['bboxes'].shape[1]
                pred_bboxes = outputs['bboxes'][:, :num_target_boxes, :]
                metrics['bbox_iou'] = self.metrics_computer.bbox_mean_iou(
                    pred_bboxes, targets['bboxes']
                )

            if 'relation_matrix' in targets and 'relation_matrix' in outputs:
                num_target_nodes = targets['relation_matrix'].shape[1]
                pred_relations = outputs['relation_matrix'][:, :num_target_nodes, :num_target_nodes, :]
                metrics['relation_accuracy'] = self.metrics_computer.relation_accuracy(
                    pred_relations, targets['relation_matrix']
                )

        # 生成任务指标（需要目标图像）
        if mode in ['generation', 'mixed1', 'mixed'] and 'target_images' in batch:
            # 使用扩散模型采样生成图像计算指标（仅验证时）
            if not self.model.training and 'generation_condition' in outputs:
                raw_model = self.model.module if hasattr(self.model, 'module') else self.model
                with torch.no_grad():
                    generated = raw_model.diffusion_module.sample(
                        outputs['generation_condition'],
                        image_size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
                    )
                target_images = batch['target_images']
                metrics['psnr'] = self.metrics_computer.psnr(generated, target_images)
                metrics['ssim'] = self.metrics_computer.ssim(generated, target_images)

        return metrics

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
        if self.loss_balancer is not None:
            self.loss_balancer.train()
        self.optimizer.zero_grad()

        images = batch['images']
        text_tokens = batch['text_tokens']
        text_mask = batch['text_mask']

        total_loss = 0.0
        metrics = {}

        # 用于多任务平衡的分组loss
        understanding_loss_total = torch.tensor(0.0, device=images.device)
        generation_loss_total = torch.tensor(0.0, device=images.device)
        has_understanding = False
        has_generation = False

        # 理解任务训练
        # 在 'understanding', 'mixed2', 'mixed' 模式下训练（mixed1 已合并到下方单次 forward）
        if mode in ['understanding', 'mixed2', 'mixed']:
            understanding_outputs = self.model(
                images, text_tokens, text_mask,
                mode='understanding'
            )

            if 'targets' in batch:
                understanding_losses = self.compute_understanding_losses(
                    understanding_outputs,
                    batch['targets']
                )

                for loss_name, loss_value in understanding_losses.items():
                    understanding_loss_total = understanding_loss_total + loss_value
                    metrics[loss_name] = loss_value.item()
                has_understanding = True

        # mixed1 专用：理解+生成合并为单次 forward
        if mode == 'mixed1':
            target_images = batch.get('target_images')
            mixed_outputs = self.model(
                images, text_tokens, text_mask,
                target_images=target_images,
                mode='mixed1'
            )

            if 'targets' in batch:
                understanding_losses = self.compute_understanding_losses(
                    mixed_outputs,
                    batch['targets']
                )
                for loss_name, loss_value in understanding_losses.items():
                    understanding_loss_total = understanding_loss_total + loss_value
                    metrics[loss_name] = loss_value.item()
                has_understanding = True

            if 'diffusion_loss' in mixed_outputs:
                generation_loss_total = generation_loss_total + mixed_outputs['diffusion_loss']
                metrics['diffusion_loss'] = mixed_outputs['diffusion_loss'].item()
                has_generation = True

            # 改进方向5: 分层一致性损失
            if 'lot_outputs' in mixed_outputs:
                hier_loss = self.compute_hierarchical_consistency_loss(mixed_outputs['lot_outputs'])
                understanding_loss_total = understanding_loss_total + \
                    self.hierarchical_consistency_weight * hier_loss
                metrics['hierarchical_consistency_loss'] = hier_loss.item()

            # 改进方向5: 特征匹配损失
            if target_images is not None and 'generation_condition' in mixed_outputs:
                fm_loss = self.compute_feature_matching_loss(mixed_outputs, target_images)
                generation_loss_total = generation_loss_total + \
                    self.feature_matching_weight * fm_loss
                metrics['feature_matching_loss'] = fm_loss.item()

        # 生成任务训练
        elif mode in ['generation', 'mixed'] and 'target_images' in batch:
            target_images = batch['target_images']

            generation_outputs = self.model(
                images, text_tokens, text_mask,
                target_images=target_images,
                mode='generation'
            )

            diffusion_loss = generation_outputs['diffusion_loss']
            generation_loss_total = generation_loss_total + diffusion_loss
            metrics['diffusion_loss'] = diffusion_loss.item()
            has_generation = True

        # 编辑任务训练
        if mode in ['editing', 'mixed2', 'mixed'] and 'edited_images' in batch:
            source_images = batch['images']
            edited_images = batch['edited_images']
            edit_instruction = batch['text_tokens']

            editing_outputs = self.model(
                source_images, edit_instruction, text_mask,
                target_images=edited_images,
                mode='editing'
            )

            editing_losses = self.compute_editing_losses(
                editing_outputs,
                batch.get('targets', {}),
                source_images,
                edited_images
            )

            for loss_name, loss_value in editing_losses.items():
                generation_loss_total = generation_loss_total + loss_value
                metrics[loss_name] = loss_value.item()
            has_generation = True

        # 改进方向3: 多任务损失平衡
        if self.loss_balancer is not None and has_understanding and has_generation:
            total_loss, task_weights = self.loss_balancer(
                [understanding_loss_total, generation_loss_total]
            )
            metrics['understanding_weight'] = task_weights[0]
            metrics['generation_weight'] = task_weights[1]
            metrics['understanding_loss_group'] = understanding_loss_total.item()
            metrics['generation_loss_group'] = generation_loss_total.item()
        else:
            # 固定权重（原始逻辑）
            total_loss = understanding_loss_total + generation_loss_total

        # 反向传播
        if isinstance(total_loss, torch.Tensor) and total_loss.item() > 0:
            if self.use_deepspeed:
                self.ds_engine.backward(total_loss)
                self.ds_engine.step()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self.loss_balancer is not None:
                    torch.nn.utils.clip_grad_norm_(self.loss_balancer.parameters(), max_norm=1.0)
                self.optimizer.step()

            metrics['total_loss'] = total_loss.item()

        return metrics

    def train_epoch(self, dataloader, epoch, mode='mixed'):
        """
        训练一个epoch
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
            postfix = {}
            if 'total_loss' in metrics:
                postfix['loss'] = f"{metrics['total_loss']:.4f}"
            if 'understanding_weight' in metrics:
                postfix['u_w'] = f"{metrics['understanding_weight']:.2f}"
                postfix['g_w'] = f"{metrics['generation_weight']:.2f}"
            if postfix:
                progress_bar.set_postfix(postfix)

        # 计算平均指标
        avg_metrics = {key: sum(values) / len(values)
                      for key, values in total_metrics.items()}

        return avg_metrics

    def validate(self, dataloader, mode='mixed'):
        """
        验证模型（改进方向4: 增加评估指标）

        除了计算loss外，还计算以下指标：
        - 理解任务: task_accuracy, semantic_miou, bbox_iou, relation_accuracy
        - 生成任务: psnr, ssim
        """
        self.model.eval()
        if self.loss_balancer is not None:
            self.loss_balancer.eval()

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

                    # 改进方向4: 计算理解任务评估指标
                    eval_metrics = self.compute_eval_metrics(
                        understanding_outputs, batch_device, mode=mode
                    )
                    for metric_name, metric_value in eval_metrics.items():
                        if metric_name not in total_metrics:
                            total_metrics[metric_name] = []
                        total_metrics[metric_name].append(metric_value)

                # 生成任务验证
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

                    # 改进方向4: 计算生成任务评估指标（PSNR, SSIM）
                    gen_metrics = self.compute_eval_metrics(
                        generation_outputs, batch_device, mode=mode
                    )
                    for metric_name, metric_value in gen_metrics.items():
                        if metric_name not in total_metrics:
                            total_metrics[metric_name] = []
                        total_metrics[metric_name].append(metric_value)

                # 编辑任务验证
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

    def check_early_stopping(self, val_loss):
        """
        检查是否应该Early Stop（改进方向2）

        Args:
            val_loss: 当前验证损失
        Returns:
            bool: 是否应该停止训练
        """
        return self.early_stopping(val_loss)

    def save_checkpoint(self, path, epoch, metrics):
        """保存检查点（兼容 DDP 和 DeepSpeed）"""
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        save_dict = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'metrics': metrics
        }
        if self.optimizer is not None and not self.use_deepspeed:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.loss_balancer is not None:
            save_dict['loss_balancer_state_dict'] = self.loss_balancer.state_dict()
        torch.save(save_dict, path)

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'loss_balancer_state_dict' in checkpoint and self.loss_balancer is not None:
            self.loss_balancer.load_state_dict(checkpoint['loss_balancer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
