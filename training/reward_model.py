"""
奖励模型 - 用于CGPO强化学习
分为理解部分和生成部分，分层次打分
"""

import torch
import torch.nn as nn
from ..models.vit_encoder import ViTEncoder

class UnderstandingRewardModel(nn.Module):
    """
    理解部分奖励模型
    评估维度：完整性、语义逻辑、空间合理性、推理语义和输入的一致性
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 低层评估：完整性 - 评估元素识别的完整性
        self.completeness_scorer = nn.Sequential(
            nn.Linear(config.LOT_LOW_DIM, config.LOT_LOW_DIM // 2),
            nn.GELU(),
            nn.Linear(config.LOT_LOW_DIM // 2, 1),
            nn.Sigmoid()
        )

        # 中层评估：语义逻辑 - 评估语义标注的逻辑性
        self.semantic_logic_scorer = nn.Sequential(
            nn.Linear(config.LOT_MID_DIM, config.LOT_MID_DIM // 2),
            nn.GELU(),
            nn.Linear(config.LOT_MID_DIM // 2, 1),
            nn.Sigmoid()
        )

        # 中层评估：空间合理性 - 评估空间定位的合理性
        self.spatial_reason_scorer = nn.Sequential(
            nn.Linear(config.LOT_MID_DIM, config.LOT_MID_DIM // 2),
            nn.GELU(),
            nn.Linear(config.LOT_MID_DIM // 2, 1),
            nn.Sigmoid()
        )

        # 高层评估：一致性 - 评估推理与输入的一致性
        self.consistency_scorer = nn.Sequential(
            nn.Linear(config.LOT_HIGH_DIM + config.VIT_DIM, config.LOT_HIGH_DIM),
            nn.GELU(),
            nn.Linear(config.LOT_HIGH_DIM, 1),
            nn.Sigmoid()
        )

        # 权重
        self.weights = config.REWARD_UNDERSTANDING_WEIGHTS

    def forward(self, lot_outputs, vision_cls):
        """
        Args:
            lot_outputs: LoT模块的输出
            vision_cls: (B, VIT_DIM) 视觉全局特征
        Returns:
            reward: (B,) 理解部分的奖励分数
            details: dict 各维度的详细分数
        """
        B = vision_cls.shape[0]

        # 1. 完整性评分 - 基于低层特征
        low_features = lot_outputs['low']['element_features'].mean(dim=1)  # (B, LOT_LOW_DIM)
        completeness_score = self.completeness_scorer(low_features).squeeze(-1)  # (B,)

        # 2. 语义逻辑评分 - 基于中层语义特征
        semantic_features = lot_outputs['mid']['semantic_features'].mean(dim=1)  # (B, LOT_MID_DIM)
        semantic_logic_score = self.semantic_logic_scorer(semantic_features).squeeze(-1)  # (B,)

        # 3. 空间合理性评分 - 基于中层空间特征
        spatial_features = lot_outputs['mid']['spatial_features'].mean(dim=1)  # (B, LOT_MID_DIM)
        spatial_reason_score = self.spatial_reason_scorer(spatial_features).squeeze(-1)  # (B,)

        # 4. 一致性评分 - 结合高层推理和原始视觉特征
        high_features = lot_outputs['high']['scene_features']  # (B, LOT_HIGH_DIM)
        combined = torch.cat([high_features, vision_cls], dim=-1)
        consistency_score = self.consistency_scorer(combined).squeeze(-1)  # (B,)

        # 加权求和
        total_reward = (
            self.weights['completeness'] * completeness_score +
            self.weights['semantic_logic'] * semantic_logic_score +
            self.weights['spatial_reason'] * spatial_reason_score +
            self.weights['consistency'] * consistency_score
        )

        details = {
            'completeness': completeness_score,
            'semantic_logic': semantic_logic_score,
            'spatial_reason': spatial_reason_score,
            'consistency': consistency_score
        }

        return total_reward, details

class GenerationRewardModel(nn.Module):
    """
    生成部分奖励模型
    评估维度：颜色质量、语义准确性、视觉效果
    使用参考模型进行评估
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 使用预训练的ViT作为参考模型提取特征
        self.reference_encoder = ViTEncoder(config)
        self.reference_encoder.freeze()

        # 低层评估：颜色质量 - 评估颜色分布和饱和度
        self.color_quality_scorer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 中层评估：语义准确性 - 比较生成图像与条件的语义匹配
        self.semantic_accuracy_scorer = nn.Sequential(
            nn.Linear(config.VIT_DIM * 2, config.VIT_DIM),
            nn.GELU(),
            nn.Linear(config.VIT_DIM, 1),
            nn.Sigmoid()
        )

        # 高层评估：视觉效果 - 评估整体视觉质量
        self.visual_quality_scorer = nn.Sequential(
            nn.Linear(config.VIT_DIM, config.VIT_DIM // 2),
            nn.GELU(),
            nn.Linear(config.VIT_DIM // 2, 1),
            nn.Sigmoid()
        )

        # 权重
        self.weights = config.REWARD_GENERATION_WEIGHTS

    def forward(self, generated_images, condition_features):
        """
        Args:
            generated_images: (B, C, H, W) 生成的图像
            condition_features: (B, VIT_DIM) 条件特征（来自原始图像）
        Returns:
            reward: (B,) 生成部分的奖励分数
            details: dict 各维度的详细分数
        """
        B = generated_images.shape[0]

        # 1. 颜色质量评分 - 基于生成图像的像素分布
        color_quality_score = self.color_quality_scorer(generated_images).squeeze(-1)  # (B,)

        # 2. 语义准确性评分 - 比较生成图像和条件的语义特征
        with torch.no_grad():
            gen_features, gen_cls = self.reference_encoder(generated_images)

        combined_semantic = torch.cat([gen_cls, condition_features], dim=-1)
        semantic_accuracy_score = self.semantic_accuracy_scorer(combined_semantic).squeeze(-1)  # (B,)

        # 3. 视觉效果评分 - 基于生成图像的全局特征
        visual_quality_score = self.visual_quality_scorer(gen_cls).squeeze(-1)  # (B,)

        # 加权求和
        total_reward = (
            self.weights['color_quality'] * color_quality_score +
            self.weights['semantic_accuracy'] * semantic_accuracy_score +
            self.weights['visual_quality'] * visual_quality_score
        )

        details = {
            'color_quality': color_quality_score,
            'semantic_accuracy': semantic_accuracy_score,
            'visual_quality': visual_quality_score
        }

        return total_reward, details

class RewardModel(nn.Module):
    """
    完整奖励模型：整合理解和生成两部分
    自适应权重调整：着重关注分数低的方面
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.understanding_model = UnderstandingRewardModel(config)
        self.generation_model = GenerationRewardModel(config)

        # 总权重
        self.understanding_weight = config.REWARD_UNDERSTANDING_TOTAL
        self.generation_weight = config.REWARD_GENERATION_TOTAL

    def forward(self, lot_outputs, vision_cls, generated_images, condition_features):
        """
        Args:
            lot_outputs: LoT模块输出
            vision_cls: (B, VIT_DIM) 原始视觉特征
            generated_images: (B, C, H, W) 生成的图像
            condition_features: (B, VIT_DIM) 条件特征
        Returns:
            total_reward: (B,) 总奖励分数
            all_details: dict 所有详细分数
        """
        # 理解部分奖励
        understanding_reward, understanding_details = self.understanding_model(
            lot_outputs, vision_cls
        )

        # 生成部分奖励
        generation_reward, generation_details = self.generation_model(
            generated_images, condition_features
        )

        # 自适应权重：分数低的方面获得更高权重
        # 使用softmax的反向 - 分数低的维度权重大
        understanding_adaptive = torch.exp(-understanding_reward * 2)
        generation_adaptive = torch.exp(-generation_reward * 2)

        total_adaptive = understanding_adaptive + generation_adaptive
        understanding_weight_adaptive = understanding_adaptive / total_adaptive
        generation_weight_adaptive = generation_adaptive / total_adaptive

        # 计算总奖励（结合固定权重和自适应权重）
        fixed_reward = (
            self.understanding_weight * understanding_reward +
            self.generation_weight * generation_reward
        )

        adaptive_reward = (
            understanding_weight_adaptive * understanding_reward +
            generation_weight_adaptive * generation_reward
        )

        # 混合固定和自适应权重
        total_reward = 0.7 * fixed_reward + 0.3 * adaptive_reward

        all_details = {
            'understanding': understanding_details,
            'generation': generation_details,
            'understanding_reward': understanding_reward,
            'generation_reward': generation_reward,
            'adaptive_weights': {
                'understanding': understanding_weight_adaptive,
                'generation': generation_weight_adaptive
            }
        }

        return total_reward, all_details

    def get_low_score_aspects(self, details):
        """
        识别分数低的方面，用于优化时着重关注
        Returns:
            low_aspects: list 分数低于阈值的维度
        """
        threshold = 0.6
        low_aspects = []

        for category in ['understanding', 'generation']:
            for aspect, score in details[category].items():
                if score.mean() < threshold:
                    low_aspects.append(f"{category}.{aspect}")

        return low_aspects
