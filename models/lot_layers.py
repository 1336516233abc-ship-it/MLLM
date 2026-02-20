"""
LoT分层推理模块 - 核心创新
低层：任务理解、元素识别
中层：元素语义、空间定位
高层：高级语义关系、空间关系
"""

import torch
import torch.nn as nn

class LowLayerReasoning(nn.Module):
    """
    低层推理：任务理解和元素识别
    - 任务理解：理解用户意图（生成/编辑/理解）
    - 元素识别：识别图像中的基本元素（物体、颜色、形状等）
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 融合视觉和文本特征
        self.vision_proj = nn.Linear(config.VIT_DIM, config.LOT_LOW_DIM)
        self.text_proj = nn.Linear(config.TEXT_DIM, config.LOT_LOW_DIM)

        # 任务理解分支
        self.task_understanding = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.LOT_LOW_DIM,
                nhead=config.LOT_LOW_HEADS,
                dim_feedforward=config.LOT_LOW_DIM * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.LOT_LOW_LAYERS)
        ])

        # 元素识别分支
        self.element_detection = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.LOT_LOW_DIM,
                nhead=config.LOT_LOW_HEADS,
                dim_feedforward=config.LOT_LOW_DIM * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.LOT_LOW_LAYERS)
        ])

        # 任务分类头（生成/编辑/理解）
        self.task_classifier = nn.Sequential(
            nn.Linear(config.LOT_LOW_DIM, config.LOT_LOW_DIM // 2),
            nn.GELU(),
            nn.Linear(config.LOT_LOW_DIM // 2, 3)  # 3类任务
        )

        # 元素特征提取
        self.element_extractor = nn.Sequential(
            nn.Linear(config.LOT_LOW_DIM, config.LOT_LOW_DIM),
            nn.LayerNorm(config.LOT_LOW_DIM)
        )

        self.norm = nn.LayerNorm(config.LOT_LOW_DIM)

    def forward(self, vision_features, text_features):
        """
        Args:
            vision_features: (B, num_patches, VIT_DIM)
            text_features: (B, seq_len, TEXT_DIM)
        Returns:
            low_reasoning: dict containing:
                - task_logits: (B, 3) 任务分类
                - task_features: (B, LOT_LOW_DIM) 任务理解特征
                - element_features: (B, num_patches, LOT_LOW_DIM) 元素特征
        """
        B = vision_features.shape[0]

        # 投影到统一维度
        vision_proj = self.vision_proj(vision_features)  # (B, num_patches, LOT_LOW_DIM)
        text_proj = self.text_proj(text_features)  # (B, seq_len, LOT_LOW_DIM)

        # 任务理解分支 - 主要关注文本意图
        task_tokens = text_proj
        for layer in self.task_understanding:
            task_tokens = layer(task_tokens)
        task_features = task_tokens.mean(dim=1)  # (B, LOT_LOW_DIM) 池化
        task_logits = self.task_classifier(task_features)  # (B, 3)

        # 元素识别分支 - 主要关注视觉元素
        element_tokens = vision_proj
        for layer in self.element_detection:
            element_tokens = layer(element_tokens)
        element_features = self.element_extractor(element_tokens)  # (B, num_patches, LOT_LOW_DIM)

        return {
            'task_logits': task_logits,
            'task_features': task_features,
            'element_features': element_features
        }

class MidLayerReasoning(nn.Module):
    """
    中层推理：元素语义和空间定位
    - 元素语义：理解元素的语义含义
    - 空间定位：理解元素的空间位置和布局
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 从低层特征投影到中层
        self.low_to_mid_proj = nn.Linear(config.LOT_LOW_DIM, config.LOT_MID_DIM)

        # 元素语义理解
        self.semantic_reasoning = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.LOT_MID_DIM,
                nhead=config.LOT_MID_HEADS,
                dim_feedforward=config.LOT_MID_DIM * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.LOT_MID_LAYERS)
        ])

        # 空间定位
        self.spatial_reasoning = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.LOT_MID_DIM,
                nhead=config.LOT_MID_HEADS,
                dim_feedforward=config.LOT_MID_DIM * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.LOT_MID_LAYERS)
        ])

        # 位置编码生成器
        self.position_encoder = nn.Sequential(
            nn.Linear(config.LOT_MID_DIM, config.LOT_MID_DIM),
            nn.GELU(),
            nn.Linear(config.LOT_MID_DIM, 4)  # (x, y, w, h) 边界框
        )

        # 语义标签预测
        self.semantic_predictor = nn.Sequential(
            nn.Linear(config.LOT_MID_DIM, config.LOT_MID_DIM // 2),
            nn.GELU(),
            nn.Linear(config.LOT_MID_DIM // 2, 100)  # 假设100个语义类别
        )

        self.norm = nn.LayerNorm(config.LOT_MID_DIM)

    def forward(self, low_reasoning):
        """
        Args:
            low_reasoning: dict from LowLayerReasoning
        Returns:
            mid_reasoning: dict containing:
                - semantic_features: (B, num_patches, LOT_MID_DIM) 语义特征
                - spatial_features: (B, num_patches, LOT_MID_DIM) 空间特征
                - semantic_logits: (B, num_patches, 100) 语义分类
                - bboxes: (B, num_patches, 4) 边界框预测
        """
        # 获取低层特征
        element_features = low_reasoning['element_features']  # (B, num_patches, LOT_LOW_DIM)
        task_features = low_reasoning['task_features']  # (B, LOT_LOW_DIM)

        # 投影到中层维度
        mid_features = self.low_to_mid_proj(element_features)  # (B, num_patches, LOT_MID_DIM)

        # 元素语义理解 - 结合任务信息
        task_mid = self.low_to_mid_proj(task_features).unsqueeze(1)  # (B, 1, LOT_MID_DIM)
        semantic_tokens = torch.cat([task_mid, mid_features], dim=1)  # (B, num_patches+1, LOT_MID_DIM)

        for layer in self.semantic_reasoning:
            semantic_tokens = layer(semantic_tokens)

        semantic_features = semantic_tokens[:, 1:, :]  # (B, num_patches, LOT_MID_DIM)
        semantic_logits = self.semantic_predictor(semantic_features)  # (B, num_patches, 100)

        # 空间定位 - 理解元素的空间关系
        spatial_tokens = mid_features
        for layer in self.spatial_reasoning:
            spatial_tokens = layer(spatial_tokens)

        spatial_features = spatial_tokens  # (B, num_patches, LOT_MID_DIM)
        bboxes = torch.sigmoid(self.position_encoder(spatial_features))  # (B, num_patches, 4)

        return {
            'semantic_features': semantic_features,
            'spatial_features': spatial_features,
            'semantic_logits': semantic_logits,
            'bboxes': bboxes
        }

class HighLayerReasoning(nn.Module):
    """
    高层推理：高级语义关系和空间关系
    - 高级语义关系：理解元素之间的语义联系
    - 空间关系：理解元素之间的相对空间关系
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 从中层投影到高层
        self.mid_to_high_proj = nn.Linear(config.LOT_MID_DIM, config.LOT_HIGH_DIM)

        # 高级语义关系推理
        self.semantic_relation = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.LOT_HIGH_DIM,
                nhead=config.LOT_HIGH_HEADS,
                dim_feedforward=config.LOT_HIGH_DIM * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.LOT_HIGH_LAYERS)
        ])

        # 空间关系推理
        self.spatial_relation = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.LOT_HIGH_DIM,
                nhead=config.LOT_HIGH_HEADS,
                dim_feedforward=config.LOT_HIGH_DIM * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.LOT_HIGH_LAYERS)
        ])

        # 关系图生成器 - 预测元素间的关系
        self.relation_predictor = nn.Sequential(
            nn.Linear(config.LOT_HIGH_DIM * 2, config.LOT_HIGH_DIM),
            nn.GELU(),
            nn.Linear(config.LOT_HIGH_DIM, 10)  # 10种关系类型
        )

        # 全局场景理解
        self.scene_understanding = nn.Sequential(
            nn.Linear(config.LOT_HIGH_DIM, config.LOT_HIGH_DIM),
            nn.GELU(),
            nn.Linear(config.LOT_HIGH_DIM, config.LOT_HIGH_DIM)
        )

        self.norm = nn.LayerNorm(config.LOT_HIGH_DIM)

    def forward(self, mid_reasoning):
        """
        Args:
            mid_reasoning: dict from MidLayerReasoning
        Returns:
            high_reasoning: dict containing:
                - semantic_relation_features: (B, num_patches, LOT_HIGH_DIM)
                - spatial_relation_features: (B, num_patches, LOT_HIGH_DIM)
                - relation_matrix: (B, num_patches, num_patches, 10) 关系矩阵
                - scene_features: (B, LOT_HIGH_DIM) 全局场景特征
        """
        B, num_patches, _ = mid_reasoning['semantic_features'].shape

        # 融合语义和空间特征
        semantic_features = mid_reasoning['semantic_features']
        spatial_features = mid_reasoning['spatial_features']
        combined_features = (semantic_features + spatial_features) / 2

        # 投影到高层维度
        high_features = self.mid_to_high_proj(combined_features)  # (B, num_patches, LOT_HIGH_DIM)

        # 高级语义关系推理
        semantic_relation_tokens = high_features
        for layer in self.semantic_relation:
            semantic_relation_tokens = layer(semantic_relation_tokens)

        semantic_relation_features = semantic_relation_tokens  # (B, num_patches, LOT_HIGH_DIM)

        # 空间关系推理
        spatial_relation_tokens = high_features
        for layer in self.spatial_relation:
            spatial_relation_tokens = layer(spatial_relation_tokens)

        spatial_relation_features = spatial_relation_tokens  # (B, num_patches, LOT_HIGH_DIM)

        # 构建关系矩阵 - 计算每对元素之间的关系
        # 扩展特征以计算成对关系
        feat_i = semantic_relation_features.unsqueeze(2).expand(-1, -1, num_patches, -1)  # (B, num_patches, num_patches, LOT_HIGH_DIM)
        feat_j = semantic_relation_features.unsqueeze(1).expand(-1, num_patches, -1, -1)  # (B, num_patches, num_patches, LOT_HIGH_DIM)

        # 拼接成对特征
        pair_features = torch.cat([feat_i, feat_j], dim=-1)  # (B, num_patches, num_patches, LOT_HIGH_DIM*2)

        # 预测关系
        relation_matrix = self.relation_predictor(pair_features)  # (B, num_patches, num_patches, 10)

        # 全局场景理解
        scene_features = semantic_relation_features.mean(dim=1)  # (B, LOT_HIGH_DIM)
        scene_features = self.scene_understanding(scene_features)  # (B, LOT_HIGH_DIM)

        return {
            'semantic_relation_features': semantic_relation_features,
            'spatial_relation_features': spatial_relation_features,
            'relation_matrix': relation_matrix,
            'scene_features': scene_features
        }

class LoTModule(nn.Module):
    """完整的分层推理模块"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.low_layer = LowLayerReasoning(config)
        self.mid_layer = MidLayerReasoning(config)
        self.high_layer = HighLayerReasoning(config)

    def forward(self, vision_features, text_features):
        """
        分层推理：低层 -> 中层 -> 高层
        每层的输出会作为下一层的输入
        """
        # 低层推理
        low_reasoning = self.low_layer(vision_features, text_features)

        # 中层推理（使用低层结果）
        mid_reasoning = self.mid_layer(low_reasoning)

        # 高层推理（使用中层结果）
        high_reasoning = self.high_layer(mid_reasoning)

        return {
            'low': low_reasoning,
            'mid': mid_reasoning,
            'high': high_reasoning
        }
