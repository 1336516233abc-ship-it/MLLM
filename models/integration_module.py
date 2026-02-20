"""
整合模块 - 类似Transformer的交叉网络结构
自适应地整合分层推理结果
"""

import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    """交叉注意力层"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, query, key_value):
        """
        Args:
            query: (B, seq_len_q, dim)
            key_value: (B, seq_len_kv, dim)
        """
        # Cross attention
        attn_out, _ = self.attention(query, key_value, key_value)
        query = self.norm1(query + attn_out)

        # FFN
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query

class IntegrationModule(nn.Module):
    """
    整合模块：自适应整合低、中、高层推理结果
    使用交叉注意力机制融合不同层次的信息
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 投影各层特征到统一维度
        self.low_proj = nn.Linear(config.LOT_LOW_DIM, config.INTEGRATION_DIM)
        self.mid_proj = nn.Linear(config.LOT_MID_DIM, config.INTEGRATION_DIM)
        self.high_proj = nn.Linear(config.LOT_HIGH_DIM, config.INTEGRATION_DIM)

        # 自适应权重学习
        self.layer_weights = nn.Parameter(torch.ones(3) / 3)  # 初始化为均等权重

        # 交叉注意力层 - 融合不同层次
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(config.INTEGRATION_DIM, config.INTEGRATION_HEADS)
            for _ in range(config.INTEGRATION_LAYERS)
        ])

        # 自注意力层 - 内部整合
        self.self_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.INTEGRATION_DIM,
                nhead=config.INTEGRATION_HEADS,
                dim_feedforward=config.INTEGRATION_DIM * 4,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.INTEGRATION_LAYERS)
        ])

        # 输出投影
        self.output_norm = nn.LayerNorm(config.INTEGRATION_DIM)

        # 用于图像生成的条件特征
        self.generation_proj = nn.Sequential(
            nn.Linear(config.INTEGRATION_DIM, config.DIFFUSION_DIM),
            nn.LayerNorm(config.DIFFUSION_DIM)
        )

    def forward(self, lot_outputs):
        """
        Args:
            lot_outputs: dict containing 'low', 'mid', 'high' reasoning results
        Returns:
            integrated_features: dict containing:
                - features: (B, num_patches, INTEGRATION_DIM) 整合后的特征
                - generation_condition: (B, DIFFUSION_DIM) 用于图像生成的条件
                - layer_weights: (3,) 各层的自适应权重
        """
        # 提取各层特征
        low_features = lot_outputs['low']['element_features']  # (B, num_patches, LOT_LOW_DIM)
        mid_features = lot_outputs['mid']['semantic_features']  # (B, num_patches, LOT_MID_DIM)
        high_features = lot_outputs['high']['semantic_relation_features']  # (B, num_patches, LOT_HIGH_DIM)

        # 投影到统一维度
        low_proj = self.low_proj(low_features)  # (B, num_patches, INTEGRATION_DIM)
        mid_proj = self.mid_proj(mid_features)  # (B, num_patches, INTEGRATION_DIM)
        high_proj = self.high_proj(high_features)  # (B, num_patches, INTEGRATION_DIM)

        # 计算自适应权重（softmax归一化）
        weights = torch.softmax(self.layer_weights, dim=0)

        # 加权融合
        weighted_features = (
            weights[0] * low_proj +
            weights[1] * mid_proj +
            weights[2] * high_proj
        )  # (B, num_patches, INTEGRATION_DIM)

        # 通过交叉注意力和自注意力层进行深度整合
        integrated = weighted_features

        for cross_attn, self_attn in zip(self.cross_attention_layers, self.self_attention_layers):
            # 交叉注意力：让当前特征关注高层特征（全局指导）
            integrated = cross_attn(integrated, high_proj)

            # 自注意力：内部整合
            integrated = self_attn(integrated)

        integrated = self.output_norm(integrated)

        # 生成用于图像生成的条件特征
        generation_condition = integrated.mean(dim=1)  # (B, INTEGRATION_DIM) 池化
        generation_condition = self.generation_proj(generation_condition)  # (B, DIFFUSION_DIM)

        return {
            'features': integrated,  # (B, num_patches, INTEGRATION_DIM)
            'generation_condition': generation_condition,  # (B, DIFFUSION_DIM)
            'layer_weights': weights  # (3,)
        }
