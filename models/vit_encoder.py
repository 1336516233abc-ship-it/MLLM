"""
ViT图像编码器 - 预训练冻结模块
"""

import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    """图像分块嵌入"""
    def __init__(self, image_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.projection(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class ViTEncoder(nn.Module):
    """Vision Transformer编码器"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            config.IMAGE_SIZE,
            config.PATCH_SIZE,
            embed_dim=config.VIT_DIM
        )

        # Position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.VIT_DIM)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.VIT_DIM))

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.VIT_DIM,
            nhead=config.VIT_HEADS,
            dim_feedforward=config.VIT_DIM * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.VIT_DEPTH
        )

        self.norm = nn.LayerNorm(config.VIT_DIM)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入图像
        Returns:
            features: (B, num_patches+1, VIT_DIM) 图像特征
            cls_token: (B, VIT_DIM) 全局特征
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, VIT_DIM)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, VIT_DIM)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x)

        cls_token = x[:, 0]  # (B, VIT_DIM)
        features = x  # (B, num_patches+1, VIT_DIM)

        return features, cls_token

    def freeze(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
