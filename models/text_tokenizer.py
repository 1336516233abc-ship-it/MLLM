"""
文本分词器和编码器 - 预训练冻结模块
"""

import torch
import torch.nn as nn

class TextTokenizer(nn.Module):
    """文本分词和编码器"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(
            config.TEXT_VOCAB_SIZE,
            config.TEXT_DIM
        )

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.TEXT_MAX_LENGTH, config.TEXT_DIM)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.TEXT_DIM,
            nhead=8,
            dim_feedforward=config.TEXT_DIM * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )

        self.norm = nn.LayerNorm(config.TEXT_DIM)

    def forward(self, token_ids, attention_mask=None):
        """
        Args:
            token_ids: (B, seq_len) 词元ID
            attention_mask: (B, seq_len) 注意力掩码
        Returns:
            features: (B, seq_len, TEXT_DIM) 文本特征
            pooled: (B, TEXT_DIM) 池化特征
        """
        B, seq_len = token_ids.shape

        # Token embedding
        x = self.token_embed(token_ids)  # (B, seq_len, TEXT_DIM)

        # Add position embedding
        x = x + self.pos_embed[:, :seq_len, :]

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = masked)
            mask = (attention_mask == 0)
        else:
            mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        # Pooling (mean of non-masked tokens)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = x.mean(dim=1)

        return x, pooled

    def freeze(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False

    def tokenize(self, texts):
        """
        简单的分词函数（实际应用中应使用专业分词器如BPE）
        Args:
            texts: list of strings
        Returns:
            token_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        """
        # 这里是简化实现，实际应使用专业分词器
        token_ids = []
        attention_masks = []

        for text in texts:
            # 简单的字符级分词
            tokens = [ord(c) % self.config.TEXT_VOCAB_SIZE for c in text]

            # 截断或填充
            if len(tokens) > self.config.TEXT_MAX_LENGTH:
                tokens = tokens[:self.config.TEXT_MAX_LENGTH]
                mask = [1] * self.config.TEXT_MAX_LENGTH
            else:
                mask = [1] * len(tokens) + [0] * (self.config.TEXT_MAX_LENGTH - len(tokens))
                tokens = tokens + [0] * (self.config.TEXT_MAX_LENGTH - len(tokens))

            token_ids.append(tokens)
            attention_masks.append(mask)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long)
        )
