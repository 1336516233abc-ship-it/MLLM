"""
主MLLM模型 - 整合所有模块
"""

import torch
import torch.nn as nn
from .vit_encoder import ViTEncoder
from .text_tokenizer import TextTokenizer
from .lot_layers import LoTModule
from .integration_module import IntegrationModule
from .diffusion_module import DiffusionModule

class MLLMModel(nn.Module):
    """
    多模态大模型：图像理解+生成+编辑统一架构
    流程：输入 -> ViT+Text Encoder -> LoT分层推理 -> 整合模块 -> Diffusion
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 编码器（预训练冻结）
        self.vit_encoder = ViTEncoder(config)
        self.text_tokenizer = TextTokenizer(config)

        # 核心模块（需要训练）
        self.lot_module = LoTModule(config)
        self.integration_module = IntegrationModule(config)
        self.diffusion_module = DiffusionModule(config)

        # 理解任务的输出头
        self.understanding_head = nn.Sequential(
            nn.Linear(config.INTEGRATION_DIM, config.INTEGRATION_DIM // 2),
            nn.GELU(),
            nn.Linear(config.INTEGRATION_DIM // 2, config.TEXT_VOCAB_SIZE)
        )

    def freeze_encoders(self):
        """冻结预训练的编码器"""
        self.vit_encoder.freeze()
        self.text_tokenizer.freeze()

    def forward(self, images, text_tokens, text_mask=None, target_images=None, mode='understanding'):
        """
        Args:
            images: (B, C, H, W) 输入图像
            text_tokens: (B, seq_len) 文本tokens
            text_mask: (B, seq_len) 文本掩码
            target_images: (B, C, H, W) 目标图像（生成/编辑模式）
            mode: 'understanding' | 'generation' | 'editing'
        Returns:
            outputs: dict containing各种输出
        """
        # 1. 编码输入
        vision_features, vision_cls = self.vit_encoder(images)  # (B, num_patches, VIT_DIM)
        text_features, text_pooled = self.text_tokenizer(text_tokens, text_mask)  # (B, seq_len, TEXT_DIM)

        # 2. LoT分层推理
        lot_outputs = self.lot_module(vision_features, text_features)

        # 3. 整合模块
        integrated = self.integration_module(lot_outputs)

        outputs = {
            'lot_outputs': lot_outputs,
            'integrated_features': integrated['features'],
            'generation_condition': integrated['generation_condition'],
            'layer_weights': integrated['layer_weights']
        }

        # 4. 根据模式执行不同任务
        if mode == 'understanding':
            # 图像理解：输出文本描述或分析
            pooled_features = integrated['features'].mean(dim=1)  # (B, INTEGRATION_DIM)
            understanding_logits = self.understanding_head(pooled_features)
            outputs['understanding_logits'] = understanding_logits

            # 任务分类
            outputs['task_logits'] = lot_outputs['low']['task_logits']

            # 语义分割
            outputs['semantic_logits'] = lot_outputs['mid']['semantic_logits']

            # 边界框
            outputs['bboxes'] = lot_outputs['mid']['bboxes']

            # 关系矩阵
            outputs['relation_matrix'] = lot_outputs['high']['relation_matrix']

        elif mode in ['generation', 'editing']:
            # 图像生成/编辑
            if target_images is not None:
                # 训练模式：计算扩散损失
                diffusion_loss = self.diffusion_module(
                    target_images,
                    integrated['generation_condition']
                )
                outputs['diffusion_loss'] = diffusion_loss
            else:
                # 推理模式：生成图像
                generated_images = self.diffusion_module.sample(
                    integrated['generation_condition'],
                    image_size=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
                )
                outputs['generated_images'] = generated_images

        return outputs

    def generate_image(self, text_prompt, reference_image=None):
        """
        便捷方法：根据文本提示生成图像
        Args:
            text_prompt: str 文本提示
            reference_image: (1, C, H, W) 参考图像（可选）
        Returns:
            generated_image: (1, C, H, W) 生成的图像
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # 准备输入
            if reference_image is None:
                # 如果没有参考图像，使用零图像
                reference_image = torch.zeros(1, 3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)

            reference_image = reference_image.to(device)

            # 文本分词
            text_tokens, text_mask = self.text_tokenizer.tokenize([text_prompt])
            text_tokens = text_tokens.to(device)
            text_mask = text_mask.to(device)

            # 前向传播
            outputs = self.forward(
                reference_image,
                text_tokens,
                text_mask,
                mode='generation'
            )

            return outputs['generated_images']

    def understand_image(self, image, question=None):
        """
        便捷方法：理解图像
        Args:
            image: (1, C, H, W) 输入图像
            question: str 问题（可选）
        Returns:
            understanding: dict 包含各种理解结果
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            image = image.to(device)

            # 准备文本输入
            if question is None:
                question = "Describe this image"

            text_tokens, text_mask = self.text_tokenizer.tokenize([question])
            text_tokens = text_tokens.to(device)
            text_mask = text_mask.to(device)

            # 前向传播
            outputs = self.forward(
                image,
                text_tokens,
                text_mask,
                mode='understanding'
            )

            return {
                'task_type': torch.argmax(outputs['task_logits'], dim=-1).item(),
                'semantic_segmentation': outputs['semantic_logits'],
                'bounding_boxes': outputs['bboxes'],
                'relations': outputs['relation_matrix'],
                'lot_analysis': outputs['lot_outputs']
            }
