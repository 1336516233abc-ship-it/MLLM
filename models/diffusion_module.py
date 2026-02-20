"""
扩散模块 - 用于图像生成和编辑
基于DDPM (Denoising Diffusion Probabilistic Models)
"""

import torch
import torch.nn as nn
import numpy as np

class SinusoidalPositionEmbedding(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, time_dim, condition_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_channels),
            nn.GELU()
        )

        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, out_channels),
            nn.GELU()
        )

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb, condition_emb):
        """
        Args:
            x: (B, C, H, W)
            time_emb: (B, time_dim)
            condition_emb: (B, condition_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)

        # 添加时间条件
        time_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_proj

        # 添加语义条件
        cond_proj = self.condition_mlp(condition_emb)[:, :, None, None]
        h = h + cond_proj

        h = nn.functional.gelu(h)
        h = self.conv2(h)
        h = self.norm2(h)

        return nn.functional.gelu(h + self.shortcut(x))

class DiffusionUNet(nn.Module):
    """U-Net架构的扩散模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 时间编码
        time_dim = config.DIFFUSION_DIM
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # 编码器
        channels = [config.DIFFUSION_DIM, config.DIFFUSION_DIM * 2, config.DIFFUSION_DIM * 4]
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(
                config.DIFFUSION_CHANNELS if i == 0 else channels[i-1],
                channels[i],
                time_dim,
                config.DIFFUSION_DIM
            ) for i in range(len(channels))
        ])

        self.downsample = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i], 3, stride=2, padding=1)
            for i in range(len(channels) - 1)
        ] + [nn.Identity()])

        # 中间层
        self.middle_block = ResidualBlock(
            channels[-1], channels[-1], time_dim, config.DIFFUSION_DIM
        )

        # 解码器
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(
                channels[i] * 2 if i < len(channels) - 1 else channels[i],
                channels[i],
                time_dim,
                config.DIFFUSION_DIM
            ) for i in reversed(range(len(channels)))
        ])

        self.upsample = nn.ModuleList([
            nn.Identity()
        ] + [
            nn.ConvTranspose2d(channels[i], channels[i], 4, stride=2, padding=1)
            for i in reversed(range(1, len(channels)))
        ])

        # 输出层
        self.output = nn.Conv2d(channels[0], config.DIFFUSION_CHANNELS, 1)

    def forward(self, x, timesteps, condition):
        """
        Args:
            x: (B, C, H, W) 噪声图像
            timesteps: (B,) 时间步
            condition: (B, DIFFUSION_DIM) 条件特征
        Returns:
            predicted_noise: (B, C, H, W) 预测的噪声
        """
        # 时间编码
        time_emb = self.time_embedding(timesteps)

        # 编码器
        encoder_features = []
        h = x
        for block, downsample in zip(self.encoder_blocks, self.downsample):
            h = block(h, time_emb, condition)
            encoder_features.append(h)
            h = downsample(h)

        # 中间层
        h = self.middle_block(h, time_emb, condition)

        # 解码器
        for i, (block, upsample) in enumerate(zip(self.decoder_blocks, self.upsample)):
            h = upsample(h)
            if i > 0:  # 跳跃连接
                h = torch.cat([h, encoder_features[-(i)]], dim=1)
            h = block(h, time_emb, condition)

        # 输出
        return self.output(h)

class DiffusionModule(nn.Module):
    """完整的扩散模块"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.timesteps = config.DIFFUSION_TIMESTEPS

        # U-Net模型
        self.unet = DiffusionUNet(config)

        # 噪声调度
        self.register_buffer('betas', self._cosine_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def _cosine_beta_schedule(self):
        """余弦调度"""
        steps = self.timesteps
        s = 0.008
        x = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x_start, condition):
        """
        训练时的前向传播
        Args:
            x_start: (B, C, H, W) 原始图像
            condition: (B, DIFFUSION_DIM) 条件特征
        Returns:
            loss: 扩散损失
        """
        B = x_start.shape[0]
        device = x_start.device

        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        # 添加噪声
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        # 预测噪声
        predicted_noise = self.unet(x_noisy, t, condition)

        # 计算损失
        loss = nn.functional.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, condition, image_size=(224, 224), batch_size=1):
        """
        生成图像
        Args:
            condition: (B, DIFFUSION_DIM) 条件特征
            image_size: (H, W) 图像尺寸
            batch_size: 批次大小
        Returns:
            images: (B, C, H, W) 生成的图像
        """
        device = condition.device
        B = condition.shape[0]

        # 从纯噪声开始
        img = torch.randn(B, self.config.DIFFUSION_CHANNELS, *image_size, device=device)

        # 逐步去噪
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # 预测噪声
            predicted_noise = self.unet(img, t_batch, condition)

            # 去噪
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]

            if t > 0:
                noise = torch.randn_like(img)
                alpha_cumprod_prev = self.alphas_cumprod[t - 1]
            else:
                noise = 0
                alpha_cumprod_prev = 1.0

            # DDPM采样公式
            pred_x0 = (img - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            direction = torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise
            img = torch.sqrt(alpha_cumprod_prev) * pred_x0 + direction

            if t > 0:
                img = img + torch.sqrt(self.betas[t]) * noise

        return torch.clamp(img, -1, 1)
