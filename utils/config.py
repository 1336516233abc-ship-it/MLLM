"""
配置文件 - 定义模型架构和训练超参数
"""

class Config:
    # 模型架构配置
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    VIT_DIM = 768
    VIT_DEPTH = 12
    VIT_HEADS = 12

    TEXT_VOCAB_SIZE = 50000
    TEXT_MAX_LENGTH = 512
    TEXT_DIM = 768

    # LoT分层推理配置
    LOT_LOW_DIM = 512    # 低层：任务理解、元素识别
    LOT_MID_DIM = 768    # 中层：元素语义、空间定位
    LOT_HIGH_DIM = 1024  # 高层：高级语义关系、空间关系

    LOT_LOW_HEADS = 8
    LOT_MID_HEADS = 12
    LOT_HIGH_HEADS = 16

    LOT_LOW_LAYERS = 4
    LOT_MID_LAYERS = 4
    LOT_HIGH_LAYERS = 4

    # 整合模块配置
    INTEGRATION_DIM = 1024
    INTEGRATION_HEADS = 16
    INTEGRATION_LAYERS = 6

    # 扩散模型配置
    DIFFUSION_TIMESTEPS = 1000
    DIFFUSION_DIM = 512
    DIFFUSION_CHANNELS = 3

    # 训练配置
    BATCH_SIZE = 8
    PRETRAIN_EPOCHS = 100
    PRETRAIN_LR = 1e-4

    CGPO_EPOCHS = 50
    CGPO_LR = 5e-5
    CGPO_BETA = 0.1  # KL散度权重

    # 奖励模型配置
    # 理解部分权重
    REWARD_UNDERSTANDING_WEIGHTS = {
        'completeness': 0.25,      # 完整性
        'semantic_logic': 0.25,    # 语义逻辑
        'spatial_reason': 0.25,    # 空间合理性
        'consistency': 0.25        # 推理语义和输入的一致性
    }

    # 生成图像部分权重
    REWARD_GENERATION_WEIGHTS = {
        'color_quality': 0.3,      # 颜色性
        'semantic_accuracy': 0.4,  # 语义准确性
        'visual_quality': 0.3      # 视觉效果
    }

    # 理解和生成的总权重
    REWARD_UNDERSTANDING_TOTAL = 0.5
    REWARD_GENERATION_TOTAL = 0.5

    # 设备配置
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    NUM_WORKERS = 4

    # 保存路径
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
