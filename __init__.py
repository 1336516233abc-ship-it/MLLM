"""MLLM包初始化"""

from .models.mllm_model import MLLMModel
from .utils.config import Config
from .utils.data_loader import create_dataloaders, MultiModalDataset
from .training.pretrain import PreTrainer
from .training.cgpo import CGPOTrainer
from .training.reward_model import RewardModel

__version__ = '1.0.0'

__all__ = [
    'MLLMModel',
    'Config',
    'create_dataloaders',
    'MultiModalDataset',
    'PreTrainer',
    'CGPOTrainer',
    'RewardModel'
]
