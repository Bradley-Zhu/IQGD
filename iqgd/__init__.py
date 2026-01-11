"""
IQGD: Iterative Q-Guided Diffusion
===================================

Main components:
- data_loader: Dataset wrapper for MGDM fluid data
- diffusion_env: RL environment wrapping diffusion process
- q_network: Q-function architecture
- iqgd_agent: Main IQGD agent with Q-learning
- replay_buffer: Experience replay for training
"""

__version__ = "0.1.0"

from .data_loader import IQGDDataset
from .diffusion_env import DiffusionEnv
from .q_network import QNetwork
from .iqgd_agent import IQGDAgent
from .replay_buffer import ReplayBuffer

__all__ = [
    "IQGDDataset",
    "DiffusionEnv",
    "QNetwork",
    "IQGDAgent",
    "ReplayBuffer",
]
