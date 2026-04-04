"""DINOv2 One-Shot Segmentation Research Module.

Modular ablation framework for testing fusion methods
with frozen DINOv2 backbones on MPS (Apple Silicon).
"""

from .config import Config
from .backbone import DINOv2Backbone, SIZE_TO_DIM
from .fusion import FUSION_REGISTRY
from .model import OneShotModel, build_model
from .trainer import Trainer

__all__ = [
    "Config",
    "DINOv2Backbone",
    "SIZE_TO_DIM",
    "FUSION_REGISTRY",
    "OneShotModel",
    "build_model",
    "Trainer",
]
