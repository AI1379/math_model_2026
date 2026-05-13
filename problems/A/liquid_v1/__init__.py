"""V1 training stack for bottle liquid recognition."""

from .datasets import LCDTCCropDataset, TransparentObjectSegDataset
from .model import BottleLiquidNet

__all__ = [
    "BottleLiquidNet",
    "LCDTCCropDataset",
    "TransparentObjectSegDataset",
]
