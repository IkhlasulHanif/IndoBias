"""Model implementations for inference.

Note: This package (indobias.models.*) contains inference models.
Data models (Demography, Group, etc.) are in indobias.data_models.
"""

from .qwen import QwenModel, get_device_info

__all__ = ["QwenModel", "get_device_info"]

