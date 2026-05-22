"""Backward-compatible shim for the old Qwen-specific module name."""

from .hf_chat import HFChatModel, QwenModel

__all__ = ["HFChatModel", "QwenModel"]
