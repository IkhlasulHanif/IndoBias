"""Compatibility helpers for lmppl with newer transformers releases."""

from __future__ import annotations

from typing import Callable


def _patch_init(cls) -> None:
    original_init = cls.__init__

    if getattr(original_init, "_indobias_auth_patch", False):
        return

    def patched_init(self, *args, **kwargs):
        kwargs.pop("use_auth_token", None)
        return original_init(self, *args, **kwargs)

    patched_init._indobias_auth_patch = True
    cls.__init__ = patched_init


def patch_lmppl_transformers_compat() -> None:
    """Strip legacy kwargs forwarded by lmppl into model constructors.

    `lmppl` still forwards `use_auth_token` into several masked-LM model
    constructors. Recent `transformers` versions no longer accept that kwarg in
    those constructors, so model loading fails before evaluation starts.
    """

    try:
        from transformers import (
            AlbertForMaskedLM,
            BertForMaskedLM,
            CamembertForMaskedLM,
            DebertaForMaskedLM,
            DebertaV2ForMaskedLM,
            DistilBertForMaskedLM,
            ElectraForMaskedLM,
            RobertaForMaskedLM,
            XLMRobertaForMaskedLM,
        )
    except Exception:
        return

    for cls in (
        AlbertForMaskedLM,
        BertForMaskedLM,
        CamembertForMaskedLM,
        DebertaForMaskedLM,
        DebertaV2ForMaskedLM,
        DistilBertForMaskedLM,
        ElectraForMaskedLM,
        RobertaForMaskedLM,
        XLMRobertaForMaskedLM,
    ):
        _patch_init(cls)
