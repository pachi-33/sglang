"""Qwen3.5 NVFP4 expert-pack format, offline tools, and runtime store."""

from __future__ import annotations

from typing import Any

from .format import (
    ALIGNMENT,
    COMPONENTS,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPERT_LAYERS,
    EXPERTS_PER_LAYER,
    FORMAT_ID,
    MANIFEST_FILENAME,
    PACK_FILENAME,
    PACK_SIZE,
    PAYLOAD_SIZE,
    RECORD_COUNT,
    RECORD_STRIDE,
    ComponentSpec,
    ExpertPackManifest,
    RecordInfo,
    load_manifest,
    record_index,
    record_key,
    record_offset,
)
from .store import ExpertLease, ExpertOffloadConfig, ExpertPackStore


def build_expert_pack(*args: Any, **kwargs: Any) -> ExpertPackManifest:
    # Keep ``python -m ...expert_pack.build`` free of runpy import warnings.
    from .build import build_expert_pack as implementation

    return implementation(*args, **kwargs)


def validate_expert_pack(*args: Any, **kwargs: Any) -> Any:
    from .validate import validate_expert_pack as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "ALIGNMENT",
    "COMPONENTS",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_INDEX_SHA256",
    "EXPERTS_PER_LAYER",
    "EXPERT_LAYERS",
    "FORMAT_ID",
    "MANIFEST_FILENAME",
    "PACK_FILENAME",
    "PACK_SIZE",
    "PAYLOAD_SIZE",
    "RECORD_COUNT",
    "RECORD_STRIDE",
    "ComponentSpec",
    "ExpertPackManifest",
    "ExpertLease",
    "ExpertOffloadConfig",
    "ExpertPackStore",
    "RecordInfo",
    "build_expert_pack",
    "load_manifest",
    "record_index",
    "record_key",
    "record_offset",
    "validate_expert_pack",
]
