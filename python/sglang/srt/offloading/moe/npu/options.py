"""Host-only configuration boundary for Ascend expert offload modes."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

_CACHE_OPTIONS = frozenset(
    {
        "offline_stats_path",
        "offline_model_id",
        "decode_mask_capacity",
        "update_interval",
        "min_usage",
        "replacement_min_gain",
        "max_swaps",
        "usage_decay",
    }
)
_COMMON_OPTIONS = frozenset(
    {
        "mode",
        "cache_ratio",
        "host_pool_gib",
        "execution_slots",
        "native_extension_directory",
    }
)


def _finite_number(name, value):
    # bool is an int subclass, but never a meaningful memory/cache setting.
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    return value


@dataclass(frozen=True)
class AscendOffloadOptions:
    """Validate once before choosing the mixed or prefill-only lifecycle.

    ``values`` retains the supplied options so the existing mixed-mode policy
    keeps its defaults and relative offline-profile path resolution.
    """

    values: dict
    mode: str
    cache_ratio: float

    @property
    def prefill_only(self):
        return self.mode == "prefill_only"

    @classmethod
    def from_path(cls, path, *, model_id, disaggregation_mode):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Offload configuration must be a JSON object")
        unknown = set(values) - _COMMON_OPTIONS - _CACHE_OPTIONS
        if unknown:
            raise ValueError(
                "Unknown offload configuration keys: " + ", ".join(sorted(unknown))
            )

        mode = values.get("mode", "mixed")
        if not isinstance(mode, str) or mode not in ("mixed", "prefill_only"):
            raise ValueError("mode must be 'mixed' or 'prefill_only'")
        if mode == "prefill_only":
            if disaggregation_mode != "prefill":
                raise ValueError(
                    "prefill_only offload requires --disaggregation-mode prefill"
                )
            incompatible = set(values) & _CACHE_OPTIONS
            if incompatible:
                raise ValueError(
                    "prefill_only does not use cache/decode options: "
                    + ", ".join(sorted(incompatible))
                )
            default_ratio = 0.0
        else:
            default_ratio = 0.25 if model_id == "GLM-5.2" else 0.28125

        ratio = _finite_number("cache_ratio", values.get("cache_ratio", default_ratio))
        if mode == "prefill_only":
            if ratio != 0:
                raise ValueError("prefill_only requires cache_ratio=0")
        elif not 0 < ratio <= 1:
            raise ValueError("cache_ratio must be in (0, 1] for mixed mode")

        for name in (
            "host_pool_gib",
            "execution_slots",
            "decode_mask_capacity",
            "update_interval",
        ):
            if name in values and (type(values[name]) is not int or values[name] <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if "max_swaps" in values and (
            type(values["max_swaps"]) is not int or values["max_swaps"] < 0
        ):
            raise ValueError("max_swaps must be a nonnegative integer")
        for name in ("min_usage", "replacement_min_gain", "usage_decay"):
            if name in values:
                value = _finite_number(name, values[name])
                if value < 0 or (name == "usage_decay" and value > 1):
                    domain = "in [0, 1]" if name == "usage_decay" else "nonnegative"
                    raise ValueError(f"{name} must be {domain}")
        for name in (
            "offline_stats_path",
            "offline_model_id",
            "native_extension_directory",
        ):
            if name not in values:
                continue
            value = values[name]
            if value is None and name != "offline_model_id":
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")

        return cls(values=values, mode=mode, cache_ratio=float(ratio))
