"""Stateless Qwen3.5-MoE building blocks for the Volta compatibility path.

The package deliberately keeps checkpoint tensors compressed.  Model wiring,
which lives outside this package, chooses the appropriate operation for each
``Weight`` instance.
"""

from .weights import QuantActivation, Weight

__all__ = ["QuantActivation", "Weight"]
