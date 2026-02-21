"""
SURGE: Surprise-Guided Token Reduction for efficient video understanding in VLMs.

Usage:
    from surge.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from surge.surge_core import SurgeConfig

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.set_surge_config(SurgeConfig(rho=0.25))
"""

from .surge_core import SurgeConfig, SurgeOutput, compute_surge, apply_surge_to_sequence
from .surge_star import SurgeStarConfig, apply_surge_star
from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel

__all__ = [
    "SurgeConfig",
    "SurgeOutput",
    "SurgeStarConfig",
    "compute_surge",
    "apply_surge_to_sequence",
    "apply_surge_star",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
]
