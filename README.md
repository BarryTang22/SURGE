# SURGE: Surprise-Guided Token Reduction

SURGE prunes redundant video tokens in Vision-Language Models by computing temporal surprise scores. Tokens that are predictable from previous frames are dropped, keeping only the informative ones.

## How it works

1. Compute per-token surprise via constant-velocity prediction with drift correction
2. Keep the top-ρ fraction of tokens by surprise score (frame 0 always kept)
3. Remove pruned tokens from the multimodal sequence before the LLM forward pass

## Quick start

```bash
pip install transformers torch qwen-vl-utils scipy
```

```python
from surge.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from surge.surge_core import SurgeConfig

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
model.set_surge_config(SurgeConfig(rho=0.25))  # keep 25% of video tokens
```

Then use `model.generate()` as usual — pruning happens automatically during the prefill pass.

## Inference script

```bash
python scripts/run_inference.py --video path/to/video.mp4 --query "What happens?" --rho 0.25
```

## End-to-end test

```bash
python tests/test_surge_e2e.py
```

Downloads a sample video and runs baseline vs SURGE at rho=1.0 and rho=0.25, verifying correctness and reporting speedup.

## Key files

| File | Description |
|------|-------------|
| `surge/surge_core.py` | Backbone-agnostic SURGE algorithm |
| `surge/modeling_qwen2_5_vl.py` | Qwen2.5-VL integration |
| `scripts/run_inference.py` | CLI inference script |
| `tests/test_surge_e2e.py` | End-to-end test |

## SurgeConfig parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho` | 0.25 | Fraction of video tokens to retain |
| `ema_gamma` | 0.9 | EMA smoothing for surprise curve |
| `ema_var_decay` | 0.9 | Decay for running variance normalization |
| `min_peak_separation` | 8 | Minimum frames between detected peaks |
| `enable_drift_correction` | True | Spatial drift correction via OLS |
| `enable_variance_norm` | True | Variance-normalized surprise scores |
