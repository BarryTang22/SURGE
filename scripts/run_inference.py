"""
Example inference script for SURGE + Qwen2.5-VL.

Usage:
    python scripts/run_inference.py --video path/to/video.mp4 --query "What happens in this video?"

Requires:
    pip install transformers torch qwen-vl-utils
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoProcessor

from surge.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from surge.surge_core import SurgeConfig


def main():
    parser = argparse.ArgumentParser(description="SURGE inference with Qwen2.5-VL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to video file")
    parser.add_argument("--query", type=str, default="Describe this video in detail.",
                        help="Query text")
    parser.add_argument("--rho", type=float, default=0.25,
                        help="SURGE retention ratio (0.25 = keep 25%% of tokens)")
    parser.add_argument("--no-surge", action="store_true",
                        help="Disable SURGE (baseline)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    if not args.no_surge:
        surge_config = SurgeConfig(rho=args.rho)
        model.set_surge_config(surge_config)
        print(f"SURGE enabled: rho={args.rho} (keeping {args.rho*100:.0f}% of video tokens)")
    else:
        print("SURGE disabled (baseline mode)")

    processor = AutoProcessor.from_pretrained(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": args.video},
                {"type": "text", "text": args.query},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    print(f"Input sequence length: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\nQuery: {args.query}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
