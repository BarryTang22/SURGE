"""
End-to-end test for SURGE with Qwen2.5-VL.

Downloads a real video (Chromecast ad, 15s) and tests:
  1. Baseline (no SURGE) generates a coherent answer
  2. SURGE rho=1.0 produces identical output to baseline
  3. SURGE rho=0.25 produces a reasonable answer with fewer tokens
  4. Surprise curve analysis
"""

import sys
import os
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

VIDEO_URL = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"
VIDEO_PATH = "/tmp/surge_test_video.mp4"


def download_video():
    """Download test video if not already cached."""
    if os.path.exists(VIDEO_PATH):
        print(f"Video already cached: {VIDEO_PATH}")
        return
    print(f"Downloading test video...")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
    print(f"Downloaded to {VIDEO_PATH}")


def run_test():
    """Run all SURGE end-to-end tests."""
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    from surge.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from surge.surge_core import SurgeConfig, compute_surge

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    download_video()

    query = "Describe what happens in this video in detail."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": VIDEO_PATH, "fps": 2.0},
                {"type": "text", "text": query},
            ],
        }
    ]

    print(f"\nLoading model: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    input_len = inputs["input_ids"].shape[1]
    video_token_count = (inputs["input_ids"] == model.config.video_token_id).sum().item()
    print(f"Input sequence length: {input_len}")
    print(f"Video tokens: {video_token_count}")
    print(f"Video grid THW: {inputs['video_grid_thw']}")

    gen_kwargs = dict(max_new_tokens=512, do_sample=False)

    print("\n" + "=" * 60)
    print("TEST 1: Baseline (no SURGE)")
    print("=" * 60)
    model.disable_surge()
    t0 = time.time()
    with torch.no_grad():
        output_ids_baseline = model.generate(**inputs, **gen_kwargs)
    t_baseline = time.time() - t0
    response_baseline = processor.tokenizer.decode(
        output_ids_baseline[0, input_len:], skip_special_tokens=True
    )
    print(f"Time: {t_baseline:.2f}s")
    print(f"Response:\n{response_baseline}\n")

    print("=" * 60)
    print("TEST 2: SURGE rho=1.0 (should match baseline)")
    print("=" * 60)
    model.set_surge_config(SurgeConfig(rho=1.0))
    t0 = time.time()
    with torch.no_grad():
        output_ids_rho1 = model.generate(**inputs, **gen_kwargs)
    t_rho1 = time.time() - t0
    response_rho1 = processor.tokenizer.decode(
        output_ids_rho1[0, input_len:], skip_special_tokens=True
    )
    print(f"Time: {t_rho1:.2f}s")
    match = response_baseline == response_rho1
    print(f"Exact match with baseline: {match}")
    if not match:
        baseline_tokens = output_ids_baseline[0, input_len:].tolist()
        rho1_tokens = output_ids_rho1[0, input_len:].tolist()
        min_len = min(len(baseline_tokens), len(rho1_tokens))
        first_diff = next((i for i in range(min_len) if baseline_tokens[i] != rho1_tokens[i]), min_len)
        print(f"  First token difference at position {first_diff}/{min_len}")

    print("\n" + "=" * 60)
    print("TEST 3: SURGE rho=0.25 (aggressive pruning)")
    print("=" * 60)
    model.set_surge_config(SurgeConfig(rho=0.25))
    t0 = time.time()
    with torch.no_grad():
        output_ids_rho025 = model.generate(**inputs, **gen_kwargs)
    t_rho025 = time.time() - t0
    response_rho025 = processor.tokenizer.decode(
        output_ids_rho025[0, input_len:], skip_special_tokens=True
    )
    print(f"Time: {t_rho025:.2f}s")
    print(f"Response:\n{response_rho025}\n")

    print("=" * 60)
    print("TEST 4: Surprise curve analysis")
    print("=" * 60)
    model.disable_surge()
    with torch.no_grad():
        pixel_values_videos = inputs["pixel_values_videos"].type(model.visual.dtype)
        video_grid_thw = inputs["video_grid_thw"]
        video_embeds = model.model.get_video_features(pixel_values_videos, video_grid_thw)

    spatial_merge_size = model.config.vision_config.spatial_merge_size
    for vid_idx, vid_embed in enumerate(video_embeds):
        T = video_grid_thw[vid_idx][0].item()
        H = video_grid_thw[vid_idx][1].item() // spatial_merge_size
        W = video_grid_thw[vid_idx][2].item() // spatial_merge_size
        m = H * W

        surge_out = compute_surge(vid_embed, T, H, W, SurgeConfig(rho=0.25))

        print(f"Video {vid_idx}: T={T}, H'={H}, W'={W}, m={m}")
        print(f"  Total tokens: {T * m}, Kept: {surge_out.keep_mask.sum().item()} ({surge_out.keep_mask.sum().item() / (T * m) * 100:.1f}%)")
        print(f"  Peak frames: {surge_out.peak_indices.tolist()}")
        print(f"  Event intervals: {surge_out.event_intervals}")

        curve = surge_out.smoothed_curve.cpu().numpy()
        print(f"  Smoothed surprise curve:")
        max_val = max(curve) if max(curve) > 0 else 1
        for t in range(T):
            bar_len = int(curve[t] / max_val * 40)
            peak_marker = " <-- PEAK" if t in surge_out.peak_indices else ""
            print(f"    t={t:2d}: {'█' * bar_len}{'░' * (40 - bar_len)} ({curve[t]:.2f}){peak_marker}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline tokens:    {video_token_count}")
    print(f"SURGE rho=0.25:     {surge_out.keep_mask.sum().item()} tokens kept")
    print(f"Token reduction:    {(1 - surge_out.keep_mask.sum().item() / video_token_count) * 100:.1f}%")
    print(f"Baseline time:      {t_baseline:.2f}s")
    print(f"SURGE rho=1.0 time: {t_rho1:.2f}s")
    print(f"SURGE rho=0.25 time:{t_rho025:.2f}s")
    print(f"Speedup:            {t_baseline / t_rho025:.2f}x")

    print("\nAll tests completed!")


if __name__ == "__main__":
    run_test()
