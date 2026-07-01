# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: Count MOTIP parameters and estimate Deformable DETR forward FLOPs (single-scale dummy input).

"""
Usage (from repo root, in conda env MOTIP):

    conda activate MOTIP
    python tools/count_params_flops.py --config-path ./configs/r50_deformable_detr_motip_dancetrack_v3pts.yaml

Optional profiling shape (default 640x640, batch 1):

    python tools/count_params_flops.py --config-path ./configs/xxx.yaml --profile-height 800 --profile-width 1333

Other CLI flags are the same as train.py (e.g. --batch-size) and override yaml via update_config.

FLOPs come from torch.profiler(with_flops=True); custom CUDA (e.g. MSDeformableAttention) may not
report full cost, so treat the number as a lower bound / rough estimate unless you verify separately.
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from configs.util import load_super_config, update_config
from models.motip import build as build_motip
from runtime_option import runtime_option
from utils.misc import yaml_to_dict, set_seed
from utils.nested_tensor import NestedTensor


def _parse_profile_argv() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile-height", type=int, default=640)
    pre.add_argument("--profile-width", type=int, default=640)
    pre.add_argument("--profile-batch-size", type=int, default=1)
    pre.add_argument("--skip-flops", action="store_true")
    args, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return args


def _count_params(module: torch.nn.Module, trainable_only: bool) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _load_config():
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)
    if opt.super_config_path is not None:
        cfg = load_super_config(cfg, opt.super_config_path)
    else:
        super_path = cfg.get("SUPER_CONFIG_PATH")
        cfg = load_super_config(cfg, super_path if super_path else None)
    cfg = update_config(config=cfg, option=opt)
    return cfg


def main():
    prof_args = _parse_profile_argv()
    cfg = _load_config()
    set_seed(cfg.get("SEED", 42))

    device_s = cfg.get("DEVICE", "cuda")
    if device_s == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Warning: DEVICE is cuda but CUDA is unavailable; using CPU.")
    else:
        device = torch.device(device_s)

    cfg["DEVICE"] = str(device)
    model, _ = build_motip(config=cfg)
    model = model.to(device)
    model.eval()

    total = _count_params(model, trainable_only=False)
    total_train = _count_params(model, trainable_only=True)
    detr = _count_params(model.detr, trainable_only=False)
    detr_train = _count_params(model.detr, trainable_only=True)
    motip_rest = total - detr
    motip_rest_train = total_train - detr_train

    print("=== Parameters ===")
    print(f"  Total:              {total / 1e6:.3f} M  (trainable: {total_train / 1e6:.3f} M)")
    print(f"  DETR submodule:     {detr / 1e6:.3f} M  (trainable: {detr_train / 1e6:.3f} M)")
    print(f"  MOTIP (non-DETR):   {motip_rest / 1e6:.3f} M  (trainable: {motip_rest_train / 1e6:.3f} M)")

    if prof_args.skip_flops:
        print("\nSkipped FLOPs (--skip-flops).")
        return

    if model.only_detr:
        print("\nONLY_DETR is True; profiling still runs the DETR backbone+transformer only.")

    H, W = prof_args.profile_height, prof_args.profile_width
    B = prof_args.profile_batch_size
    if B < 1:
        raise ValueError("--profile-batch-size must be >= 1")

    tensors = torch.randn(B, 3, H, W, device=device, dtype=torch.float32)
    mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    frames = NestedTensor(tensors=tensors, mask=mask)

    with torch.no_grad():
        with torch.profiler.profile(with_flops=True) as prof:
            model.detr(frames)

    events = prof.key_averages()
    flops = sum(e.flops for e in events if e.flops is not None)

    print("\n=== DETR forward FLOPs (profiler sum) ===")
    print(f"  Input: NestedTensor  tensors (B,3,H,W)=({B},3,{H},{W}), mask (B,H,W)")
    print(f"  Summed FLOPs (ops that report): {flops / 1e9:.3f} G")
    print("  Note: custom CUDA ops may be under-counted; compare with papers/tools if needed.")

    top = sorted((e for e in events if e.flops), key=lambda x: x.flops, reverse=True)[:10]
    if top:
        print("\n  Top ops by reported FLOPs:")
        for e in top:
            print(f"    {e.key:50s}  {e.flops / 1e9:.4f} G")


if __name__ == "__main__":
    main()
