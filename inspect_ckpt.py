#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查看 checkpoint 文件内容，以可读文本形式打印。"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import json

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    ap = argparse.ArgumentParser(description="查看 checkpoint 内容")
    ap.add_argument("ckpt", nargs="?", default="checkpoints/best.pt", help="checkpoint 路径")
    ap.add_argument("--detail", action="store_true", help="显示每层权重的形状和部分数值")
    ap.add_argument("--meta", default="data/meta.json", help="meta.json 路径（用于显示词汇表大小）")
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"找不到文件：{args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    print("=" * 60)
    print(f"  Checkpoint: {args.ckpt}")
    print("=" * 60)

    print(f"\n训练步数 (step):     {ckpt.get('step', 'N/A')}")
    print(f"验证损失 (val_loss): {ckpt.get('val_loss', 'N/A'):.4f}")

    print("\n--- 模型结构 (config) ---")
    cfg = ckpt.get("config", {})
    total_params = 0
    for k, v in cfg.items():
        print(f"  {k:16s}: {v}")
    vocab_size = cfg.get("vocab_size", "?")
    n_embd = cfg.get("n_embd", 0)
    n_layer = cfg.get("n_layer", 0)
    block_size = cfg.get("block_size", 0)
    n_head = cfg.get("n_head", 0)

    print("\n--- 训练参数 (args) ---")
    train_args = ckpt.get("args", {})
    for k, v in train_args.items():
        print(f"  {k:16s}: {v}")

    model_state = ckpt.get("model", {})
    print(f"\n--- 模型权重 ({len(model_state)} 个参数矩阵) ---")
    print(f"{'层名':50s} {'形状':20s} {'参数量':>10s}")
    print("-" * 85)

    for name, tensor in model_state.items():
        shape = str(tuple(tensor.shape))
        num = tensor.numel()
        total_params += num
        print(f"  {name:48s} {shape:20s} {num:>10,}")

    print("-" * 85)
    print(f"  {'总计':48s} {'':20s} {total_params:>10,}")

    if total_params >= 1_000_000:
        print(f"\n  约 {total_params/1_000_000:.1f}M 参数（百万）")

    if args.detail:
        print("\n" + "=" * 60)
        print("  各层权重数值预览（前 5 个元素）")
        print("=" * 60)
        for name, tensor in model_state.items():
            flat = tensor.flatten()[:5]
            val_str = ", ".join(f"{v:.6f}" for v in flat.tolist())
            print(f"\n  {name} [{tuple(tensor.shape)}]")
            print(f"    均值={tensor.float().mean():.6f}  标准差={tensor.float().std():.6f}")
            print(f"    前5个: [{val_str}, ...]")

    print("\n--- 词汇表 ---")
    if os.path.exists(args.meta):
        with open(args.meta, encoding="utf-8") as fh:
            meta = json.load(fh)
        itos = meta["itos"]
        print(f"  词汇量: {len(itos)}")
        specials = meta.get("specials", {})
        print(f"  特殊标记: {json.dumps(specials, ensure_ascii=False)}")
        normal_chars = [ch for ch in itos if not ch.startswith("<")]
        print(f"  普通汉字: {len(normal_chars)} 个")
        print(f"  前50个汉字: {''.join(normal_chars[:50])}")
    else:
        print(f"  (找不到 {args.meta}，跳过)")

    print()


if __name__ == "__main__":
    main()