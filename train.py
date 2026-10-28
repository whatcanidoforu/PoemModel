#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py —— 训练五言诗字符级 GPT。

示例：
  python train.py --max-steps 8000 --batch-size 32
  python train.py --resume checkpoints/last.pt --max-steps 12000   # 接着训

训练日志会同时打印到屏幕并写入 checkpoints/train.log。
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from model import GPT, GPTConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "checkpoints"))
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--sample-interval", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default="")
    return ap.parse_args()


def load_data(data_dir):
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    train = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)
    if len(train) == 0 or len(val) == 0:
        raise SystemExit("data/train.bin 或 val.bin 为空，请先运行 data_prep.py")
    return meta, train, val


def get_batch(data, block_size, batch_size, device, gen):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def get_lr(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if step >= args.max_steps:
        return args.min_lr
    ratio = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def make_autocast(device):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def decode(itos, ids):
    return "".join(itos[i] for i in ids)


@torch.no_grad()
def estimate_loss(model, data, args, gen, device):
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(data, model.cfg.block_size, args.batch_size, device, gen)
        with make_autocast(device):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_poem(model, itos, meta, device, title="秋夜", max_new_tokens=90):
    """训练过程中随便看看效果：给标题，自由续写。"""
    specials = meta["specials"]
    stoi = {ch: i for i, ch in enumerate(itos)}
    ids = [specials["bos"], specials["title"]]
    ids += [stoi.get(ch, specials["unk"]) for ch in title]
    ids.append(specials["sep"])
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.9,
                         top_k=40, end_id=specials["end"])
    model.train()
    txt = decode(itos, out[0].tolist())
    return txt.replace("<title>", "").replace("<sep>", "\n").replace("<bos>", "").replace("<end>", "")


def save_ckpt(path, model, args, step, val_loss, optimizer=None):
    ckpt = {
        "model": model.state_dict(),
        "config": vars(model.cfg),
        "step": step,
        "val_loss": val_loss,
        "args": vars(args),
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")
    log_file = open(log_path, "a", encoding="utf-8")
    start_time = time.time()

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    meta, train_data, val_data = load_data(args.data_dir)
    itos = meta["itos"]
    cfg = GPTConfig(
        vocab_size=len(itos),
        block_size=meta["block_size"],
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=args.dropout,
    )
    device = args.device
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    gen = torch.Generator().manual_seed(args.seed)

    model = GPT(cfg).to(device)
    n_params = model.num_params()
    log("=" * 70)
    log(f"设备: {device}  |  模型参数: {n_params/1e6:.2f}M  |  词表: {cfg.vocab_size}  |  上下文: {cfg.block_size}")
    log(f"训练 token: {len(train_data):,}  |  验证 token: {len(val_data):,}")
    log(f"预计每 epoch token: {len(train_data):,}  |  每步 token: {args.batch_size*cfg.block_size:,}")
    log("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)

    start_step = 0
    best_val = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_val = ckpt.get("val_loss", float("inf"))
        log(f"从 {args.resume} 恢复训练，起始 step={start_step}，历史最优 val_loss={best_val:.4f}")

    model.train()
    t0 = time.time()
    tokens_seen = 0
    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, cfg.block_size, args.batch_size, device, gen)
        tokens_seen += x.numel()

        with make_autocast(device):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0 or step == args.max_steps - 1:
            elapsed = time.time() - t0
            tok_s = tokens_seen / max(elapsed, 1e-6)
            log(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{tok_s:,.0f} tok/s | 已用 {elapsed/60:.1f} min")

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            val_loss = estimate_loss(model, val_data, args, gen, device)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                save_ckpt(os.path.join(args.out_dir, "best.pt"), model, args, step, val_loss)
            save_ckpt(os.path.join(args.out_dir, "last.pt"), model, args, step, val_loss, optimizer)
            log(f"  [eval] step {step} | val_loss {val_loss:.4f} | best {best_val:.4f}"
                f"{'  <- 保存 best.pt' if improved else ''}")

        if step % args.sample_interval == 0 and step > 0:
            txt = sample_poem(model, itos, meta, device)
            log("  [样例] " + txt[:160].replace("\n", " / "))

    total = time.time() - start_time
    log(f"训练结束，用时 {total/60:.1f} 分钟，最优 val_loss={best_val:.4f}")
    if device == "cuda":
        log(f"显存峰值: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    log(f"模型保存在 {os.path.join(args.out_dir, 'best.pt')}")
    log_file.close()


if __name__ == "__main__":
    main()