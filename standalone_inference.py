#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
standalone_inference.py —— 在任意项目中加载五言诗模型的最小示例。

使用方法：
  1. 把本文件和 model.py、checkpoints/best.pt、data/meta.json 放到同一个项目里
  2. python standalone_inference.py

在其他项目中复用你只需要：
  - model.py（GPT 模型定义）
  - checkpoints/best.pt（训练好的权重）
  - data/meta.json（词汇表）
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys

import torch

from model import load_checkpoint, GPTConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def is_hanzi(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


@torch.no_grad()
def generate_poem(model, meta, title: str, n_clauses: int = 4,
                  temperature: float = 0.8, top_k: int = 15,
                  rep_penalty: float = 1.15, device: str = "cpu"):
    """
    给定标题，生成一首五言诗（带格式约束）。

    参数：
      model:       GPT 模型实例
      meta:        meta.json 内容 (itos + specials)
      title:       诗题，如 "秋夜"
      n_clauses:   句数，4=五绝，8=五律
      temperature: 采样温度，越高越随机，0.8 是默认
      top_k:       top-k 采样，只从概率最高的 k 个候选里选
      rep_penalty: 重复惩罚，>1 减少重复字
      device:      "cuda" 或 "cpu"
    """
    itos = meta["itos"]
    sp = meta["specials"]
    stoi = {ch: i for i, ch in enumerate(itos)}

    comma_id = stoi.get("，")
    period_id = stoi.get("。")
    hanzi_ids = [i for i, ch in enumerate(itos) if is_hanzi(ch)]
    hanzi_tensor = torch.tensor(hanzi_ids, dtype=torch.long, device=device)
    neg_inf = -1e9

    # 编码标题
    seq = [sp["bos"], sp["title"]]
    seq += [stoi.get(ch, sp["unk"]) for ch in title]
    seq.append(sp["sep"])

    generated = []
    clauses_text = []

    for k in range(n_clauses):
        clause = []
        for _ in range(5):
            idx = torch.tensor([seq[-model.cfg.block_size:]], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                logits, _ = model(idx)
            logits = logits[0, -1, :].float()

            # 重复惩罚
            if generated and rep_penalty != 1.0:
                for t in set(generated):
                    logits[t] = logits[t] / rep_penalty if logits[t] > 0 \
                                else logits[t] * rep_penalty

            # 只允许汉字
            mask = torch.full_like(logits, neg_inf)
            mask[hanzi_tensor] = 0.0
            logits = logits + mask
            if clause:
                logits[clause[-1]] = neg_inf

            # top-k 截断
            kk = min(top_k, logits.numel())
            v, _ = torch.topk(logits, kk)
            logits[logits < v[-1]] = neg_inf
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).item()

            seq.append(nxt)
            clause.append(nxt)
            generated.append(nxt)

        clauses_text.append("".join(itos[i] for i in clause))
        punct = period_id if (k % 2 == 1 or k == n_clauses - 1) else comma_id
        seq.append(punct)

    # 排版
    lines = []
    for k in range(0, n_clauses, 2):
        if k + 1 < n_clauses:
            lines.append(clauses_text[k] + "，" + clauses_text[k + 1] + "。")
        else:
            lines.append(clauses_text[k] + "。")
    return "\n".join(lines)


# ============================================================
# 使用示例：在任意项目中调用
# ============================================================
if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))

    # ---- 步骤 1：加载模型（一行搞定）----
    ckpt_path = os.path.join(HERE, "checkpoints", "best.pt")
    meta_path = os.path.join(HERE, "data", "meta.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_checkpoint(ckpt_path, device=device)
    model.eval()

    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    print(f"模型已加载：{cfg.n_layer} 层, {sum(p.numel() for p in model.parameters())/1e6:.1f}M 参数, "
          f"词表 {cfg.vocab_size} 字, 设备 {device}")
    print()

    # ---- 步骤 2：生成诗词 ----
    titles = ["秋夜", "送别", "山中", "月下独酌"]

    for title in titles:
        poem = generate_poem(model, meta, title, n_clauses=4,
                             temperature=0.8, device=device)
        print(f"《{title}》")
        print(poem)
        print()