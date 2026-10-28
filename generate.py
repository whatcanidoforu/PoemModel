#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate.py —— 用训练好的模型写五言诗（带格式约束）。

示例：
  python generate.py --title "秋夜"
  python generate.py --titles "秋夜,送别,山中" --count 2 --temperature 0.9
  python generate.py --title "月下" --clauses 8          # 五言律诗（8 句）
  python generate.py --free --title "秋天"                # 自由发挥，不强制五言

约束解码原理：
  每句强制采样 5 个汉字（屏蔽标点/特殊字符），然后强制补上「，」或「。」，
  所以无论模型多不靠谱，输出的格式一定是规整的五言诗。
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys

import torch

from model import load_checkpoint

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))


def is_hanzi(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "best.pt"))
    ap.add_argument("--meta", default=os.path.join(HERE, "data", "meta.json"))
    ap.add_argument("--title", default="秋夜", help="诗题，如：秋夜")
    ap.add_argument("--titles", default="", help="多个标题用逗号分隔，如：秋夜,送别,山中")
    ap.add_argument("--clauses", type=int, default=4, help="句数：4=五绝，8=五律")
    ap.add_argument("--count", type=int, default=1, help="每个标题写几首")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--free", action="store_true", help="不强制五言格式，自由续写")
    ap.add_argument("--max-tokens", type=int, default=80, help="自由模式最多生成多少字")
    ap.add_argument("--out", default="", help="把结果写入文件")
    return ap.parse_args()


def encode_title(meta, title):
    stoi = {ch: i for i, ch in enumerate(meta["itos"])}
    sp = meta["specials"]
    ids = [sp["bos"], sp["title"]]
    ids += [stoi.get(ch, sp["unk"]) for ch in title]
    ids.append(sp["sep"])
    return ids


@torch.no_grad()
def write_poem(model, meta, title, n_clauses, temperature, top_k, rep_penalty, device):
    itos = meta["itos"]
    sp = meta["specials"]
    stoi = {ch: i for i, ch in enumerate(itos)}
    comma_id = stoi.get("，")
    period_id = stoi.get("。")
    hanzi_ids = [i for i, ch in enumerate(itos) if is_hanzi(ch)]
    hanzi_tensor = torch.tensor(hanzi_ids, dtype=torch.long, device=device)
    neg_inf = -1e9

    seq = encode_title(meta, title)
    generated = []  # 本首诗已生成的汉字 id（用于重复惩罚）
    clauses_text = []  # 生成出的每一句（5 个字）

    for k in range(n_clauses):
        clause = []
        for _ in range(5):
            idx = torch.tensor([seq[-model.cfg.block_size:]], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits, _ = model(idx)
            logits = logits[0, -1, :].float()

            # 重复惩罚（CTRL 式）
            if generated and rep_penalty != 1.0:
                for t in set(generated):
                    logits[t] = logits[t] / rep_penalty if logits[t] > 0 else logits[t] * rep_penalty

            # 只允许汉字；另外屏蔽上一句刚出现的字，减少「卡壳」重复
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
        # 强制标点：句中对「，」，句末对「。」
        punct = period_id if (k % 2 == 1 or k == n_clauses - 1) else comma_id
        seq.append(punct)

    # 排版：两句一行
    lines = []
    for k in range(0, n_clauses, 2):
        if k + 1 < n_clauses:
            lines.append(clauses_text[k] + "，" + clauses_text[k + 1] + "。")
        else:
            lines.append(clauses_text[k] + "。")
    return "\n".join(lines)


@torch.no_grad()
def write_free(model, meta, title, max_tokens, temperature, top_k, device):
    itos = meta["itos"]
    sp = meta["specials"]
    seq = encode_title(meta, title)
    idx = torch.tensor([seq], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature,
                         top_k=top_k, end_id=sp["end"])
    ids = out[0].tolist()
    # 只保留 <sep> 之后的内容
    if sp["sep"] in ids:
        ids = ids[ids.index(sp["sep"]) + 1:]
    text = "".join(itos[i] for i in ids if i != sp["end"])
    return text


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"找不到模型文件：{args.ckpt}，请先运行 train.py")

    with open(args.meta, encoding="utf-8") as fh:
        meta = json.load(fh)
    model, cfg, _ = load_checkpoint(args.ckpt, device=args.device)
    model.eval()

    titles = [t.strip() for t in args.titles.split(",") if t.strip()] if args.titles else [args.title]
    outputs = []
    for title in titles:
        for i in range(args.count):
            if args.free:
                body = write_free(model, meta, title, args.max_tokens,
                                  args.temperature, args.top_k, args.device)
                text = f"《{title}》\n{body}"
            else:
                body = write_poem(model, meta, title, args.clauses, args.temperature,
                                  args.top_k, args.repetition_penalty, args.device)
                text = f"《{title}》\n{body}"
            outputs.append(text)
            print(text)
            print()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(outputs) + "\n")
        print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()