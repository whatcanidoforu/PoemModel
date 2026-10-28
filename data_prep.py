#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_prep.py —— 把 chinese-poetry 的「全唐诗 / 全宋诗」整理成训练数据。

流程：
  1. 读取 全唐诗/poet.tang.*.json 与 poet.song.*.json
  2. 按标点切句，只保留「每句 5 个汉字」的五言诗
  3. 用 OpenCC 把繁体转成简体
  4. 拼成 <bos> <title> 标题 <sep> 诗句 <end> 的字符流
  5. 统计字符表，保存 data/train.bin、data/val.bin、data/meta.json

用法：
  python data_prep.py --data-root "D:\\MyTools\\zlf\\chinese-poetry\\全唐诗"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

import numpy as np

try:
    from opencc import OpenCC
except ImportError:
    print("[错误] 缺少 opencc，请先安装：pip install opencc-python-reimplemented")
    sys.exit(1)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PUNCT = set("，。！？；：、,.!?;:()（）《》〈〉「」『』“”‘’[]【】…—· \t\r\n\u3000")
SPECIALS = ["<pad>", "<unk>", "<bos>", "<end>", "<title>", "<sep>"]
PAD, UNK, BOS, END, TITLE, SEP = range(len(SPECIALS))

# 五言诗的上下限（句数）
MIN_CLAUSES, MAX_CLAUSES = 2, 64
# 标题最长保留多少字（避免极端长标题）
MAX_TITLE_LEN = 24


def is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def split_clauses(paragraphs):
    """把 chinese-poetry 的 paragraphs 按标点切成一个个「句」。"""
    clauses = []
    for para in paragraphs or []:
        buf = []
        for ch in para:
            if ch in PUNCT:
                if buf:
                    clauses.append("".join(buf))
                    buf = []
            else:
                buf.append(ch)
        if buf:
            clauses.append("".join(buf))
    return clauses


def build_body(clauses):
    """把句子拼回带标点的诗：五言绝句 -> a，b。c，d。"""
    n = len(clauses)
    out = []
    for i, cl in enumerate(clauses):
        out.append(cl)
        if i == n - 1 or i % 2 == 1:
            out.append("。")
        else:
            out.append("，")
    return "".join(out)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"D:\MyTools\zlf\chinese-poetry\全唐诗",
                    help="chinese-poetry 仓库里『全唐诗』文件夹的路径")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--min-count", type=int, default=2, help="字符至少出现几次才进词表，其余归为 <unk>")
    ap.add_argument("--no-song", action="store_true", help="只训练全唐诗，不含全宋诗")
    ap.add_argument("--val-ratio", type=float, default=0.02, help="验证集比例")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tang_files = sorted(glob.glob(os.path.join(args.data_root, "poet.tang.*.json")))
    song_files = [] if args.no_song else sorted(glob.glob(os.path.join(args.data_root, "poet.song.*.json")))
    files = tang_files + song_files
    if not files:
        print(f"[错误] 在 {args.data_root} 下没找到 poet.tang.*.json / poet.song.*.json")
        sys.exit(1)
    print(f"[1/5] 找到 {len(tang_files)} 个唐诗文件、{len(song_files)} 个宋诗文件")

    cc = OpenCC("t2s")

    # 边读边过滤，避免把所有诗都堆在内存里
    kept = []          # (title, body, n_clauses)
    seen = set()
    n_total = 0
    n_tang = n_song = 0
    for fi, path in enumerate(files, 1):
        name = os.path.basename(path)
        is_tang = "tang" in name
        try:
            with open(path, encoding="utf-8") as fh:
                poems = json.load(fh)
        except Exception as e:  # 损坏文件跳过
            print(f"  [跳过] {name}: {e}")
            continue
        if not isinstance(poems, list):
            continue
        for p in poems:
            n_total += 1
            if is_tang:
                n_tang += 1
            else:
                n_song += 1
            clauses = split_clauses(p.get("paragraphs"))
            # 只保留五言：每一句都是 5 个汉字
            if not (MIN_CLAUSES <= len(clauses) <= MAX_CLAUSES):
                continue
            if not all(len(c) == 5 and all(is_cjk(ch) for ch in c) for c in clauses):
                continue
            title = (p.get("title") or "").strip()
            title = "".join(ch for ch in title if ch not in "\r\n\t")
            if not title:
                title = "无题"
            if len(title) > MAX_TITLE_LEN:
                title = title[:MAX_TITLE_LEN]
            # 繁 -> 简
            title_s = cc.convert(title)
            clauses_s = [cc.convert(c) for c in clauses]
            if not all(len(c) == 5 and all(is_cjk(ch) for ch in c) for c in clauses_s):
                continue
            body = build_body(clauses_s)
            key = (title_s, body)
            if key in seen:
                continue
            seen.add(key)
            kept.append((title_s, body, len(clauses_s)))
        if fi % 50 == 0 or fi == len(files):
            print(f"  已处理 {fi}/{len(files)} 个文件，保留 {len(kept)} 首")

    print(f"[2/5] 原始 {n_total} 首（唐诗 {n_tang}、宋诗 {n_song}），五言诗保留 {len(kept)} 首")
    if len(kept) < 1000:
        print("[错误] 语料太少，检查 --data-root 是否正确")
        sys.exit(1)

    # 字符表（标题 + 正文），低频字归 <unk>
    counter = Counter()
    for title, body, _ in kept:
        counter.update(title)
        counter.update(body)
    chars = sorted([ch for ch, c in counter.items() if c >= args.min_count and is_cjk(ch)])
    punct_chars = sorted([ch for ch in PUNCT if ch in counter])  # 保留正文用到的标点
    itos = SPECIALS + chars + punct_chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    print(f"[3/5] 字符表大小 {len(itos)}（汉字 {len(chars)}，标点 {len(punct_chars)}）")

    # 划分训练 / 验证
    rng = random.Random(args.seed)
    order = list(range(len(kept)))
    rng.shuffle(order)
    n_val = max(1, int(len(kept) * args.val_ratio))
    val_idx = set(order[:n_val])
    train_poems = [kept[i] for i in range(len(kept)) if i not in val_idx]
    val_poems = [kept[i] for i in range(len(kept)) if i in val_idx]
    print(f"[4/5] 训练 {len(train_poems)} 首，验证 {len(val_poems)} 首")

    def encode(poems):
        ids = []
        for title, body, _ in poems:
            ids.append(BOS)
            ids.append(TITLE)
            ids.extend(stoi.get(ch, UNK) for ch in title)
            ids.append(SEP)
            ids.extend(stoi.get(ch, UNK) for ch in body)
            ids.append(END)
        return np.array(ids, dtype=np.uint16)

    train_ids = encode(train_poems)
    val_ids = encode(val_poems)
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    print(f"[5/5] 训练 token {len(train_ids):,}，验证 token {len(val_ids):,}")
    print(f"      已保存：{train_path}")
    print(f"      已保存：{val_path}")

    clause_dist = Counter(n for _, _, n in kept)
    stats = {
        "raw_poems": n_total,
        "raw_tang": n_tang,
        "raw_song": n_song,
        "kept_poems": len(kept),
        "train_poems": len(train_poems),
        "val_poems": len(val_poems),
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "vocab_size": len(itos),
        "hanzi_vocab_size": len(chars),
        "clause_count_dist": {str(k): v for k, v in clause_dist.most_common(12)},
        "data_root": args.data_root,
        "include_song": not args.no_song,
    }
    meta = {
        "itos": itos,
        "specials": {"pad": PAD, "unk": UNK, "bos": BOS, "end": END, "title": TITLE, "sep": SEP},
        "block_size": 192,
        "stats": stats,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # 顺手存 20 首样例，方便肉眼检查质量
    sample_path = os.path.join(args.out_dir, "samples.txt")
    with open(sample_path, "w", encoding="utf-8") as fh:
        for title, body, _ in train_poems[:20]:
            fh.write(f"《{title}》\n{body}\n\n")
    print(f"      样例：{sample_path}")
    print("完成！")


if __name__ == "__main__":
    main()
