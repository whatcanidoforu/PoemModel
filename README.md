# 五言诗 GPT（firstModel）

从零训练的中文五言诗生成模型：**1300 万参数**的字符级 GPT，用《全唐诗》+《全宋诗》里筛出的 **10.7 万首五言诗**训练，输入标题就能写出一首格式规整的五言绝句/律诗。

这个项目的目标不是"做出多强的模型"，而是**完整走一遍大模型的核心流程**：

> 数据清洗 → 字符级分词 → Transformer → 预训练 → 采样生成

全部代码纯 PyTorch、不到 800 行，不依赖任何现成的大模型框架。

---

## 1. 先看效果

```bash
python generate.py --title "月下"
```

实际生成结果（每次不同）：

```
《月下》
月上桂林中，清光自满空。
明珠随海近，素魄与天同。
```

```
《秋夜》
月在高梧上，秋生古木边。
孤灯寒更冷，一榻独无眠。
```

五言律诗（8 句）：

```bash
python generate.py --title "山中" --clauses 8
```

```
《山中》
一径入深坞，三年无俗情。
山中无客到，石上有人行。
水竹闲眠石，云泉冷漱冰。
何时寻道侣，来往复相迎。
```

> 生成时用了**约束解码**：每句强制 5 个汉字、两句一顿，所以格式一定是五言，不会跑偏。

---

## 2. 目录结构

```
firstModel/
├── data_prep.py    # 数据准备：筛选五言诗、繁体转简体、生成 train.bin / val.bin
├── model.py        # 迷你 GPT（6 层 / 384 维 / 6 头，约 1300 万参数）
├── train.py        # 训练脚本（bf16 混合精度 + 余弦学习率）
├── generate.py     # 生成脚本（标题 → 五言诗，带格式约束）
├── data/           # 生成的数据（meta.json / train.bin / val.bin / samples.txt）
├── checkpoints/    # 训练产物（best.pt / last.pt / train.log）
├── demo_poems.txt  # 一次生成的示例输出
└── README.md
```

---

## 3. 环境

| 项目 | 说明 |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop 8GB（Blackwell，sm_120） |
| Python | 3.11（conda 环境 `firstmodel`） |
| PyTorch | **2.11.0+cu128**（必须是 CUDA 12.8+，RTX 50 系才支持） |
| 其他 | `opencc-python-reimplemented`（仅数据预处理用，繁体转简体） |

5060 是 Blackwell 架构（sm_120），旧版 PyTorch 会直接报
`sm_120 is not compatible with the current PyTorch installation`。
本机可用的环境：`D:\conda\envs\firstmodel`（从 `weld_yolo` 克隆，已验证 CUDA 可用）。

```bash
conda activate firstmodel
```

---

## 4. 数据

- 开源数据集：[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT 协议）
- 本地语料：`D:\MyTools\zlf\chinese-poetry`，使用其中 `全唐诗/` 目录：
  - `poet.tang.*.json`：58 个文件
  - `poet.song.*.json`：255 个文件
- 原始 **311,855 首** → 筛选「每句 5 个汉字」→ 得到 **107,311 首五言诗**
- 原始数据是繁体，用 OpenCC 统一转成简体
- 编码后的样本格式：

```
<bos> <title> 秋夜 <sep> 空山新雨后，天气晚来秋。明月松间照，清泉石上流。<end>
```

| 统计项 | 数值 |
|---|---|
| 五言诗总数 | 107,311 首 |
| 训练 / 验证 | 105,165 / 2,146 首 |
| 字符表 | 8,107（8,085 汉字 + 16 标点 + 6 特殊符） |
| 训练 token | 7,826,509 |
| 验证 token | 159,510 |

重新生成数据：

```bash
python data_prep.py --data-root "D:\MyTools\zlf\chinese-poetry\全唐诗"
```

---

## 5. 训练

```bash
python train.py --max-steps 8000
```

实测配置与结果（RTX 5060 Laptop 8GB）：

| 项目 | 实测值 |
|---|---|
| 参数量 | 13.81M（6 层 / 384 维 / 6 头，权重绑定） |
| 上下文长度 | 192 字符 |
| batch size | 32 |
| 精度 | bf16 混合精度 |
| 学习率 | 1e-3 → 1e-4，余弦衰减 + 100 步 warmup |
| 训练步数 | 8,000 步（约 6.3 个 epoch） |
| **训练耗时** | **5.9 分钟**（约 14 万 token/s） |
| 验证集最优 loss | **3.9495** |
| **显存峰值** | **1,313 MB** |

日志会写入 `checkpoints/train.log`，每 250 步评估一次，最优模型存为 `checkpoints/best.pt`。

继续训练：

```bash
python train.py --max-steps 12000 --resume checkpoints/last.pt
```

---

## 6. 生成

```bash
# 五言绝句（4 句，默认）
python generate.py --title "秋夜"

# 五言律诗（8 句）
python generate.py --title "秋夜" --clauses 8

# 多个标题，每个写 2 首
python generate.py --titles "秋夜,送别,山中" --count 2

# 调整采样：温度越低越保守工整，越高越放飞
python generate.py --title "月下" --temperature 0.7
python generate.py --title "月下" --temperature 1.05 --top-k 30

# 不强制五言格式，自由续写
python generate.py --free --title "秋天"

# 保存到文件
python generate.py --titles "秋夜,送别" --out poems.txt
```

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--title` | 秋夜 | 诗题 |
| `--clauses` | 4 | 句数：4=五绝，8=五律 |
| `--temperature` | 0.8 | 采样温度 |
| `--top-k` | 15 | 每步只从概率最高的 k 个字里采样 |
| `--repetition-penalty` | 1.15 | 重复字惩罚 |
| `--free` | 关 | 自由续写，不约束五言 |

---

## 7. 模型是怎么工作的（一句话版）

1. **字符级分词**：每个汉字/标点就是一个 token，词表 8,107；
2. **Transformer**：6 层因果自注意力，每个位置根据前文预测下一个字；
3. **训练目标**：给标题 + 前文，预测下一个字符（交叉熵损失）；
4. **生成**：输入标题逐字续写；约束解码保证"每句 5 字、两句一顿"。

---

## 8. 效果边界（重要）

- ✅ 能学到五言诗的**用字、意象、对仗语感**，多数句子读起来像古诗
- ⚠️ **不保证平仄格律和押韵**：1300 万参数 + 780 万字符的数据，学不会严格的格律规则
- ⚠️ 质量波动明显：同一标题多生成几次，有工整的也有别扭的（比如重复用字、语义不通）
- ⚠️ 标题只是"提示"，模型对标题的理解很弱，写"秋夜"未必句句写秋

想更进一步，可以做的方向：
1. **加押韵约束**：用拼音词典限制每句末字的韵母
2. **加平仄模板**：按五绝/五律的平仄格式约束解码
3. **换更大模型**：3000 万~1 亿参数，或加入七言、宋词数据
4. **只训练绝句/律诗**：去掉长篇古体，让短诗结构更稳定

---

## 9. 常见问题

**Q：报错 "sm_120 is not compatible with the current PyTorch installation"？**
用了旧版 PyTorch。RTX 50 系必须装 CUDA 12.8+ 版本：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Q：没有 GPU 能跑吗？**
能，加 `--device cpu`，但训练会慢几十倍、生成慢几秒。

**Q：显存不够？**
这个模型训练峰值约 1.3GB，8GB 完全够。

**Q：数据下载不下来？**
`raw.githubusercontent.com` 国内不稳定，可用 jsDelivr 镜像：

```
https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/全唐诗/...
```
