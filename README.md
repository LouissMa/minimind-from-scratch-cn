# minimind-from-scratch-cn

> 从零复现一个 26M 参数的中文小语言模型，走通 **预训练 → SFT → LoRA → DPO → 蒸馏 → RLHF** 全流程。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)

---

## 📌 关于本项目

本项目是我对开源项目 **[MiniMind](https://github.com/jingyaogong/minimind)**（作者 [@jingyaogong](https://github.com/jingyaogong)）的**学习复现**。

我的目标不是「造一个新轮子」，而是**亲手把一个大语言模型从零到一的完整训练链路跑通**，在这个过程中真正理解每个阶段在做什么、为什么这么做、会踩哪些坑。所有原创的技术设计、模型结构、算法实现均归功于原作者，本仓库遵循原项目的 **Apache 2.0** 许可证发布。

**我在复现中做的工作：**
- ✅ 在本地（Windows / 单卡）环境下完整搭建训练环境，记录依赖与踩坑（见 [环境搭建](docs/环境搭建.md)）
- ✅ 逐阶段跑通训练流程，记录真实命令、超参、loss 曲线与产出（见 [复现日志](docs/复现日志.md)）
- ✅ 阅读并逐行理解核心代码（模型结构 / 数据管线 / 各训练范式）
- 🔲 *（进行中）* 对关键阶段做实验对比与结果分析

> 📖 原项目完整的技术文档、原理讲解非常详尽，我已归档在 [`docs/reference/`](docs/reference/) 供参考，强烈建议对照阅读。

---

## 🧠 复现内容概览

MiniMind 是一个极简的类 GPT 结构（Transformer Decoder，支持 RoPE、RMSNorm、可选 MoE），最小版本仅 **25.8M 参数**，可在个人 GPU 上快速训练。完整复现涵盖以下阶段：

| 阶段 | 脚本 | 说明 | 我的复现状态 |
|------|------|------|:---:|
| 训练分词器 | `trainer/train_tokenizer.py` | 从零训练 BPE 分词器 | 🔲 |
| 预训练 Pretrain | `trainer/train_pretrain.py` | 学习语言知识 | 🔲 |
| 监督微调 SFT | `trainer/train_full_sft.py` | 学习对话格式 | 🔲 |
| LoRA 微调 | `trainer/train_lora.py` | 参数高效微调 | 🔲 |
| 偏好优化 DPO | `trainer/train_dpo.py` | 对齐人类偏好 | 🔲 |
| 知识蒸馏 | `trainer/train_distillation.py` | 大模型教小模型 | 🔲 |
| 推理蒸馏 | `trainer/train_reason.py` | 复现 R1 式推理 | 🔲 |
| 强化学习 | `trainer/train_ppo.py` / `train_grpo.py` / `train_spo.py` | RLHF | 🔲 |

> ✅ = 已跑通并记录结果　🔲 = 待复现。真实进度与结果见 [复现日志](docs/复现日志.md)。

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 建议 Python 3.10+，创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
```
详细的环境说明与踩坑记录见 [docs/环境搭建.md](docs/环境搭建.md)。

### 2. 下载数据集
数据集需自行下载后放入 `dataset/` 目录（体积较大，已在 `.gitignore` 中排除）。
下载地址见原项目说明：[dataset/dataset.md](dataset/dataset.md)。

### 3. 预训练（示例命令）
```bash
cd trainer
python train_pretrain.py \
  --data_path ../dataset/pretrain_hq.jsonl \
  --hidden_size 512 --num_hidden_layers 8 \
  --epochs 1 --batch_size 32 --learning_rate 5e-4
```

### 4. 监督微调（示例命令）
```bash
cd trainer
python train_full_sft.py \
  --data_path ../dataset/sft_mini_512.jsonl \
  --from_weight pretrain --epochs 2
```

### 5. 测试对话
```bash
python eval_llm.py            # 命令行对话
python scripts/web_demo.py    # 网页 demo
```

> 完整参数说明请 `python trainer/train_pretrain.py --help` 查看。

---

## 📂 目录结构

```
.
├── model/              # 模型结构（MiniMind / LoRA）与分词器
├── trainer/            # 各阶段训练脚本
├── dataset/            # 数据加载代码（数据文件需自行下载）
├── scripts/            # 推理服务 / 网页 demo / 模型转换
├── eval_llm.py         # 模型评测 / 对话
├── docs/
│   ├── 复现日志.md      # ⭐ 我的逐阶段复现记录（核心）
│   ├── 环境搭建.md      # 环境与依赖踩坑
│   └── reference/      # 原作者 README 归档
├── results/            # 我的 loss 曲线 / 对话样例
├── requirements.txt
├── LICENSE             # Apache 2.0（沿用原项目）
└── NOTICE              # 基于原项目的修改说明
```

---

## 🙏 致谢与许可

- 原项目：**[MiniMind](https://github.com/jingyaogong/minimind)** by [@jingyaogong](https://github.com/jingyaogong)，本项目的一切模型设计与算法实现均源于此，特此致谢。
- 本项目在 **Apache License 2.0** 下发布，与原项目一致。修改与再分发说明见 [NOTICE](NOTICE)。
- 本仓库为**个人学习复现**用途，不用于任何商业目的。
