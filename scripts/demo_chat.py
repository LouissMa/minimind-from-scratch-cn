"""
官方权重推理 Demo
=================
本脚本加载【原作者 jingyaogong 发布的 MiniMind2 权重】（从 HuggingFace 下载），
用于验证本仓库的推理链路是否跑通、并直观展示 MiniMind 模型的对话能力。

⚠️ 诚实声明：此处使用的是原作者训练好的权重，**不是**本人从零训练的成果。
   本人从零训练的部分见 docs/复现日志.md 中的「迷你预训练」记录。

用法：
    python scripts/demo_chat.py
输出：
    终端打印问答，同时写入 results/demo_chat_official_weights.md
"""
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "MiniMind2")
OUT_MD = os.path.join(os.path.dirname(__file__), "..", "results", "demo_chat_official_weights.md")

PROMPTS = [
    "你好，请简单介绍一下你自己。",
    "为什么天空是蓝色的？",
    "用一句话解释什么是机器学习。",
    "推荐三种适合新手的编程语言，并说明理由。",
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[加载] 设备={device}，权重目录={os.path.abspath(MODEL_DIR)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).to(device).eval()

    lines = [
        "# 推理 Demo（使用原作者发布的 MiniMind2 权重）",
        "",
        "> ⚠️ 本页对话使用的是 **原作者 [jingyaogong](https://github.com/jingyaogong) 训练好的 "
        "MiniMind2 权重**，用于验证本仓库推理链路可用、展示模型对话效果。",
        "> **这不是本人从零训练的成果**——本人的迷你预训练记录见 "
        "[docs/复现日志.md](../docs/复现日志.md)。",
        "",
        f"- 模型：MiniMind2（Llama 架构，hidden=768，layers=16，vocab=6400）",
        f"- 运行设备：{device}",
        "",
        "---",
        "",
    ]

    for i, prompt in enumerate(PROMPTS, 1):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
        st = time.time()
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=256, do_sample=True,
                top_p=0.85, temperature=0.85,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        speed = (gen[0].shape[0] - inputs["input_ids"].shape[1]) / (time.time() - st)
        print(f"\n💬 {prompt}\n🤖 {answer}\n[{speed:.1f} tok/s]")
        lines += [f"### {i}. 💬 {prompt}", "", f"🤖 {answer}", "", f"`生成速度: {speed:.1f} tokens/s`", "", "---", ""]

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[完成] 已写入 {os.path.abspath(OUT_MD)}")


if __name__ == "__main__":
    main()
