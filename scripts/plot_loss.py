"""
从训练日志解析 loss 并绘制曲线
==============================
用法：
    python scripts/plot_loss.py <日志文件> <输出图片> [标题]
示例：
    python scripts/plot_loss.py results/pretrain_demo.log results/pretrain_demo_loss.png "迷你预训练 Loss 曲线"
"""
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 匹配形如: Epoch:[1/1](120/1250), loss: 6.1234, ...
PATTERN = re.compile(r"\((\d+)/(\d+)\),\s*loss:\s*([0-9.]+)")


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "results/pretrain_demo.log"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results/pretrain_demo_loss.png"
    title = sys.argv[3] if len(sys.argv) > 3 else "迷你预训练 Loss 曲线"

    steps, losses = [], []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))

    if not losses:
        print(f"[警告] 未从 {log_path} 解析到 loss 数据")
        sys.exit(1)

    plt.figure(figsize=(9, 5))
    plt.plot(steps, losses, color="#2563eb", linewidth=1.6)
    plt.scatter(steps, losses, color="#2563eb", s=12, zorder=3)
    plt.xlabel("训练步数 (step)")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.annotate(f"起始 {losses[0]:.3f}", (steps[0], losses[0]),
                 textcoords="offset points", xytext=(10, 10), color="#dc2626")
    plt.annotate(f"结束 {losses[-1]:.3f}", (steps[-1], losses[-1]),
                 textcoords="offset points", xytext=(-40, 10), color="#16a34a")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"[完成] 共 {len(losses)} 个点，loss {losses[0]:.3f} -> {losses[-1]:.3f}，图已存 {out_path}")


if __name__ == "__main__":
    main()
