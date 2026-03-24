"""生成更真实和多样化的合成 Advertising 数据。

特点：
- 使用可复现的随机种子
- TV、radio、newspaper 使用不同分布（近似真实广告数据的分布）
- 引入交互项与异方差噪声
- 支持少量异常值（outliers）以增加多样性

运行：
    python generate_ad_data.py  # 在 dataprocessing 目录下运行
"""

import csv
import math
import random
from pathlib import Path
import sys


def box_muller(mean=0.0, std=1.0):
    # 生成标准正态变量（Box-Muller），使用 random.random 保证无外部依赖
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
    return mean + z0 * std


def generate(n=100, seed=2026, out_path=None):
    random.seed(seed)
    out = Path(out_path) if out_path else Path(__file__).parent / 'data' / 'Advertising.csv'
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        # TV: 混合分布（主峰在 100-200），加少量偏移和周期性变化
        tv_base = 140 + box_muller(0, 40)
        tv_season = 20 * math.sin(i / 6.0)
        TV = max(0.0, min(300.0, tv_base + tv_season))

        # radio: 较小方差，主要集中在 0-50
        radio = max(0.0, min(50.0, 20 + box_muller(0, 8)))

        # newspaper: 偏斜分布（大部分较小，少量大值）
        newspaper = max(0.0, min(120.0, abs(box_muller(10, 20)) ) )

        # 引入交互项和异方差噪声（噪声随 TV 增大略增）
        interaction = 0.00025 * TV * radio
        noise_scale = 0.7 + 0.003 * TV
        noise = box_muller(0, noise_scale)

        sales = 3.0 + 0.04 * TV + 0.19 * radio + 0.0012 * newspaper + interaction + noise

        # 少量异常值（1-3 个）
        if i % 37 == 0:
            # 人为放大某些样本的广告投入，造成异常高销量
            TV *= 1.6
            radio *= 1.4
            sales += 5 * abs(box_muller(0, 1.5))

        rows.append((TV, radio, newspaper, sales))

    # 写入 CSV
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['TV', 'radio', 'newspaper', 'sales'])
        for TV, radio, newspaper, sales in rows:
            writer.writerow([f"{TV:.4f}", f"{radio:.4f}", f"{newspaper:.4f}", f"{sales:.4f}"])

    print(f"Wrote {n} samples to {out}")


if __name__ == '__main__':
    # 支持命令行参数：样本数、随机种子
    n = 100
    seed = 2026
    if len(sys.argv) >= 2:
        try:
            n = int(sys.argv[1])
        except Exception:
            pass
    if len(sys.argv) >= 3:
        try:
            seed = int(sys.argv[2])
        except Exception:
            pass

    generate(n=n, seed=seed)
