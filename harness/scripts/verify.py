"""
比对算子输出与 golden。

用法：
    python3 scripts/verify.py <case_id> [标签]

判据用赛题的精度要求（float32 输出）：
    相对误差 < 1e-4 且 绝对误差 < 1e-4
    （判定式：|got - golden| <= atol + rtol * |golden|）

退出码：0 = 通过，1 = 不通过，3 = 文件缺失
"""

import os
import sys

import numpy as np

RTOL = 1e-4
ATOL = 1e-4

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    if len(sys.argv) < 2:
        print("用法: python3 scripts/verify.py <case_id> [标签]")
        return 3

    cid = int(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else ""

    got_path = os.path.join(ROOT, "output", "y.bin")
    gld_path = os.path.join(ROOT, "work", f"case{cid:02d}", "golden_y.bin")

    if not os.path.exists(got_path):
        print(f"  [缺失] {got_path} —— 算子没产出输出（可能卡死或提前 return）")
        return 3
    if not os.path.exists(gld_path):
        print(f"  [缺失] {gld_path} —— 先跑 gen_cases.py")
        return 3

    got = np.fromfile(got_path, dtype=np.float32)
    gld = np.fromfile(gld_path, dtype=np.float32)

    if got.size != gld.size:
        print(f"  [FAIL] 长度不符：算子输出 {got.size} 个，期望 {gld.size} 个")
        return 1

    ok = np.isclose(got, gld, rtol=RTOL, atol=ATOL)
    n_bad = int((~ok).sum())
    diff = np.abs(got.astype(np.float64) - gld.astype(np.float64))
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(gld.astype(np.float64)), 1e-12)
    max_rel = float((diff / denom).max()) if diff.size else 0.0

    prefix = f"  case{cid:02d}"
    if tag:
        prefix += f" [{tag}]"

    if n_bad == 0:
        print(f"{prefix} ✅ 通过   最大绝对误差={max_abs:.3e}  最大相对误差={max_rel:.3e}")
        return 0

    rate = n_bad / gld.size * 100.0
    print(f"{prefix} ❌ 不通过  错误 {n_bad}/{gld.size} ({rate:.2f}%)  "
          f"最大绝对误差={max_abs:.3e}  最大相对误差={max_rel:.3e}")

    # 打印前几个错的，方便定位
    idx = np.where(~ok)[0][:6]
    print(f"          期望: {np.round(gld[idx], 6).tolist()}")
    print(f"          实际: {np.round(got[idx], 6).tolist()}")
    if rate == 100.0:
        print("          提示：100% 全错通常是结构性问题（维度/布局读错），不是精度问题")
    return 1


if __name__ == "__main__":
    sys.exit(main())
