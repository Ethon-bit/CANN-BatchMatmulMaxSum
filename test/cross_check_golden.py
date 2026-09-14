"""
交叉验证：golden_pure.py  vs  golden_torch.py

为什么要做这件事？
------------------
我们有两个"标准答案"：
    golden_pure.py  —— 手写的三层循环，零依赖
    golden_torch.py —— 官方给出的 torch.bmm + amax + sum

如果这两个算出来的结果一致，我们才有信心说"我理解的题意是对的"。
如果它们不一致，说明其中一个是错的，必须先查清楚，不能拿去验证 NPU 算子。

这个脚本会在 4 种存储布局组合 × 多种形状下随机造数据，逐一比对。

运行方式（需要 torch）：
    uv run --with torch python test/cross_check_golden.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from golden_pure import batch_matmul_max_sum_pure  # noqa: E402


def main() -> int:
    import torch
    from golden_torch import golden_fp64, make_storage

    sys.stdout.reconfigure(encoding="utf-8")

    # (B, M, N, K) —— 覆盖小 batch、非对齐尾块、较大形状
    shapes = [
        (1, 1, 1, 32),      # 最小规模
        (1, 2, 3, 32),      # 赛题示例规模
        (2, 5, 7, 40),      # M/N 非 16 对齐，考尾块处理
        (3, 16, 16, 64),    # 整齐对齐
        (4, 17, 33, 128),   # 明显的非对齐尾块
        (8, 64, 64, 256),   # 中等规模
    ]

    total = 0
    bad = 0

    for (B, M, N, K) in shapes:
        for t1 in (False, True):
            for t2 in (False, True):
                total += 1
                tag = f"B={B} M={M} N={N} K={K} tX1={int(t1)} tX2={int(t2)}"

                x1_st, x2_st = make_storage(B, M, N, K, t1, t2,
                                            dtype=torch.float32, seed=B * 1000 + M * 10 + N)

                # 官方语义（FP64 计算）
                y_torch = golden_fp64(x1_st, x2_st, t1, t2).tolist()

                # 纯 Python 版
                y_pure = batch_matmul_max_sum_pure(
                    x1_st.tolist(), x2_st.tolist(), t1, t2
                )

                # 比对：两条路径都应是 FP64 语义，容差给到 1e-6 足够
                max_diff = max(abs(a - b) for a, b in zip(y_pure, y_torch))
                ok = max_diff < 1e-6
                if not ok:
                    bad += 1
                print(f"{'OK  ' if ok else 'FAIL'} {tag}  最大差异={max_diff:.3e}")

    print()
    print(f"交叉验证：{total - bad}/{total} 通过")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
