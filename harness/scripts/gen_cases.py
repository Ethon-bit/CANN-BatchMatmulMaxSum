"""
按 cases.txt 生成全部测试数据 + golden。

产出：
    work/case<id>/x1.bin           输入（物理布局，裸二进制）
    work/case<id>/x2.bin
    work/case<id>/golden_y.bin     期望输出（B 个 float32）

用法：
    python3 scripts/gen_cases.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from golden import impl, make_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CASES = os.path.join(ROOT, "cases.txt")
WORK = os.path.join(ROOT, "work")


def parse_cases(path):
    """读 cases.txt：跳过空行和 # 注释行。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 9:
                print(f"  跳过第 {lineno} 行（字段数 {len(parts)} != 9）: {line}")
                continue
            try:
                cid, B, M, N, K = (int(parts[0]), int(parts[1]), int(parts[2]),
                                   int(parts[3]), int(parts[4]))
                tx1, tx2, pattern = int(parts[6]), int(parts[7]), int(parts[8])
            except ValueError:
                # 表头或说明行，不是数据 —— 直接跳过
                continue
            dtype = parts[5]
            if dtype not in ("float16", "bfloat16", "float32"):
                print(f"  跳过第 {lineno} 行（dtype 不认识: {dtype}）")
                continue
            out.append(dict(id=cid, B=B, M=M, N=N, K=K, dtype=dtype,
                            tx1=tx1, tx2=tx2, pattern=pattern))
    return out


def main():
    cases = parse_cases(CASES)
    if not cases:
        print("cases.txt 里没有有效用例")
        return 1

    os.makedirs(WORK, exist_ok=True)
    ok = 0

    for c in cases:
        d = os.path.join(WORK, f"case{c['id']:02d}")
        os.makedirs(d, exist_ok=True)

        # 种子只由 (B,M,N,K,pattern) 决定，**不含 case id 和 dtype**。
        # 这样形状相同、只是 transpose 摆放不同的用例（比如 case01~04）
        # 会拿到同一份逻辑数据，golden 必然相同 —— 这正是「四象限」用例的意义。
        seed = (20260918
                + c["B"] * 1000003
                + c["M"] * 10007
                + c["N"] * 101
                + c["K"] * 3
                + c["pattern"] * 7919)
        try:
            x1, x2 = make_inputs(c["B"], c["M"], c["N"], c["K"],
                                 bool(c["tx1"]), bool(c["tx2"]),
                                 c["dtype"], c["pattern"], seed=seed)
        except RuntimeError as e:
            # 典型情况：本机没装 ml_dtypes，跑不了 bfloat16 用例。
            # 明确报出来并跳过，不要静默漏掉。
            print(f"  [跳过] case{c['id']:02d} ({c['dtype']}): {e}")
            continue
        y = impl(x1, x2, bool(c["tx1"]), bool(c["tx2"]))

        c["x1_shape"] = x1.shape
        c["x2_shape"] = x2.shape

        x1.tofile(os.path.join(d, "x1.bin"))
        x2.tofile(os.path.join(d, "x2.bin"))
        y.tofile(os.path.join(d, "golden_y.bin"))

        # 自检：全负用例必须真的全负
        if c["pattern"] == 1 and not np.all(y < 0):
            print(f"  [WARN] case{c['id']:02d} 标了全负但 golden 不是全负: {y}")

        print(f"  case{c['id']:02d}  B={c['B']} M={c['M']} N={c['N']} K={c['K']} "
              f"{c['dtype']} tx=({c['tx1']},{c['tx2']}) pat={c['pattern']}  "
              f"x1{x1.shape} x2{x2.shape}  y={y[:4].tolist()}{'...' if len(y) > 4 else ''}")
        ok += 1

    # 写一份机器可读的清单，给 run_all.sh 用
    import json
    with open(os.path.join(WORK, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)

    print(f"\n共生成 {ok} 个用例 -> {WORK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
