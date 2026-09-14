"""
BatchMatmulMaxSum —— 精度测试脚本

它做三件事
----------
1. 按 test/cases/cases.json 的定义造测试数据
2. 用 Python 参考实现算出"标准答案"（golden），存到 test/cases/golden/
3. 如果你的 NPU 算子已经跑出了结果，拿它和 golden 比对，按赛题的精度
   要求判定通过/失败

用法
----
    # 第 1 步：先生成 golden（在没接 NPU 的机器上也能跑）
    python test/test_accuracy.py --gen

    # 第 2 步：把 NPU 算出的结果放到 test/cases/npu_out/ 目录下
    #         文件名要和用例名对应，例如 示例1_基础计算.bin（float32 小端裸数据）

    # 第 3 步：比对
    python test/test_accuracy.py --check

关于依赖
--------
装了 numpy 会快很多。没装也能跑，但大形状的用例会非常慢，
所以会自动跳过超过阈值的用例，**并且明确打印跳过了哪些**。

赛题的精度要求（第五节）
----------------------
    float32            : 相对误差 < 1e-4，绝对误差 < 1e-4
    float16 / bfloat16 : 相对误差 < 1e-3，绝对误差 < 1e-3
"""

import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from golden_pure import batch_matmul_max_sum_pure  # noqa: E402

CASES_PATH = os.path.join(HERE, "cases", "cases.json")
GOLDEN_DIR = os.path.join(HERE, "cases", "golden")
NPU_OUT_DIR = os.path.join(HERE, "cases", "npu_out")

# 纯 Python 模式下单个用例允许的最大计算量（元素个数），超过就跳过。
# 2_000_000 次乘加在纯 Python 里大约要几秒。
PURE_PY_LIMIT = 2_000_000

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ======================================================================
# 数据生成
# ======================================================================

def make_inputs(case, seed):
    """按用例定义生成一对输入张量（以逻辑形状返回）。

    返回 (x1_logical, x2_logical)。
    如果是"固定输入"用例，就用 cases.json 里写死的值。
    """
    B, M, N, K = case["B"], case["M"], case["N"], case["K"]

    fixed = case.get("fixed_input")
    if fixed is not None:
        # 固定输入：用例里给的是什么就是什么，其余位置补 0
        x1 = [[[0.0] * K for _ in range(M)] for _ in range(B)]
        x2 = [[[0.0] * N for _ in range(K)] for _ in range(B)]
        for b, rows in enumerate(fixed.get("x1_logical", [])):
            for m, row in enumerate(rows):
                for k, v in enumerate(row):
                    x1[b][m][k] = float(v)
        for b, rows in enumerate(fixed.get("x2_logical", [])):
            for k, row in enumerate(rows):
                for n, v in enumerate(row):
                    x2[b][k][n] = float(v)
        return x1, x2

    if HAS_NUMPY:
        rng = np.random.default_rng(seed)
        x1 = rng.standard_normal((B, M, K))
        x2 = rng.standard_normal((B, K, N))
        # 模拟上游已做过 L2 归一化（赛题说明归一化不归本算子管）
        x1 = x1 / np.maximum(np.linalg.norm(x1, axis=-1, keepdims=True), 1e-12)
        x2 = x2 / np.maximum(np.linalg.norm(x2, axis=-2, keepdims=True), 1e-12)
        # 转成目标精度再转回来，模拟真实输入已被量化的事实
        import numpy as _np
        dt = _np.float16 if case["dtype"] == "float16" else None
        if dt is not None:
            x1 = x1.astype(dt).astype(_np.float64)
            x2 = x2.astype(dt).astype(_np.float64)
        return x1.tolist(), x2.tolist()

    # 没有 numpy：用 Python 自带的 random，固定种子保证可复现
    #
    # 注意这里的嵌套顺序：x1 逻辑形状是 [B, M, K]，所以 x1[b] 是 M 行 K 列；
    # x2 逻辑形状是 [B, K, N]，所以 x2[b] 是 K 行 N 列。
    # 写反了会导致后面按列归一化时下标越界。
    import random
    rnd = random.Random(seed)
    x1 = [[[rnd.gauss(0, 1) for _ in range(K)] for _ in range(M)] for _ in range(B)]
    x2 = [[[rnd.gauss(0, 1) for _ in range(N)] for _ in range(K)] for _ in range(B)]

    def l2_norm_rows(mat, rows, cols, row_major=True):
        for i in range(rows):
            s = 0.0
            for j in range(cols):
                v = mat[i][j] if row_major else mat[j][i]
                s += v * v
            s = max(s ** 0.5, 1e-12)
            for j in range(cols):
                if row_major:
                    mat[i][j] /= s
                else:
                    mat[j][i] /= s

    for b in range(B):
        l2_norm_rows(x1[b], M, K, row_major=True)   # x1: 每行（每个 token）归一化
        l2_norm_rows(x2[b], N, K, row_major=False)  # x2 逻辑上是 [K,N]，按列归一化
    return x1, x2


def to_storage(x1_logical, x2_logical, transpose_x1, transpose_x2):
    """把逻辑形状的输入转成实际存储形状。

    逻辑 x1 [B,M,K] -> transpose 时存成 [B,K,M]
    逻辑 x2 [B,K,N] -> transpose 时存成 [B,N,K]
    """
    B = len(x1_logical)
    x1_st = []
    for b in range(B):
        if transpose_x1:
            x1_st.append([list(col) for col in zip(*x1_logical[b])])  # 转置
        else:
            x1_st.append(x1_logical[b])

    x2_st = []
    for b in range(B):
        if transpose_x2:
            x2_st.append([list(col) for col in zip(*x2_logical[b])])  # 转置
        else:
            x2_st.append(x2_logical[b])
    return x1_st, x2_st


def calc_ops(case):
    """估算这个用例的计算量（乘加次数），用来决定纯 Python 模式跑不跑。"""
    return case["B"] * case["M"] * case["N"] * case["K"]


# ======================================================================
# 生成 golden
# ======================================================================

def cmd_gen():
    with open(CASES_PATH, encoding="utf-8") as f:
        spec = json.load(f)

    os.makedirs(GOLDEN_DIR, exist_ok=True)

    skipped = []
    done = 0

    for idx, case in enumerate(spec["cases"]):
        name = case["name"]
        ops = calc_ops(case)

        if not HAS_NUMPY and ops > PURE_PY_LIMIT:
            skipped.append((name, ops))
            continue

        print(f"[{idx + 1}/{len(spec['cases'])}] {name} "
              f"(B={case['B']} M={case['M']} N={case['N']} K={case['K']}, "
              f"计算量={ops:,}) ...", end=" ", flush=True)

        x1_log, x2_log = make_inputs(case, seed=1000 + idx)
        x1_st, x2_st = to_storage(x1_log, x2_log,
                                  case["transposeX1"], case["transposeX2"])

        y = batch_matmul_max_sum_pure(
            x1_st, x2_st, case["transposeX1"], case["transposeX2"]
        )

        # 如果用例写了期望值，顺便校验一下我们的 golden 本身对不对
        if "expect_y" in case:
            want = case["expect_y"]
            ok = len(y) == len(want) and all(
                abs(a - b) < 1e-6 for a, b in zip(y, want)
            )
            print(f"golden={[round(v, 6) for v in y]} 期望={want} "
                  f"{'✓自洽' if ok else '✗golden与期望不符!'}")
            if not ok:
                print("    ⚠️ 这说明 golden 实现或用例定义有问题，请先排查再继续")
        else:
            print(f"golden={[round(v, 6) for v in y[:4]]}{'...' if len(y) > 4 else ''}")

        # 保存 golden 和输入（输入也存下来，方便 NPU 侧用同一份数据）
        with open(os.path.join(GOLDEN_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "case": case,
                "y": y,
                "x1_storage": x1_st,
                "x2_storage": x2_st,
            }, f, ensure_ascii=False)
        done += 1

    print()
    print(f"完成：生成 {done} 个 golden"
          + (f"，跳过 {len(skipped)} 个（纯Python模式算不动）" if skipped else ""))
    if skipped:
        print("跳过的用例（装了 numpy 就能跑）：")
        for name, ops in skipped:
            print(f"    - {name}  计算量={ops:,}")


# ======================================================================
# 精度比对
# ======================================================================

def read_npu_output(path):
    """读 NPU 输出的 float32 裸数据（小端）。"""
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 4
    return list(struct.unpack(f"<{n}f", raw[:n * 4]))


def compare(golden, got, rtol, atol):
    """按赛题规则判定精度是否达标。

    判定条件：对每个元素，|got - golden| <= atol + rtol * |golden|
    这是"双阈值"判定，绝对误差和相对误差满足其一即可。

    返回 (是否通过, 最大绝对误差, 最大相对误差, 第一个失败的位置)
    """
    max_abs = 0.0
    max_rel = 0.0
    first_bad = None

    for i, (g, a) in enumerate(zip(golden, got)):
        diff = abs(a - g)
        rel = diff / max(abs(g), 1e-12)
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, rel)
        tol = atol + rtol * abs(g)
        if diff > tol and first_bad is None:
            first_bad = (i, g, a, diff, tol)

    return first_bad is None, max_abs, max_rel, first_bad


def cmd_check():
    with open(CASES_PATH, encoding="utf-8") as f:
        spec = json.load(f)

    # 精度阈值：以 JSON 里声明的为准，避免和赛题脱节
    prec = spec["_精度要求"]

    if not os.path.isdir(NPU_OUT_DIR):
        print(f"找不到目录 {NPU_OUT_DIR}")
        print("请先跑 NPU 算子，把结果按 <用例名>.bin 的名字放进该目录。")
        return 1

    total = 0
    passed = 0
    missing = []

    for case in spec["cases"]:
        name = case["name"]
        golden_path = os.path.join(GOLDEN_DIR, f"{name}.json")
        npu_path = os.path.join(NPU_OUT_DIR, f"{name}.bin")

        if not os.path.exists(golden_path):
            missing.append((name, "没有 golden，请先跑 --gen"))
            continue
        if not os.path.exists(npu_path):
            missing.append((name, "没有 NPU 输出"))
            continue

        with open(golden_path, encoding="utf-8") as f:
            golden = json.load(f)["y"]
        got = read_npu_output(npu_path)

        total += 1

        if len(got) != len(golden):
            print(f"✗ {name}: 输出长度不对，golden={len(golden)} 得到={len(got)}")
            continue

        # 输出必须是 fp32，所以按 fp32 的阈值判
        rule = prec["float32"]
        ok, max_abs, max_rel, first_bad = compare(
            golden, got, rule["rtol"], rule["atol"]
        )

        if ok:
            passed += 1
            print(f"✓ {name}  最大绝对误差={max_abs:.3e}  最大相对误差={max_rel:.3e}")
        else:
            i, g, a, diff, tol = first_bad
            print(f"✗ {name}  第 {i} 个元素不达标：")
            print(f"      golden={g!r}  你的结果={a!r}")
            print(f"      误差={diff:.3e}  允许上限={tol:.3e}")
            print(f"      （注意：如果 golden 是负数而你的结果是 0，"
                  f"说明 MaxSim 初值错误地设成了 0）")

    print()
    print(f"精度比对：{passed}/{total} 通过")
    if missing:
        print(f"另有 {len(missing)} 个用例未参与比对：")
        for name, why in missing:
            print(f"    - {name}: {why}")

    return 0 if (total > 0 and passed == total) else 1


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="BatchMatmulMaxSum 精度测试")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--gen", action="store_true", help="生成 golden 标准答案")
    g.add_argument("--check", action="store_true", help="比对 NPU 输出与 golden")
    args = parser.parse_args()

    print(f"numpy: {'可用（快速模式）' if HAS_NUMPY else '不可用（纯Python模式，大用例会跳过）'}")
    print()

    if args.gen:
        cmd_gen()
        return 0
    return cmd_check()


if __name__ == "__main__":
    sys.exit(main())
