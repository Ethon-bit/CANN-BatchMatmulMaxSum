"""
C++ 参考实现的编译与验证桥接脚本

它做什么
--------
1. 编译 cpp/golden.cpp
2. 把 test/cases/cases.json 里的用例导出成 C++ 程序能读的格式
3. 跑 C++ 程序
4. 把 C++ 结果和 Python golden 逐元素比对

为什么值得做这件事
------------------
cp/golden.cpp 里的三层循环，和 op_kernel/ 里要写的 Ascend C 代码
**结构完全一样**。先确认这份 C++ 版算得对，再往 NPU 上迁移，
就只需要操心"怎么搬数据、怎么用 Cube"这些硬件问题，
而不用同时怀疑算法本身对不对。

用法
----
    python test/cpp_bridge.py            # 编译 + 验证
    python test/cpp_bridge.py --keep     # 保留中间文件，方便自己看
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

from golden_pure import batch_matmul_max_sum_pure  # noqa: E402
from test_accuracy import make_inputs, to_storage, calc_ops  # noqa: E402

CASES_PATH = os.path.join(HERE, "cases", "cases.json")
CPP_SRC = os.path.join(ROOT, "cpp", "golden.cpp")
CPP_BIN = os.path.join(ROOT, "cpp", "golden.exe")
WORK_DIR = os.path.join(HERE, "cases", "cpp_work")

# 纯 Python 计算太慢的用例，用一个上限卡掉。
# 这里只用于生成"输入数据"，计算由 C++ 完成，所以其实只有 Python
# 造数据那一步会慢。造数据是 O(B*M*K)，比 O(B*M*N*K) 小得多。
MAX_ELEMS_FOR_PYTHON = 40_000_000


def build():
    """编译 C++ 参考实现。"""
    print(f"编译 {CPP_SRC} ...")
    cmd = ["g++", "-O2", "-std=c++11", "-o", CPP_BIN, CPP_SRC]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("编译失败：")
        print(r.stderr)
        return False
    print(f"编译成功 -> {CPP_BIN}")
    return True


def export_cases():
    """把用例导出成 C++ 能读的文本格式，返回可参与比对的用例名列表。

    格式：
        <用例数>
        每用例: <B> <M> <N> <K> <tX1> <tX2>
                <B*M*K 个 x1 存储数据>
                <B*K*N 个 x2 存储数据>
    """
    with open(CASES_PATH, encoding="utf-8") as f:
        spec = json.load(f)

    os.makedirs(WORK_DIR, exist_ok=True)
    usable = []
    skipped = []

    lines = []
    for idx, case in enumerate(spec["cases"]):
        B, M, N, K = case["B"], case["M"], case["N"], case["K"]
        elems = max(B * M * K, B * K * N)   # 造数据的规模
        if elems > MAX_ELEMS_FOR_PYTHON:
            skipped.append((case["name"], elems))
            continue

        x1_log, x2_log = make_inputs(case, seed=1000 + idx)
        x1_st, x2_st = to_storage(x1_log, x2_log,
                                  case["transposeX1"], case["transposeX2"])

        # 展平成一维（行优先），和 C++ 侧的 offset 公式对应
        def flat(t):
            out = []
            for a in t:
                for row in a:
                    out.extend(float(v) for v in row)
            return out

        f1 = flat(x1_st)
        f2 = flat(x2_st)

        # 自检：元素总数必须对得上，否则说明维度解释错了
        assert len(f1) == B * M * K, f"{case['name']}: x1 元素数 {len(f1)} != {B*M*K}"
        assert len(f2) == B * K * N, f"{case['name']}: x2 元素数 {len(f2)} != {B*K*N}"

        lines.append(f"{B} {M} {N} {K} "
                     f"{1 if case['transposeX1'] else 0} "
                     f"{1 if case['transposeX2'] else 0}")
        lines.append(" ".join(repr(v) for v in f1))
        lines.append(" ".join(repr(v) for v in f2))
        usable.append(case["name"])

    in_path = os.path.join(WORK_DIR, "input.txt")
    with open(in_path, "w", encoding="utf-8") as f:
        f.write(f"{len(usable)}\n")
        f.write("\n".join(lines))
        f.write("\n")

    print(f"导出 {len(usable)} 个用例 -> {in_path}")
    if skipped:
        print(f"跳过 {len(skipped)} 个（Python 造数据太慢）：")
        for name, e in skipped:
            print(f"    - {name}  {e:,} 元素")
    return usable, skipped, in_path


def run_cpp(in_path):
    """跑 C++ 程序，返回解析后的结果列表。"""
    out_path = os.path.join(WORK_DIR, "output.txt")
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        r = subprocess.run([CPP_BIN], stdin=fin, stdout=fout,
                           stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("C++ 程序执行失败：")
        print(r.stderr)
        return None, out_path

    results = []
    with open(out_path, encoding="utf-8") as f:
        n = int(f.readline().strip())
        for _ in range(n):
            vals = [float(x) for x in f.readline().split()]
            results.append(vals[1:])   # 第 1 个数是 B，后面才是 y
    return results, out_path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true",
                        help="保留中间文件（默认也会保留，此参数仅为兼容）")
    parser.parse_args()

    # 1. 编译
    if not build():
        return 1

    # 2. 导出用例
    usable, skipped, in_path = export_cases()
    if not usable:
        print("没有可用的用例")
        return 1

    # 3. 跑 C++
    cpp_results, out_path = run_cpp(in_path)
    if cpp_results is None:
        return 1
    print(f"C++ 输出 -> {out_path}")
    print()

    # 4. 与 Python golden 比对
    with open(CASES_PATH, encoding="utf-8") as f:
        spec = json.load(f)
    by_name = {c["name"]: c for c in spec["cases"]}

    passed = 0
    failed = 0

    for idx, name in enumerate(usable):
        case = by_name[name]
        x1_log, x2_log = make_inputs(case, seed=1000 + idx)
        x1_st, x2_st = to_storage(x1_log, x2_log,
                                  case["transposeX1"], case["transposeX2"])
        y_py = batch_matmul_max_sum_pure(
            x1_st, x2_st, case["transposeX1"], case["transposeX2"]
        )
        y_cpp = cpp_results[idx]

        if len(y_py) != len(y_cpp):
            print(f"✗ {name}: 长度不一致 python={len(y_py)} cpp={len(y_cpp)}")
            failed += 1
            continue

        max_diff = max(abs(a - b) for a, b in zip(y_py, y_cpp))
        # C++ 最后转了 float，Python 保留 double，所以容差按 float 精度给
        ok = max_diff < 1e-5 * max(1.0, max(abs(v) for v in y_py))
        if ok:
            passed += 1
            print(f"✓ {name}  最大差异={max_diff:.3e}")
        else:
            failed += 1
            print(f"✗ {name}  最大差异={max_diff:.3e}")
            print(f"    python={y_py[:4]}")
            print(f"    cpp   ={y_cpp[:4]}")

    print()
    print(f"C++ 参考实现 vs Python golden：{passed}/{passed + failed} 通过")
    if skipped:
        print(f"（另有 {len(skipped)} 个用例因数据量过大未参与）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
