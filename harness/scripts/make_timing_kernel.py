"""
从 op_kernel/kernel_cachefix.asc 生成 op_kernel/kernel_timing.asc。

差别只有一处：在 host 侧 run_kernel() 里插 5 个计时点，把一次调用的总耗时
拆成四段打印出来：

    [phase] tiling=... malloc=... kernel=... free=... || 合计=...

为什么要拆这四段：
    平台上看到的「用时」包含整个 run_kernel，而 run_kernel 里有
      - 生成 tiling（MultiCoreMatmulTiling::GetTiling）
      - 4 次 aclrtMalloc + 2 次 aclrtMemcpy
      - 真正跑 kernel
      - 4 次 aclrtFree
    aclrtMalloc / aclrtFree 是重操作。如果小形状用例的耗时几乎全在这几段上，
    那优化方向就完全不是"改算子算法"，而是"别每次重新申请显存"。

用法：
    python3 scripts/make_timing_kernel.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))       # 仓库根
SRC = os.path.join(ROOT, "op_kernel", "kernel_cachefix.asc")
DST = os.path.join(ROOT, "op_kernel", "kernel_timing.asc")


def main():
    if not os.path.isfile(SRC):
        print(f"找不到源文件：{SRC}")
        print("（它在 fix/gm-cache-writeback 分支上，先 git checkout 过来）")
        return 1

    text = open(SRC, encoding="utf-8").read()

    # 用一个可变容器累积，避免"每次替换都基于原文"的经典 bug
    box = [text]

    def rep(tag, old, new):
        n = box[0].count(old)
        if n != 1:
            print(f"  [失败] {tag}: 匹配到 {n} 处（应为 1 处）")
            sys.exit(1)
        box[0] = box[0].replace(old, new)
        print(f"  [ok] {tag}")

    # ---- 0) 引入 <chrono> ----
    rep("include <chrono>",
        "#include <cmath>",
        "#include <cmath>\n#include <chrono>")

    # ---- 1) t0：刚进 run_kernel ----
    rep("t0 进入函数",
        "    (void)info_y;",
        "    (void)info_y;\n"
        "\n"
        "    // ===== 分阶段计时（仅 harness 使用，不影响算子逻辑）=====\n"
        "    using BmmsClock = std::chrono::steady_clock;\n"
        "    static int s_bmmsTimingCalls = 0;\n"
        "    const auto tp0 = BmmsClock::now();")

    # ---- 2) 形状回显（放在 tiling 之前，和 [shape] 行对齐）----
    rep("形状回显",
        "    const int64_t numMBlocks = (M + kBaseM - 1) / kBaseM;",
        "    const int64_t numMBlocks = (M + kBaseM - 1) / kBaseM;\n"
        "\n"
        "    const int64_t dbgBlockNum = (B < availableCoreNum) ? B : availableCoreNum;\n"
        "    if (s_bmmsTimingCalls < 8) {\n"
        "        printf(\"[shape] B=%lld M=%lld N=%lld K=%lld  numMBlocks=%lld  核数=%lld\\n\",\n"
        "               (long long)B, (long long)M, (long long)N, (long long)K,\n"
        "               (long long)numMBlocks, (long long)dbgBlockNum);\n"
        "    }")

    # ---- 3) t1：tiling 生成结束 ----
    rep("t1 tiling 结束",
        "    // 不再 SetFixSplit / SetSplitRange：kernel 侧不依赖 baseN，",
        "    const auto tp1 = BmmsClock::now();   // ← tiling 生成结束\n"
        "\n"
        "    // 不再 SetFixSplit / SetSplitRange：kernel 侧不依赖 baseN，")

    # ---- 4) t2：显存申请 + tiling 搬运结束 ----
    rep("t2 malloc 结束",
        "    aclrtMemcpy(myTilingDev, sizeof(BmmsTiling), hostMyTiling.data(), sizeof(BmmsTiling),\n"
        "                ACL_MEMCPY_HOST_TO_DEVICE);",
        "    aclrtMemcpy(myTilingDev, sizeof(BmmsTiling), hostMyTiling.data(), sizeof(BmmsTiling),\n"
        "                ACL_MEMCPY_HOST_TO_DEVICE);\n"
        "\n"
        "    const auto tp2 = BmmsClock::now();   // ← 显存申请 + 搬运结束")

    # ---- 5) t3：kernel 执行结束 ----
    rep("t3 kernel 结束",
        "    // ---- 6. 同步后释放 ----\n"
        "    aclrtSynchronizeStream(stream);",
        "    // ---- 6. 同步后释放 ----\n"
        "    aclrtSynchronizeStream(stream);\n"
        "    const auto tp3 = BmmsClock::now();   // ← kernel 执行结束")

    # ---- 6) t4：释放结束 + 打印 ----
    rep("t4 打印",
        "    aclrtFree(cScratchDev);\n}",
        "    aclrtFree(cScratchDev);\n"
        "\n"
        "    const auto tp4 = BmmsClock::now();\n"
        "    if (s_bmmsTimingCalls < 8) {\n"
        "        auto bmmsUs = [](BmmsClock::time_point a, BmmsClock::time_point b) {\n"
        "            return std::chrono::duration<double, std::micro>(b - a).count();\n"
        "        };\n"
        "        const double tTiling = bmmsUs(tp0, tp1);\n"
        "        const double tMalloc = bmmsUs(tp1, tp2);\n"
        "        const double tKernel = bmmsUs(tp2, tp3);\n"
        "        const double tFree   = bmmsUs(tp3, tp4);\n"
        "        const double tTotal  = bmmsUs(tp0, tp4);\n"
        "        printf(\"[phase] tiling=%8.1fus  malloc=%8.1fus  kernel=%9.1fus  \"\n"
        "               \"free=%8.1fus  ||  合计=%9.1fus\\n\",\n"
        "               tTiling, tMalloc, tKernel, tFree, tTotal);\n"
        "    }\n"
        "    ++s_bmmsTimingCalls;\n"
        "}")

    open(DST, "w", encoding="utf-8", newline="\n").write(box[0])
    print(f"\n已写出 {DST}（{box[0].count(chr(10)) + 1} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
