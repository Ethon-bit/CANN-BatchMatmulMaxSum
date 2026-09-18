"""
从 op_kernel/kernel_cachefix.asc 生成一组"计时变体" kernel。

为什么需要变体：
    本地实测发现一次 run_kernel 要 ~80us，而其中真正算矩阵的时间约等于 0
    （case01 和 case09 数据量差 21 倍，耗时却一模一样）。也就是说时间全在
    "准备工作"上。但光知道"全是固定开销"还不够，得知道**具体是哪一段**，
    才能决定改哪里。

    单看一个数分不清，所以做**受控对比**：只改一个地方，其余完全不动，
    两版相减，差值就是那一处的代价。

生成三个文件（都在 op_kernel/ 下）：

    kernel_timing.asc          V0 基线：完整 + 5 个计时点
    kernel_timing_nocache.asc  V1：只删掉 DataCacheCleanAndInvalid 那一行
    kernel_timing_empty.asc    V2：设备侧立刻 return（空 kernel）

    ★ V0 - V1 = 整 cache 刷新的代价（launch/sync 开销在两者里相同，相减自动抵消）
    ★ V2      = 我们这套测量方法的固定底噪（launch + sync 的地板）

    ⚠️ V2 的算术结果是错的（它什么都不算）—— 这是**故意的**，
       它只用来量时间，不要拿它的正确性做任何判断。

用法：
    python3 scripts/make_timing_kernel.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))       # 仓库根
SRC = os.path.join(ROOT, "op_kernel", "kernel_cachefix.asc")


def build_variant(text, name, extra=None):
    """在 cachefix 基础上插入计时点；extra 是 (锚点, 替换内容) 的额外改动。"""

    box = [text]

    def rep(tag, old, new):
        n = box[0].count(old)
        if n != 1:
            print(f"    [失败] {tag}: 匹配到 {n} 处（应为 1 处）")
            sys.exit(1)
        box[0] = box[0].replace(old, new)

    # ---- 0) 引入 <chrono> ----
    rep("include", "#include <cmath>", "#include <cmath>\n#include <chrono>")

    # ---- 1) t0：刚进 run_kernel ----
    rep("t0", "    (void)info_y;",
        "    (void)info_y;\n"
        "\n"
        "    // ===== 分阶段计时（仅 harness 使用，不影响算子逻辑）=====\n"
        "    using BmmsClock = std::chrono::steady_clock;\n"
        "    static int s_bmmsTimingCalls = 0;\n"
        "    const auto tp0 = BmmsClock::now();")

    # ---- 2) 形状回显 ----
    rep("shape", "    const int64_t numMBlocks = (M + kBaseM - 1) / kBaseM;",
        "    const int64_t numMBlocks = (M + kBaseM - 1) / kBaseM;\n"
        "\n"
        "    const int64_t dbgBlockNum = (B < availableCoreNum) ? B : availableCoreNum;\n"
        "    if (s_bmmsTimingCalls < 8) {\n"
        "        printf(\"[shape] B=%lld M=%lld N=%lld K=%lld  numMBlocks=%lld  核数=%lld\\n\",\n"
        "               (long long)B, (long long)M, (long long)N, (long long)K,\n"
        "               (long long)numMBlocks, (long long)dbgBlockNum);\n"
        "    }")

    # ---- 3) t1 ----
    rep("t1", "    // 不再 SetFixSplit / SetSplitRange：kernel 侧不依赖 baseN，",
        "    const auto tp1 = BmmsClock::now();   // ← tiling 生成结束\n"
        "\n"
        "    // 不再 SetFixSplit / SetSplitRange：kernel 侧不依赖 baseN，")

    # ---- 4) t2 ----
    rep("t2",
        "    aclrtMemcpy(myTilingDev, sizeof(BmmsTiling), hostMyTiling.data(), sizeof(BmmsTiling),\n"
        "                ACL_MEMCPY_HOST_TO_DEVICE);",
        "    aclrtMemcpy(myTilingDev, sizeof(BmmsTiling), hostMyTiling.data(), sizeof(BmmsTiling),\n"
        "                ACL_MEMCPY_HOST_TO_DEVICE);\n"
        "\n"
        "    const auto tp2 = BmmsClock::now();   // ← 显存申请 + 搬运结束")

    # ---- 5) t3 ----
    rep("t3", "    // ---- 6. 同步后释放 ----\n    aclrtSynchronizeStream(stream);",
        "    // ---- 6. 同步后释放 ----\n"
        "    aclrtSynchronizeStream(stream);\n"
        "    const auto tp3 = BmmsClock::now();   // ← kernel 执行结束")

    # ---- 6) t4 + 打印 ----
    rep("t4", "    aclrtFree(cScratchDev);\n}",
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

    # ---- 7) 变体专属改动 ----
    if extra:
        rep(extra[0], extra[1], extra[2])

    out = os.path.join(ROOT, "op_kernel", name)
    open(out, "w", encoding="utf-8", newline="\n").write(box[0])
    print(f"    -> {name}")


# ---------------------------------------------------------------------------

V1_ANCHOR = ("删掉 cache 回写",
    "    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(yGm);",
    "    // ★★ V1 变体：这一行被**故意删掉**了。\n"
    "    //   原版靠它把 aicore cache 里的脏数据刷回 GM，是拿到 15/15 的关键。\n"
    "    //   这里删掉只为测量它的耗时，**算术结果是错的，不要用它提交**。\n"
    "    // AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(yGm);")

V2_ANCHOR = ("设备侧提前返回",
    "    const int32_t isTransB = t->isTransB;",
    "    const int32_t isTransB = t->isTransB;\n"
    "\n"
    "    // ★★ V2 变体：立刻返回，什么都不算。\n"
    "    //   目的：量出 launch + sync 的**地板开销**，作为解读 V0/V1 的参照。\n"
    "    //   条件恒成立（B/M/N/K 等都是非负数），只是为了不让变量变成未使用。\n"
    "    //   ⚠️ 它不写 y，算术结果必然是错的 —— 这是故意的。\n"
    "    if (B + M + N + K + numMBlocks + isTransA + isTransB >= 0) {\n"
    "        return;\n"
    "    }")


def main():
    if not os.path.isfile(SRC):
        print(f"找不到源文件：{SRC}")
        return 1

    text = open(SRC, encoding="utf-8").read()

    print("生成计时变体：")
    build_variant(text, "kernel_timing.asc")
    build_variant(text, "kernel_timing_nocache.asc", V1_ANCHOR)
    build_variant(text, "kernel_timing_empty.asc", V2_ANCHOR)

    print("\n完成。用 bench_variants.sh 一键对比。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
