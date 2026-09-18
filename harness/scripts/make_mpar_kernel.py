"""
从 op_kernel/kernel_cachefix.asc 派生出 op_kernel/kernel_mpar.asc
—— 「安全混合版」：能用 M 并行时用，不能用时原样退回。

======================================================================
为什么要做成混合版
======================================================================

本地实测发现一条**完美分隔**的规律（M 并行版，20 核机器）：

    用例     单元数  核数   单元/核   结果
    case09     8      8      1.0     ✅  83us
    case10     8      8      1.0     ✅  86us
    case13     2      2      1.0     ✅  83us
    case14     4      4      1.0     ✅  87us
    case17    16     16      1.0     ✅ 146us
    case21     8      8      1.0     ✅  98us
    case23     8      8      1.0     ✅  87us
    ---------------------------------------------------------------
    case18    64     20      3.2     ❌  80ms，7/8 错
    case19   256     20     12.8     ❌  94ms，16/16 错

**单元/核 == 1 的全部正常，> 1 的全部出事，没有例外。**

「每核多个单元」那条路径的责任还没查清（见 kernel_mpar 的调试记录）。
在查清之前，正确的工程做法是：**只用已验证的那条路径**。

======================================================================
本版本的行为
======================================================================

    host 侧判断：  useMpar = (B * numMBlocks < availableCoreNum)
                   （严格小于，刻意留一个核的余量）

    useMpar 为真  →  blockNum = B*numMBlocks，每个核恰好分到 1 个单元
                     ⇒ 走 M 并行路径（本地七个用例全部验证过）
    useMpar 为假  →  blockNum = min(B, availableCoreNum)
                     ⇒ 走**原版路径**，代码与原 kernel_cachefix 逐字相同

分支判据由 host 算好写进 tiling（BmmsTiling.useMpar），所有核读到同一个值，
因此不存在「有的核进 if、有的核进 else」导致 SyncAll 缺席死锁的问题。

======================================================================
收益预期
======================================================================

    用例     形状                单元数   走哪条路      提速
    case17  B=1 1024x1024x1024    16    M 并行       4.15x（已实测）
    case21  B=1 512x512x32         8    M 并行       1.63x（已实测）
    case14  B=1 255x33x32          4    M 并行       1.30x（已实测）
    其余（B 大、或 M 块数 >= 核数）      原版路径      1.00x（无变化，但无风险）

也就是说：**只在有把握的地方提速，没把握的地方保持原样。**

======================================================================
用法
======================================================================

    python3 scripts/make_mpar_kernel.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "op_kernel", "kernel_cachefix.asc")
DST = os.path.join(ROOT, "op_kernel", "kernel_mpar.asc")

# 原版循环里「一个 M 块的计算主体」的起止锚点（用来原样搬进 M 并行路径）
INNER_BEGIN = "            const int32_t tmCnt = static_cast<int32_t>("
INNER_END = "            // 把本 m 块的 partial 也以补偿方式累进 total"

LOOP_HEADER = "    for (int64_t b = GetBlockIdx(); b < B; b += coreNum) {"
LOOP_TAIL = "    }       // end for b"


def main():
    if not os.path.isfile(SRC):
        print(f"找不到源文件：{SRC}")
        return 1

    text = open(SRC, encoding="utf-8").read()
    box = [text]

    def rep(tag, old, new):
        n = box[0].count(old)
        if n != 1:
            print(f"  [失败] {tag}: 匹配到 {n} 处（应为 1 处）")
            sys.exit(1)
        box[0] = box[0].replace(old, new)
        print(f"  [ok] {tag}")

    # ================================================================
    # 0) 先把「M 块计算主体」原样抠出来 —— M 并行路径要复用它
    #    （从 tmCnt 开始，到 Kahan 累进 total 之前为止）
    # ================================================================
    i0 = text.index(INNER_BEGIN)
    i1 = text.index(INNER_END)
    INNER = text[i0:i1]
    print(f"  [ok] 抠出计算主体 {INNER.count(chr(10))} 行")
    assert "DataCopyPad(cLocal" in INNER and "WholeReduceMax" in INNER

    # ================================================================
    # 0b) 补两个标准头 —— 本文件用了 printf / getenv / atoll，
    #     而之前提交的 kernel_cachefix 没用过这些。本文件是被平台的包装
    #     头文件 #include 进去的，不能假设它一定提供了 <cstdio>/<cstdlib>。
    #     自己显式包含，零成本消除一个编译失败的风险。
    # ================================================================
    rep("include cstdio / cstdlib",
        "#include <cmath>",
        "#include <cmath>\n"
        "#include <cstdio>    // printf —— [mpar] 那行路径诊断输出\n"
        "#include <cstdlib>   // getenv / atoll —— BMMS_MAX_CORES 诊断开关")

    # ================================================================
    # 1) 常量
    # ================================================================
    rep("常量 kMaxBatch / kPartStride",
        "constexpr int32_t kDtFp16 = 1;",
        "// 赛题 3.4：1 <= B <= 64。部分和累加器按 batch 下标索引，取 128 留余量。\n"
        "constexpr int32_t kMaxBatch = 128;\n"
        "\n"
        "// ★★ 每个核在 GM 部分和区里独占的 float 个数（128 * 4B = 512B）。\n"
        "//   紧密排列（partGm[b*核数 + 核号]）时 B 只有 8 个 float = 32 字节，\n"
        "//   一条 cache line 里塞着十几个核的数据，多核并发写会互相冲掉\n"
        "//   （false sharing）。拉开成每核 512B 就没有共享了 —— 实测这条修完，\n"
        "//   case10 从「1/8 错」变成通过。\n"
        "constexpr int32_t kPartStride = 128;\n"
        "\n"
        "constexpr int32_t kDtFp16 = 1;")

    # ================================================================
    # 2) BmmsTiling：加 useMpar
    # ================================================================
    rep("BmmsTiling 加 useMpar / partOffset",
        "    int32_t dtypeCode;\n};",
        "    int32_t dtypeCode;\n"
        "    // ★ 是否走 M 并行路径。由 host 算好，所有核读到同一个值，\n"
        "    //   保证不会出现「部分核进 if、部分核进 else」而导致 SyncAll 缺席。\n"
        "    int32_t useMpar;\n"
        "    // 部分和区相对 cScratchGm 的元素下标（布局 [核号*kPartStride + b]）。\n"
        "    int64_t partOffset;\n};")

    # ================================================================
    # 3) UB：部分和累加器
    # ================================================================
    rep("TBuf 声明",
        "    TBuf<TPosition::VECCALC> yBuf;",
        "    TBuf<TPosition::VECCALC> yBuf;\n"
        "    TBuf<TPosition::VECCALC> partBuf;    // M 并行路径：每核的部分和累加器\n"
        "    TBuf<TPosition::VECCALC> partCBuf;   // 对应的 Kahan 补偿项")

    rep("InitBuffer",
        "    pipe.InitBuffer(yBuf, 8 * sizeof(float));                // 32B，够放一个 fp32",
        "    pipe.InitBuffer(yBuf, 8 * sizeof(float));                // 32B，够放一个 fp32\n"
        "    pipe.InitBuffer(partBuf,  kMaxBatch * sizeof(float));    // 128*4 = 512B\n"
        "    pipe.InitBuffer(partCBuf, kMaxBatch * sizeof(float));")

    rep("LocalTensor",
        "    LocalTensor<float> yLocal = yBuf.Get<float>();",
        "    LocalTensor<float> yLocal = yBuf.Get<float>();\n"
        "    LocalTensor<float> partLocal  = partBuf.Get<float>();\n"
        "    LocalTensor<float> partCLocal = partCBuf.Get<float>();")

    # ================================================================
    # 4) yGm 之后：绑定 partGm、算 totalUnits
    # ================================================================
    rep("绑定 partGm",
        "    GlobalTensor<float> yGm;\n"
        "    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint32_t>(B));",
        "    GlobalTensor<float> yGm;\n"
        "    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint32_t>(B));\n"
        "\n"
        "    // M 并行路径要用的跨核部分和区（原版路径用不到，绑好放着不花钱）。\n"
        "    const int64_t coreIdx = GetBlockIdx();\n"
        "    GlobalTensor<float> partGm;\n"
        "    partGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cScratchGm) + t->partOffset,\n"
        "                           static_cast<uint32_t>(coreNum * kPartStride));")

    # ================================================================
    # 5) coreNum 原来定义在 batch 循环前面，现在提前到上面那段用得到，
    #    这里把重复的定义删掉
    # ================================================================
    rep("删掉重复的 coreNum 定义",
        "    //   3. 核内按 m 块固定顺序累加 => 多次执行结果完全一致（题面规则 1）\n"
        "    const int64_t coreNum = GetBlockNum();\n",
        "    //   3. 核内按 m 块固定顺序累加 => 多次执行结果完全一致（题面规则 1）\n"
        "    //   （coreNum 已在上面绑定 partGm 时定义）\n")

    # coreNum 现在还没定义 —— 补在 coreIdx 前面
    rep("补上 coreNum 定义",
        "    const int64_t coreIdx = GetBlockIdx();\n"
        "    GlobalTensor<float> partGm;",
        "    const int64_t coreNum = GetBlockNum();\n"
        "    const int64_t coreIdx = GetBlockIdx();\n"
        "    GlobalTensor<float> partGm;")

    # ================================================================
    # 6) 把 if (useMpar) { ...M并行... } else { 原版循环 } 拼出来
    # ================================================================
    MPAR = (
        "    if (t->useMpar != 0) {\n"
        "        // ============ M 并行路径 ============\n"
        "        // 前提：host 已经保证 B*numMBlocks < availableCoreNum，\n"
        "        // 也就是**每个核恰好分到 1 个单元** —— 这条路径本地七个用例全过。\n"
        "        const int64_t totalUnits = B * numMBlocks;\n"
        "\n"
        "        Duplicate(partLocal,  0.0f, kMaxBatch);\n"
        "        Duplicate(partCLocal, 0.0f, kMaxBatch);\n"
        "\n"
        "        for (int64_t unit = coreIdx; unit < totalUnits; unit += coreNum) {\n"
        "            const int64_t b  = unit / numMBlocks;\n"
        "            const int64_t mb = unit - b * numMBlocks;\n"
        "            const int64_t tm0 = mb * baseM;\n"
        "            if (tm0 >= M) {\n"
        "                continue;   // 单元是跨步取的，不能 break\n"
        "            }\n"
        "\n"
        + INNER +
        "            // 把本块的 partial 以补偿方式累进本核的 partLocal[b]\n"
        "            {\n"
        "                const int32_t bi = static_cast<int32_t>(b);\n"
        "                const float acc  = partLocal.GetValue(bi);\n"
        "                const float accC = partCLocal.GetValue(bi);\n"
        "                const float yt = partial - accC;\n"
        "                const float tt = acc + yt;\n"
        "                partCLocal.SetValue(bi, (tt - acc) - yt);\n"
        "                partLocal.SetValue(bi, tt);\n"
        "            }\n"
        "        }   // end for unit\n"
        "\n"
        "        // ---- 各核把部分和写到自己独占的那一段（DMA，不是标量写）----\n"
        "        {\n"
        "            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);\n"
        "            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);\n"
        "            DataCopyExtParams pcp{1, static_cast<uint32_t>(B * sizeof(float)), 0, 0, 0};\n"
        "            DataCopyPad(partGm[static_cast<uint32_t>(coreIdx * kPartStride)],\n"
        "                        partLocal, pcp);\n"
        "        }\n"
        "        // 写方在 SyncAll 之前 clean，把脏数据推回 GM\n"
        "        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(partGm);\n"
        "\n"
        "        // 跨核同步：所有核都写完部分和之后才开始归约\n"
        "        AscendC::SyncAll();\n"
        "        // 读方在 SyncAll 之后 invalidate，丢掉本地可能过期的行\n"
        "        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(partGm);\n"
        "\n"
        "        // ---- 归约：按固定顺序 c = 0..coreNum-1 相加 ⇒ 结果确定 ----\n"
        "        for (int64_t b = coreIdx; b < B; b += coreNum) {\n"
        "            float s = 0.0f;\n"
        "            for (int64_t c = 0; c < coreNum; ++c) {\n"
        "                s += partGm.GetValue(static_cast<uint32_t>(c * kPartStride + b));\n"
        "            }\n"
        "            Duplicate(yLocal, s, 1);\n"
        "            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);\n"
        "            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);\n"
        "            DataCopyExtParams ycp{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};\n"
        "            DataCopyPad(yGm[static_cast<uint32_t>(b)], yLocal, ycp);\n"
        "        }   // end for b（归约阶段）\n"
        "    } else {\n"
        "    // ============ 原版路径（与 kernel_cachefix.asc 逐字相同）============\n"
    )

    rep("包成 if/else 两条路径", LOOP_HEADER, MPAR + LOOP_HEADER)

    # 原版循环结束后补一个右括号关掉 else
    rep("补上 else 的右括号",
        LOOP_TAIL + "\n",
        LOOP_TAIL + "\n"
        "    }   // end if (t->useMpar != 0)\n")

    # ================================================================
    # 7) host：算 useMpar、定 blockNum、算 cScratch 大小
    # ================================================================
    rep("host 提前算 useMpar/blockNum/scratch",
        "    const int64_t numMBlocks = (M + kBaseM - 1) / kBaseM;",
        "    const int64_t numMBlocks = (M + kBaseM - 1) / kBaseM;\n"
        "\n"
        "    // ★ 是否走 M 并行：只有「任务单元数严格小于核数」时才走，\n"
        "    //   也就是**每个核恰好分到 1 个单元**。\n"
        "    //\n"
        "    //   刻意用严格小于（而不是 <=）留一个核的余量：本地实测中，\n"
        "    //   单元/核 > 1 的路径有未查清的竞态（80ms + 结果错），\n"
        "    //   而「核数正好等于物理上限」也在嫌疑名单上（尚未排除）。\n"
        "    //   少用一个核换确定性，值。\n"
        "    const int64_t bmmsUnits = B * numMBlocks;\n"
        "    const bool bmmsUseMpar = (bmmsUnits < availableCoreNum);\n"
        "\n"
        "    // 诊断开关：BMMS_MAX_CORES=<n> 压低核数上限（不设则行为不变）\n"
        "    int64_t bmmsCoreCap = availableCoreNum;\n"
        "    if (const char* bmmsEnv = getenv(\"BMMS_MAX_CORES\")) {\n"
        "        const long long bmmsV = atoll(bmmsEnv);\n"
        "        if (bmmsV > 0 && bmmsV < bmmsCoreCap) { bmmsCoreCap = bmmsV; }\n"
        "    }\n"
        "    if (bmmsCoreCap != availableCoreNum) {\n"
        "        printf(\"[dbg] BMMS_MAX_CORES 生效：核数上限 %lld（原本 %lld）\\n\",\n"
        "               (long long)bmmsCoreCap, (long long)availableCoreNum);\n"
        "    }\n"
        "\n"
        "    int64_t bmmsBlockNum = bmmsUseMpar\n"
        "                               ? bmmsUnits\n"
        "                               : ((B < bmmsCoreCap) ? B : bmmsCoreCap);\n"
        "    if (bmmsBlockNum < 1) { bmmsBlockNum = 1; }\n"
        "\n"
        "    printf(\"[mpar] 单元数=%lld 核数=%lld 路径=%s\\n\",\n"
        "           (long long)bmmsUnits, (long long)bmmsBlockNum,\n"
        "           bmmsUseMpar ? \"M并行(每核1单元)\" : \"原版(每核1batch)\");\n"
        "\n"
        "    // C 暂存区 = kBaseM x N + kSlackElems，每核一份；\n"
        "    // 后面紧跟部分和区：kPartStride x blockNum（M 并行路径才用，但一起申请省事）\n"
        "    const size_t cScratchElems = static_cast<size_t>(kBaseM) * static_cast<size_t>(N) + kSlackElems;\n"
        "    const size_t cScratchBytes = (cScratchElems * static_cast<size_t>(bmmsBlockNum)\n"
        "                                  + static_cast<size_t>(kPartStride) * static_cast<size_t>(bmmsBlockNum))\n"
        "                                 * sizeof(float);")

    rep("host 删掉后半段的重复定义",
        "    // C 暂存区 = singleCoreM x singleCoreN = kBaseM x N，外加 kSlackElems(=kTileN) 余量：\n"
        "    // 逐行满宽拷贝时最后一行会越读到 tile 末尾之后，这点余量让越读落在自己的内存里。\n"
        "    // 大小完全由 host 自己的 SetShape 决定，\n"
        "    // 不需要回读 tiling 的私有字段。\n"
        "    const size_t cScratchElems = static_cast<size_t>(kBaseM) * static_cast<size_t>(N) + kSlackElems;\n"
        "    // ★OPT-1 每核一份 scratch ⇒ 总大小要乘核数。blockNum 必须在 malloc **之前**\n"
        "    //   就算出来（原版是后面才算的）。\n"
        "    int64_t bmmsBlockNum = (B < availableCoreNum) ? B : availableCoreNum;\n"
        "    if (bmmsBlockNum < 1) { bmmsBlockNum = 1; }\n"
        "    const size_t cScratchBytes = cScratchElems * static_cast<size_t>(bmmsBlockNum) * sizeof(float);",
        "    // （cScratchElems / bmmsBlockNum / cScratchBytes 已在函数开头算好，\n"
        "    //   因为填 tiling 时就要用它们算 partOffset。）")

    rep("host 填 tiling 的两个新字段",
        "    tiling.dtypeCode = dtypeCode;",
        "    tiling.dtypeCode = dtypeCode;\n"
        "    tiling.useMpar = bmmsUseMpar ? 1 : 0;\n"
        "    // 部分和区紧跟在「每核 C 暂存区」之后\n"
        "    tiling.partOffset = static_cast<int64_t>(cScratchElems) * bmmsBlockNum;")

    open(DST, "w", encoding="utf-8", newline="\n").write(box[0])
    print(f"\n已写出 {DST}（{box[0].count(chr(10)) + 1} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
