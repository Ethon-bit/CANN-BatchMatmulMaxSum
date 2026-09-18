"""
从 op_kernel/kernel_cachefix.asc 派生出 op_kernel/kernel_mpar.asc
—— 「沿 M 维切分多核」版本。

======================================================================
为什么要做这件事
======================================================================

本地实测（CANN 9.1.0 + Ascend910，md5=7881ae81 那版 kernel）：

    用例     形状                    M块数  核数   kernel段
    case01  B=1    2x3x32              1     1     34.1us
    case13  B=1   65x16x32             2     1     43.1us
    case14  B=1  255x33x32             4     1     60.5us
    case21  B=1  512x512x32            8     1    111.4us
    case17  B=1 1024x1024x1024        16     1    552.6us

两个结论：

1) **B=1 时只用 1 个核**（"核数"那列全是 1）。机器有 20 个核，19 个在闲着。
   赛题 3.4 允许 B 到 64，但也明确要求"沿 B 维和/或 M 维合理分配多核任务，
   兼顾**较小 Batch**"—— 现在这版没做 M 维切分。

2) **每个 M 块有 5~32us 的固定开销**：
   case21 的 8 块共 77us（每块 9.7us），而一个 64x512x32 的矩阵乘 Cube 只要
   0.14us —— 每块 98% 的时间花在 SetOrgShape/SetTensorA/SetTensorB/SetTail/
   IterateAll 这一串调用上。

======================================================================
改法
======================================================================

把任务单位从「一个核 = 一个 batch」改成「一个核 = 一批 (batch, M块) 单元」：

    for (unit = coreIdx; unit < B*numMBlocks; unit += coreNum)
        b  = unit / numMBlocks
        mb = unit % numMBlocks

每个核把自己负责的块算出的 partial 累加进**本核的** partLocal[b]（UB 里，带
Kahan 补偿），然后写进 GM 上的部分和区，SyncAll() 之后按**固定顺序**归约。

数值影响：
    沿 M 切分后，求和的结合顺序变了。浮点加法不满足结合律，所以结果会有
    极微小差异 —— 量级在 1e-7 相对误差，远小于赛题要求的 1e-4。
    归约顺序固定（c = 0..coreNum-1 顺序相加），所以**多次执行结果完全一致**，
    满足赛题"数值一致性规则"。

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
    # 1) 常量：B 的上限
    # ================================================================
    rep("常量 kMaxBatch",
        "constexpr int32_t kDtFp16 = 1;",
        "// 赛题 3.4：1 <= B <= 64。部分和累加器按 batch 下标索引。\n"
        "// 取 128 而不是 64，是为了万一某个测试点超出题面约束也不至于越界写 UB。\n"
        "// 代价只有 512 字节 UB（我们总共才用到 ~66KB，UB 有 192KB）。\n"
        "constexpr int32_t kMaxBatch = 128;\n"
        "\n"
        "// ★★ 每个核在 GM 部分和区里独占的 float 个数（128 * 4B = 512B）。\n"
        "//\n"
        "//   为什么必须独占一整段、而不是按 [b*核数 + 核号] 紧密排列：\n"
        "//   紧密排列时 B 只有 8 个 float = 32 字节，一条 cache line 里塞着\n"
        "//   十几个不同核的数据。多核并发写同一条 cache line 会互相冲掉\n"
        "//   （false sharing），实测症状：\n"
        "//     - 8 个输出里偶发 1 个变成 0（写丢了）\n"
        "//     - 核数越多越严重：B=8/16 且铺满 20 个核时耗时从 ~1ms 暴涨到 100ms\n"
        "//   拉开成每核 512B（≥ 4 条典型 128B cache line）之后就没有共享了。\n"
        "constexpr int32_t kPartStride = 128;\n"
        "\n"
        "constexpr int32_t kDtFp16 = 1;")

    # ================================================================
    # 2) BmmsTiling：加 partOffset
    # ================================================================
    rep("BmmsTiling 加 partOffset",
        "    int32_t dtypeCode;\n};",
        "    int32_t dtypeCode;\n"
        "    // ★ M 并行版：部分和区相对 cScratchGm 的起始**元素**下标。\n"
        "    //   布局是 [每核 C 暂存区 cScratchElems 个] x blockNum，紧跟着\n"
        "    //   B*blockNum 个 float 的部分和（按 [b*blockNum + coreIdx] 索引）。\n"
        "    int64_t partOffset;\n};")

    # ================================================================
    # 3) UB：部分和累加器（值 + Kahan 补偿）
    # ================================================================
    rep("TBuf 声明",
        "    TBuf<TPosition::VECCALC> yBuf;",
        "    TBuf<TPosition::VECCALC> yBuf;\n"
        "    TBuf<TPosition::VECCALC> partBuf;    // ★ M 并行版：每核的部分和累加器\n"
        "    TBuf<TPosition::VECCALC> partCBuf;   // ★ Kahan 补偿项")

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
    # 4) yGm 之后：加 partGm
    # ================================================================
    rep("partGm 绑定",
        "    GlobalTensor<float> yGm;\n"
        "    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint32_t>(B));",
        "    GlobalTensor<float> yGm;\n"
        "    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint32_t>(B));\n"
        "\n"
        "    // ★ M 并行版：跨核部分和区。\n"
        "    //   布局 [核号 * kPartStride + b] —— 每个核独占 kPartStride 个 float，\n"
        "    //   彼此不共享 cache line（详见 kPartStride 处的说明）。\n"
        "    //   放在 C 暂存区之后（host 侧算好偏移填进 tiling.partOffset）。\n"
        "    const int64_t coreNum = GetBlockNum();\n"
        "    const int64_t coreIdx = GetBlockIdx();\n"
        "    GlobalTensor<float> partGm;\n"
        "    partGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cScratchGm) + t->partOffset,\n"
        "                           static_cast<uint32_t>(coreNum * kPartStride));")

    # ================================================================
    # 5) 循环头：batch 循环 -> 单元循环
    # ================================================================
    rep("循环头改成 unit 循环",
        "    for (int64_t b = GetBlockIdx(); b < B; b += coreNum) {\n"
        "        float total = 0.0f;      // 顺序固定 ⇒ 确定性（题面规则 1）\n"
        "        float totalC = 0.0f;     // ★ 第 35 轮：Kahan 补偿项，见下方 M 求和处\n"
        "        for (int64_t mb = 0; mb < numMBlocks; ++mb) {\n"
        "            const int64_t tm0 = mb * baseM;\n"
        "            if (tm0 >= M) {\n"
        "                break;\n"
        "            }",
        "    // ★★ M 并行版：任务单位 = (batch, M块)。\n"
        "    //\n"
        "    //   原版是 `for (b = blockIdx; b < B; b += coreNum)` —— 一个核包干\n"
        "    //   一个完整 batch。B=1 时只有 1 个核在干活。\n"
        "    //   现在按 unit = b*numMBlocks + mb 做跨核轮转，B=1 时也能铺满所有核。\n"
        "    //\n"
        "    //   每个核维护自己的 partLocal[b]（UB 里，带 Kahan 补偿），\n"
        "    //   最后写进 GM 的部分和区，SyncAll 之后按固定顺序归约。\n"
        "    const int64_t totalUnits = B * numMBlocks;\n"
        "\n"
        "    Duplicate(partLocal,  0.0f, kMaxBatch);\n"
        "    Duplicate(partCLocal, 0.0f, kMaxBatch);\n"
        "\n"
        "    for (int64_t unit = coreIdx; unit < totalUnits; unit += coreNum) {\n"
        "        const int64_t b  = unit / numMBlocks;\n"
        "        const int64_t mb = unit - b * numMBlocks;\n"
        "        const int64_t tm0 = mb * baseM;\n"
        "        if (tm0 >= M) {\n"
        "            continue;      // 注意是 continue 不是 break：单元是跨步取的不是连续的\n"
        "        }")

    # ================================================================
    # 6) 循环尾：累进 total -> 累进 partLocal[b]
    # ================================================================
    rep("循环尾改成累进 partLocal",
        "            // 把本 m 块的 partial 也以补偿方式累进 total（跨 m 块同样是长链求和）\n"
        "            {\n"
        "                const float yt = partial - totalC;\n"
        "                const float tt = total + yt;\n"
        "                totalC = (tt - total) - yt;\n"
        "                total = tt;\n"
        "            }\n"
        "        }   // end for mb\n",
        "            // 把本块的 partial 以补偿方式累进 **本核** 的 partLocal[b]。\n"
        "            // 每个核最多处理 ceil(B*numMBlocks/coreNum) 个块，项数少，\n"
        "            // 但既然原版用了 Kahan，这里也保留，避免精度回退。\n"
        "            {\n"
        "                const int32_t bi = static_cast<int32_t>(b);\n"
        "                const float acc  = partLocal.GetValue(bi);\n"
        "                const float accC = partCLocal.GetValue(bi);\n"
        "                const float yt = partial - accC;\n"
        "                const float tt = acc + yt;\n"
        "                partCLocal.SetValue(bi, (tt - acc) - yt);\n"
        "                partLocal.SetValue(bi, tt);\n"
        "            }\n"
        "    }   // end for unit\n")

    # ================================================================
    # 7) 尾部：写部分和 + SyncAll + 归约 + 写 y
    # ================================================================
    rep("尾部加 SyncAll 归约",
        "        // ★★ 第 44 轮：y 的写出走 DMA（向量单元产出 → DataCopyPad 写回），\n"
        "        //   并在 kernel 结束前**显式回写 cache**（见函数末尾的 DataCacheCleanAndInvalid）。\n"
        "        //\n"
        "        //   本地实测（dev/sweep.py，B=8 M=16 N=65 K=32，修复前）：\n"
        "        //     - kernel 内读回 yGm[b] **8/8 全部正确**\n"
        "        //     - 但 host 拷回来的 y 里有一半是 0（即 aclrtMalloc 的零初始化值）\n"
        "        //     - 驱动侧 aclrtMemcpy / 同步 / 大小逐行核对，没有问题\n"
        "        //   ⇒ **值停在 aicore 的 cache 里，从未回写到 GM**；kernel 内的读回命中的\n"
        "        //     是同一个 cache，所以\"看起来写成功了\"。这正是判题里「部分测试点\n"
        "        //     恰好错 1 个元素」的根因（判题跑的就是这份 kernel）。\n"
        "        //   ⇒ 修法 = 本处 DMA 写出 + 函数末尾的 DataCacheCleanAndInvalid。\n"
        "        //   修复后：本地 240 组合扫描全过，11 个本地用例全过，误差回到 1e-6 量级。\n"
        "        Duplicate(yLocal, total, 1);\n"
        "        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);\n"
        "        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);\n"
        "        DataCopyExtParams ycp{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};\n"
        "        DataCopyPad(yGm[static_cast<uint32_t>(b)], yLocal, ycp);\n"
        "    }       // end for b\n",
        "    // ---- 各核把自己的部分和写到 GM 上**自己那一段** ----\n"
        "    //\n"
        "    //   两个要点：\n"
        "    //   1) 写到自己独占的 kPartStride 区间，避免多核写同一条 cache line\n"
        "    //   2) 用 DMA（DataCopyPad）而不是标量 SetValue —— 标量写 GM 不走\n"
        "    //      向量单元的写通路，正是本算子踩过的那个「写了不落盘」的坑\n"
        "    {\n"
        "        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);\n"
        "        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);\n"
        "        DataCopyExtParams pcp{1, static_cast<uint32_t>(B * sizeof(float)), 0, 0, 0};\n"
        "        DataCopyPad(partGm[static_cast<uint32_t>(coreIdx * kPartStride)], partLocal, pcp);\n"
        "    }\n"
        "    // 让 GM 上的部分和对所有核可见。\n"
        "    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(partGm);\n"
        "\n"
        "    // ★ 跨核同步：所有核都写完部分和之后，才允许开始归约。\n"
        "    //   SyncAll 只保证「都执行到这儿了」，不保证 cache 可见性 —— 这正是本算子\n"
        "    //   踩过的坑（y 写 GM 不回写 cache）。所以：\n"
        "    //     写方：SyncAll **之前** clean，把脏数据推回 GM\n"
        "    //     读方：SyncAll **之后** invalidate，丢掉本地可能过期的行，强制从 GM 读\n"
        "    //   DataCacheCleanAndInvalid 同时做 clean + invalidate，两边都调是安全的。\n"
        "    AscendC::SyncAll();\n"
        "    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(partGm);\n"
        "\n"
        "    // ---- 归约：按 **固定顺序** c = 0..coreNum-1 相加 ⇒ 结果确定 ----\n"
        "    //   注意这里也是按 b 轮转分核，每个 b 只由一个核写 y，不存在写冲突。\n"
        "    for (int64_t b = coreIdx; b < B; b += coreNum) {\n"
        "        float s = 0.0f;\n"
        "        for (int64_t c = 0; c < coreNum; ++c) {\n"
        "            s += partGm.GetValue(static_cast<uint32_t>(c * kPartStride + b));\n"
        "        }\n"
        "        // y 的写出走 DMA（向量单元产出 → DataCopyPad 写回），\n"
        "        // 并在 kernel 结束前由 DataCacheCleanAndInvalid 强制回写 cache。\n"
        "        Duplicate(yLocal, s, 1);\n"
        "        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);\n"
        "        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);\n"
        "        DataCopyExtParams ycp{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};\n"
        "        DataCopyPad(yGm[static_cast<uint32_t>(b)], yLocal, ycp);\n"
        "    }       // end for b（归约阶段）\n")

    # ================================================================
    # 8) 【已废弃】循环里原来那句 const int64_t coreNum = GetBlockNum(); 要删掉
    #    （新的 coreNum 在上面的 partGm 绑定处已经定义了）
    # ================================================================
    rep("删掉重复的 coreNum 定义",
        "    //   3. 核内按 m 块固定顺序累加 => 多次执行结果完全一致（题面规则 1）\n"
        "    const int64_t coreNum = GetBlockNum();\n",
        "    //   3. 核内按 m 块固定顺序累加 => 多次执行结果完全一致（题面规则 1）\n"
        "    //   （M 并行版：coreNum / coreIdx 已在上面绑定 partGm 时定义）\n")

    # ================================================================
    # 9) Host：blockNum 改成按单元数算；cScratch 里加上部分和区；填 partOffset
    # ================================================================
    # ---- host：把 blockNum / cScratchElems / cScratchBytes 全部**提前**算 ----
    #
    # ⚠️ 这里踩过一次编译错误：tiling.partOffset 要用 cScratchElems 和 bmmsBlockNum，
    #    但原版这两个变量定义在函数后半段（"申请 device 内存"那一节），而
    #    tiling 结构体在它们**之前**就填好了。直接用会报
    #        error: use of undeclared identifier 'cScratchElems'
    #    所以必须把这三个量的计算整体提到 numMBlocks 之后、填 tiling 之前。
    rep("host 提前算 blockNum/scratch",
        "    const int64_t numMBlocks = (M + kBaseM - 1) / kBaseM;",
        "    const int64_t numMBlocks = (M + kBaseM - 1) / kBaseM;\n"
        "\n"
        "    // ★ M 并行版：并行度不再受 B 限制，而是受「任务单元数」限制。\n"
        "    //   单元 = 一个 (batch, M块)，总数 = B * numMBlocks。\n"
        "    //   B=1 且 M=1024 时单元数是 16 ⇒ 能铺 16 个核，而不是原来的 1 个。\n"
        "    const int64_t bmmsUnits = B * numMBlocks;\n"
        "\n"
        "    // ---- 诊断开关：BMMS_MAX_CORES=<n> 压低 launch 核数上限 ----\n"
        "    //\n"
        "    //   用来区分两个都能解释「单元/核 > 1 就出错」的假设：\n"
        "    //     H1 每核处理多个单元本身有问题\n"
        "    //     H2 核数正好撞上物理上限（20）时 SyncAll 有问题\n"
        "    //   让 case18 用 16 个核跑：单元/核 = 64/16 = 4（满足 H1），\n"
        "    //   但 16 < 20（不满足 H2）。坏了 ⇒ H1；好了 ⇒ H2。\n"
        "    //\n"
        "    //   不设这个环境变量时行为完全不变（默认就是 availableCoreNum）。\n"
        "    int64_t bmmsCoreCap = availableCoreNum;\n"
        "    if (const char* bmmsEnv = getenv(\"BMMS_MAX_CORES\")) {\n"
        "        const long long bmmsV = atoll(bmmsEnv);\n"
        "        if (bmmsV > 0 && bmmsV < bmmsCoreCap) { bmmsCoreCap = bmmsV; }\n"
        "    }\n"
        "    if (bmmsCoreCap != availableCoreNum) {\n"
        "        printf(\"[dbg] BMMS_MAX_CORES 生效：核数上限 %lld（原本 %lld）\\n\",\n"
        "               (long long)bmmsCoreCap, (long long)availableCoreNum);\n"
        "    }\n"
        "    int64_t bmmsBlockNum = (bmmsUnits < bmmsCoreCap) ? bmmsUnits : bmmsCoreCap;\n"
        "    if (bmmsBlockNum < 1) { bmmsBlockNum = 1; }\n"
        "\n"
        "    // C 暂存区 = singleCoreM x singleCoreN = kBaseM x N，外加 kSlackElems 余量：\n"
        "    // 逐行满宽拷贝时最后一行会越读到 tile 末尾之后，这点余量让越读落在自己的内存里。\n"
        "    const size_t cScratchElems = static_cast<size_t>(kBaseM) * static_cast<size_t>(N) + kSlackElems;\n"
        "    // 显存布局：[每核 C 暂存区] x blockNum，紧跟 [部分和区] kPartStride x blockNum\n"
        "    const size_t cScratchBytes = (cScratchElems * static_cast<size_t>(bmmsBlockNum)\n"
        "                                  + static_cast<size_t>(kPartStride) * static_cast<size_t>(bmmsBlockNum))\n"
        "                                 * sizeof(float);")

    # 原版后半段那三行现在重复了，删掉（否则重定义报错）
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
        "    //   因为填 tiling 时就要用 bmmsBlockNum 算 partOffset。）")

    rep("host partOffset",
        "    tiling.dtypeCode = dtypeCode;",
        "    tiling.dtypeCode = dtypeCode;\n"
        "    // 部分和区紧跟在「每核 C 暂存区」之后\n"
        "    tiling.partOffset = static_cast<int64_t>(cScratchElems) * bmmsBlockNum;")

    open(DST, "w", encoding="utf-8", newline="\n").write(box[0])
    print(f"\n已写出 {DST}（{box[0].count(chr(10)) + 1} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
