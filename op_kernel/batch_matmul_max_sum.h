/**
 * BatchMatmulMaxSum —— Ascend C 核函数实现
 *
 * ============================ 阅读前必看 ============================
 * 本文件是**骨架**，不是可以直接拿分的完整实现。
 *
 * 已写出：整体流水结构、多核任务划分、两级归约的顺序、尾块与全负数
 *         的处理思路、关键 API 的调用位置。
 * 待补齐：从 GM 搬运数据到片上的具体 Tiling 细节、Matmul 对象初始化、
 *         分形（fractal）格式处理、Double Buffer 优化。
 *
 * 我在开发机上没有昇腾硬件，**无法编译验证**这份代码。
 * 上板前请对照比赛环境的 CANN 版本逐个核对 API 签名。
 * ==================================================================
 *
 * 整体思路（先求对，再求快）
 * -------------------------
 * 第 0 版（本骨架采用的策略）：
 *   1. 把 (b, m) 拉平成一维行号，行号区间分给各个核
 *   2. 每个核负责若干行，行与行之间完全独立，天然无数据竞争
 *   3. 每行沿 N 方向切块，逐块算点积、逐块更新最大值
 *   4. 每行算完得到一个标量，原子加到 y[b] 上
 *
 * 这版的性能不怎么样（每行都要重新读 x2 的一整行），但**逻辑正确、
 * 容易验证**。先让它精度全过，再按 README 里的优化路线图提速。
 *
 * 关于 y 的累加
 * ------------
 * 多个核可能同时处理同一个 b 的不同 m，都要往 y[b] 上加数。
 * 所以用 AtomicAdd（原子加）避免写冲突。
 * B 最大 64，冲突很少，原子操作的开销可以接受。
 */

#ifndef BATCH_MATMUL_MAX_SUM_KERNEL_H
#define BATCH_MATMUL_MAX_SUM_KERNEL_H

#include "kernel_operator.h"
#include "../op_host/batch_matmul_max_sum_tiling.h"

using namespace AscendC;

class KernelBatchMatmulMaxSum {
 public:
  __aicore__ inline KernelBatchMatmulMaxSum() {}

  /**
   * 初始化：把 GM 地址包装成 GlobalTensor，并解析 tiling 参数。
   *
   * GM_ADDR 是 NPU 全局内存的裸地址（可以理解成 C 里的 void*）。
   * GlobalTensor 是 Ascend C 给的"带形状的视图"，用法类似 numpy 数组。
   */
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                              GM_ADDR tiling, TPipe* pipe) {
    // ---------- 解析 tiling ----------
    auto* t = reinterpret_cast<__gm__ BatchMatmulMaxSumTilingData*>(tiling);
    batch_ = t->batch;
    m_ = t->m;
    n_ = t->n;
    k_ = t->k;
    transposeX1_ = (t->transposeX1 != 0);
    transposeX2_ = (t->transposeX2 != 0);
    totalRows_ = t->totalRows;
    rowsPerCore_ = t->rowsPerCore;
    tailRows_ = t->tailRows;
    nTileLen_ = t->nTileLen;
    nTileCount_ = t->nTileCount;
    nTailLen_ = t->nTailLen;
    kTileLen_ = t->kTileLen;
    kTileCount_ = t->kTileCount;
    kTailLen_ = t->kTailLen;
    mAlign_ = t->mAlign;
    nAlign_ = t->nAlign;
    kAlign_ = t->kAlign;
    isFp16_ = (t->isFp16 != 0);

    // ---------- 计算本核负责的行区间 [rowStart_, rowEnd_) ----------
    // 前 tailRows_ 个核各多拿 1 行，用来消化除不尽的余数。
    //
    // 例：totalRows_=10, usedCoreNum=3 -> rowsPerCore_=3, tailRows_=1
    //     核0: [0, 4)   核1: [4, 7)   核2: [7, 10)
    const uint32_t coreId = GetBlockIdx();
    if (coreId < tailRows_) {
      rowStart_ = coreId * (rowsPerCore_ + 1);
      rowEnd_ = rowStart_ + rowsPerCore_ + 1;
    } else {
      rowStart_ = tailRows_ * (rowsPerCore_ + 1) + (coreId - tailRows_) * rowsPerCore_;
      rowEnd_ = rowStart_ + rowsPerCore_;
    }
    // 防御：核数多于总行数时，本核可能分到空区间
    if (rowStart_ > totalRows_) rowStart_ = totalRows_;
    if (rowEnd_ > totalRows_) rowEnd_ = totalRows_;

    // ---------- 包装 GM 张量 ----------
    // 注意：这里用的是 **storage shape**（实际存储），不是逻辑形状。
    // transposeX1_=true 时 x1 在内存里是 [B, K, M]，x1Gm_ 就按这个来。
    //
    // 取数据时通过 Stride 来"跳着读"，从而在不真正搬动数据的前提下
    // 完成逻辑上的转置语义——这是赛题"存储布局规则"的关键。
    if (transposeX1_) {
      x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x1),
                            batch_ * k_ * m_);
    } else {
      x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x1),
                            batch_ * m_ * k_);
    }

    if (transposeX2_) {
      x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x2),
                            batch_ * n_ * k_);
    } else {
      x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x2),
                            batch_ * k_ * n_);
    }

    // 输出 y 是 fp32、长度 B
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), batch_);

    // ---------- 申请片上缓存（UB） ----------
    // TODO(需实测调整)：下面两块 buffer 的大小直接决定能开多大的 tile，
    //   开太大会申请失败，开太小则搬运次数变多、性能下降。
    //   建议先跑通，再用 msprof 看 UB 占用和 MTE 带宽来调。
    //
    // x1 一行（一个 K 向量）需要 k_ 个半精度数
    pipe->InitBuffer(x1Buf_, 1, kAlign_ * sizeof(half));
    // x2 一块（K 行 x nTileLen 列）
    pipe->InitBuffer(x2Buf_, 1, kAlign_ * nTileLen_ * sizeof(half));
    // 相似度结果 + 归约用的临时空间
    pipe->InitBuffer(simBuf_, 1, nTileLen_ * sizeof(float));

    // 单个标量的最大值累加器
    pipe->InitBuffer(maxBuf_, 1, 32);  // 32 字节对齐
  }

  /**
   * 主流程：本核处理自己负责的每一行。
   */
  __aicore__ inline void Process() {
    for (uint32_t row = rowStart_; row < rowEnd_; ++row) {
      // 把一维行号还原成 (b, m)
      const uint32_t b = row / m_;
      const uint32_t m = row % m_;

      // 算这一行的 MaxSim 值
      const float rowMax = ProcessOneRow(b, m);

      // 累加到 y[b]。
      // 用 AtomicAdd 是因为处理同一个 b 的不同行的核可能不止一个。
      // TODO(上板核对)：确认当前 CANN 版本 SetAtomicAdd 的调用方式；
      //   部分版本需要在 SetAtomicAdd 之后再 SetAtomicNone 复位。
      SetAtomicAdd<float>();
      // TODO(实现)：用 DataCopy 把 rowMax 写入 yGm_[b] 并做原子加。
      //   提示：可以用一个长度为 1 的 LocalTensor<float> 作为中转，
      //         调用 DataCopy(yGm_[b], localTensor, 1)。
      SetAtomicNone();
    }
  }

 private:
  /**
   * 计算第 b 组第 m 行的 R[b,m] = max over n of ( sum over k of X1*X2 )
   *
   * 这是整个算子的核心，也是最需要优化的地方。
   */
  __aicore__ inline float ProcessOneRow(uint32_t b, uint32_t m) {
    // ---------- 第 1 步：把 x1[b, m, :] 这一整行 K 个元素搬进片上 ----------
    // 逻辑上要取 X1[b, m, k]，k 从 0 到 K-1。
    //
    // 存储布局决定了怎么读：
    //   transposeX1_=false，存储 [B,M,K]：元素在 offset = b*M*K + m*K + k
    //      -> 连续！一次 DataCopy 就能搬完
    //   transposeX1_=true， 存储 [B,K,M]：元素在 offset = b*K*M + k*M + m
    //      -> 不连续，stride = M。DataCopy 支持带 stride 的搬运，
    //         但效率较差。这是本方案的一个性能瓶颈点。
    //
    // TODO(实现)：用 DataCopy 把这一行搬进 x1Buf_。
    //   非转置（连续）：DataCopy(x1Local, x1Gm_[b * m_ * k_ + m * k_], k_);
    //   转置（stride=M）：需要用 DataCopyPad 或循环逐块搬，
    //                     或改用 Matmul 的 transpose 能力在 Cube 侧解决。

    // ---------- 第 2 步：沿 N 方向分块，逐块求最大相似度 ----------
    // 关键：初值必须是"负无穷"，不能是 0！
    // 这正是赛题示例3 要考的点。
    //
    // 这里用 -3.4e38（float 能表示的最小值附近）作为负无穷的近似。
    // 之所以不用 -INFINITY，是因为部分 NPU 指令对 Inf 的处理
    // 与 CPU 不同，用有限大负数更稳妥。
    constexpr float kNegInf = -3.4e38f;
    float runningMax = kNegInf;

    for (uint32_t nTile = 0; nTile < nTileCount_; ++nTile) {
      const uint32_t nStart = nTile * nTileLen_;
      // 最后一块可能不满，要用 nTailLen_
      const uint32_t curLen = (nTile == nTileCount_ - 1) ? nTailLen_ : nTileLen_;

      // ---------- 第 2a 步：搬 x2 的一块 [K, curLen] ----------
      // 逻辑上要取 X2[b, k, n]，k=0..K-1，n=nStart..nStart+curLen-1
      //   transposeX2_=false，存储 [B,K,N]：offset = b*K*N + k*N + n
      //      -> 每行连续 curLen 个，行间 stride = N
      //   transposeX2_=true， 存储 [B,N,K]：offset = b*N*K + n*K + k
      //      -> 每列连续 K 个，需要按转置方式读
      //
      // TODO(实现)：把这块数据搬进 x2Buf_。

      // ---------- 第 2b 步：算点积，得到 curLen 个相似度 ----------
      // sim[n] = sum_k X1[b,m,k] * X2[b,k,n]
      //
      // 这一步应该交给 **Cube 单元**（矩阵乘单元）来做，而不是用
      // Vector 逐元素乘加——Cube 的算力高一个数量级。
      //
      // TODO(实现)：用 matmul::Matmul 完成这一步。
      //   形状：[1, K] x [K, curLen] -> [1, curLen]
      //   注意：M 方向只有 1 行，Cube 利用率极低（Cube 最小处理 16x16）。
      //   这正是"先把 B 组行凑成 16 行一起算"这种优化的动机所在，
      //   属于 README 优化路线图的第 4 步。
      //
      // 本骨架暂时演示 Vector 写法（正确但慢）：
      //   Mul(x1Broadcast, x2Local, tmp); 然后沿 K 维 ReduceSum

      // ---------- 第 2c 步：对本块的相似度取最大值，更新 runningMax ----------
      // TODO(实现)：
      //   ReduceMax(maxLocal, simLocal, simLocal, curLen);
      //   float tileMax = maxLocal.GetValue(0);
      //   runningMax = (tileMax > runningMax) ? tileMax : runningMax;
      //
      // ⚠️ 尾块陷阱：最后一块长度不足时，simBuf_ 里可能残留上一轮的
      //    旧数据，或者补零产生的 0。如果不处理，当整行真值都是负数时，
      //    这个残留的 0 会错误地成为最大值 —— 直接踩中示例3 的坑。
      //    处理办法：把尾块之后的位置显式填成 kNegInf。
    }

    // ---------- 第 3 步：返回本行的 MaxSim 值 ----------
    // 注意此处**还没有求和**。求和发生在 Process() 里对 y[b] 的原子加上，
    // 因为 y[b] 是所有 m 的 R[b,m] 之和，需要跨核汇总。
    return runningMax;
  }

  // ---------------- 成员变量 ----------------
  GlobalTensor<half> x1Gm_, x2Gm_;   // 输入（fp16 / bf16，这里以 half 占位）
  GlobalTensor<float> yGm_;          // 输出

  TBuf<TPosition::VECCALC> x1Buf_, x2Buf_, simBuf_, maxBuf_;

  uint32_t batch_ = 0, m_ = 0, n_ = 0, k_ = 0;
  bool transposeX1_ = false, transposeX2_ = false;

  uint32_t totalRows_ = 0, rowsPerCore_ = 0, tailRows_ = 0;
  uint32_t rowStart_ = 0, rowEnd_ = 0;

  uint32_t nTileLen_ = 0, nTileCount_ = 0, nTailLen_ = 0;
  uint32_t kTileLen_ = 0, kTileCount_ = 0, kTailLen_ = 0;
  uint32_t mAlign_ = 0, nAlign_ = 0, kAlign_ = 0;

  bool isFp16_ = true;
};

#endif  // BATCH_MATMUL_MAX_SUM_KERNEL_H
