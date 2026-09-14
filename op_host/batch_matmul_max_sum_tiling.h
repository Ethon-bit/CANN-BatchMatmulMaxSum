/**
 * BatchMatmulMaxSum —— Tiling 数据结构定义
 *
 * 什么是 Tiling？
 * --------------
 * "Tiling" 直译是"铺瓷砖"。在 NPU 上它的意思是：
 *
 *   数据太大了，塞不进片上缓存，所以要**切成小块（tile）**，
 *   一块一块搬进来算。怎么切、切多大、哪个核算哪一块 —— 这就是 Tiling。
 *
 * Host 侧（CPU）负责算好切分方案，把方案写进下面这个结构体，
 * 然后传给 Kernel 侧（NPU）让它照着执行。
 *
 * 对 Python 选手的类比
 * -------------------
 * 就像你把一个大任务拆成批次：
 *
 *     for i in range(0, len(data), batch_size):
 *         process(data[i : i + batch_size])
 *
 * 只不过这里切的是"哪些行归哪个核管""每次搬多少进缓存"，
 * 而且这些数字必须提前算好、写死在一个结构体里传给 NPU。
 *
 * ⚠️ 重要约束：这个结构体的成员必须是固定大小的基础类型（uint32_t 等），
 *    不能用 std::vector / 指针。因为它要被**按字节原样拷贝**到 NPU 侧，
 *    两边看到的内存布局必须完全一致。
 */

#ifndef BATCH_MATMUL_MAX_SUM_TILING_H
#define BATCH_MATMUL_MAX_SUM_TILING_H

#include <cstdint>

// Tiling 策略枚举：不同的形状规模走不同的切分策略
enum class BmmMaxSumTilingKey : uint32_t {
  // 策略0：B 足够大，直接按 B 维切给多个核（最理想，无数据竞争）
  kSplitByBatch = 0,
  // 策略1：B 很小（比如 B=1），核用不满，需要把 M 维也切开分给多个核
  kSplitByBatchAndM = 1,
};

struct BatchMatmulMaxSumTilingData {
  // ---------- 原始问题规模 ----------
  uint32_t batch = 0;        // B：批大小
  uint32_t m = 0;            // M：query token 数
  uint32_t n = 0;            // N：document token 数
  uint32_t k = 0;            // K：特征维

  // ---------- 属性（用 0/1 表示 bool，避免跨平台 bool 长度不一致） ----------
  uint32_t transposeX1 = 0;  // 0=false, 1=true
  uint32_t transposeX2 = 0;  // 0=false, 1=true

  // ---------- 多核切分方案 ----------
  uint32_t usedCoreNum = 0;  // 实际启用的核数
  uint32_t tilingKey = 0;    // 对应 BmmMaxSumTilingKey

  // 每个核负责的 (batch, m) 范围。
  // 采用"线性编号"方式：把 (b, m) 拉平成一个一维序号
  //     linear_id = b * M + m
  // 然后按 usedCoreNum 平均切分。这样无论是按 B 切还是按 B×M 切，
  // kernel 侧的处理逻辑都是同一套，不用写两份代码。
  uint32_t totalRows = 0;      // = B * M，需要归约的总行数
  uint32_t rowsPerCore = 0;    // 每个核分到多少行
  uint32_t tailRows = 0;       // 前 tailRows 个核要多处理 1 行（余数分配）

  // ---------- 片上缓存切分方案 ----------
  // N 维可能高达 8192，一次搬不进 UB，需要沿 N 维再切
  uint32_t nTileLen = 0;       // 每次沿 N 维处理多少列
  uint32_t nTileCount = 0;     // 沿 N 维一共切几块
  uint32_t nTailLen = 0;       // 最后一块的实际长度（可能不足 nTileLen）

  // K 维切分（Cube 单元的 K 方向分块）
  uint32_t kTileLen = 0;       // 每次沿 K 维处理多少
  uint32_t kTileCount = 0;     // 沿 K 维一共切几块
  uint32_t kTailLen = 0;       // 最后一块的实际长度

  // ---------- Cube（矩阵乘）分形大小 ----------
  // 昇腾 Cube 单元要求矩阵按 16x16 的分形（fractal）格式搬运，
  // 所以 M / N 方向需要向上对齐到 16 的倍数
  uint32_t mAlign = 0;         // M 向上对齐到 16 的结果
  uint32_t nAlign = 0;         // N 向上对齐到 16 的结果
  uint32_t kAlign = 0;         // K 向上对齐到 16 的结果

  // ---------- 其他 ----------
  uint32_t isFp16 = 0;         // 1=FLOAT16, 0=BFLOAT16（决定用哪个 Matmul 模板）
  uint32_t reserved = 0;       // 占位，保持结构体 8 字节对齐，方便以后加字段
};

#endif  // BATCH_MATMUL_MAX_SUM_TILING_H
