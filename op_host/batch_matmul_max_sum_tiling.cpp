/**
 * BatchMatmulMaxSum —— Tiling 计算实现（Host 侧 / CPU 上执行）
 *
 * 这个文件在什么时候跑？
 * ---------------------
 * 在你调用算子的那一刻，**在 CPU 上先跑一遍**，算出"怎么切分"，
 * 然后才把切分方案和真实数据一起丢给 NPU。
 *
 * 它跑一次，NPU kernel 跑很多次。所以这里可以稍微"奢侈"一点，
 * 但也不能太慢——Tiling 的耗时也算在端到端时间里的。
 *
 * 核心任务有三件：
 *   1. 校验输入合法性（形状、类型、K 是否为 8 的倍数……）
 *   2. 决定用几个核、每个核算哪部分
 *   3. 决定沿 N 和 K 怎么切块
 *
 * ⚠️ 状态说明
 * -----------
 * 本文件是**可编译的骨架**：校验逻辑和切分逻辑已写出完整思路与公式，
 * 但**未在真实 NPU 环境验证过**（开发机上没有昇腾硬件）。
 * 上板前请对照比赛环境的 CANN 版本核对 API 名称与参数。
 */

#include "batch_matmul_max_sum_tiling.h"

#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// ---------------- 一些经验常数 ----------------
// 这些值决定"一块搬多大"，直接影响性能，需要实测调优。

// 单个核最多处理多少行 (b,m)。太大则负载不均，太小则调度开销占比高。
constexpr uint32_t kMaxRowsPerCore = 256;

// UB（统一缓冲区）一次能容纳的 N 方向长度上限（元素个数，按 fp32 计）。
// 昇腾 910B 的 UB 大小约 192KB，这里保守取一半留作双缓冲。
constexpr uint32_t kMaxNTileLen = 2048;

// K 方向切块长度：Cube 的 K 方向以 16 为分形单位，取 128 是常见的甜点值
constexpr uint32_t kKTileLen = 128;

// Cube 分形对齐要求
constexpr uint32_t kFractalAlign = 16;

// K 必须是 8 的倍数（赛题 3.4 节硬性约束）
constexpr uint32_t kKAlignRequirement = 8;

/**
 * 把 v 向上对齐到 align 的整数倍。
 * 例：CeilAlign(17, 16) = 32；CeilAlign(16, 16) = 16
 */
static inline uint32_t CeilAlign(uint32_t v, uint32_t align) {
  return (v + align - 1) / align * align;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  // ==================== 第 1 步：拿到输入形状 ====================
  const gert::StorageShape* x1Shape = context->GetInputShape(0);  // x1
  const gert::StorageShape* x2Shape = context->GetInputShape(1);  // x2
  if (x1Shape == nullptr || x2Shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  // 注意：这里拿到的是 **storage shape**（数据在内存里的实际摆法），
  // 不是逻辑形状。所以当 transposeX1=true 时，x1Shape 是 [B, K, M]。
  auto x1Storage = x1Shape->GetStorageShape();
  auto x2Storage = x2Shape->GetStorageShape();
  if (x1Storage.GetDimNum() != 3 || x2Storage.GetDimNum() != 3) {
    // 赛题要求：x1 和 x2 均必须为 3 维 Tensor
    return ge::GRAPH_FAILED;
  }

  // ==================== 第 2 步：读属性 ====================
  auto attrs = context->GetAttrs();
  bool transposeX1 = false;
  bool transposeX2 = false;
  if (attrs != nullptr) {
    const bool* p1 = attrs->GetAttrPointer<bool>(0);
    const bool* p2 = attrs->GetAttrPointer<bool>(1);
    if (p1 != nullptr) transposeX1 = *p1;
    if (p2 != nullptr) transposeX2 = *p2;
  }

  // ==================== 第 3 步：从 storage shape 反推逻辑形状 ====================
  // 逻辑形状永远是 x1:[B,M,K]，x2:[B,K,N]。
  // 这一步和 python/golden_pure.py 里那段推理是**完全一样的逻辑**，
  // 可以对照着看。
  const uint32_t B = static_cast<uint32_t>(x1Storage.GetDim(0));

  uint32_t M = 0, K1 = 0;
  if (transposeX1) {
    // 存储是 [B, K, M]
    K1 = static_cast<uint32_t>(x1Storage.GetDim(1));
    M = static_cast<uint32_t>(x1Storage.GetDim(2));
  } else {
    // 存储是 [B, M, K]
    M = static_cast<uint32_t>(x1Storage.GetDim(1));
    K1 = static_cast<uint32_t>(x1Storage.GetDim(2));
  }

  uint32_t N = 0, K2 = 0;
  if (transposeX2) {
    // 存储是 [B, N, K]
    N = static_cast<uint32_t>(x2Storage.GetDim(1));
    K2 = static_cast<uint32_t>(x2Storage.GetDim(2));
  } else {
    // 存储是 [B, K, N]
    K2 = static_cast<uint32_t>(x2Storage.GetDim(1));
    N = static_cast<uint32_t>(x2Storage.GetDim(2));
  }

  // ==================== 第 4 步：合法性校验 ====================
  // 赛题"一一配对规则"：两个输入的 B 和 K 必须分别相等
  if (K1 != K2) return ge::GRAPH_FAILED;
  if (static_cast<uint32_t>(x2Storage.GetDim(0)) != B) return ge::GRAPH_FAILED;
  const uint32_t K = K1;

  // 维度范围校验（赛题 3.4 节）
  if (B < 1 || B > 64) return ge::GRAPH_FAILED;
  if (M < 1 || M > 8192) return ge::GRAPH_FAILED;
  if (N < 1 || N > 8192) return ge::GRAPH_FAILED;
  if (K < 32 || K > 8192) return ge::GRAPH_FAILED;
  if (K % kKAlignRequirement != 0) return ge::GRAPH_FAILED;

  // ==================== 第 5 步：决定多核切分方案 ====================
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
  if (aicCoreNum == 0) return ge::GRAPH_FAILED;

  BatchMatmulMaxSumTilingData tiling;
  tiling.batch = B;
  tiling.m = M;
  tiling.n = N;
  tiling.k = K;
  tiling.transposeX1 = transposeX1 ? 1u : 0u;
  tiling.transposeX2 = transposeX2 ? 1u : 0u;

  // 把 (b, m) 拉平成一维序号，统一按行切分。
  // 这样做的好处：无论 B 大还是 M 大，kernel 侧都只需要一套"我负责
  // [rowStart, rowEnd) 这些行"的逻辑，不用为两种策略写两份代码。
  const uint32_t totalRows = B * M;
  tiling.totalRows = totalRows;

  uint32_t usedCoreNum = aicCoreNum;
  if (totalRows < aicCoreNum) {
    // 行数比核数还少，多出来的核没活干，直接少启几个核更省电也更稳
    usedCoreNum = totalRows;
  }
  if (usedCoreNum == 0) usedCoreNum = 1;
  tiling.usedCoreNum = usedCoreNum;

  // 平均分行数，余数分给前面的核（每个核最多多 1 行）
  // 例：totalRows=10, usedCoreNum=3 -> rowsPerCore=3, tailRows=1
  //     核0 拿 4 行，核1 拿 3 行，核2 拿 3 行
  tiling.rowsPerCore = totalRows / usedCoreNum;
  tiling.tailRows = totalRows % usedCoreNum;

  // 选择 tilingKey：区分"核够用"和"核用不满"两种情况。
  // 现阶段两者切分公式相同，保留 key 是为了后续给不同情况挂不同的
  // kernel 实现（比如小 batch 时换更激进的并行策略）。
  if (B >= usedCoreNum) {
    tiling.tilingKey = static_cast<uint32_t>(BmmMaxSumTilingKey::kSplitByBatch);
  } else {
    tiling.tilingKey = static_cast<uint32_t>(BmmMaxSumTilingKey::kSplitByBatchAndM);
  }

  // ==================== 第 6 步：沿 N 维切块 ====================
  // N 最大 8192，一次搬不完，要切。
  // 这里按 UB 容量估一个上限，再向上对齐到 16（Cube 分形要求）。
  uint32_t nTileLen = std::min(N, kMaxNTileLen);
  nTileLen = CeilAlign(nTileLen, kFractalAlign);
  if (nTileLen > N) nTileLen = CeilAlign(N, kFractalAlign);

  tiling.nTileLen = nTileLen;
  tiling.nTileCount = (N + nTileLen - 1) / nTileLen;
  tiling.nTailLen = N - (tiling.nTileCount - 1) * nTileLen;

  // ==================== 第 7 步：沿 K 维切块 ====================
  uint32_t kTileLen = std::min(K, kKTileLen);
  kTileLen = CeilAlign(kTileLen, kFractalAlign);
  if (kTileLen > K) kTileLen = CeilAlign(K, kFractalAlign);

  tiling.kTileLen = kTileLen;
  tiling.kTileCount = (K + kTileLen - 1) / kTileLen;
  tiling.kTailLen = K - (tiling.kTileCount - 1) * kTileLen;

  // ==================== 第 8 步：对齐后的尺寸 ====================
  // Cube 要求 M/N/K 方向都对齐到 16，不足的补零。
  // 注意：M、N 赛题说"建议为 16 的整数倍"，意味着**会出现不对齐的情况**，
  // 尾块必须正确处理，补的零不能污染 MaxSim 和 Sum 的结果。
  tiling.mAlign = CeilAlign(M, kFractalAlign);
  tiling.nAlign = CeilAlign(N, kFractalAlign);
  tiling.kAlign = CeilAlign(K, kFractalAlign);

  // ==================== 第 9 步：数据类型 ====================
  auto x1Desc = context->GetInputDesc(0);
  if (x1Desc == nullptr) return ge::GRAPH_FAILED;
  tiling.isFp16 = (x1Desc->GetDataType() == ge::DT_FLOAT16) ? 1u : 0u;

  // ==================== 第 10 步：写回 context ====================
  // 把算好的方案写入 context，框架会负责把它传给 NPU kernel。
  tiling.reserved = 0;
  context->SetBlockDim(usedCoreNum);

  size_t* workspaces = context->GetWorkspaceSizes(1);
  if (workspaces != nullptr) {
    // 当前设计不需要额外 workspace（中间结果都在片上）。
    // 如果后续改成"先算相似度矩阵写回 GM 再归约"，这里要相应加大。
    workspaces[0] = 0;
  }

  auto rawTilingData = context->GetRawTilingData();
  if (rawTilingData == nullptr) return ge::GRAPH_FAILED;
  if (rawTilingData->GetCapacity() < sizeof(BatchMatmulMaxSumTilingData)) {
    return ge::GRAPH_FAILED;
  }
  errno_t ret = memcpy_s(rawTilingData->GetData(),
                         rawTilingData->GetCapacity(),
                         &tiling,
                         sizeof(BatchMatmulMaxSumTilingData));
  if (ret != EOK) return ge::GRAPH_FAILED;
  rawTilingData->SetDataSize(sizeof(BatchMatmulMaxSumTilingData));

  return ge::GRAPH_SUCCESS;
}

// 注册 Tiling 函数，框架会在需要时调用它
IMPL_OP_OPTILING(BatchMatmulMaxSum).Tiling(TilingFunc);

}  // namespace optiling
