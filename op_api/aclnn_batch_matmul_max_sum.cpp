/**
 * BatchMatmulMaxSum —— aclnn 单算子接口（实现）
 *
 * ⚠️ 状态：骨架，未经编译验证。
 *    本文件展示 aclnn 接口的标准写法与校验逻辑，
 *    标 TODO 的地方需要按比赛环境 CANN 版本补齐。
 */

#include "aclnn_batch_matmul_max_sum.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

// ---------------- 本算子的约束（与赛题 3.4 节一致） ----------------
static constexpr int64_t kDimNum = 3;          // x1/x2 必须是 3 维
static constexpr int64_t kMaxBatch = 64;
static constexpr int64_t kMaxMN = 8192;
static constexpr int64_t kMinK = 32;
static constexpr int64_t kMaxK = 8192;

/**
 * 检查输入的合法性。
 *
 * 好的算子要把错误挡在入口，而不是让它在 NPU 上跑出个莫名其妙的结果。
 * 类比 Python：这就是函数开头的 assert / raise ValueError。
 */
static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2,
                               const aclTensor* y) {
  // 1. 空指针检查
  CHECK_RET(x1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(x2 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(y != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 2. 维度检查：x1 和 x2 都必须是 3 维
  CHECK_RET(x1->GetViewShape().GetDimNum() == static_cast<size_t>(kDimNum),
            ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(x2->GetViewShape().GetDimNum() == static_cast<size_t>(kDimNum),
            ACLNN_ERR_PARAM_INVALID);
  // 输出 y 必须是 1 维，长度 B
  CHECK_RET(y->GetViewShape().GetDimNum() == 1, ACLNN_ERR_PARAM_INVALID);

  // 3. 数据类型检查：x1 和 x2 类型必须相同，且只能是 fp16 / bf16
  auto dt1 = x1->GetDataType();
  auto dt2 = x2->GetDataType();
  CHECK_RET(dt1 == dt2, ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(dt1 == DataType::DT_FLOAT16 || dt1 == DataType::DT_BFLOAT16,
            ACLNN_ERR_PARAM_INVALID);
  // 输出必须是 fp32（赛题 3.6 节硬性要求）
  CHECK_RET(y->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID);

  // 4. 形状匹配检查（注意：这里看到的是 storage shape，
  //    所以 transpose 打开时维度顺序和逻辑形状不同，
  //    真正的 B/K 一致性检查放到 Tiling 里做更稳妥）
  // TODO(实现)：按 transposeX1/transposeX2 把 storage shape 还原成逻辑形状，
  //   再校验 B 相等、K 相等、各维在合法范围内。
  //   参考 op_host/batch_matmul_max_sum_tiling.cpp 第 3、4 步的写法。

  return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnBatchMatmulMaxSumGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    bool transposeX1,
    bool transposeX2,
    const aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  // 出参先置空，避免调用方拿到脏数据
  CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  *workspaceSize = 0;
  *executor = nullptr;

  // 参数校验
  auto ret = CheckParams(x1, x2, y);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 创建执行器。
  // L0_DFX 这一串是 CANN 的埋点参数（记录调用了哪个 API、哪个算子），
  // 照抄格式即可。
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 把属性打包成框架认识的形式
  // TODO(上板核对)：不同 CANN 版本这里构造 attr 的 API 名称略有差异，
  //   较新版本用 op::CastToBool 或直接传 bool 数组。
  //   请对照你环境里的头文件 aclnn_kernels/common/op_error_check.h 调整。
  const bool attrVals[2] = {transposeX1, transposeX2};

  // 把这次调用记录到执行器里。
  // 注意：这里**不会真的去算**，只是把"要算什么"记下来。
  // TODO(实现)：用 l0op 命名空间下的算子调用把 BatchMatmulMaxSum 挂进图。
  //   标准写法形如：
  //     auto yOut = l0op::BatchMatmulMaxSum(x1, x2, attrVals, uniqueExecutor.get());
  //     CHECK_RET(yOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  //     auto viewCopyResult = l0op::ViewCopy(yOut, y, uniqueExecutor.get());
  //     CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  //
  //   前提是 op_host 里定义的原型已经通过 GE 注册好了。

  // 本算子当前不需要额外 workspace（中间结果都在片上）。
  // 如果后续改成"先写相似度矩阵到 GM 再归约"，这里要返回真实大小。
  *workspaceSize = 0;

  // 把执行器交给调用方
  *executor = uniqueExecutor.get();
  // 引用计数 +1，防止 uniqueExecutor 析构时把执行器释放掉
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnBatchMatmulMaxSum(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream) {
  // 真正触发执行。框架会拿着第 1 段记录好的执行器，在指定的流上跑。
  auto ret = ACLNN_SUCCESS;
  // TODO(实现)：标准写法是
  //   ret = l0op::RunOpApi(executor, workspace, workspaceSize, stream);
  //   具体函数名随 CANN 版本变化，请对照环境头文件。
  (void)workspace;
  (void)workspaceSize;
  (void)executor;
  (void)stream;
  return ret;
}
