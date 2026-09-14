/**
 * BatchMatmulMaxSum —— aclnn 单算子接口（头文件）
 *
 * 这个文件是干什么的？
 * -------------------
 * 让**别人能调用你的算子**。没有这一层，你的算子只能被图编译器
 * （GE）用；有了这一层，就能像调普通函数一样在 Python/C++ 里调它。
 *
 * 对 Python 选手的类比
 * -------------------
 * 就是你写的 Python 函数的"公开签名"：
 *
 *     def batch_matmul_max_sum(x1, x2, transposeX1=False, transposeX2=False) -> y
 *
 * aclnn 接口必须写成**两段式**，这是 CANN 的固定套路：
 *
 *   第 1 段 GetWorkspaceSize —— "先告诉我需要多少临时显存？"
 *   第 2 段 Execute          —— "显存我准备好了，开始算"
 *
 * 为什么要分两段？因为调用方（比如 PyTorch 的适配层）需要提前
 * 一次性把所有算子要的临时显存都申请好，再统一执行，这样效率最高。
 */

#ifndef ACLNN_BATCH_MATMUL_MAX_SUM_H
#define ACLNN_BATCH_MATMUL_MAX_SUM_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 第 1 段：查询执行本算子需要多少 workspace，并拿到执行器句柄。
 *
 * @param x1            输入张量 1，逻辑形状 [B, M, K]
 * @param x2            输入张量 2，逻辑形状 [B, K, N]
 * @param transposeX1   属性，默认 false。仅声明 x1 的存储布局
 * @param transposeX2   属性，默认 false。仅声明 x2 的存储布局
 * @param y             输出张量，形状 [B]，类型 float32
 * @param workspaceSize 出参：返回需要的 workspace 字节数
 * @param executor      出参：返回执行器句柄，第 2 段要用
 * @return ACLNN_SUCCESS(0) 表示成功，其他值表示失败
 */
__attribute__((visibility("default"))) aclnnStatus aclnnBatchMatmulMaxSumGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    bool transposeX1,
    bool transposeX2,
    const aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * 第 2 段：真正执行。
 *
 * @param workspace     第 1 段申请到的显存地址（不需要就传 nullptr）
 * @param workspaceSize 第 1 段返回的大小
 * @param executor      第 1 段返回的执行器
 * @param stream        要挂在哪个 ACL 流上执行（类比 CUDA stream）
 */
__attribute__((visibility("default"))) aclnnStatus aclnnBatchMatmulMaxSum(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_BATCH_MATMUL_MAX_SUM_H
