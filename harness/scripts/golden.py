"""
BatchMatmulMaxSum —— golden 参考实现（本地测试用）

语义与赛题 3.1 节一致：
    阶段一：A = bmm(x1', x2')          x1' 逻辑 (B,M,K)、x2' 逻辑 (B,K,N)
    阶段二：R = amax(A, axis=-1)       沿 N 取最大
    阶段三：y = sum(R, axis=-1)        沿 M 求和

精度：全程 float64 累加，最后转 float32（赛题要求"标准 golden 使用输入实际
      存储值在 FP64 精度下计算，最后转换为 FP32"）。

输入的 x1/x2 是**物理布局**的数组：
    transposeX1=False -> (B, M, K)      True -> (B, K, M)
    transposeX2=False -> (B, K, N)      True -> (B, N, K)
本函数内部会 swap 成逻辑形状再算。
"""

import numpy as np


def impl(x1, x2, transpose_x1=False, transpose_x2=False):
    x1f = np.asarray(x1).astype(np.float64)
    x2f = np.asarray(x2).astype(np.float64)

    if transpose_x1:
        x1f = np.swapaxes(x1f, -1, -2)
    if transpose_x2:
        x2f = np.swapaxes(x2f, -1, -2)

    if x1f.shape[0] != x2f.shape[0]:
        raise ValueError("batch 维不一致")
    if x1f.shape[2] != x2f.shape[1]:
        raise ValueError("K 维不一致")

    sim = np.matmul(x1f, x2f)                 # (B, M, N)
    row_max = np.max(sim, axis=-1)            # (B, M)
    y = np.sum(row_max, axis=-1)              # (B,)
    return y.astype(np.float32)


def physical_shapes(B, M, N, K, transpose_x1, transpose_x2):
    """按 transpose 标志给出物理（存储）形状。"""
    x1_shape = (B, K, M) if transpose_x1 else (B, M, K)
    x2_shape = (B, N, K) if transpose_x2 else (B, K, N)
    return x1_shape, x2_shape


def make_inputs(B, M, N, K, transpose_x1, transpose_x2, dtype, pattern, seed):
    """造一对输入（按物理布局摆放）。

    pattern:
        0 = 均匀 [-1, 1]          （普通用例）
        1 = x1 在 [0,1]、x2 在 [-1,0]  ⇒ 所有点积 ≤ 0，全是负数
                                   （对应赛题示例3「全负相似度」）
    """
    rng = np.random.default_rng(seed)

    # ★ 关键：永远先在【逻辑形状】(B,M,K) / (B,K,N) 上生成数据，
    #   再按 transpose 摆成物理布局。
    #   这样只要 seed 相同，四种 transpose 组合拿到的是**同一份逻辑数据**，
    #   golden 必然相同 —— 正是「四象限」用例要的行为。
    #   （早期版本直接按物理形状生成，导致四个象限各是各的随机数据，无法对比。）
    if pattern == 1:
        x1_log = rng.uniform(0.0, 1.0, (B, M, K)).astype(np.float32)
        x2_log = rng.uniform(-1.0, 0.0, (B, K, N)).astype(np.float32)
    else:
        x1_log = rng.uniform(-1.0, 1.0, (B, M, K)).astype(np.float32)
        x2_log = rng.uniform(-1.0, 1.0, (B, K, N)).astype(np.float32)

    # 先降到目标精度再摆放：量化应该与布局无关
    def cast(a):
        if dtype == "float16":
            return a.astype(np.float16)
        if dtype == "bfloat16":
            try:
                from ml_dtypes import bfloat16
            except ImportError as e:
                raise RuntimeError(
                    "需要 ml_dtypes 才能生成 bfloat16 数据：pip install ml_dtypes") from e
            return a.astype(bfloat16)
        raise ValueError(f"不支持的 dtype: {dtype}")

    x1_log = cast(x1_log)
    x2_log = cast(x2_log)

    x1 = np.swapaxes(x1_log, -1, -2) if transpose_x1 else x1_log
    x2 = np.swapaxes(x2_log, -1, -2) if transpose_x2 else x2_log

    return np.ascontiguousarray(x1), np.ascontiguousarray(x2)


if __name__ == "__main__":
    # 自检：跑赛题的三个示例
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # 示例1
    a = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float16)
    b = np.array([[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]], dtype=np.float16)
    print("示例1:", impl(a, b).tolist(), "期望 [2.0]")

    # 示例2（转置存储）
    a2 = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float16)          # [B,K,M]
    b2 = np.array([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]], dtype=np.float16)  # [B,N,K]
    print("示例2:", impl(a2, b2, True, True).tolist(), "期望 [2.0]")

    # 示例3（全负）
    a3 = np.array([[[1.0, 0.0]]], dtype=np.float16)
    b3 = np.array([[[-1.0, -2.0], [0.0, 0.0]]], dtype=np.float16)
    print("示例3:", impl(a3, b3).tolist(), "期望 [-1.0]")
