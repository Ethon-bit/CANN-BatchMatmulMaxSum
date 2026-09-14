"""
BatchMatmulMaxSum —— PyTorch 参考实现（官方语义版）

这个文件严格照抄赛题 3.1 节给出的"等价 python 实现"。
它是本赛题精度比对的最终依据。

和 golden_pure.py 的分工：
    golden_torch.py —— 官方语义，做精度比对时以它为准（需要 pip install torch）
    golden_pure.py  —— 零依赖，用来理解算法每一步在做什么

运行前需要安装 torch：
    pip install torch --index-url https://download.pytorch.org/whl/cpu

注意赛题里的一句话：
    "标准 golden 使用输入实际存储值在 FP64 精度下计算，最后转换为 FP32。"
所以下面额外提供了 golden_fp64()，它先把输入升到 float64 再算，最后转回 float32。
做精度比对时应该用它，而不是直接用 float32 算——否则误差会算重。
"""

from typing import Optional, Tuple

import torch


def batch_matmul_max_sum(
    x1_logical: torch.Tensor,
    x2_logical: torch.Tensor,
) -> torch.Tensor:
    """赛题 3.1 节原文给出的等价实现（原样保留，未做任何修改）。

    参数
    ----
    x1_logical : [B, M, K]
    x2_logical : [B, K, N]

    返回
    ----
    y : [B]
    """
    similarity = torch.bmm(
        x1_logical.to(torch.float32),
        x2_logical.to(torch.float32),
    )
    max_sim = torch.amax(similarity, dim=-1)
    return torch.sum(max_sim, dim=-1, dtype=torch.float32)


def golden_fp64(
    x1_storage: torch.Tensor,
    x2_storage: torch.Tensor,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
) -> torch.Tensor:
    """按赛题要求的标准 golden：用 FP64 计算，最后转 FP32。

    这个函数额外处理了 transpose 存储布局，方便直接喂入原始存储张量。

    参数
    ----
    x1_storage : x1 的实际存储张量
                 transpose_x1=False -> [B, M, K]
                 transpose_x1=True  -> [B, K, M]
    x2_storage : x2 的实际存储张量
                 transpose_x2=False -> [B, K, N]
                 transpose_x2=True  -> [B, N, K]

    返回
    ----
    y : [B]，float32
    """
    x1 = to_logical_x1(x1_storage, transpose_x1)
    x2 = to_logical_x2(x2_storage, transpose_x2)

    # 升到 float64 再做全部计算，避免低精度长归约累积误差
    sim = torch.bmm(x1.to(torch.float64), x2.to(torch.float64))
    max_sim = torch.amax(sim, dim=-1)          # 阶段二：沿 N 维取最大值
    y = torch.sum(max_sim, dim=-1)             # 阶段三：沿 M 维求和
    return y.to(torch.float32)


def to_logical_x1(x1_storage: torch.Tensor, transpose_x1: bool) -> torch.Tensor:
    """把 x1 的实际存储转成逻辑形状 [B, M, K]。

    transpose_x1=True 时存储是 [B, K, M]，用 transpose 换成 [B, M, K]。
    注意：这里之所以能"换回来"，是因为对参考实现来说数据已经在内存里了；
    真正的 NPU 算子不允许真的做一次转置搬运，必须直接按存储布局读取。
    """
    if transpose_x1:
        return x1_storage.transpose(-1, -2)  # [B,K,M] -> [B,M,K]
    return x1_storage


def to_logical_x2(x2_storage: torch.Tensor, transpose_x2: bool) -> torch.Tensor:
    """把 x2 的实际存储转成逻辑形状 [B, K, N]。

    transpose_x2=True 时存储是 [B, N, K]，用 transpose 换成 [B, K, N]。
    """
    if transpose_x2:
        return x2_storage.transpose(-1, -2)  # [B,N,K] -> [B,K,N]
    return x2_storage


def make_storage(
    B: int,
    M: int,
    N: int,
    K: int,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    dtype: torch.dtype = torch.float16,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """造一对随机输入（按指定存储布局摆放），返回 (x1_storage, x2_storage)。

    模拟真实场景：embedding 已经做过 L2 归一化，所以数值范围大致在 [-1, 1]。
    """
    if seed is not None:
        torch.manual_seed(seed)

    # 先生成逻辑形状 [B, M, K] 和 [B, K, N]
    x1_logical = torch.randn(B, M, K, dtype=torch.float32)
    x2_logical = torch.randn(B, K, N, dtype=torch.float32)

    # L2 归一化（赛题说明：这一步由上游网络完成，本算子不负责）
    x1_logical = x1_logical / x1_logical.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x2_logical = x2_logical / x2_logical.norm(dim=-2, keepdim=True).clamp_min(1e-12)

    # 降精度到目标类型，再按需要的存储布局摆放
    x1_logical = x1_logical.to(dtype)
    x2_logical = x2_logical.to(dtype)

    x1_storage = x1_logical.transpose(-1, -2).contiguous() if transpose_x1 else x1_logical.contiguous()
    x2_storage = x2_logical.transpose(-1, -2).contiguous() if transpose_x2 else x2_logical.contiguous()

    return x1_storage, x2_storage


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    # 用赛题 3 个示例验证（这里转为张量喂进去）
    print("=== 用赛题示例验证 golden_torch ===")

    # 示例1
    x1 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    x2 = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])
    y = batch_matmul_max_sum(x1, x2)
    print(f"示例1: {y.tolist()}  期望 [2.0]")

    # 示例2（transpose 存储）
    x1_t = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])          # [B,K,M]
    x2_t = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])  # [B,N,K]
    y = golden_fp64(x1_t, x2_t, transpose_x1=True, transpose_x2=True)
    print(f"示例2: {y.tolist()}  期望 [2.0]")

    # 示例3（全负相似度）
    x1 = torch.tensor([[[1.0, 0.0]]])
    x2 = torch.tensor([[[-1.0, -2.0], [0.0, 0.0]]])
    y = batch_matmul_max_sum(x1, x2)
    print(f"示例3: {y.tolist()}  期望 [-1.0]")
