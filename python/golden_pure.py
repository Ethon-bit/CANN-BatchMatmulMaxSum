"""
BatchMatmulMaxSum —— 纯 Python 参考实现（零依赖版 / Golden Reference）

这个文件的作用
--------------
它是整个赛题的"标准答案"。你用 Ascend C 写出来的 NPU 算子，算出来的结果
必须和这个文件算出来的结果一致（在允许的误差范围内）。

为什么要有两个版本？
--------------------
1. golden_pure.py（本文件）：只用 Python 自带的语法，不需要装任何库。
   好处是每一步都看得见，适合理解算法到底在算什么。
2. golden_torch.py：严格照抄赛题给的 PyTorch 写法。
   好处是它就是官方定义的语义，做精度比对时以它为准。

赛题里的三个公式，对应下面代码的三层循环：
    阶段一 批量矩阵乘:  A[b,m,n] = Σ_k X1[b,m,k] * X2[b,k,n]
    阶段二 MaxSim 归约: R[b,m]   = max_n A[b,m,n]
    阶段三 求和归约:    y[b]     = Σ_m R[b,m]

重要提醒：Max 和 Sum 的顺序不能交换！
    先 max 再 sum  ≠  先 sum 再 max
    这是赛题"归约顺序规则"明确要求的。
"""

from typing import List


def batch_matmul_max_sum_pure(
    x1: List,
    x2: List,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
) -> List[float]:
    """按赛题定义计算 BatchMatmulMaxSum。

    参数
    ----
    x1 : 嵌套列表，代表 x1 张量。
         逻辑形状固定是 [B, M, K]（B=批大小, M=query token 数, K=特征维）。
         实际存储形状由 transpose_x1 决定：
             transpose_x1=False -> 存储就是 [B, M, K]，取值 x1[b][m][k]
             transpose_x1=True  -> 存储是 [B, K, M]，取值 x1[b][k][m]
         注意：transpose 只改变"数据怎么摆放"，不改变逻辑形状，
               也不代表算子要真的去做一次转置运算。

    x2 : 嵌套列表，代表 x2 张量。
         逻辑形状固定是 [B, K, N]（N=document token 数）。
         实际存储形状由 transpose_x2 决定：
             transpose_x2=False -> 存储就是 [B, K, N]，取值 x2[b][k][n]
             transpose_x2=True  -> 存储是 [B, N, K]，取值 x2[b][n][k]

    transpose_x1 / transpose_x2 : 对应赛题的 transposeX1 / transposeX2 属性。

    返回
    ----
    y : 长度为 B 的列表，第 b 个元素是第 b 组 query-document 的相关性分数。
    """

    # ---------- 第 0 步：读清楚存储形状，推出 B / M / N / K ----------
    # 逻辑形状永远是 x1:[B,M,K]、x2:[B,K,N]，但存储摆放方式有 4 种组合。
    # 这里根据 transpose 标志，从实际存储里把四个维度"认"出来。
    B = len(x1)  # batch 维在最外层，四种布局下都是 x1[0] 这一层的长度

    if transpose_x1:
        # 存储是 [B, K, M]
        K = len(x1[0])
        M = len(x1[0][0])
    else:
        # 存储是 [B, M, K]
        M = len(x1[0])
        K = len(x1[0][0])

    # x2 的 K 必须和 x1 的 K 相等（赛题要求：两个输入的 B 和 K 必须分别相等）
    if transpose_x2:
        # 存储是 [B, N, K]
        N = len(x2[0])
        K2 = len(x2[0][0])
    else:
        # 存储是 [B, K, N]
        K2 = len(x2[0])
        N = len(x2[0][0])

    if K != K2:
        raise ValueError(f"x1 和 x2 的 K 维不一致: {K} != {K2}")
    if len(x2) != B:
        raise ValueError(f"x1 和 x2 的 batch 维不一致: {B} != {len(x2)}")

    # ---------- 定义两个"取值小助手" ----------
    # 有了它们，后面就不用到处写 if transpose 了，代码干净很多。
    def get_x1(b: int, m: int, k: int) -> float:
        """按逻辑坐标 (b, m, k) 取出 x1 的值，自动适配存储布局。"""
        if transpose_x1:
            return float(x1[b][k][m])  # 存储是 [B,K,M]，所以下标是 k 在前
        return float(x1[b][m][k])      # 存储是 [B,M,K]

    def get_x2(b: int, k: int, n: int) -> float:
        """按逻辑坐标 (b, k, n) 取出 x2 的值，自动适配存储布局。"""
        if transpose_x2:
            return float(x2[b][n][k])  # 存储是 [B,N,K]，所以下标是 n 在前
        return float(x2[b][k][n])      # 存储是 [B,K,N]

    # ---------- 主计算：对每一组 b 独立计算 ----------
    y: List[float] = []

    for b in range(B):
        # 赛题"一一配对规则"：第 b 组 x1 只和第 b 组 x2 计算，
        # 不做 batch broadcast，也不做跨 batch 的两两组合。
        total = 0.0  # 这一组最终的分数（阶段三的累加器）

        for m in range(M):
            # ===== 阶段二：MaxSim 归约 =====
            # 关键坑：初值不能用 0！
            # 如果某一行相似度全是负数，初值设 0 会错误地返回 0 而不是最大负数。
            # 赛题示例3 就是专门考这个点的。
            # 这里用 None 表示"还没取到任何值"，等价于 -inf，且不依赖任何库。
            best = None

            for n in range(N):
                # ===== 阶段一：批量矩阵乘（K 维点积） =====
                acc = 0.0
                for k in range(K):
                    acc += get_x1(b, m, k) * get_x2(b, k, n)

                # 取最大值，同时正确处理"全负数"的情况
                if best is None or acc > best:
                    best = acc

            # ===== 阶段三：求和归约 =====
            # 必须放在 MaxSim 之后，顺序不可交换
            total += best

        y.append(total)

    return y


# ======================================================================
# 下面是自测代码：直接运行本文件时，会用赛题里的 3 个示例验证实现是否正确。
# 运行方式： python golden_pure.py
# ======================================================================
if __name__ == "__main__":
    import sys

    # 让 Windows 终端也能正常打印中文，避免 GBK 编码报错
    sys.stdout.reconfigure(encoding="utf-8")

    passed = 0
    failed = 0

    # ---------- 示例1：基础计算 ----------
    # x1 = [[[1,0],[0,1]]]  逻辑 [B=1, M=2, K=2]
    # x2 = [[[1,0,-1],[0,1,0]]]  逻辑 [B=1, K=2, N=3]
    # BatchMatMul -> [[[1,0,-1],[0,1,0]]]
    # MaxSim      -> [[1,1]]
    # y           -> [2.0]
    x1 = [[[1, 0], [0, 1]]]
    x2 = [[[1, 0, -1], [0, 1, 0]]]
    got = batch_matmul_max_sum_pure(x1, x2)
    want = [2.0]
    ok = got == want
    passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
    print(f"[示例1 基础计算]      {'通过' if ok else '失败'}  得到={got}  期望={want}")

    # ---------- 示例2：转置存储布局 ----------
    # 逻辑内容与示例1完全相同，但物理摆放方式变了：
    #   x1 按 [B,K,M] 存，x2 按 [B,N,K] 存
    # 输出必须仍然是 [2.0]（transpose 只影响存储，不影响计算结果）
    x1_t = [[[1, 0], [0, 1]]]           # 存储 [B=1, K=2, M=2]
    x2_t = [[[1, 0], [0, 1], [-1, 0]]]  # 存储 [B=1, N=3, K=2]
    got = batch_matmul_max_sum_pure(x1_t, x2_t, transpose_x1=True, transpose_x2=True)
    want = [2.0]
    ok = got == want
    passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
    print(f"[示例2 转置存储布局]  {'通过' if ok else '失败'}  得到={got}  期望={want}")

    # ---------- 示例3：全负相似度（最容易被坑的用例） ----------
    # x1 = [[[1,0]]]        逻辑 [B=1, M=1, K=2]
    # x2 = [[[-1,-2],[0,0]]] 逻辑 [B=1, K=2, N=2]
    # BatchMatMul -> [[[-1,-2]]]
    # MaxSim      -> [[-1]]     <- 如果初值错误地设成 0，这里会输出 0
    # y           -> [-1.0]
    x1 = [[[1, 0]]]
    x2 = [[[-1, -2], [0, 0]]]
    got = batch_matmul_max_sum_pure(x1, x2)
    want = [-1.0]
    ok = got == want
    passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
    print(f"[示例3 全负相似度]    {'通过' if ok else '失败'}  得到={got}  期望={want}")

    print()
    print(f"自测结果：{passed} 个通过，{failed} 个失败")
    sys.exit(1 if failed else 0)
