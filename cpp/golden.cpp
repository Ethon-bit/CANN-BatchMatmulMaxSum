// ======================================================================
// BatchMatmulMaxSum —— C++ 参考实现（标准答案）
//
// 为什么要有这个文件？
// ------------------
// 1. 你有 C++ 基础，用 C++ 写一遍比看 Python 更直观
// 2. 它是 op_kernel/ 里 Ascend C 代码的"直系祖先"——
//    Ascend C 本质上就是带 NPU 扩展的 C++，三层循环的结构完全一样，
//    区别只在于：数据要手动从 GM 搬到片上、矩阵乘要交给 Cube 单元。
//    先把这份 CPU 版写对、跑通，再往 Ascend C 迁移，路径最短。
//
// 精度说明
// --------
// 赛题要求"标准 golden 使用输入实际存储值在 FP64 精度下计算，最后转 FP32"。
// 所以本程序全程用 double 累加，输出时转成 float。
// 这和 python/golden_pure.py 的行为一致（Python 的 float 就是 double）。
//
// 编译运行
// --------
//     g++ -O2 -std=c++11 -o golden.exe cpp/golden.cpp
//     ./golden.exe < 输入文件 > 输出文件
//
// 输入输出格式见 test/cpp_bridge.py 的说明。
// ======================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>

typedef std::vector<double> VecD;

// ----------------------------------------------------------------------
// 核心计算
//
// 逻辑形状：x1 = [B, M, K]，x2 = [B, K, N]
//
// 存储形状由 transpose 标志决定：
//   transposeX1 = false -> x1Storage 按 [B, M, K] 摆，取 x1Storage[b][m][k]
//   transposeX1 = true  -> x1Storage 按 [B, K, M] 摆，取 x1Storage[b][k][m]
//   transposeX2 = false -> x2Storage 按 [B, K, N] 摆，取 x2Storage[b][k][n]
//   transposeX2 = true  -> x2Storage 按 [B, N, K] 摆，取 x2Storage[b][n][k]
//
// 关键：transpose 只是"数据怎么摆"，不改变计算结果。
//       下面用两个内联取值函数把布局差异封装起来，主循环就不用管了。
// ----------------------------------------------------------------------

// 按逻辑坐标 (b, m, k) 从 x1 存储中取值
static inline double getX1(const VecD& x1, unsigned B, unsigned M, unsigned K,
                           bool tX1, unsigned b, unsigned m, unsigned k) {
    if (tX1) {
        // 存储是 [B, K, M]：offset = b*K*M + k*M + m
        return x1[((size_t)b * K + k) * M + m];
    } else {
        // 存储是 [B, M, K]：offset = b*M*K + m*K + k
        return x1[((size_t)b * M + m) * K + k];
    }
}

// 按逻辑坐标 (b, k, n) 从 x2 存储中取值
static inline double getX2(const VecD& x2, unsigned B, unsigned K, unsigned N,
                           bool tX2, unsigned b, unsigned k, unsigned n) {
    if (tX2) {
        // 存储是 [B, N, K]：offset = b*N*K + n*K + k
        return x2[((size_t)b * N + n) * K + k];
    } else {
        // 存储是 [B, K, N]：offset = b*K*N + k*N + n
        return x2[((size_t)b * K + k) * N + n];
    }
}

// 主计算：返回长度为 B 的向量 y
static VecD compute(const VecD& x1Storage, const VecD& x2Storage,
                    unsigned B, unsigned M, unsigned N, unsigned K,
                    bool tX1, bool tX2) {
    VecD y(B, 0.0);

    // 注意三层循环的嵌套顺序：
    //   最外层 b  —— 每组数据独立计算（赛题"一一配对规则"）
    //   中间层 m  —— 每个 query token
    //   内层   n  —— 每个 document token，然后里面还有一层 k 做点积
    //
    // 这个顺序对应赛题公式的三步：
    //   b, m 循环里的 k 内积  -> 阶段一：矩阵乘
    //   b, m 循环里的 n 取最大 -> 阶段二：MaxSim
    //   y[b] 累加             -> 阶段三：求和
    for (unsigned b = 0; b < B; ++b) {
        for (unsigned m = 0; m < M; ++m) {

            // ★★★ 全场最关键的一行 ★★★
            // MaxSim 的初值必须是负无穷，不能是 0！
            // 如果某一行相似度全是负数，初值设 0 会错误地返回 0。
            // 这是赛题示例3 专门考的坑。
            double best = -std::numeric_limits<double>::infinity();

            for (unsigned n = 0; n < N; ++n) {
                // 阶段一：K 维点积
                // 用 double 累加，满足赛题"FP32 累加或具有等效精度"的要求
                double acc = 0.0;
                for (unsigned k = 0; k < K; ++k) {
                    acc += getX1(x1Storage, B, M, K, tX1, b, m, k) *
                           getX2(x2Storage, B, K, N, tX2, b, k, n);
                }
                // 阶段二：取最大值
                if (acc > best) best = acc;
            }

            // 阶段三：求和
            // 顺序绝对不能反 —— max 和 sum 不可交换
            y[b] += best;
        }
    }

    return y;
}

// ----------------------------------------------------------------------
// 输入输出
//
// 输入格式（全部用空白分隔，cin 直接读）：
//     <用例数 C>
//     对每个用例：
//         <B> <M> <N> <K> <tX1> <tX2>
//         <B*M*K 个 double>      x1 的存储数据
//         <B*K*N 个 double>      x2 的存储数据
//
// 输出格式：
//     <用例数 C>
//     对每个用例：
//         <B> <y[0]> <y[1]> ... <y[B-1]>
//
// 注意：x1 存储无论是 [B,M,K] 还是 [B,K,M]，元素总数都是 B*M*K，
//       所以读取时不用关心布局，只按总数读，解释留给 getX1/getX2。
// ----------------------------------------------------------------------

int main() {
    unsigned numCases = 0;
    if (scanf("%u", &numCases) != 1) {
        fprintf(stderr, "读取用例数失败\n");
        return 1;
    }

    printf("%u\n", numCases);

    for (unsigned c = 0; c < numCases; ++c) {
        unsigned B, M, N, K, tX1i, tX2i;
        if (scanf("%u %u %u %u %u %u", &B, &M, &N, &K, &tX1i, &tX2i) != 6) {
            fprintf(stderr, "读取第 %u 个用例的形状失败\n", c + 1);
            return 1;
        }
        const bool tX1 = (tX1i != 0);
        const bool tX2 = (tX2i != 0);

        // 读 x1（元素总数固定为 B*M*K）
        VecD x1((size_t)B * M * K);
        for (size_t i = 0; i < x1.size(); ++i) {
            if (scanf("%lf", &x1[i]) != 1) {
                fprintf(stderr, "读取 x1 数据失败（用例 %u，第 %zu 个元素）\n", c + 1, i);
                return 1;
            }
        }

        // 读 x2（元素总数固定为 B*K*N）
        VecD x2((size_t)B * K * N);
        for (size_t i = 0; i < x2.size(); ++i) {
            if (scanf("%lf", &x2[i]) != 1) {
                fprintf(stderr, "读取 x2 数据失败（用例 %u，第 %zu 个元素）\n", c + 1, i);
                return 1;
            }
        }

        VecD y = compute(x1, x2, B, M, N, K, tX1, tX2);

        printf("%u", B);
        for (unsigned b = 0; b < B; ++b) {
            // 转成 float 输出（赛题要求输出类型是 FLOAT32）
            printf(" %.17g", (double)(float)y[b]);
        }
        printf("\n");

        fflush(stdout);
    }

    return 0;
}
