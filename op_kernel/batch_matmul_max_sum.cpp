/**
 * BatchMatmulMaxSum —— Kernel 入口函数
 *
 * 这个文件是整个算子在 NPU 上执行的**唯一入口**。
 * 框架（CANN runtime）会为每个 AI Core 调用一次这个函数，
 * 每个核拿到的 __gm__ 指针相同，但 GetBlockIdx() 不同，
 * 于是各自去处理自己那一份任务。
 *
 * 对 Python 选手的类比
 * -------------------
 * 有点像 multiprocessing：
 *
 *     def worker(core_id):
 *         my_rows = split_rows(core_id)     # 每个进程分一段
 *         for row in my_rows:
 *             ...
 *
 *     # 主进程启动 N 个 worker
 *     Pool(N).map(worker, range(N))
 *
 * 区别在于：这里没有"主进程"，每个核都是独立启动、独立跑完的，
 * 核与核之间只能通过 GM 通信（所以我们用原子加来汇总 y）。
 *
 * ⚠️ 状态：骨架，未经编译验证。详见 batch_matmul_max_sum.h 顶部说明。
 */

#include "batch_matmul_max_sum.h"

/**
 * 核函数入口。
 *
 * 参数说明（这些由框架自动传入，顺序是固定的，不能改）：
 *   x1        —— 输入 x1 在 GM 上的起始地址
 *   x2        —— 输入 x2 在 GM 上的起始地址
 *   y         —— 输出 y 在 GM 上的起始地址
 *   workspace —— 框架分配的临时显存（本算子当前用不到，tiling 里设为 0）
 *   tiling    —— tiling 数据的地址，里面是 Host 侧算好的切分方案
 *
 * __global__ __aicore__ 是 Ascend C 的固定修饰符：
 *   __global__  表示这是"核函数"，会被每个核分别执行
 *   __aicore__  表示这段代码运行在 AI Core 上（而不是 CPU 上）
 */
extern "C" __global__ __aicore__ void batch_matmul_max_sum(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  // 把 workspace 标记为已使用，避免某些 CANN 版本报"未使用参数"警告。
  // 本算子目前不需要额外显存。
  (void)workspace;

  // TPipe 是片上内存（UB）的管理器。
  // 类比：它像是给这个核分配的"工作台"，所有临时数据都要在它上面申请空间。
  TPipe pipe;

  KernelBatchMatmulMaxSum op;
  op.Init(x1, x2, y, tiling, &pipe);
  op.Process();
}

// ======================================================================
// 上板前检查清单（Checklist）
// ======================================================================
//
// 【编译前】
//   [ ] 确认 op_host/batch_matmul_max_sum.cpp 里的 AddConfig("ascend910b")
//       与比赛环境芯片型号一致（用 npu-smi info 查）
//   [ ] 确认 include 路径能找到 kernel_operator.h
//   [ ] 确认 tiling 头文件的相对路径 ../op_host/ 在构建系统中被正确解析
//       （如果不方便，把 tiling 结构体抽到 op_kernel/ 下的独立头文件更省事）
//
// 【功能正确性】
//   [ ] ProcessOneRow 里 DataCopy 全部补齐
//   [ ] MaxSim 初值确认为 -inf 等价物，**不是 0**
//   [ ] 尾块之后的多余位置显式填成 -inf，避免残留值污染最大值
//   [ ] 四种 transpose 组合都跑一遍（transposeX1 x transposeX2 = 4 种）
//   [ ] 用 test/ 下的脚本对比 NPU 结果与 golden 结果
//
// 【数值精度】
//   [ ] K 维点积确认用 FP32 累加
//   [ ] 精度误差 < 1e-4（相对 & 绝对），这是赛题 float32 的要求
//
// 【性能】
//   [ ] 用 msprof 采集 Task Duration / Cube 利用率 / Vector 利用率 / MTE 带宽
//   [ ] 确认每个核的负载大致均衡（行数分配是否均匀）
//   [ ] 考虑 Double Buffer：搬运和计算重叠
//
// ======================================================================
