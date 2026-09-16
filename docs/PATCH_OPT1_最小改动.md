# ★OPT-1 最小改动补丁（只修竞态，不动其他）

## ⚠️ 背景：为什么不能抽公共函数（实测教训）

上一版优化引入了这个函数，导致 **15/15 测试点全部 Compile Error**：

```cpp
__aicore__ inline constexpr int64_t ScratchStrideElems(int64_t N)   // ← 错
{
    return static_cast<int64_t>(kBaseM) * N + kSlackElems;
}
```

然后在 **host 侧** `run_kernel()` 里调用它。**编译失败。**

依据 —— Ascend C《函数执行空间限定符》表 1：

| 限定符 | 执行空间 | 说明 |
| --- | --- | --- |
| `__host__` | host | 只能被 Host 侧函数调用；**无限定符的函数默认是 host 函数** |
| `__global__` | device | 核函数入口，只能被 Host 侧调用 |
| `__aicore__` | device | **只能在 Device 侧执行，只能被 `__global__` 或其他 `__aicore__` 函数调用** |

⇒ `__aicore__`（device）函数被 host 函数调用 = **编译错误**。

**结论：跨 host/device 的公共函数一律不要抽。公式在哪一侧用，就在哪一侧写一遍。**
多写一行换掉一个编译错误，值得。

---

## 为什么用补丁而不是整份替换

`op_kernel/bmms_optimized.asc` 那一版引入了 4 处改动，其中 ★OPT-1 抽了一个
`__aicore__ constexpr` 公共函数，并在 **host 侧** `run_kernel()` 里调用它 ——
host 不能调用 device 函数，**直接编译失败**。

本补丁改为：**以你的原版为基础，只改 4 个位置**，不新增任何函数、类型或变量名。
把编译风险压到最低，同时把「正确性修复」和「性能优化」两个变量彻底分开。

---

## 改法

### 改动 1／4 —— kernel 侧：给 C scratch 加核间偏移

**位置**：原版第 177–178 行附近

**原来**：

```cpp
    GlobalTensor<float> cGm;
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cScratchGm), baseM * N);
```

**改成**：

```cpp
    GlobalTensor<float> cGm;
    const int64_t bmmsCoreIdx = GetBlockIdx();
    const int64_t bmmsScratchStride = static_cast<int64_t>(baseM) * N + kSlackElems;
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cScratchGm)
                            + bmmsCoreIdx * bmmsScratchStride,
                        static_cast<uint32_t>(baseM * N));
```

> 说明：`GetBlockIdx()` 返回类型在不同 CANN 版本可能是 `int32_t` 或 `int64_t`，
> 这里显式用 `int64_t` 接收，避免隐式转换告警。

---

### 改动 2／4 —— host 侧：把 blockNum 提前算出来，并让 scratch 乘以核数

**位置**：原版第 382–383 行附近

**原来**：

```cpp
    const size_t cScratchElems = static_cast<size_t>(kBaseM) * static_cast<size_t>(N) + kSlackElems;
    const size_t cScratchBytes = cScratchElems * sizeof(float);
```

**改成**：

```cpp
    const size_t cScratchElems = static_cast<size_t>(kBaseM) * static_cast<size_t>(N) + kSlackElems;
    // ★ 每核一份 scratch。blockNum 必须在 malloc 之前算出来。
    int64_t bmmsBlockNum = (B < availableCoreNum) ? B : availableCoreNum;
    if (bmmsBlockNum < 1) { bmmsBlockNum = 1; }
    const size_t cScratchBytes = cScratchElems * static_cast<size_t>(bmmsBlockNum) * sizeof(float);
```

---

### 改动 3／4 —— host 侧：删掉后面重复的 blockNum 计算

**位置**：原版第 405–407 行附近

**原来**：

```cpp
    // ---- 5. 启动（只启动 1 个 kernel）----
    int64_t blockNum = (B < availableCoreNum) ? B : availableCoreNum;
    if (blockNum < 1) { blockNum = 1; }
```

**改成**：

```cpp
    // ---- 5. 启动（只启动 1 个 kernel）----
    // blockNum 已在第 4 步（申请内存）之前算好，这里直接复用 bmmsBlockNum
```

---

### 改动 4／4 —— host 侧：两处启动语句改用 `bmmsBlockNum`

**位置**：原版第 409、413 行附近

**原来**：

```cpp
        bmms_main<bfloat16_t><<<blockNum, nullptr, stream>>>(...);
        bmms_main<half><<<blockNum, nullptr, stream>>>(...);
```

**改成**：

```cpp
        bmms_main<bfloat16_t><<<bmmsBlockNum, nullptr, stream>>>(...);
        bmms_main<half><<<bmmsBlockNum, nullptr, stream>>>(...);
```

---

## 为什么不直接用 bmms_optimized.asc

那个文件除了 ★OPT-1，还带了三处性能改动：

| 标记 | 内容 | 编译风险 | 正确性风险 |
| :--- | :--- | :--- | :--- |
| ★OPT-1 | 每核独立 scratch | 低（补丁后） | **必须改** |
| ★OPT-2 | 合并 DataCopyPad | 中 | **中**（`dstStride` 单位不确定） |
| ★OPT-3 | ORDER_VALUE_ONLY | 高（API 不确定） | — |
| ★OPT-4 | 提循环不变量 | 无 | 无 |

原版已经能编译、能拿 3 个 pass。**同时引入正确性修复 + 一个单位存疑的搬运改动，
一旦失败就无法定位。** 先只上 ★OPT-1。

---

## 验证顺序

1. 拿回原版 → 打上面 4 处补丁 → 提交
2. **如果 pass 数 ≥ 3 且没变差** → 竞态修复生效，可以继续做优化
3. **如果还是 Compile Error** → 说明我判断错了，把**编译器的完整报错文本**贴出来
4. **如果 pass 数反而下降** → ★OPT-1 有问题，回退改动 1 和 2

---

## ⚠️ 我仍然无法验证

本补丁**没有编译过**（手上没有 NPU 和 CANN 工具链）。它是纯代码阅读的结论。

补丁刻意避免了所有我不确定的构造：没有新增函数、没有新增类型、
没有用 `static_cast<uint16_t>`、没有 `constexpr` 局部变量。
只用最普通的算术和变量赋值。
