# BatchMatmulMaxSum 根因报告：kernel 写 GM 未回写 cache

> 平台提交 ID 344474，状态 **Pass，15 / 15 全部通过**（2026-09-18）
> 环境：CANN 9.0.0 + Ascend 910B3（dav-2201）

---

## 一句话结论

**kernel 里往 GM 写的结果值，停在 aicore 的 cache 里，从未回写到 GM。**
host 侧读回时拿到的是 `aclrtMalloc` 的零初始化内存。

修法是在 kernel 结束前显式回写 cache：

```cpp
AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(yGm);
```

---

## 这个 bug 为什么极难定位

它的**三个观测通道全都给出误导性的"正常"信号**：

| 观测手段 | 看到的 | 为什么骗人 |
|---|---|---|
| kernel 内读回 `yGm[b]` | **完全正确** | 命中的是和写入**同一个 cache**，所以"看起来写成功了" |
| 本地小用例（B ≤ 4） | **全部通过** | y 只有 8~16 字节、核数少，撞不上 |
| 平台反馈 | 只有"错误占比 12.50%" | 看不到错值是 0 还是算错了 |

平台上的表现是「**某些测试点恰好只错 1 个输出元素**」，且那个元素的位置在不同提交之间会漂移。
很容易被误判成精度问题或归约 bug。

---

## 现象与排查路径

### 1. 平台现象

- 通过 13 / 15，失败点各错 **恰好 1 个元素**（错误占比 12.50% = 1/8、7.69% = 1/13、6.25% = 1/16）
- 某个失败点（测试点 5）的 12.50% **在十几次提交里一位小数都没变过**，
  期间改过归约的**全部**参数（列块宽度、UB 行距、掩码、repeat 次数）都动不了它

**⇒ 对归约所有参数免疫的误差，不可能在归约里。** 这条推理是最终破案的起点。

### 2. 本地复现（关键一步）

本地 11 个用例（B ≤ 4）**全部通过**，误差只有 1e-6 量级。
但生成器 `gen_cases.py` 的 `BIG_IDS` 默认排除了大用例，**B 维度零覆盖**——
而平台失败的测试点从错误占比反推恰好是 **B = 8 / 13 / 16**。

按这个特征写网格扫描（B=8/13/16 × N≤256 × 多种子）后**立刻复现**：

```
B=8 M=16 N=65 K=32 float16 → 错 4/8，max|diff|=7.3e+01，max_rel=1.0000e+00
```

`max_rel = 1.0` 说明错值是 **0**（不是算错），且**一半输出是 0**。

### 3. 缩小到写回环节

在 kernel 里加打印，输出每个 block 算出的 `total`：

```
[dbg] blk=0 b=0 total=-74.458313 yWrote=74.458313
[dbg] blk=1 b=1 total=-60.239826 yWrote=60.239826
...
```

**8 个 batch 算出的值与 FP64 golden 逐位吻合**，读回也全部正确 —— 计算完全没问题。

而 host 拷回来的 y 里一半是 0，且逐行核对驱动侧
（`aclrtMemcpy` / 同步 / 大小 / `memset` 顺序）**没有任何问题**。

**⇒ 只能是"值停在 aicore cache 里，没回写到 GM"。**

---

## 修法

```cpp
// 1) 输出张量统一用 GlobalTensor，不要裸 __gm__ 指针强转
GlobalTensor<float> yGm;
yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), B);

// 2) 写出走 DMA（向量单元产出 → DataCopyPad）
Duplicate(yLocal, total, 1);
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtV2E3);
DataCopyPad(yGm[b], yLocal, ycp);

// 3) ★ 决定性的一步：kernel 结束前把本核 cache 的脏数据强制写回 GM
AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(yGm);
```

> ⚠️ **只做第 1、2 步不够**——实测仍然丢写。
> **`DataCacheCleanAndInvalid` 才是决定性的。**

### 修复后

- 本地 240 组合（shape × 数据模式 × 随机种子）**全部通过**
- 11 个本地用例 **全部通过**，误差回到 `1e-6` 量级
- 平台提交 **15 / 15**

---

## 给其他选手的三条建议

1. **只要 kernel 写 GM 的结果要被 host 读取，就必须考虑 cache 回写。**
   尤其是"每个核写一小段标量结果"的场景（比如每个 batch 一个分数）。

2. **本地用例的覆盖盲区会伪装成"没有 bug"。**
   本例中生成器默认排除大用例，导致 B 维度零覆盖——而 bug 恰好藏在 B ≥ 8。
   **务必核对用例表的每个维度都有覆盖。**

3. **拿到本地复现能力，比任何推理都重要。**
   本项目接上本地 NPU 环境后，**五轮**就定位到真因；
   在此之前靠平台提交盲试了**三十多轮**，其中两次还把 13/15 打回 1/15 和 7/15。
