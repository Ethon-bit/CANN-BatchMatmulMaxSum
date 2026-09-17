# CANN-BatchMatmulMaxSum

> **2026 年 CANN 挑战赛 · 上合赛区 · 初赛** 赛题专题仓库
> 赛题：**BatchMatmulMaxSum** —— 基于 Ascend C 的昇腾 NPU 算子开发
> 赛程：2026/09/04 —— 2026/10/17

---

## 这是什么题？

在 RAG（检索增强生成）和神经信息检索场景里，ColBERT 模型用 **Late Interaction** 判断"这句话和这篇文档有多相关"。整个过程分三步：

```
① 批量矩阵乘     query 的每个 token × document 的每个 token  →  相似度矩阵
② MaxSim 归约    每个 query token 挑出最像的那个 document token  →  取最大值
③ Sum 归约       把所有 query token 的最高分加起来              →  一个总分
```

PyTorch 里这是三个独立操作（`torch.bmm` → `torch.amax` → `torch.sum`），每一步都要把数据在显存里读一遍写一遍。中间那个相似度矩阵最大能到 8192×8192 = 6700 万个数字，**搬运开销比计算还大**。

**这道题要你把三步融合成一个算子，用 Ascend C 写在昇腾 NPU 上跑，并且要比拆分实现快。**

核心难点：

| 难点 | 说明 |
| :--- | :--- |
| 四种存储布局 | `transposeX1` × `transposeX2` 的 4 种组合都要正确支持 |
| 非对齐尾块 | M、N 只是"建议"16 的倍数，实际会出现 17、33 这种形状 |
| 较小 Batch | B 最小是 1，核用不满，得沿 M 维再切 |
| 大维度 | M/N/K 最大都能到 8192 |
| Cube/Vector 协同 | 矩阵乘归 Cube 管，归约归 Vector 管，要让它们流水起来 |

---

## 仓库结构

按 **CANN 算子工程标准布局** 组织：

```
CANN-BatchMatmulMaxSum/
│
├── op_host/                          【Host 侧】算子原型 + 切分方案
│   ├── batch_matmul_max_sum.cpp           算子原型定义（"算子长什么样"）
│   └── batch_matmul_max_sum_tiling.{h,cpp}  Tiling：怎么切给多个核
│
├── op_kernel/                        【Device 侧】真正在 NPU 上跑的计算
│   ├── batch_matmul_max_sum.h            核函数实现主体  ★ 主要工作在这里
│   └── batch_matmul_max_sum.cpp          核函数入口 + 上板检查清单
│
├── op_api/                           【接口层】让外部能调用这个算子
│   ├── aclnn_batch_matmul_max_sum.h       两段式接口声明
│   └── aclnn_batch_matmul_max_sum.cpp     参数校验与执行器组装
│
├── python/                           【基准】标准答案，用来验证你写的算子
│   ├── golden_pure.py                     零依赖版，逐行注释，适合理解算法
│   └── golden_torch.py                    官方 torch 版，精度比对以此为准
│
├── test/                             【测试】数据生成与精度比对
│   ├── test_accuracy.py                   生成 golden / 比对 NPU 结果
│   ├── cross_check_golden.py              两版 golden 的交叉验证
│   └── cases/cases.json                   14 个测试用例定义
│
├── docs/                             【文档】
│   ├── 02_写给Python初学者的导读.md        零基础入门路径
│   ├── 03_项目记录与调试日志.md            调试过程与结论
│   └── PATCH_OPT1_最小改动.md              竞态补丁说明
│
├── cmake/                            构建配置
├── CMakeLists.txt
├── build.sh
└── README.md                         ← 你在这里
```

---

## ⚠️ 当前状态（请先读这一段）

这个仓库**不是一份可以直接拿分的完整答案**。它是一套骨架 + 一个已验证的基准。诚实的状态如下：

| 部分 | 状态 | 说明 |
| :--- | :---: | :--- |
| `python/golden_pure.py` | ✅ **已完成并验证** | 赛题 3 个示例全部通过 |
| `python/golden_torch.py` | ✅ **已完成** | 严格照抄赛题原文 |
| `test/cross_check_golden.py` | ✅ **已验证** | 24 种组合（6 形状 × 4 布局）全部一致 |
| `test/test_accuracy.py` | ✅ **已跑通** | 14 个用例 golden 全部生成成功 |
| `test/cases/cases.json` | ✅ **已完成** | 覆盖边界、尾块、4 种布局、两种精度 |
| `docs/` | ✅ **已完成** | 初学者导读 + 项目记录 |
| `op_host/` 算子原型 | 🟡 **骨架** | 结构完整，属性/类型声明可直接用 |
| `op_host/` Tiling | 🟡 **骨架** | 切分公式已写出，**未在真实环境编译验证** |
| `op_kernel/` | 🔴 **未完成** | 流水结构已搭，关键的 DataCopy / Matmul 待实现 |
| `op_api/` | 🟡 **骨架** | 两段式结构完整，内部调用待按 CANN 版本补齐 |

> **关于「未验证」**：开发这份骨架的机器上没有昇腾 NPU，`op_host` / `op_kernel` / `op_api` 下的 C++ 代码**一次都没有编译过**。里面的 API 名称和参数是照 CANN 文档写的，但不同 CANN 版本之间会有差异，上板前必须逐个核对。**请不要直接假设它能编译通过。**

---

## 快速开始

### 1. 验证算法理解（不需要 NPU，不需要装任何库）

```bash
python python/golden_pure.py
```

预期输出：

```
[示例1 基础计算]      通过  得到=[2.0]  期望=[2.0]
[示例2 转置存储布局]  通过  得到=[2.0]  期望=[2.0]
[示例3 全负相似度]    通过  得到=[-1.0]  期望=[-1.0]

自测结果：3 个通过，0 个失败
```

### 2. 交叉验证两个基准是否一致（需要 torch）

```bash
uv run --with torch python test/cross_check_golden.py
```

预期：`交叉验证：24/24 通过`

### 3. 生成测试用例的标准答案

```bash
# 装了 numpy 能跑全部 14 个用例（推荐）
uv run --with numpy python test/test_accuracy.py --gen

# 没装 numpy 也能跑，但会跳过 4 个大形状用例（会明确打印跳过了哪些）
python test/test_accuracy.py --gen
```

> 生成的数据放在 `test/cases/golden/`，约 45MB，已在 `.gitignore` 里排除。
> 它完全可以从 `cases.json` + 固定随机种子重新生成，不需要进版本库。

### 4. 编译并上板（需要在有昇腾 NPU 的环境）

```bash
# 先确认芯片型号，然后改 op_host/batch_matmul_max_sum.cpp 里的 AddConfig
npu-smi info

# 编译
bash build.sh

# 跑完算子后，把输出按 <用例名>.bin 放进 test/cases/npu_out/，然后比对精度
python test/test_accuracy.py --check
```

---

## 三个必须避开的坑

这三个坑赛题**专门设计了用例来筛人**，`test/cases/cases.json` 里都有对应用例。

### 坑 1：MaxSim 初值不能是 0

```python
# ✗ 错的
best = 0.0
for n in range(N):
    best = max(best, sim[n])
# 当整行相似度全是负数时，返回 0 —— 但正确答案是那个最大的负数
```

```python
# ✓ 对的
best = float('-inf')   # 或者直接用第一个元素当初值
for n in range(N):
    best = max(best, sim[n])
```

👉 对应赛题**示例3** 和用例 `示例3_全负相似度`。

**更隐蔽的变体**：在 NPU 上做分块归约时，最后一块长度不足会补零。如果补的零参与了 ReduceMax，同样会踩这个坑。**尾块之后的位置必须显式填成 `-inf`。**

### 坑 2：Max 和 Sum 不能交换

$$\sum_m \max_n A[m,n] \ne \max_n \sum_m A[m,n]$$

反例：矩阵 `[[1,5],[3,2]]`

- 先 max 再 sum：`5 + 3 = 8` ✓
- 先 sum 再 max：`max(4, 7) = 7` ✗

### 坑 3：transpose 只是"数据的摆放方式"，不是"要执行转置"

同一份逻辑数据，在内存里可以横着放也可以竖着放：

| `transposeX1` | `transposeX2` | x1 存储形状 | x2 存储形状 |
| :---: | :---: | :--- | :--- |
| false | false | `[B, M, K]` | `[B, K, N]` |
| false | true | `[B, M, K]` | `[B, N, K]` |
| true | false | `[B, K, M]` | `[B, K, N]` |
| true | true | `[B, K, M]` | `[B, N, K]` |

**四种组合的结果必须完全相同。** 实现上不能真的去做一次转置搬运（那会白白多一次内存往返），而应该**按对应布局直接寻址读取**。

---

## 评分规则

$$\text{score} = \frac{100}{1 + \log_{1.5}\left(t / T\right)}$$

其中 $T$ 是全场最优耗时，$t$ 是你的耗时。

| 你的耗时 | 得分 |
| :--- | ---: |
| 和最优一样快 | 100 |
| 慢 1.5 倍 | 50 |
| 慢 2.25 倍 | 33.3 |
| 慢 3.4 倍 | 25 |

**三条关键结论：**

1. **15 个测试点必须全部精度通过才计分。** 挂一个 = 0 分。所以**正确性 >> 性能**。
2. 最终分数是 15 个 case 的**均值**。
3. 分数相同时，**提交越早排名越高**。

另外赛事规定：每天最多提交 50 次，取比赛期间**最后一次**提交的成绩。

---

## 开发路线图

**不要一上来就追性能。** 按这个顺序走：

- [x] **Step 1** — Python 版跑通，理解算法 ✅
- [x] **Step 2** — 交叉验证两版 golden 一致 ✅
- [ ] **Step 3** — 编译通过（先不管对不对）
- [ ] **Step 4** — 写一个**朴素但正确**的 kernel，15 个用例精度全过
- [ ] **Step 5** — 测出基线耗时，用 `msprof` 采集 Cube/Vector/MTE 利用率
- [ ] **Step 6** — 开始优化

### Step 6 的优化优先级

| 优先级 | 优化点 | 预期收益 | 说明 |
| :---: | :--- | :--- | :--- |
| ⭐⭐⭐ | **Double Buffer** | 最大 | 让数据搬运和计算重叠，掩盖 MTE 延迟 |
| ⭐⭐⭐ | **多核负载均衡** | 大 | B 小的时候必须沿 M 维切，否则一半核在空转 |
| ⭐⭐ | **Cube 替代 Vector 做矩阵乘** | 大 | Cube 算力比 Vector 高一个数量级 |
| ⭐⭐ | **凑满 16 行再送 Cube** | 中 | Cube 最小处理 16×16，只送 1 行利用率极低 |
| ⭐ | **减少 GM 往返** | 中 | 中间结果留在片上，不要写回 GM |

### 性能分析命令

```bash
msprof op --application="your_op_binary" \
    --output=./prof_out \
    --aic-metrics=CubeUtilization,VectorUtilization,MTEBandwidth
```

---

## 赛事日程

| 环节 | 时间 |
| :--- | :--- |
| 报名 | 07/21 —— 10/16 |
| 初赛赛题发布 | 09/05 |
| 在线答题 | 09/05 —— 10/16 |
| **参赛作品提交** | **10/17（17:30 封榜、18:00 提交截止）** |
| 晋级名单 | 10/18 公示、10/19 公布 |
| 决赛 | 10/20 —— 11/07 |
| 现场决赛 | 11/07 10:00—11:30 |

> 赛区匹配按**高校所属省份**（上海、安徽），不是户籍地或居住地。同一队伍成员须属同一赛区。

---

## 注意事项

- ⚠️ **算力申请**：CANNLab 算力仅允许**队长**申请，申请后**不能更换团队名称和成员**，且仅允许队长账号登录。**务必先组建好队伍再申请。**
- ⚠️ **合规红线**：核心计算必须在昇腾 NPU 上通过 AscendC 算子实现。把计算转移到 Host CPU、或用空 kernel 占位绕过 NPU 计算，**将被取消成绩**。
- ⚠️ **算子不得修改输入**。
- ⚠️ 本仓库已 **public**。比赛进行中（至 10/17）公开解题思路可能有查重风险，请自行斟酌。

---

## 参考链接

- [赛题发布页](https://competition.gitcode.com/competition/2094688057034268674/publish)
- [CANN 学习中心仓](https://gitcode.com/cann/cann-learning-hub) —— 官方教程，支持在线互动运行
- [赛事交流与入群入口](https://gitcode.com/cann/cann-ops-competitions/discussions/12)
- [昇腾社区文档](https://www.hiascend.com/document)

---

*本仓库为个人参赛备赛资料。赛题内容版权归主办方所有，本仓库不转载赛题原文。*
