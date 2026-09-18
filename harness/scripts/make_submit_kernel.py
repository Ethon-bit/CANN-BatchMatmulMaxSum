"""
生成**提交用的干净版本** op_kernel/kernel_mpar_submit.asc。

背景：
    第一次提交 kernel_mpar_clean.asc 时，平台报「代码中有不合规的内容」。
    那是内容扫描，不是编译错误。

    对比之前**被接受**的那份（kernel_cachefix.asc），本版新增的东西只有两类：

        1) getenv("BMMS_MAX_CORES") / atoll   —— 运行时读环境变量
        2) printf(...) x2                     —— 调试输出

    其余新增内容（SyncAll、if/else 分支、partGm / partLocal / kPartStride）
    都只是普通的计算逻辑，不该被拦。

    ⇒ 本脚本把这两类**诊断代码整体删掉**，再把注释剥掉，得到纯计算代码：
      没有任何 I/O、没有环境变量访问、没有字符串字面量。

处理步骤：
    1. 删掉 #include <cstdio> / <cstdlib>（删了 printf/getenv 后就不需要了）
    2. 删掉 BMMS_MAX_CORES 整段（含 getenv/atoll 与 [dbg] printf）
    3. 删掉 [mpar] printf
    4. bmmsCoreCap 全部换回 availableCoreNum
    5. 剥掉全部注释（复用 strip_comments.py 的状态机）

用法：
    python3 scripts/make_submit_kernel.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from strip_comments import strip_comments, tidy      # noqa: E402

SRC = os.path.join(ROOT, "op_kernel", "kernel_mpar.asc")
DST = os.path.join(ROOT, "op_kernel", "kernel_mpar_submit.asc")


def main():
    if not os.path.isfile(SRC):
        print(f"找不到源文件：{SRC}")
        return 1

    text = open(SRC, encoding="utf-8").read()
    box = [text]

    def rep(tag, old, new):
        n = box[0].count(old)
        if n != 1:
            print(f"  [失败] {tag}: 匹配到 {n} 处（应为 1 处）")
            sys.exit(1)
        box[0] = box[0].replace(old, new)
        print(f"  [ok] {tag}")

    # ---- 1) 删掉两个标准头（没有 printf/getenv 就不需要了）----
    rep("删 <cstdio> / <cstdlib>",
        "#include <cstdio>    // printf —— [mpar] 那行路径诊断输出\n"
        "#include <cstdlib>   // getenv / atoll —— BMMS_MAX_CORES 诊断开关\n",
        "")

    # ---- 2) 删掉 BMMS_MAX_CORES 整段（环境变量 + [dbg] 输出）----
    rep("删 BMMS_MAX_CORES 整段",
        "    // 诊断开关：BMMS_MAX_CORES=<n> 压低核数上限（不设则行为不变）\n"
        "    int64_t bmmsCoreCap = availableCoreNum;\n"
        "    if (const char* bmmsEnv = getenv(\"BMMS_MAX_CORES\")) {\n"
        "        const long long bmmsV = atoll(bmmsEnv);\n"
        "        if (bmmsV > 0 && bmmsV < bmmsCoreCap) { bmmsCoreCap = bmmsV; }\n"
        "    }\n"
        "    if (bmmsCoreCap != availableCoreNum) {\n"
        "        printf(\"[dbg] BMMS_MAX_CORES 生效：核数上限 %lld（原本 %lld）\\n\",\n"
        "               (long long)bmmsCoreCap, (long long)availableCoreNum);\n"
        "    }\n"
        "\n",
        "")

    # ---- 3) 删掉 [mpar] 那行输出 ----
    rep("删 [mpar] printf",
        "    printf(\"[mpar] 单元数=%lld 核数=%lld 路径=%s\\n\",\n"
        "           (long long)bmmsUnits, (long long)bmmsBlockNum,\n"
        "           bmmsUseMpar ? \"M并行(每核1单元)\" : \"原版(每核1batch)\");\n"
        "\n",
        "")

    # ---- 4) bmmsCoreCap 换回 availableCoreNum ----
    rep("bmmsCoreCap -> availableCoreNum",
        "    int64_t bmmsBlockNum = bmmsUseMpar\n"
        "                               ? bmmsUnits\n"
        "                               : ((B < bmmsCoreCap) ? B : bmmsCoreCap);",
        "    int64_t bmmsBlockNum = bmmsUseMpar\n"
        "                               ? bmmsUnits\n"
        "                               : ((B < availableCoreNum) ? B : availableCoreNum);")

    # ---- 5) 剥注释 ----
    stripped = tidy(strip_comments(box[0]))
    open(DST, "w", encoding="utf-8", newline="\n").write(stripped)

    print(f"\n已写出 {DST}（{stripped.count(chr(10))} 行）")

    # ---- 6) 自检 ----
    checks = {
        "printf": "printf" in stripped,
        "getenv": "getenv" in stripped,
        "atoll": "atoll" in stripped,
        "cstdio": "cstdio" in stripped,
        "cstdlib": "cstdlib" in stripped,
    }
    bad = [k for k, v in checks.items() if v]
    print("  自检:", "干净 ✓" if not bad else f"仍有残留 {bad}")

    has_cjk = any("一" <= ch <= "鿿" for ch in stripped)
    print("  中文字符:", "无 ✓" if not has_cjk else "仍有！（可能在字符串里）")

    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
