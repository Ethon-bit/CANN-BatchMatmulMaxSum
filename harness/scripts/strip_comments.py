"""
把 .asc 里的 C++ 注释剥掉，生成一份"纯代码"版本。

为什么不能直接用正则 `//.*$` 删：
    字符串字面量里也可能出现 `//` 或 `/*`。比如

        printf("[mpar] 路径=%s\n", ...);

    里的 `\n` 如果被当成转义处理错了，后面的内容就全乱了。
    本脚本用一个**逐字符状态机**，严格按 C++ 的词法规则走：

        code  --//-->  line   --换行--> code
        code  --/*-->  block  --*/-->   code
        code  --"-->   str    --"-->    code   （\\ 转义）
        code  --'-->   chr    --'-->    code   （\\ 转义）

    只有处于 code 状态时才会识别注释起始符，字符串/字符字面量内容
    原样保留。

副作用（有意为之）：
    - 块注释里的换行会被保留，行号不会整体塌掉
    - 连续 3 行以上的空行压缩成 1 行
    - 行尾空白去掉

用法：
    python3 scripts/strip_comments.py <源文件> <目标文件>
"""

import os
import re
import sys


def strip_comments(src):
    out = []
    i = 0
    n = len(src)
    state = "code"

    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""

        if state == "code":
            if c == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if c == '"':
                state = "str"
                out.append(c)
                i += 1
                continue
            if c == "'":
                state = "chr"
                out.append(c)
                i += 1
                continue
            out.append(c)
            i += 1
            continue

        if state == "line":
            if c == "\n":
                state = "code"
                out.append(c)
            i += 1
            continue

        if state == "block":
            if c == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
            if c == "\n":
                out.append(c)      # 保留换行，行号不塌
            i += 1
            continue

        if state in ("str", "chr"):
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1])   # 转义序列整体带过
                i += 2
                continue
            if (state == "str" and c == '"') or (state == "chr" and c == "'"):
                state = "code"
            i += 1
            continue

    return "".join(out)


def tidy(text):
    """去掉行尾空白，把 3 行以上连续空行压成 1 行，去掉首部空行。"""
    lines = [ln.rstrip() for ln in text.split("\n")]
    out = []
    blank = 0
    for ln in lines:
        if ln == "":
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(ln)
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    src_path, dst_path = sys.argv[1], sys.argv[2]
    if not os.path.isfile(src_path):
        print(f"找不到源文件：{src_path}")
        return 1

    raw = open(src_path, encoding="utf-8").read()
    stripped = tidy(strip_comments(raw))
    open(dst_path, "w", encoding="utf-8", newline="\n").write(stripped)

    n_before = raw.count("\n") + 1
    n_after = stripped.count("\n")
    print(f"  {os.path.basename(src_path)}: {n_before} 行 -> {n_after} 行 "
          f"（去掉 {n_before - n_after} 行，{100 * (n_before - n_after) // n_before}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
