"""
当 github.com:443 连不上时，用 GitHub 的 Git Data API 把提交推上去。

背景：
    `git push` 走的是 github.com:443，在国内经常被重置/超时；
    而 `api.github.com` 通常仍然可达。两者是不同的域名、不同的链路，
    所以「git push 失败但 gh api 正常」是很常见的情况。

用法（在仓库根目录，先确保改动已经 git commit）：

    python tools/api_push.py <改动文件1> [改动文件2 ...] <提交信息文件>

例：
    python tools/api_push.py harness/bench.sh tools/api_push.py _msg.txt

原理：
    1. 读远程 main 当前的 commit 和它的 tree
    2. 把每个文件的内容做成 blob
    3. 以远程 tree 为 base_tree，建一棵只替换这几个文件的新 tree
    4. 以远程 HEAD 为父提交，建一个新 commit
    5. 把 refs/heads/main 指过去

    因为父提交就是远程当前的 HEAD，所以结果与本地 `git push` 等价，
    只是历史里那条提交的 SHA 会不同（提交人/时间戳不一致）。

⚠️ 前提：本地改动必须已经 commit —— 本脚本读的是**工作区文件内容**，
   不读 git 对象。所以先 `git add -A && git commit`，再跑本脚本。

⚠️ 与本地 git 的关系：推完之后远程 main 会比本地多一条「内容相同但 SHA
   不同」的提交。等网络恢复后，在本地执行
       git fetch origin && git reset --hard origin/main
   即可对齐，不会丢任何内容。
"""

import base64
import json
import os
import subprocess
import sys


def gh(args, stdin_bytes=None):
    """调用 gh api，返回 stdout（去掉首尾空白）。失败则打印 stderr 并退出。"""
    r = subprocess.run(["gh"] + args, input=stdin_bytes, capture_output=True)
    if r.returncode != 0:
        sys.stderr.write("[gh api 失败] " + r.stderr.decode("utf-8", "replace") + "\n")
        sys.exit(1)
    return r.stdout.decode("utf-8", "replace").strip()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    repo = os.environ.get("GH_REPO", "Ethon-bit/CANN-BatchMatmulMaxSum")
    branch = os.environ.get("GH_BRANCH", "main")

    files = sys.argv[1:-1]
    msg_file = sys.argv[-1]

    for f in files:
        if not os.path.isfile(f):
            print(f"找不到文件：{f}")
            return 1
    if not os.path.isfile(msg_file):
        print(f"找不到提交信息文件：{msg_file}")
        return 1

    print(f"仓库   : {repo}")
    print(f"分支   : {branch}")
    print(f"文件   : {', '.join(files)}")
    print()

    # 1) 远程当前 HEAD 与 tree
    head = gh(["api", f"repos/{repo}/git/ref/heads/{branch}", "--jq", ".object.sha"])
    base_tree = gh(["api", f"repos/{repo}/git/commits/{head}", "--jq", ".tree.sha"])
    print(f"远程 HEAD : {head[:10]}")

    # 2) 每个文件建 blob
    tree = []
    for f in files:
        content = base64.b64encode(open(f, "rb").read()).decode("ascii")
        payload = json.dumps({"encoding": "base64", "content": content}).encode("utf-8")
        sha = gh(["api", f"repos/{repo}/git/blobs", "--input", "-", "--jq", ".sha"], payload)
        path = f.replace(os.sep, "/")
        print(f"  blob  {path:<34} {sha[:10]}")
        tree.append({"path": path, "mode": "100644", "type": "blob", "sha": sha})

    # 3) 建 tree
    payload = json.dumps({"base_tree": base_tree, "tree": tree}).encode("utf-8")
    new_tree = gh(["api", f"repos/{repo}/git/trees", "--input", "-", "--jq", ".sha"], payload)
    print(f"tree      : {new_tree[:10]}")

    # 4) 建 commit
    message = open(msg_file, encoding="utf-8").read()
    payload = json.dumps(
        {"message": message, "tree": new_tree, "parents": [head]}
    ).encode("utf-8")
    commit = gh(["api", f"repos/{repo}/git/commits", "--input", "-", "--jq", ".sha"], payload)
    print(f"commit    : {commit[:10]}")

    # 5) 移动 ref（用 force，因为本地那条同名提交的 SHA 不同，不是快进）
    payload = json.dumps({"sha": commit, "force": True}).encode("utf-8")
    gh(["api", "-X", "PATCH", f"repos/{repo}/git/refs/heads/{branch}", "--input", "-"],
       payload)

    print(f"\n已推送 -> {branch}  ({commit[:10]})")
    print("网络恢复后，本地执行下面这条对齐（不会丢内容）：")
    print("    git fetch origin && git reset --hard origin/main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
