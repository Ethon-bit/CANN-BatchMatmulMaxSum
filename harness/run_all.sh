#!/bin/bash
# ============================================================================
# BatchMatmulMaxSum —— 本地全量测试
#
# 用法：
#   bash run_all.sh                 # 编译 + 跑全部用例 × 两种 shape_mode
#   bash run_all.sh 1 2 3 4         # 只跑指定 case
#   bash run_all.sh --build-only    # 只编译
#   REPEAT=5 bash run_all.sh 1      # 同一用例连跑 5 次（查竞态/卡死）
#
# 每个用例会跑两遍：
#   shape_mode=0  TensorGroupInfo 填【逻辑形状】
#   shape_mode=1  TensorGroupInfo 填【存储形状】
# 哪种对得上 golden，哪种解释就是正确的 —— 这就一次性判定那个悬案。
# ============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP=bmms_local_test
REPEAT="${REPEAT:-1}"

# ---------- 环境 ----------
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "错误：ASCEND_HOME_PATH 未设置。先执行："
    echo "    source /home/developer/Ascend/cann-9.1.0/set_env.sh"
    exit 1
fi
echo "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}"
echo "NPU_ARCH         = ${NPU_ARCH:-(未设置，将用 CMakeLists 里的默认值)}"

# ---------- 编译 ----------
if [ "${1:-}" = "--build-only" ]; then
    rm -rf build && mkdir -p build && cd build
    cmake .. && make -j4
    exit $?
fi

echo
echo "=== [1/3] 编译 ==="
if [ ! -x "build/${OP}" ] || [ ! -f build/Makefile ]; then
    rm -rf build && mkdir -p build
fi
(
  cd build
  cmake .. > /tmp/cmake.log 2>&1 || { echo "cmake 失败："; tail -30 /tmp/cmake.log; exit 1; }
  make -j4  > /tmp/make.log  2>&1 || { echo "make 失败：";  tail -60 /tmp/make.log; exit 1; }
) || exit 1
echo "编译成功 -> build/${OP}"

# ---------- 生成数据 ----------
echo
echo "=== [2/3] 生成测试数据 + golden ==="
python3 scripts/gen_cases.py || exit 1

# ---------- 跑用例 ----------
echo
echo "=== [3/3] 跑用例 ==="
mkdir -p input output

if [ $# -gt 0 ]; then
    WANT="$*"
else
    WANT=""
fi

pass0=0; fail0=0; err0=0
pass1=0; fail1=0; err1=0

for d in work/case*/; do
    cid=$(basename "$d" | sed 's/case//' | sed 's/^0*//')
    [ -z "$cid" ] && cid=0

    if [ -n "$WANT" ]; then
        echo "$WANT" | tr ' ' '\n' | grep -qx "$cid" || continue
    fi

    # 取该用例的输入
    cp -f "$d/x1.bin" input/x1.bin
    cp -f "$d/x2.bin" input/x2.bin

    # 从 cases.json 取形状（用 python 解析，避免 shell 里再写一遍）
    read -r B M N K DT TX1 TX2 PAT <<< "$(python3 - "$cid" <<'PY'
import json,sys,os
cid=int(sys.argv[1])
cs=json.load(open(os.path.join("work","cases.json"),encoding="utf-8"))
c=next(x for x in cs if x["id"]==cid)
dt={"float16":1,"bfloat16":2,"float32":0}[c["dtype"]]
print(c["B"],c["M"],c["N"],c["K"],dt,c["tx1"],c["tx2"],c["pattern"])
PY
)"

    for mode in 0 1; do
        for rep in $(seq 1 "$REPEAT"); do
            echo "$B $M $N $K $DT $TX1 $TX2 $mode" > case.txt
            rm -f output/y.bin
            ./build/${OP} > /tmp/run.log 2>&1
            rc=$?

            label="shape_mode=$mode"
            [ "$REPEAT" -gt 1 ] && label="$label rep$rep"

            if [ $rc -eq 2 ]; then
                echo "  case${cid} [$label] ⏱  卡死/超时 (退出码 2)"
                tail -3 /tmp/run.log | sed 's/^/        /'
                [ "$mode" = "0" ] && err0=$((err0+1)) || err1=$((err1+1))
                continue
            fi
            if [ $rc -ne 0 ]; then
                echo "  case${cid} [$label] ⚠  运行失败 (退出码 $rc)"
                tail -5 /tmp/run.log | sed 's/^/        /'
                [ "$mode" = "0" ] && err0=$((err0+1)) || err1=$((err1+1))
                continue
            fi

            python3 scripts/verify.py "$cid" "$label"
            if [ $? -eq 0 ]; then
                [ "$mode" = "0" ] && pass0=$((pass0+1)) || pass1=$((pass1+1))
            else
                [ "$mode" = "0" ] && fail0=$((fail0+1)) || fail1=$((fail1+1))
            fi
        done
    done
done

echo
echo "============================================================"
echo "  shape_mode=0（填逻辑形状）: 通过 $pass0 / 失败 $fail0 / 异常 $err0"
echo "  shape_mode=1（填存储形状）: 通过 $pass1 / 失败 $fail1 / 异常 $err1"
echo "============================================================"
if [ "$pass0" -gt "$pass1" ]; then
    echo "  ⇒ 结论：TensorGroupInfo.shape 是【逻辑形状】"
elif [ "$pass1" -gt "$pass0" ]; then
    echo "  ⇒ 结论：TensorGroupInfo.shape 是【存储形状】"
else
    echo "  ⇒ 两种模式结果相同，本批用例无法区分（可能都不含 transpose）"
fi
