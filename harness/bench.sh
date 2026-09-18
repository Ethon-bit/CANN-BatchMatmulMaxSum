#!/bin/bash
# ============================================================================
# BatchMatmulMaxSum —— 单用例性能测量
#
# 用法：
#   bash bench.sh 9                # 测 case09，默认重复 50 次
#   bash bench.sh 9 200            # 测 case09，重复 200 次
#   bash bench.sh 9 200 1          # 同上，指定 shape_mode=1（存储形状）
#
# 输出三样东西：
#   1. [shape]  本次的形状参数 —— 用来核对和平台上哪个测试点对应
#   2. [phase]  run_kernel 内部四段耗时（tiling / malloc / kernel / free）
#   3. [bench]  重复 N 次的平均单次耗时
#   最后还会跑一次正确性比对，确认优化没有把结果改坏。
#
# ⚠️ 前提：编译用的 kernel.asc 必须是 **kernel_timing.asc**，
#    否则不会打印 [phase] 行（但 [bench] 行仍然有效）。
#
#       cp op_kernel/kernel_timing.asc kernel.asc
#
# ============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP=bmms_local_test
CID="${1:-}"
REP="${2:-50}"
MODE="${3:-1}"          # ★ 默认 1：平台传的是【存储形状】，已由实测确认

if [ -z "${CID}" ]; then
    echo "用法: bash bench.sh <case_id> [重复次数] [shape_mode]"
    echo "现有用例： $(ls work/case* -d 2>/dev/null | sed 's|work/case||' | tr '\n' ' ')"
    exit 1
fi

CID=$(printf '%02d' "${CID}" 2>/dev/null || echo "${CID}")

# ---------- 环境 ----------
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "错误：ASCEND_HOME_PATH 未设置。先执行："
    echo "    source /home/developer/Ascend/cann-9.1.0/set_env.sh"
    exit 1
fi

# ---------- 编译 ----------
#
# ⚠️ 这里必须检测「源文件比二进制新」，不能只看 build/ 在不在。
#
# 踩过的坑：bench_variants.sh 会连续换 kernel.asc 编译多个变体。它跑完后
# build/ 里留下的是**最后一个变体**的二进制。之后如果只换 kernel.asc 而不
# 重新编译，跑的就还是上一个变体 —— 现象是「所有用例都输出 0」，
# 极容易被误读成「kernel 在大形状上静默失败」。
#
# 教训：只要 build/ 存在就跳过编译 = 悄悄跑错二进制。必须比时间戳。
NEED_BUILD=0
if [ ! -x "build/${OP}" ] || [ ! -f build/Makefile ]; then
    NEED_BUILD=1
fi
for f in kernel.asc main.asc CMakeLists.txt data_utils.h; do
    if [ -f "$f" ] && [ "$f" -nt "build/${OP}" ]; then
        NEED_BUILD=1
        echo "（检测到 ${f} 比编译产物新 ⇒ 需要重新编译）"
    fi
done

if [ "${NEED_BUILD}" = "1" ]; then
    echo "=== 编译 ==="
    rm -rf build && mkdir -p build
    (
      cd build
      cmake .. > /tmp/cmake.log 2>&1 || { echo "cmake 失败："; tail -30 /tmp/cmake.log; exit 1; }
      make -j4  > /tmp/make.log  2>&1 || { echo "make 失败：";  tail -60 /tmp/make.log; exit 1; }
    ) || exit 1
    echo "编译成功"
fi

# ---------- 数据 ----------
D="work/case${CID}"
if [ ! -f "${D}/x1.bin" ]; then
    echo "=== 生成测试数据 ==="
    python3 scripts/gen_cases.py || exit 1
fi
if [ ! -f "${D}/x1.bin" ]; then
    echo "错误：找不到 ${D}/x1.bin"
    exit 1
fi

mkdir -p input output
cp -f "${D}/x1.bin" input/x1.bin
cp -f "${D}/x2.bin" input/x2.bin

# ---------- 取形状 ----------
read -r B M N K DT TX1 TX2 PAT <<< "$(python3 - "${CID}" <<'PY'
import json,sys,os
cid=int(sys.argv[1])
cs=json.load(open(os.path.join("work","cases.json"),encoding="utf-8"))
c=next(x for x in cs if x["id"]==cid)
dt={"float16":1,"bfloat16":2,"float32":0}[c["dtype"]]
print(c["B"],c["M"],c["N"],c["K"],dt,c["tx1"],c["tx2"],c["pattern"])
PY
)"

echo "============================================================"
echo "  case${CID}   B=${B} M=${M} N=${N} K=${K}  dtype=${DT}  tx=(${TX1},${TX2})"
echo "  shape_mode=${MODE}（1=存储形状，平台实际约定）   重复 ${REP} 次"
# ★ 打印当前用的是哪个 kernel —— 防止"换了文件却跑着旧二进制"这类错误再发生
if [ -f kernel.asc ]; then
    echo "  kernel.asc : $(wc -c < kernel.asc) 字节   md5=$(md5sum kernel.asc 2>/dev/null | cut -c1-8)"
fi
echo "============================================================"

echo "${B} ${M} ${N} ${K} ${DT} ${TX1} ${TX2} ${MODE}" > case.txt
rm -f output/y.bin

./build/${OP} bench "${REP}" > /tmp/bench.log 2>&1
rc=$?

if [ $rc -eq 2 ]; then
    echo "⏱  卡死/超时（退出码 2）"
    tail -20 /tmp/bench.log
    exit 2
fi
if [ $rc -ne 0 ]; then
    echo "⚠  运行失败（退出码 ${rc}）"
    tail -30 /tmp/bench.log
    exit $rc
fi

# ---------- 打印 [shape] / [phase] ----------
grep -E '^\[mpar\]|^\[shape\]|^\[phase\]' /tmp/bench.log | head -20

# ---------- 打印 [bench] ----------
echo
grep -E '^\[bench\]' /tmp/bench.log

# ---------- 正确性 ----------
echo
echo "---------- 正确性 ----------"
python3 scripts/verify.py "${CID}" "bench" || echo "  ↑ 结果不对，性能数据没有意义，先修正确性"
