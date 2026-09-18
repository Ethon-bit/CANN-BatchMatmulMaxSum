#!/bin/bash
# ============================================================================
# 变体对比测量
#
# 用法（在仓库根目录执行，就一行）：
#     bash harness/bench_variants.sh 1 200
#
# 参数：<case_id> [重复次数]
#
# 它会依次编译并测量三个 kernel 变体，最后打一张对比表：
#
#     V0 基线      完整版本
#     V1 无 cache  只删掉 DataCacheCleanAndInvalid 那一行
#     V2 空内核     设备侧立刻 return
#
# 怎么读结果：
#     V2          = 测量方法的地板（launch + sync 的最小开销）
#     V0 - V1     = 刷整个 data cache 的代价   ← 本次最想知道的数
#     V0 - V2     = 一次真实 kernel 执行的全部开销
#     V1 - V2     = 除 cache 刷新外，其余部分的开销
#
# ⚠️ V2 不写 y，算术结果必然错；V1 丢了 cache 回写，结果也可能错。
#    这两个变体**只用来量时间**，不要拿它们的正确性下任何结论。
# ============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
REPO="$(dirname "${SCRIPT_DIR}")"

CID="${1:-1}"
REP="${2:-200}"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "❌ ASCEND_HOME_PATH 未设置。先执行："
    echo "     source /home/developer/Ascend/cann-9.1.0/set_env.sh"
    exit 1
fi

# 变体表：文件名 | 说明 | 是否预期算术正确
VARIANTS=(
  "kernel_timing.asc|V0 基线（原版）|yes"
  "kernel_timing_nocache.asc|V1 删掉 cache 回写|no"
  "kernel_timing_empty.asc|V2 空内核（立刻返回）|no"
)

# ---------- 预检 ----------
for v in "${VARIANTS[@]}"; do
    f="${v%%|*}"
    if [ ! -f "${REPO}/op_kernel/${f}" ]; then
        echo "❌ 缺少 ${REPO}/op_kernel/${f}"
        echo "   先生成： python3 harness/scripts/make_timing_kernel.py"
        exit 1
    fi
done

echo "============================================================"
echo "  变体对比测量   case=${CID}   重复 ${REP} 次"
echo "============================================================"

RESULT_FILE=/tmp/bmms_variants.txt
: > "${RESULT_FILE}"

for v in "${VARIANTS[@]}"; do
    IFS='|' read -r file desc expect <<< "$v"

    echo
    echo "############################################################"
    echo "#  ${desc}"
    echo "############################################################"

    cp -f "${REPO}/op_kernel/${file}" "${SCRIPT_DIR}/kernel.asc"
    rm -rf "${SCRIPT_DIR}/build"          # kernel 换了，必须重新编译

    out="$(bash bench.sh "${CID}" "${REP}" 2>&1)"
    echo "${out}"

    # 稳态的 [phase]（最后一行）与 [bench]
    phase="$(echo "${out}" | grep '^\[phase\]' | tail -1)"
    bench="$(echo "${out}" | grep '^\[bench\]' | tail -1)"
    ok="$(echo "${out}" | grep -cE '✅ 通过')"

    echo "${file}|${desc}|${bench}|${phase}|${ok}|${expect}" >> "${RESULT_FILE}"
done

# ---------- 汇总 ----------
echo
echo "============================================================"
echo "                    对 比 汇 总"
echo "============================================================"
printf "%-26s %-22s %s\n" "变体" "平均单次" "正确性"
printf "%-26s %-22s %s\n" "--------------------------" "----------------------" "--------"

while IFS='|' read -r file desc bench phase ok expect; do
    us="$(echo "${bench}" | sed -n 's/.*平均 \([0-9.]*\) us.*/\1/p')"
    if [ "${ok}" = "1" ]; then
        corr="✅ 通过"
    else
        corr="❌ 不通过（预期内）"
    fi
    [ "${expect}" = "yes" ] && [ "${ok}" != "1" ] && corr="❌ 不通过 ⚠️ 意外！"
    printf "%-26s %-22s %s\n" "${desc}" "${us} us" "${corr}"
done < "${RESULT_FILE}"

echo
echo "---- 各变体的 run_kernel 分段（steady state）----"
while IFS='|' read -r file desc bench phase ok expect; do
    echo "${desc}"
    echo "    ${phase:-（无 [phase] 行）}"
done < "${RESULT_FILE}"

echo
echo "============================================================"
echo "  怎么看："
echo "    V2            = launch+sync 地板"
echo "    V0 - V1       = 刷整个 data cache 的代价"
echo "    V0 - V2       = 一次 kernel 执行的全部开销"
echo "============================================================"
