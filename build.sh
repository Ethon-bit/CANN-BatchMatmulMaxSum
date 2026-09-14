#!/usr/bin/env bash
# ======================================================================
# BatchMatmulMaxSum —— 构建脚本
#
# ⚠️ 本脚本是骨架，未在真实 CANN 环境验证过。
#    正式编译请以 msopgen 生成的工程为准（详见 CMakeLists.txt 顶部说明）。
#
# 用法：
#     bash build.sh              # 默认编译
#     bash build.sh clean        # 清理构建目录
# ======================================================================

set -e  # 任何一条命令失败就立刻退出，不要带着错误继续往下跑

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# ---------- 清理 ----------
if [ "$1" == "clean" ]; then
    echo "清理构建目录: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    exit 0
fi

# ---------- 检查环境 ----------
if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "错误：未设置 ASCEND_HOME_PATH 环境变量。"
    echo ""
    echo "请先执行昇腾环境变量脚本，例如："
    echo "    source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo ""
    echo "如果不确定装在哪，可以试着手动找一下："
    echo "    ls /usr/local/Ascend/ascend-toolkit/"
    exit 1
fi

echo "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}"

# ---------- 检查芯片型号 ----------
echo ""
echo "当前环境的昇腾设备："
if command -v npu-smi &> /dev/null; then
    npu-smi info 2>/dev/null | head -20 || echo "  (npu-smi 执行失败，可能没有 NPU 设备)"
else
    echo "  未找到 npu-smi 命令"
fi
echo ""
echo "⚠️  请确认 op_host/batch_matmul_max_sum.cpp 里的 AddConfig(...)"
echo "   以及 CMakeLists.txt 里的 SOC_VERSION 与上面显示的型号一致。"
echo ""

# ---------- 构建 ----------
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

make -j"$(nproc 2>/dev/null || echo 4)"

echo ""
echo "======================================================"
echo "  构建完成。"
echo ""
echo "  提醒：本脚本只编译了 Host 侧代码。"
echo "  Kernel 侧（op_kernel/）需要 ascendc 编译器，"
echo "  请使用 msopgen 生成的完整工程流程。"
echo "======================================================"
