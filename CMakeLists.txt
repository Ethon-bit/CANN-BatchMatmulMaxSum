# ======================================================================
# BatchMatmulMaxSum —— 顶层构建脚本
#
# ⚠️ 重要说明
# ----------
# CANN 自定义算子的**官方推荐做法**是用 msopgen 工具生成工程骨架，
# 它会自动处理大量平台相关的细节（ascendc 编译器调用、算子信息库
# 生成、kernel 二进制编译等）。
#
# 本文件是一个**简化版**，目的是让工程结构完整、可读。
# 真正编译上板时，建议二选一：
#
#   方案A（推荐）：用 msopgen 重新生成工程，把本仓库的源码搬进去
#       msopgen gen -i batch_matmul_max_sum.json -c ai_core-<soc_version> -out .
#
#   方案B：以本文件为起点，参照 CANN 安装目录下的样例工程补全
#       $ASCEND_HOME/ascend-toolkit/latest/tools/msopgen/template/
#
# 本文件**未在真实 CANN 环境验证过**。
# ======================================================================

cmake_minimum_required(VERSION 3.16)
project(batch_matmul_max_sum CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- 检查 CANN 环境变量 ----------
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR
        "未找到环境变量 ASCEND_HOME_PATH。\n"
        "请先 source 昇腾环境变量脚本，例如：\n"
        "    source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
        "（Windows 环境下请用对应版本的 set_env.bat）")
endif()

set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
message(STATUS "ASCEND_HOME = ${ASCEND_HOME}")

# ---------- 目标 SoC 型号 ----------
# TODO(上板必改)：用 `npu-smi info` 查真实型号后填这里。
#   常见取值：ascend910b / ascend910_93 / ascend310p ...
set(SOC_VERSION "ascend910b"
    CACHE STRING "目标昇腾芯片型号")

# ---------- 头文件路径 ----------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/include/experiment
    ${ASCEND_HOME}/include/experiment/platform
)

# ---------- 源文件 ----------
set(HOST_SRCS
    op_host/batch_matmul_max_sum.cpp
    op_host/batch_matmul_max_sum_tiling.cpp
)

set(API_SRCS
    op_api/aclnn_batch_matmul_max_sum.cpp
)

# ---------- Host 侧目标 ----------
# 注意：Kernel 侧（op_kernel/）不能直接用普通 C++ 编译器编译，
# 必须走 ascendc 编译器 + 算子信息库（op info）流程。
# 下面这块只是占位，实际请用 msopgen 生成的构建脚本。
add_library(batch_matmul_max_sum_host STATIC ${HOST_SRCS})

# ---------- 提示 ----------
message(STATUS "")
message(STATUS "======================================================")
message(STATUS "  本 CMakeLists.txt 是简化骨架，不是可直接上板的构建脚本。")
message(STATUS "  正式编译请使用 msopgen 生成的工程，或参照 CANN 官方样例。")
message(STATUS "  详见文件顶部注释。")
message(STATUS "======================================================")
message(STATUS "")
