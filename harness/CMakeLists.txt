# ============================================================================
# BatchMatmulMaxSum —— 本地测试工程构建脚本
#
# 说明：本工程用于在昇腾 NPU 上直接调用 run_kernel 做本地验证。
#       链接库列表参考了比赛方模板，保持一致：
#       tiling_api / register / platform / unified_dlog / dl / m / graph_base
#
# 用法：bash run.sh        （会自动 cmake + make + 跑测试）
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(bmms_local_test LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

# 目标芯片架构。
#   NPU_ARCH 由 CANN 环境脚本设置；未设置时回落到 dav-2201。
#   上板前请用 `echo $NPU_ARCH` 确认，与本机 npu-smi 显示的型号匹配。
if(DEFINED NPU_ARCH)
    set(SOC_ARCH ${NPU_ARCH})
else()
    set(SOC_ARCH "dav-2201")
endif()

message(STATUS "SOC_ARCH = ${SOC_ARCH}")

# 编译入口是 main.asc（它 #include "kernel.asc"）
add_executable(bmms_local_test main.asc)

target_link_libraries(bmms_local_test PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(bmms_local_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(bmms_local_test PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>
)
