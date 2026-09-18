// ============================================================================
// data_utils.h —— 裸二进制文件读写（本地测试用）
//
// ReadFile 会**严格校验文件大小**：大小不符直接返回 false。
// 这样测试数据没生成全、或者形状算错时，会立刻报错而不是读到脏数据。
// ============================================================================

#ifndef BMMS_DATA_UTILS_H
#define BMMS_DATA_UTILS_H

#include <cstddef>
#include <fstream>
#include <string>

// 读整个文件到 buffer，要求文件大小恰好等于 expectBytes
inline bool ReadFile(const std::string& path, size_t expectBytes, void* buffer, size_t bufferCap)
{
    if (buffer == nullptr) return false;
    if (expectBytes > bufferCap) return false;

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    f.seekg(0, std::ios::end);
    const std::streamoff endPos = f.tellg();
    if (endPos < 0) return false;
    if (static_cast<size_t>(endPos) != expectBytes) {
        // 大小不对：多半是造数据脚本和这里算的形状不一致
        return false;
    }
    f.seekg(0, std::ios::beg);

    f.read(static_cast<char*>(buffer), static_cast<std::streamsize>(expectBytes));
    return static_cast<bool>(f);
}

// 把 buffer 的前 size 字节写进文件（覆盖）
inline bool WriteFile(const std::string& path, const void* buffer, size_t size)
{
    if (buffer == nullptr && size > 0) return false;

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return false;

    f.write(static_cast<const char*>(buffer), static_cast<std::streamsize>(size));
    f.flush();
    return static_cast<bool>(f);
}

#endif  // BMMS_DATA_UTILS_H
