/**
 * BatchMatmulMaxSum —— 算子原型定义（OpDef）
 *
 * 这个文件是干什么的？
 * -------------------
 * 它是写给 CANN 框架看的"算子说明书"。框架读完这份说明，才知道：
 *   - 这个算子叫什么名字
 *   - 要吃几个输入、吐几个输出、分别是什么类型和形状
 *   - 有什么可调参数（属性）
 *   - 支持哪些硬件型号
 *
 * 这里面**没有任何计算逻辑**。真正的计算在 op_kernel/ 里。
 *
 * 对 Python 选手的类比
 * -------------------
 * 如果你写过 Python 函数，这个文件相当于"函数签名 + 类型注解"：
 *
 *     def batch_matmul_max_sum(
 *         x1: Tensor[B, M, K],      # float16 或 bfloat16
 *         x2: Tensor[B, K, N],      # float16 或 bfloat16
 *         transposeX1: bool = False,
 *         transposeX2: bool = False,
 *     ) -> Tensor[B]:               # float32
 *         ...
 *
 * 只不过要用 C++ 的语法、按 CANN 框架要求的格式来写。
 */

#include "register/op_def_registry.h"

namespace ops {

class BatchMatmulMaxSum : public OpDef {
 public:
  explicit BatchMatmulMaxSum(const char* name) : OpDef(name) {
    // ==================== 输入 1：x1 ====================
    // 逻辑形状固定为 [B, M, K]
    // 存储形状随 transposeX1 变化：false -> [B,M,K]，true -> [B,K,M]
    this->Input("x1")
        .ParamType(REQUIRED)                        // 必选输入，不能省略
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})    // 支持两种半精度类型
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})     // ND = 普通连续排布
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

    // ==================== 输入 2：x2 ====================
    // 逻辑形状固定为 [B, K, N]
    // 存储形状随 transposeX2 变化：false -> [B,K,N]，true -> [B,N,K]
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

    // ==================== 输出：y ====================
    // 形状固定为 [B]，类型固定为 FLOAT32（赛题 3.6 节硬性要求）
    //
    // 注意这里的写法：DataType({...}) 里有两个元素，
    // 第 1 个对应 x1/x2 是 FLOAT16 的情况，第 2 个对应 BF16 的情况。
    // 也就是说：**无论输入是 fp16 还是 bf16，输出都必须是 fp32**。
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

    // ==================== 属性 ====================
    // ATTR(可选) 对应 AttrType(OPTIONAL)，默认值 false
    //
    // 再次强调赛题的坑：这两个属性**只描述数据在内存里怎么摆**，
    // 不表示算子要真的做一次转置运算。它们不改变逻辑形状，
    // 也不改变计算结果。
    this->Attr("transposeX1").AttrType(OPTIONAL).Bool(false);
    this->Attr("transposeX2").AttrType(OPTIONAL).Bool(false);

    // ==================== 硬件配置 ====================
    // 声明本算子要在昇腾 AI Core 上跑。
    // "ascend910b" 是昇腾 910B 系列芯片的标识。
    //
    // TODO(待确认)：请到比赛环境执行 `npu-smi info` 确认真实芯片型号，
    //               如果比赛用的是别的型号（如 ascend910_93 等），
    //               这里要相应改成对应的 SocVersion，否则编译或加载会失败。
    this->AICore().AddConfig("ascend910b");
  }
};

// 把上面这个类注册到框架里。
// OP_ADD 是一个宏，展开后会生成框架需要的注册代码。
OP_ADD(BatchMatmulMaxSum);

}  // namespace ops
