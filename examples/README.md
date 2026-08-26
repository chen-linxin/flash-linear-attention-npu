# 示例

## 简介

本目录提供 `flash-linear-attention-npu` 的端到端 GDN 调用、Fast Kernel Launch 和 Add 算子工程样例。
运行示例前，请先按照仓库根目录的[构建与安装说明](../README.md)安装 `flash-linear-attention-npu`，并加载与目标产品匹配的 CANN 环境。

## 目录说明

```text
examples/
├── adapters/
│   └── triton_ascend_kda/
│       └── sitecustomize.py          # 可选的 Triton Ascend KDA 启动适配器
├── add_example/                      # Add 算子端到端工程样例
│   ├── op_host/                      # 算子定义、Tiling 和 InferShape
│   ├── op_kernel/                    # AI Core Kernel
│   ├── op_kernel_aicpu/              # AICPU Kernel（含 fallback 场景）
│   └── tests/                        # 算子测试工程
├── fast_kernel_launch_example/       # Ascend C Kernel 直调与 PyTorch Extension 样例
├── CMakeLists.txt                    # 自动加入包含 CMakeLists.txt 的示例子目录
├── flash_gated_delta_rule.py         # GDN 前向、反向和模型级冒烟示例
└── README.md
```

## GDN 端到端示例

[`flash_gated_delta_rule.py`](flash_gated_delta_rule.py) 组装 GDN 前向和反向小算子，覆盖 dense / varlen、可选初始状态、最终状态、Q/K L2Norm，以及 Ascend C 和 Triton 调用路径。

运行默认的 GDN 张量级前向、反向冒烟用例：

```sh
python3 examples/flash_gated_delta_rule.py --device 0
```

运行较小 shape 的 `DemoGatedDeltaNet` 模型级前向、反向冒烟用例：

```sh
python3 examples/flash_gated_delta_rule.py \
  --device 0 \
  --demo-model \
  --no-varlen \
  --tokens 128 \
  --query-heads 2 \
  --value-heads 2 \
  --key-dim 128 \
  --value-dim 128 \
  --chunk-size 64
```

`--demo-model` 会在 GDN 调用链外增加 Q/K/V 投影、`causal_conv1d`、门控归一化和输出投影。默认不传该参数时，脚本直接构造 Q/K/V、门控和 beta 张量，只验证 GDN 核心调用链。完整参数请运行：

```sh
python3 examples/flash_gated_delta_rule.py --help
```

### Example/ST 用例

[`ci/example_st_cases.json`](../ci/example_st_cases.json) 管理仓库启用的 Example/ST shape 和行为参数，[`ci/run_example_st_cases.py`](../ci/run_example_st_cases.py) 负责逐项运行：

```sh
python3 ci/run_example_st_cases.py \
  --device 0 \
  --cases-file ci/example_st_cases.json
```

只检查用例解析和生成的命令，不启动 NPU 计算：

```sh
python3 ci/run_example_st_cases.py \
  --device 0 \
  --cases-file ci/example_st_cases.json \
  --dry-run
```

当前启用的 Example/ST 用例使用直接张量 GDN 路径。需要覆盖 `DemoGatedDeltaNet` 时，应在用例中设置 `"demo_model": true`，或使用前述 `--demo-model` 命令直接运行。

## Fast Kernel Launch 示例

[`fast_kernel_launch_example`](fast_kernel_launch_example/README.md) 演示如何通过 PyTorch Extension 使用 `<<<>>>` 语法直接启动 Ascend C Kernel。该示例会独立构建并安装 `ascend_ops` wheel。

运行全部测试：

```sh
cd examples/fast_kernel_launch_example
bash build_and_test.sh
```

运行指定算子的测试，例如 `chunk_fwd_o`：

```sh
cd examples/fast_kernel_launch_example
bash build_and_test.sh chunk_fwd_o
```

支持的算子以 [`tests/`](fast_kernel_launch_example/tests/) 下的实际目录为准。

## Add 算子工程样例

[`add_example`](add_example/README.md) 展示一个完整的 Add 算子工程，包括算子定义、Tiling、InferShape、Kernel、AICPU fallback、aclnn 调用样例和 UT。该目录用于理解标准 AI Core 算子工程结构，不属于 GDN 端到端调用链。

## Triton Ascend KDA 适配器

[`adapters/triton_ascend_kda/sitecustomize.py`](adapters/triton_ascend_kda/sitecustomize.py) 是一个可选的 Python 启动钩子。将适配器目录加入 `PYTHONPATH`，并显式设置开关后，可在 Python 启动时安装模型透明的 KDA 前向替换：

```sh
export PYTHONPATH="$(pwd)/examples/adapters/triton_ascend_kda:${PYTHONPATH}"
export FLA_NPU_ENABLE_TRITON_ASCEND_KDA_ADAPTER=1
python3 <model_entry.py>
```

未设置 `FLA_NPU_ENABLE_TRITON_ASCEND_KDA_ADAPTER=1` 时，该启动钩子不会安装适配器。

## 新增示例要求

- 新增示例必须可独立运行，避免依赖其他示例的中间产物。
- 涉及新算子的示例，需同时提供该算子的单算子测试（`torch_custom/fla_npu/test/test_npu_<op>.py`）并接入 `test.sh`。
- 示例代码优先使用稳定 Python 入口 `fla_npu.ops.ascendc` / `fla_npu.ops.triton`，不要默认走 legacy `torch.ops.npu.*` 路径。
- 在 `examples/` 新增子目录时，如需参与统一编译，请提供对应 `CMakeLists.txt`。
