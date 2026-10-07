# AGENTS.md

本文件是 `flash-linear-attention-npu` 的仓库级 Agent 规则。根文件只保留开发流程来源和任务路由；具体方法、阶段输出和完成条件按任务读取 `docs/agents/` 中对应文件。若子目录存在更近的 `AGENTS.md`，以更近文件为准。

## 本仓开发流程来源

- 本仓 `fla/ops/ascendc/**` 下的算子均属于线性 Attention 算子域。涉及算子接口、标杆、方案、
  kernel、host tiling 或性能优化时，直接按 [`docs/agents/cannbot-workflow.md`](docs/agents/cannbot-workflow.md)
  使用 CANNBot `linear_attention` 功能分支。
- 仓库规则已经完成算法域确认：固定 `algorithm_family=linear_attention`、
  `workflow_id=catlass-linear-attention-v1`，不再执行 family 分类。
- op_api/aclnn、Stable-ABI、Python 导出、ATK、构建、安装和 CI 继续使用本仓流程；Triton 算子
  使用其现有实现约束。
- 仓内代码和其他技术文档按本文件与 `docs/agents/` 规定的阶段作为接口证据、工程资料或实现参考。
- CANNBot 路由下如阶段职责发生冲突，以 `docs/agents/cannbot-workflow.md` 的边界为准；仓库目录、
  Stable-ABI、ATK、构建、安装和 CI 要求仍以本仓文档为准。

## 任务路由

| 任务类型 | 执行流程或必读内容 |
| --- | --- |
| 新增或修改 Ascend C 算子的接口、功能、核心实现或性能 | `docs/agents/cannbot-workflow.md`；直接使用 CANNBot `linear_attention`，由 CANNBot 执行 01–04，本仓完成适配并作为 05 的验证后端 |
| 继续既有 Ascend C 算子任务 | 读取当前接口、唯一 CPU 标杆、设计、实现和测试，固定 `linear_attention` 后从最早受影响的 CANNBot 阶段继续 |
| 修改或新增算子测试，包括 ATK 用例 | `docs/agents/05-算子测试.md`、`tests/atk/README.md` 和当前算子的 ATK README；涉及 CPU 标杆对齐、输入值域、精度规则或精度失败时，同时按下一行进入精度路由 |
| 对齐 CPU 标杆、校准精度值域或定位精度问题 | 先读取 `docs/agents/reference/精度对比与定位.md` 选择场景，再按该文件指向的执行方法操作 |
| 修改公共组件、公共 ABI、代码生成模板或 Python runtime | `docs/architecture/torch-npu-decoupled-architecture.md`，并识别全部受影响算子 |
| 修改 wheel、OPP、构建或安装流程 | `docs/开发者指南.md` 和相关构建脚本 |
| 修改 PR、分支、CODEOWNERS 或 CI 规则 | `docs/repository-rules.md`、`.github/pull_request_template.md` 和现有 workflow |
| 修改 Triton 算子 | 当前 Triton 实现、导出入口、对应测试和 README，并采用 Triton 对应的实现约束 |

前一阶段的结论发生变化时，从最早受影响的阶段重新执行后续阶段。

目录索引、阶段输入输出和按任务阅读顺序见 `docs/agents/README.md`。
