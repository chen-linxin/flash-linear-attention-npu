# AGENTS.md

本文件是 `flash-linear-attention-npu` 的仓库级 Agent 规则。根文件只保留开发流程来源和任务路由；具体方法、阶段输出和完成条件按任务读取 `docs/agents/` 中对应文件。若子目录存在更近的 `AGENTS.md`，以更近文件为准。

## 本仓开发流程来源

- Linear Attention 或 Block Sparse Attention 的 CATLASS C++ 算子开发，默认按
  [`docs/agents/cannbot-workflow.md`](docs/agents/cannbot-workflow.md) 调用 CANNBot
  `catlass-cpp-generator` 五阶段工作流；本仓保留适配、测试和交付责任。
- 其它算子或不满足 CANNBot 适用条件的任务，使用本文件和 `docs/agents/` 中定义的本地五阶段流程。
- 仓内代码和其他技术文档按本文件与 `docs/agents/` 规定的阶段作为接口证据、工程资料或实现参考。
- CANNBot 路由下如阶段职责发生冲突，以 `docs/agents/cannbot-workflow.md` 的边界为准；仓库目录、
  Stable-ABI、ATK、构建、安装和 CI 要求仍以本仓文档为准。

## 任务路由

| 任务类型 | 执行流程或必读内容 |
| --- | --- |
| 新增 Linear Attention / Block Sparse Attention CATLASS C++ 算子或公开接口 | `docs/agents/cannbot-workflow.md`；由 CANNBot 执行 01–04，本仓完成适配并作为 05 的验证后端 |
| 修改上述算子的接口、功能、内部实现或性能 | 先读取当前接口、唯一 CPU 标杆、设计、实现和测试，再按 `docs/agents/cannbot-workflow.md` 从最早受影响的 CANNBot 阶段继续 |
| 新增或修改其它算子 | 按顺序执行 `docs/agents/01-接口确认.md` → `02-标杆生成.md` → `03-方案设计.md` → `04-算子开发.md` → `05-算子测试.md` |
| 修改或新增算子测试，包括 ATK 用例 | `docs/agents/05-算子测试.md`、`tests/atk/README.md` 和当前算子的 ATK README；涉及 CPU 标杆对齐、输入值域、精度规则或精度失败时，同时按下一行进入精度路由 |
| 对齐 CPU 标杆、校准精度值域或定位精度问题 | 先读取 `docs/agents/reference/精度对比与定位.md` 选择场景，再按该文件指向的执行方法操作 |
| 修改公共组件、公共 ABI、代码生成模板或 Python runtime | `docs/architecture/torch-npu-decoupled-architecture.md`，并识别全部受影响算子 |
| 修改 wheel、OPP、构建或安装流程 | `docs/开发者指南.md` 和相关构建脚本 |
| 修改 PR、分支、CODEOWNERS 或 CI 规则 | `docs/repository-rules.md`、`.github/pull_request_template.md` 和现有 workflow |
| 修改 Triton 算子 | 当前 Triton 实现、导出入口、对应测试和 README，并采用 Triton 对应的实现约束 |

前一阶段的结论发生变化时，从最早受影响的阶段重新执行后续阶段。

目录索引、阶段输入输出和按任务阅读顺序见 `docs/agents/README.md`。
