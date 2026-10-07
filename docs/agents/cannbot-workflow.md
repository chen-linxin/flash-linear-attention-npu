# CANNBot 算子开发桥接

本文定义 CANNBot `catlass-cpp-generator` 与本仓交付流程的边界。目标是让 CANNBot 完成算子核心
研发，同时复用本仓已有的适配层、ATK 和上库门禁，并且全流程只维护一份接口、标杆和设计。

CANNBot 工作流来源：
[`plugins-official/catlass-op-generator/catlass-cpp-generator`](https://gitcode.com/cann/cannbot-skills/tree/master/plugins-official/catlass-op-generator/catlass-cpp-generator)。
开始任务时记录实际使用的 CANNBot commit 或版本，后续恢复任务时继续使用同一版本；需要升级时，
先检查工作流和产物格式差异。

## 何时使用

同时满足以下条件时使用本流程：

- 算法已根据公式确认属于 Linear Attention 或 Block Sparse Attention，不能只按算子名称判断。
- 核心实现使用 CATLASS C++。
- 任务涉及接口、标杆、方案、kernel、host tiling 或性能优化。

只改 op_api、Stable-ABI、Python wrapper、ATK、构建、安装、CI 或文档时，直接使用本仓对应流程。
不属于上述算法族时，使用 [`README.md`](README.md) 中的本地五阶段流程。

## 执行顺序

```text
需求和固定版本参考资料
  -> CANNBot 01：确认接口并冻结 operator contract
  -> CANNBot 02：生成唯一 CPU 标杆并冻结 golden contract
  -> CANNBot 03：完成 Stage、资源、同步和性能方案
  -> CANNBot 04：实现 kernel、host tiling 并完成逐 Stage 定向验证
  -> 本仓接入：op_api/aclnn、Stable-ABI、Python 导出和构建注册
  -> CANNBot 05：调用本仓 ATK、Example/ST、构建和性能入口完成 full 验收
  -> 整理本仓交付件，满足 PR 和 CI 门禁
```

CANNBot 04 定向精度通过后进入 `validation`，不能在接入完成时提前标记 `complete`。只有本仓 05
要求的完整验证全部通过，才能把 CANNBot workflow 更新为 `complete`。

## 职责和产物

| 环节 | 负责方 | 唯一产物或职责 |
| --- | --- | --- |
| 接口确认 | CANNBot 01 | `docs/api.md`、冻结的 `operator_contract`、目标架构 |
| 标杆生成 | CANNBot 02 | `reference/reference.py`、`definition.json`、`precision-policy.json`、冻结的 `golden_contract` |
| 方案设计 | CANNBot 03 | `docs/design.md`、Stage/资源/同步/tiling/性能方案 |
| 核心开发 | CANNBot 04 | `op_kernel`、host tiling、TilingData/TilingKey、逐 Stage 验证证据 |
| 仓库接入 | 本仓 | 算子定义与 InferShape 的仓库接入、op_api/aclnn、Stable-ABI、Python wrapper、导出和构建注册 |
| 完整验收 | CANNBot 05 编排，本仓执行 | ATK、Example/ST、构建安装、性能和 CI 所需证据 |
| 最终交付 | 本仓 | 算子代码、适配代码、测试用例、README 和 PR 材料 |

CANNBot 产物需要映射到本仓目标算子目录，不保留两份可编辑副本。接入过程中如需转换目录或命名，
记录“来源文件 -> 目标文件 -> 仅路径/命名转换或实际语义变化”。发生语义变化时返回相应 CANNBot
阶段重新生成和验证，不能在适配层静默修正。

## 唯一事实来源

- 接口：`docs/api.md` 是输入、输出、属性、默认值、异常和支持范围的唯一事实来源。
- 标杆：`reference/reference.py` 是唯一可编辑的数学标杆。ATK executor 只能导入它或通过薄适配器
  完成参数和数据格式转换，禁止复制公式、状态更新或边界处理逻辑。
- 精度策略：`reference/precision-policy.json` 维护 CANNBot 的输入值域、有效区域、关键分区和比较
  策略。本仓正式 ATK 仍使用 YAML 中配置的原生精度标准；其值域和有效区必须与该策略一致，阈值
  映射需要留下校准记录，executor 不得另造精度指标。
- 设计：`docs/design.md` 是 Stage、任务映射、内存、同步和性能方案的唯一事实来源。

本仓现有 [`01-接口确认.md`](01-接口确认.md)、[`02-标杆生成.md`](02-标杆生成.md)、
[`03-方案设计.md`](03-方案设计.md) 和 [`04-算子开发.md`](04-算子开发.md) 在此路由下只作为
仓库接入检查项。如果检查发现缺失内容，补充到上述唯一产物并返回 CANNBot 校验，不得新建平行产物。

## 接入检查

开始写本仓适配代码前，逐项确认：

1. CANNBot workflow 状态为 `validation`、`validation_scope=full`，两个 contract 均已冻结。
2. `target_architecture` 与目标 SoC、编译标识和实际实现一致，A2/A3 与 A5 路径没有混用。
3. `docs/api.md`、`reference/reference.py`、`precision-policy.json`、`docs/design.md` 与核心代码版本一致。
4. 核心代码在本仓目标目录完成路径映射，且不存在另一份继续维护的 kernel 或标杆源码。
5. op_api/aclnn、Stable-ABI 和 Python wrapper 的名称、顺序、类型、默认值及返回值与 `docs/api.md` 一致。
6. ATK executor 只做调用与数据转换；测试 generator、YAML 和有效区处理与冻结的 golden contract 一致。

## 最终验证和失败恢复

完整验收按 [`05-算子测试.md`](05-算子测试.md)、[`tests/atk/README.md`](../../tests/atk/README.md)、
[`开发者指南`](../开发者指南.md) 和当前算子的 ATK README 执行。本仓验证结果回填 CANNBot
`docs/validation.md`，作为 CANNBot 05 的 full 验收证据。

失败后返回最早受影响的位置：

| 失败原因 | 返回位置 |
| --- | --- |
| 接口或支持范围错误 | CANNBot 01 |
| golden、值域、有效区或标杆语义错误 | CANNBot 02 |
| Stage、资源、同步、tiling 或性能方案错误 | CANNBot 03 |
| kernel 或 host tiling 实现错误 | CANNBot 04 |
| op_api、Stable-ABI、Python、构建或导出错误 | 本仓接入层 |
| 测试用例、ATK executor 或交付记录错误 | 本仓 05；若暴露上游问题，再返回对应 CANNBot 阶段 |

最终交付前，将仍有效的覆盖范围、测试结果和性能结论收敛到本仓要求的 README；CANNBot workflow
通过 full 验收后才能标记完成。性能未达标时保持在 CANNBot `validation`，按其性能优化恢复规则迭代，
不能降低目标或跳过本仓门禁。
