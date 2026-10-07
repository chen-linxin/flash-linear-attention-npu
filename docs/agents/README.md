# 算子开发 Agent 指南

本目录记录 CANNBot `linear_attention` 与本仓工程规则之间的边界。五阶段方法直接采用任务锁定
的 CANNBot 版本，仓库级路由以根目录 [`AGENTS.md`](../../AGENTS.md) 为准。

## Ascend C 算子

本仓所有 `fla/ops/ascendc/**` 算子固定使用
[`CANNBot 算子开发桥接`](cannbot-workflow.md)：

```text
CANNBot 01--03：接口、唯一标杆和方案
  -> CANNBot 04：直调工程逐 Stage 开发与定向验证
  -> 本仓：适配层与 ATK 交付件包装
  -> 适配后整链路冒烟和单 case ATK 预检
  -> CANNBot 05：本仓 ATK 唯一一次完整验收
  -> PR / CI
```

CANNBot 04 通过直调 host 启动 kernel。适配完成后，`smoke_operator.py --compare` 通过 Python
API、Stable-ABI 或 ctypes、aclnn/op_api 和 host tiling 验证完整调用链。

## 仓库专属规则

| 工作内容 | 规则来源 |
| --- | --- |
| CANNBot 与本仓产物映射、执行顺序和失败恢复 | [`cannbot-workflow.md`](cannbot-workflow.md) |
| op_api/aclnn、Stable-ABI 和 Python 接入 | [`适配层接入指南`](../architecture/适配层接入指南.md) |
| ATK 目录、executor、用例和验收 | [`tests/atk/README.md`](../../tests/atk/README.md) |
| wheel、OPP、构建和安装 | [`开发者指南`](../开发者指南.md) |
| PR 与 CI | [`repository-rules.md`](../repository-rules.md) |
| Triton 实现 | 当前实现、导出入口、对应测试和 README |

## 执行口径

- CANNBot 01--05 的执行方法以任务锁定的 CANNBot 版本为准。
- 接口、标杆和设计分别只有一个可编辑事实来源。
- `tests/atk/<op>/reference.py` 同时供 CANNBot 直调测试和 ATK executor 使用。
- CANNBot 04 执行快速定向验证，CANNBot 05 对最终候选执行完整 `scope=all`。
- CANNBot 可用、版本匹配且产物契约完整是工作流入口条件；缺少条件时记录并报告阻塞项。
