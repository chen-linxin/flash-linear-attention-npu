# CANNBot 算子开发桥接

本文记录 CANNBot `catlass-cpp-generator` 的 `linear_attention` 功能分支与本仓适配、ATK 和上库
流程之间的边界。CANNBot 五阶段的具体方法以实际使用版本为准。

CANNBot 来源：
[`plugins-official/catlass-op-generator/catlass-cpp-generator`](https://gitcode.com/cann/cannbot-skills/tree/master/plugins-official/catlass-op-generator/catlass-cpp-generator)。
开始任务时记录使用的 commit 或版本；任务恢复时继续使用同一版本，升级前先检查工作流和产物格式。

## 固定路由

本仓 `fla/ops/ascendc/**` 均属于线性 Attention 算子域，直接设置：

```text
algorithm_family=linear_attention
workflow_id=catlass-linear-attention-v1
```

上述字段直接作为算法域分类结论。CANNBot 可用且版本匹配是工作流入口条件；缺少条件时记录并
报告阻塞项。

## 执行顺序

```text
CANNBot 01：冻结接口和 operator contract
  -> CANNBot 02：生成唯一 CPU 标杆和 precision policy
  -> CANNBot 03：完成 Stage、资源、同步、tiling 和性能方案
  -> CANNBot 04：直调工程实现 kernel/host tiling，逐 Stage 定向验证
  -> 本仓适配：op_api/aclnn、Stable-ABI、Python API、构建注册和 ATK 包装
  -> 适配后预检：整链路 smoke + 单 case ATK accuracy
  -> CANNBot 05：调用本仓 ATK 完成唯一一次 full 验收
  -> workflow complete
  -> PR / CI
```

CANNBot 04 通过后进入 `validation`。最终候选的完整 ATK 精度、性能、确定性和内存检测通过后，
workflow 更新为 `complete`。

## 唯一标杆

CANNBot 02 生成的数学标杆在本仓直接交付为：

```text
tests/atk/<op>/reference.py
```

该文件就是 CANNBot 的唯一 `reference.py`，同时供 CANNBot 04 直调测试和本仓 ATK executor
导入。公式、状态更新和边界处理集中在该文件维护。

`reference.py` 应是纯 CPU/PyTorch 模块：

- 运行依赖限定为 CPU/PyTorch 和 Python 标准库；
- 以冻结接口接收输入和属性，返回完整输出 tuple；
- 内部计算精度、有效区域和边界语义与 golden contract 一致；
- 可以由直调测试和 ATK CPU 节点在不同进程中独立导入。

若当前 CANNBot 版本使用默认产物路径，通过无数学逻辑的路径转发或目标路径记录连接本仓位置，
可编辑实现仍集中在 `tests/atk/<op>/reference.py`。新建或重新进入 CANNBot 流程的算子按本约定
收敛，历史算子在后续进入流程时迁移。

## CANNBot 04：适配前直调验证

CANNBot 04 使用其直调工程完成快速反馈：host `main()` 负责 ACL 初始化、tiling、device 内存和
workspace，直接执行 `kernel<<<blockDim, ..., stream>>>`，拷回当前 Stage 或整 kernel 输出，
再使用同一 `reference.py` 和 CANNBot 精度策略比较。

每轮只运行失败用例、受影响 Stage 和最小边界集合：

1. 当前 Stage 实现后，运行从 Stage 0 到当前 Stage 的最小合法用例并比较中间输出。
2. 修复或优化只回归受影响 Stage、原失败 case、目标模型 shape 和必要边界。
3. 全部 Stage 拼接后，使用直调工程运行最小整 kernel 冒烟。
4. 每轮性能候选先过定向精度；只有最终候选进入 full 验收。

本阶段的调用边界由直调 host、原始 kernel、唯一 `reference.py` 和 CANNBot 精度比较入口组成。
仓库公开 API、适配层和 ATK 调用链在后续接入阶段启用。

## 本仓适配与 ATK 包装

CANNBot 04 直调验证通过后，按本仓文档完成算子定义与 InferShape 接入、op_api/aclnn、Stable-ABI、
Python wrapper、导出和构建注册。所有参数名称、顺序、类型、默认值和返回值必须与 `docs/api.md`
一致。

ATK executor 负责输入构造、数据和属性转换、NPU DUT 调用及 ATK `FunctionApi`：CPU 节点通过
薄 `run_cpu` 包装调用 `reference.py`，数学实现集中在 `reference.py`。YAML 和 JSON 的值域、有效区域、
shape、dtype 和属性必须与 CANNBot precision policy 及冻结 contract 一致。

## 适配后快速预检

完成并安装本仓适配层后，才能运行 `tests/atk/<op>/scripts/smoke_operator.py --compare`。该脚本会
经过 Python API、Stable-ABI 或 ctypes、aclnn/op_api、host tiling 和 kernel，用于快速确认 ABI、
kernel 启动、同步和基本精度。正式精度结论由后续 ATK 验收产生。

smoke 通过后，使用 `ACCURACY_START=<id>`、`ACCURACY_END=<id+1>` 和 `-scope=accuracy` 运行一个
代表性 ATK case，确认 executor、YAML 判据和公开调用链已正确接通。需要时再扩大到受影响 case
集合；完整 `scope=all` 安排在最终候选阶段。

## CANNBot 05：唯一完整验收

最终候选由 CANNBot 05 编排本仓入口：

```sh
bash tests/atk/run_test_cpu.sh -op=<op> -scope=all
```

该命令是本任务唯一一次完整本地验收，覆盖完整精度矩阵、模型性能 case、确定性、mssanitizer、
全部可达 TilingKey 和边界分支。Example/ST、构建安装和仓库要求的其他回归按本仓交付规则补充。
PR CI 重放同一批交付件和验收定义，作为合入门禁。

## 产物和职责

| 环节 | 唯一产物或职责 |
| --- | --- |
| CANNBot 01 | `docs/api.md`、operator contract、目标架构 |
| CANNBot 02 | `tests/atk/<op>/reference.py`、definition、precision policy、golden contract |
| CANNBot 03 | `docs/design.md`、Stage/资源/同步/tiling/性能方案 |
| CANNBot 04 | `op_kernel`、host tiling、TilingData/TilingKey、直调验证证据 |
| 本仓适配 | op_api/aclnn、Stable-ABI、Python wrapper、导出和构建注册 |
| 本仓 ATK 包装 | executor、generator、YAML、三类 JSON、算子 ATK README |
| CANNBot 05 | 编排本仓 ATK full 验收并维护 workflow 状态 |
| 本仓交付 | Example/ST、构建安装、PR 和 CI 材料 |

## 失败恢复

| 失败证据 | 返回位置 |
| --- | --- |
| 接口或支持范围错误 | CANNBot 01 |
| 标杆、值域、有效区或数学语义错误 | CANNBot 02 |
| Stage、资源、同步、tiling 或性能方案错误 | CANNBot 03 |
| 直调 Stage 或整 kernel 结果错误 | CANNBot 04 |
| 直调通过但整链路 smoke 失败 | 本仓适配层 |
| smoke 通过但单 case ATK 失败 | ATK executor、YAML 或数据转换；证据指向核心时返回相应 CANNBot 阶段 |
| full 验收失败 | 按最早受影响位置恢复；修复后最终候选重新执行 fresh full 验收 |

性能达到目标前保持 `validation`，每轮改变一个主要变量，先做直调定向精度再同条件测量。性能
目标、本仓门禁和数学语义在迭代期间保持固定。
