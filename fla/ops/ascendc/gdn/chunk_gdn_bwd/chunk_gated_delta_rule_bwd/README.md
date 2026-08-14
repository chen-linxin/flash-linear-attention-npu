# ChunkGatedDeltaRuleBwd 融合算子

本目录用于开发 GDN 训练反向大融合算子。目标是在保持当前无状态边界训练语义和精度的前提下，
将现有反向小算子拼接收敛为统一 L0 路径，并将目标场景耗时降低到当前拼接基线的 0.5 倍。

开发开始前必须遵守仓库根目录 `AGENTS.md` 和 PR 299 引入的 `docs/agents/` 指导文档。

## 当前范围

- 支持 dense 和 packed varlen。
- 支持 `K=128`、`V=128/256`、`chunkSize=64/128`。
- 支持 FP16/BF16 主输入以及现有 gate/beta dtype 组合。
- 原生支持 GVA，要求 `HV % HK == 0`。
- 当前不支持 CP 状态边界训练：
  `initial_state=None`、`output_final_state=False`、`dht=None`、`dh0=None`。
- 保持 `flash_gated_delta_rule` 公开接口兼容。

## 文档

- [设计方案](设计方案.md)
- [开发流程与迭代记录](开发流程.md)
- [精度与性能用例](test/测试用例.md)
