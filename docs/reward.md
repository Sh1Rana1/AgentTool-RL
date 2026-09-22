# Verifiable Reward Contract (v0.2)

AgentTool-RL 的奖励完全由场景隐藏状态和已记录 trajectory 计算，不使用人工打分或 LLM-as-a-Judge。同一场景、动作序列和配置始终得到相同分数。

## 成功条件

只有同时满足以下条件，episode 才获得任务成功奖励：

1. `submit_diagnosis.root_cause` 与隐藏根因一致；
2. 提交的 evidence IDs 覆盖该场景全部必要证据；
3. mitigation 与场景允许的处置方案之一匹配；
4. 所有 evidence IDs 都来自本 episode 先前真实执行的工具结果。

第四项在环境层校验。伪造或跨 episode 引用证据不会终止任务，而会返回可恢复的 `invalid_evidence_reference` 错误。

## v0.2 系数

| Component | Coefficient | Calculation |
| --- | ---: | --- |
| Task success | +1.00 | 满足完整成功条件 |
| Correct diagnosis | +0.30 | 根因标签正确 |
| Evidence coverage | +0.40 | `required evidence collected / required evidence total` |
| Correct mitigation | +0.20 | 处置命中允许集合 |
| Valid-call ratio | +0.20 | `valid calls / attempted calls` |
| Error recovery | +0.15 | 非法调用后，使用同一工具完成一次合法修正 |
| Invalid call | -0.15 | 每次未通过参数或语义校验的调用 |
| Invalid evidence | -0.25 | 每次引用本轮未获取证据；与 invalid-call 惩罚叠加 |
| Redundant call | -0.05 | 工具名与参数均相同的重复调用 |
| Call cost | -0.02 | 每次工具调用，包括非法调用 |
| Incorrect diagnosis | -0.50 | 提交了错误根因 |

默认配置定义于 [`src/agenttool_rl/reward.py`](../src/agenttool_rl/reward.py)，通过 `RewardConfig` 可以构造后续消融实验。

## 增量奖励

`score_trajectory()` 返回当前 trajectory 前缀的累计得分。环境在每一步返回：

```text
step_reward = current_total - previous_total
```

因此逐步 reward 之和严格等于 episode 最终得分，既可以用于训练，也可以在离线评测中完整复算。

## 防止奖励投机

- Agent observation 不包含 task ID、隐藏根因、必要证据或完整 world state；
- 未知服务错误不会返回隐藏的服务列表；
- 空结果是合法工具结果，但不会产生 evidence；
- evidence 只在工具真实命中记录后进入 episode evidence set；
- 重复调用和每步调用成本限制无意义的 reward farming；
- 测试集在训练前生成并记录 seed、生成器版本与 SHA-256。
