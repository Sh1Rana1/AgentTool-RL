# OpenPipe ART Rollout Integration (v0.4)

V0.4 将 AgentTool-RL 的有状态环境接入 OpenPipe ART 0.5.x trajectory contract。该阶段只负责采样、分组和记录，不执行参数更新；GRPO、LoRA 和 checkpoint 属于 V0.5。

## Rollout contract

`rollout_art` 为每个 scenario 创建独立环境，并完成以下转换：

1. 将 public observation 写入 system / user message；隐藏状态不会进入 prompt 或 trajectory metadata。
2. 将五个工具契约转换为 OpenAI function tools，并请求 ART model 的 OpenAI-compatible inference client。
3. 将每次 assistant `Choice` 原样保存在 `messages_and_choices`，作为 ART 可训练输出。
4. 执行工具后，将结构化 observation 作为带 `tool_call_id` 的 tool message 写回历史。
5. episode 结束后写入程序化 reward、成功、诊断、证据覆盖、恢复、格式重试和调用成本相关指标。
6. 推理或工具异常返回带错误日志的 partial trajectory，避免丢失已经生成的训练样本前缀。

当兼容 endpoint 忽略 `tool_choice=required` 并返回自然语言时，rollout 最多进行两次格式纠正。每次 assistant 输出都保留在 trajectory 中；只有成功解析动作或重试耗尽后才消耗环境调用预算。

## Same-scenario groups

`gather_art_rollout_groups` 对每个 scenario 构建一个 `TrajectoryGroup`，默认并发采样 4 条 trajectory。不同 scenario 的 group 也交由 `art.gather_trajectory_groups` 并发收集。group metadata 保存 `scenario_id`，便于 GRPO 只在同场景样本之间计算相对优势。

异常阈值默认按 `rollouts_per_group × scenario_count` 计算，而不是写死一个与 batch 无关的常数。ART 返回的 group 及其 exception 记录不会被二次重建或丢弃。

## Local smoke test

安装 client 依赖：

```bash
pip install -e ".[art]"
```

启动任意 OpenAI-compatible endpoint 后运行：

```bash
python scripts/smoke_art_rollout.py \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen2.5:1.5b \
  --index 0 \
  --max-steps 6
```

输出的 JSON 使用 ART `Trajectory.for_logging()` 格式，明确区分 trainable assistant choices 和环境消息。默认文件为 `reports/v0.4_art_rollout_smoke.json`。

## Training handoff

V0.5 将直接消费 `gather_art_rollout_groups` 的返回值：

```python
groups = await gather_art_rollout_groups(
    model,
    scenarios,
    rollouts_per_group=4,
)
result = await backend.train(model, groups, learning_rate=1e-5)
await model.log(groups, metrics=result.metrics, step=result.step, split="train")
```

训练前仍需选择并配置可用的 ART backend；V0.4 不伪造 GPU、LoRA 或 checkpoint 结果。
