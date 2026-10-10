# Unified Evaluation Protocol (v0.3)

V0.3 使用同一套 episode runner 和指标实现比较不同 Agent，避免每个 baseline 自定义成功标准。环境只向 Agent 暴露 public observation；隐藏根因仅在 episode 结束后参与程序化计分。

## Baselines

### Random

Random Agent 使用固定 seed 随机选择调查工具，并生成满足 Schema 的参数。它只能使用告警中的主服务和工具结果中已经发现的依赖，不能访问隐藏状态。最后一次调用提交随机根因、已收集证据和通用处置。

### Rule-based

Rule-based Agent 根据五类已知场景模板中的告警特征选择调查路径，并从真实工具结果提取服务、版本和 evidence ID。它用于验证环境是否可解、Reward 是否与任务目标一致，不代表模型泛化能力。

### Base LLM

Base LLM Agent 支持 OpenAI-compatible `/chat/completions`，包括 vLLM、SGLang 和兼容服务：

```bash
python scripts/evaluate_baselines.py \
  --agents base-llm \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output reports/qwen2.5-3b-base.json
```

默认使用原生 function calling；若兼容服务将动作写入文本，则接受以下严格 JSON 格式：

```json
{"tool": "query_logs", "arguments": {"service": "..."}}
```

API key 可通过 `OPENAI_API_KEY` 提供，也可以使用 `--api-key-env` 指定其他环境变量名。报告不会记录 key。

当模型返回自然语言而不是动作时，Base Agent 最多进行两次格式纠正重试；重试不会消耗环境工具预算，完整尝试记录在单 episode trajectory 的 `model_response.attempts` 中。环境仍会拒绝重试后无法解析的输出。

## Metrics

| Metric | Definition |
| --- | --- |
| Task Success Rate | 根因、必要证据和处置全部正确的 episode 比例 |
| Diagnosis Accuracy | 合法终止提交中的根因是否正确，按全部 episode 计 |
| Schema Valid Rate | 通过工具名、字段、类型和枚举校验的调用比例 |
| Semantic Argument Accuracy | 可执行调用数除以已通过 Schema 校验的调用数 |
| Evidence Validity | 提交时已在本 episode 真实获取的 evidence 引用比例 |
| Error Recovery Rate | 可恢复错误后立即用同一工具完成修正的比例 |
| Invalid Call Rate | 环境拒绝执行的调用比例 |
| Average Tool Calls | 每个 episode 的平均调用数 |
| Average Reward | 使用 v0.2 冻结 Reward 配置计算的平均总分 |

所有比例采用 micro aggregation；任务成功、诊断正确和截断率按 episode aggregation。

## Frozen v0.3 sanity-check

配置：`data/generated/test.jsonl`、20 个测试场景、每个 episode 最多 6 次调用、Random seed 42。

| Agent | Success | Diagnosis | Schema valid | Semantic valid | Coverage | Evidence valid | Recovery | Invalid | Avg calls | Avg reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.0% | 5.0% | 100.0% | 97.5% | 40.0% | 100.0% | 0.0% | 2.5% | 6.00 | -0.215 |
| Rule-based | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | — | 0.0% | 3.00 | 2.040 |
| Qwen2.5-1.5B Base | 0.0% | 5.0% | 92.9% | 70.2% | 17.5% | 28.6% | 5.1% | 34.8% | 5.60 | -0.390 |

Qwen2.5-1.5B Base 通过 Ollama 0.40.2 本地运行，temperature 为 0，每个动作最多两次格式纠正。它没有出现后端错误，但 90% episode 因耗尽预算而截断，主要差距集中在语义参数、调查策略、证据引用和错误恢复。机器可读结果分别位于 [`reports/v0.3_baselines.json`](../reports/v0.3_baselines.json) 与 [`reports/v0.3_qwen2.5_1.5b.json`](../reports/v0.3_qwen2.5_1.5b.json)；一个包含模型原始尝试和环境反馈的失败样例见 [`reports/qwen2.5_1.5b_trace_000_retry.json`](../reports/qwen2.5_1.5b_trace_000_retry.json)。

## Reproduction

```bash
python scripts/verify_dataset.py
python scripts/evaluate_baselines.py \
  --agents random,rule \
  --dataset data/generated/test.jsonl \
  --max-steps 6 \
  --seed 42 \
  --output reports/v0.3_baselines.json

# Optional local-model baseline (after `ollama pull qwen2.5:1.5b`)
python scripts/evaluate_baselines.py \
  --agents base-llm \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen2.5:1.5b \
  --output reports/v0.3_qwen2.5_1.5b.json
```

确定性报告不写入运行时间，因此相同代码、数据、seed 和参数会生成字节稳定的内容。真实模型报告还依赖模型 digest、推理后端及其版本，应将运行配置与结果一起保存。
