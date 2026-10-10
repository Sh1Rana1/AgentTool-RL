# Changelog

All notable changes to this project are documented here.

## [0.4.0] - 2026-10-10

### Added

- OpenPipe ART 0.5.x adapter for stateful multi-turn tool-use rollouts.
- Same-scenario `TrajectoryGroup` construction and concurrent rollout collection.
- Trainable OpenAI assistant choices with non-trainable environment observations.
- Programmatic trajectory reward, behavioral metrics, metadata, and partial-error logs.
- Bounded action-format correction without consuming environment call budget.
- Real ART 0.5.20 + Qwen2.5-1.5B/Ollama smoke trajectory.
- Unit tests for trajectory construction, correction history, and concurrent grouping.

### Changed

- Shared the Base LLM system prompt and JSON action parser with the ART adapter.
- Added ART as an optional dependency so the deterministic environment remains lightweight.

## [0.3.0] - 2026-09-22

### Added

- Shared stateful Agent interface and episode runner.
- Seeded Random and interpretable Rule-based baselines.
- OpenAI-compatible Base LLM tool-calling baseline with native and JSON fallback parsing.
- Bounded action-format correction for endpoints that ignore required tool choice.
- Unified task, schema, semantic, evidence, recovery, cost, and reward metrics.
- Deterministic JSON and Markdown evaluation reports with per-episode results.
- Frozen Random and Rule-based sanity-check results on the 20-scenario test split.
- A real Qwen2.5-1.5B/Ollama baseline report and single-episode trajectory exporter.
- Tests for baseline behavior, tool-call parsing, metric aggregation, and report output.

## [0.2.0] - 2026-09-22

### Added

- Stateful incident environment with step budgets and deterministic execution.
- Schema, value, temporal, service, and cross-turn evidence validation.
- Structured recoverable errors and complete trajectory recording.
- Programmatic dense reward with an auditable component breakdown.
- Seeded train, validation, and test generation with checksum manifests.
- A runnable recovery demo and dataset reproducibility verification.
- Environment, reward, generator, leakage, recovery, and budget tests.

### Changed

- Example scenarios now include executable hidden tool data.
- Public observations no longer expose task IDs or hidden service inventories.

## [0.1.0] - 2026-09-20

### Added

- Initial repository structure and project scope.
- Five versioned tool contracts for incident diagnosis.
- Five representative, deterministic example scenarios.
- Scenario validation command with no third-party runtime dependencies.
- Architecture, reward, evaluation, and version roadmap documentation.
- CI checks for Python 3.11 and 3.12.
