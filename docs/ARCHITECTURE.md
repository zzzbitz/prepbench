# Architecture Overview

PrepBench currently has two usage tracks:

1. **Benchmark consumer track (primary)**
- Evaluate your own framework with E2E protocol and local evaluator.
- Key docs: `docs/BYOA_E2E.md`
- Reference method: `methods/prepagent/`

2. **Paper reproduction track (secondary)**
- Internal multi-mode experiments (`orig`, `disamb`, `interact`, `disamb_only`, `flow`, `e2e`).
- Entry: `run.py`, `docs/REPRO.md`

## Runtime Components

- `src/agents/`: model-facing generation modules (clarify/profile/code/flow)
- `src/core/`: orchestration, execution, and shared runtime logic
- `src/llm_connect/`: provider abstraction and model config resolution
- `src/simulator/`: local user-simulator runtime
- `src/evaluate/`: local evaluator and batch scoring
- `src/py2flow/`: flow IR/operator system and executor

## Method Layer

- `methods/prepagent/`: fully runnable reference method for external users
  - Own prompt assets under `methods/prepagent/prompts/`
  - Isolated output root: `@output/<model_info>/prepagent/`
