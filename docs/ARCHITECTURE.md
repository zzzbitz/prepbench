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

- `agents/`: model-facing generation modules (clarify/profile/code/flow)
- `core/`: orchestration, execution, and shared runtime logic
- `llm_connect/`: provider abstraction and model config resolution
- `simulator/`: local user-simulator runtime
- `evaluate/`: local evaluator and batch scoring
- `py2flow/`: flow IR/operator system and executor

## Method Layer

- `methods/prepagent/`: fully runnable reference method for external users
  - Own prompt assets under `methods/prepagent/prompts/`
  - Isolated output root: `@output/<model_info>/prepagent/`

## Scheme-3 Migration Scaffold

To support a future `src/prepbench/...` package layout without breaking current scripts, a compatibility namespace was added:

- `src/prepbench/`
- `src/prepbench/{agents,config,core,evaluate,llm_connect,py2flow,simulator,data_synthesis}`

These bridge packages map to existing top-level packages. This allows gradual migration of imports to `prepbench.*` while keeping current runtime behavior stable.
