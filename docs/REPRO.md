# Reproducing Paper Experiments

This document is for the internal reproduction track (`run.py` + multi-mode runs).
If you are evaluating your own framework, use `docs/BYOA_E2E.md` instead.

## Entry Point

Main runner:

```bash
python run.py --case 1 --run_mode orig --model openai/gpt-5.2
```

Shortcut scripts:

```bash
./scripts/run_orig.sh --case 1 --model openai/gpt-5.2
./scripts/run_disamb.sh --case 1 --model openai/gpt-5.2
./scripts/run_interact.sh --case 1 --model openai/gpt-5.2
./scripts/run_disamb_only.sh --case 1 --model openai/gpt-5.2
./scripts/run_flow.sh --case 1 --model openai/gpt-5.2
./scripts/run_e2e.sh --case 1 --model openai/gpt-5.2
```

## Run Modes

- `orig`: raw query + profile + code
- `disamb`: disambiguated/full query + profile + code
- `interact`: raw query + clarify + profile + code
- `disamb_only`: disambiguated/full query + code (no profile)
- `e2e`: interact pipeline + code-to-flow
- `flow`: flow-only execution (requires `simulator/assets/solutions/case_XXX.py`)

## Common Selectors

Run one case:

```bash
python run.py --case 52 --run_mode disamb --model openai/gpt-5.2
```

Run a range:

```bash
python run.py --case 5-8 --run_mode orig --model openai/gpt-5.2
```

Run all cases:

```bash
python run.py --run_mode orig --model openai/gpt-5.2
```

Use config defaults (omit `--run_mode` and `--model`):

```bash
python run.py --case 52
```

## Notes

- `e2e` can run directly and reuses compatible interact artifacts when available.
- Flow mode and user simulator alignment require `simulator/assets/solutions/case_XXX.py`.
- Reference solutions are distributed on request due to leakage risk:
  - Contact: `j1n9zhe@gmail.com`

## Evaluate Reproduction Runs

Evaluate one run root:

```bash
python -m evaluate.batch --results-root @output/<model_info>/<run_mode>
```

Generated:
- `@output/<model_info>/<run_mode>/evaluation_summary.csv`
- `@output/<model_info>/<run_mode>/acc.txt`
