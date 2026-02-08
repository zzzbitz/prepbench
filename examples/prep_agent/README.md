# PrepAgent Reference (E2E)

This is a reference implementation for participants who want to build and evaluate their own agent framework on PrepBench.

Pipeline order is fixed:
1. profile
2. clarify (via local user simulator API)
3. code generation + execution
4. flow generation + execution (only when code outputs exist)

## Run

```bash
./scripts/run_prepagent.sh --case 1 --model openai/gpt-5.2
```

Outputs follow the standard evaluation layout under `@output/<model_info>/e2e/case_xxx/solution/`:
- `cand/*.csv` (code track)
- `flow_cand/*.csv` (flow track)

## Evaluate

```bash
python -m evaluate.batch --results-root @output/<model_info>/e2e --candidate-kind auto
```

## Notes

- This reference implementation reuses prompts and agent modules under `agents/`.
- User simulator and flow mode require reference solutions in `simulator/assets/solutions/case_xxx.py`.
