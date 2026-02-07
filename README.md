# PrepBench: How Far Are We from Natural-Language-Driven Data Preparation?

This repository contains the benchmark dataset and experimental runner for the paper
"PrepBench: How Far Are We from Natural-Language-Driven Data Preparation?".

## Installation

1) Clone and enter the repo:

```bash
git clone <repository_url>
cd prepbench
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

## Prerequisites

- Python 3.9+ (recommended)
- Network access to your configured LLM provider endpoint
- A valid API key in `.env`

## Configuration

- `config/experiment.yaml` holds experiment/runtime defaults (tracked).
- `config/llm.yaml` holds provider/model routing defaults (tracked).
- Optional local override (gitignored): `config/config.local.yaml`
- For the built-in OpenRouter provider, API keys must be set in `.env` (do not put keys in YAML).

Important notes:
- YAML values are **not** environment-variable expanded. Do not use `${OPENROUTER_API_KEY}` inside YAML.
- Set `experiment.run_mode` in `config/experiment.yaml` and model/provider in `config/llm.yaml` if you want defaults.

### Minimal OpenRouter example

`config/experiment.yaml`:

```yaml
experiment:
  run_mode: "orig"
```

`config/llm.yaml`:

```yaml
llm:
  active_provider: "openrouter"
  providers:
    openrouter:
      type: "openrouter"
      model: "openai/gpt-5.2"
      # OpenRouter API keys are read from .env only.
```

`.env` (local, not committed):

```bash
OPENROUTER_API_KEY="your-key"
```

## Fast Run Path (Keep It Simple)

If you only care about "how to run":

1) Set API key in `.env`.
2) Pick one mode script and run:

```bash
./scripts/run_orig.sh --case 1
# or:
./scripts/run_disamb.sh --case 1
./scripts/run_interact.sh --case 1
./scripts/run_disamb_only.sh --case 1
./scripts/run_flow.sh --case 1
./scripts/run_e2e.sh --case 1
```

3) Check run result:

```bash
cat @output/<model_info>/<run_mode>/case_001/solution/final_status.json
```

4) Evaluate a full run root:

```bash
python -m evaluate.batch --results-root @output/<model_info>/<run_mode>
```

The evaluation summary CSV will be written to:

```bash
@output/<model_info>/<run_mode>/evaluation_summary.csv
```

## Quickstart (1 minute)

1) Create `.env` at the repo root:

```bash
cat << 'EOF' > .env
OPENROUTER_API_KEY=your-key
EOF
```

2) Run a single case:

```bash
python run.py --case 1 --run_mode orig --model openai/gpt-5.2
```

Or use shortcut scripts:

```bash
./scripts/run_orig.sh --case 1 --model openai/gpt-5.2
./scripts/run_disamb.sh --case 1 --model openai/gpt-5.2
./scripts/run_interact.sh --case 1 --model openai/gpt-5.2
./scripts/run_disamb_only.sh --case 1 --model openai/gpt-5.2
./scripts/run_flow.sh --case 1 --model openai/gpt-5.2
./scripts/run_e2e.sh --case 1 --model openai/gpt-5.2
```

3) Check outputs:

- Outputs are written under `@output/<model_info>/<run_mode>/<case_name>/` by default.
- Success is indicated by `@output/<model_info>/<run_mode>/<case_name>/solution/final_status.json`.

## Run Modes (`run_mode`)

- `orig`: raw query + profile + code
- `disamb`: disambiguated/full query + profile + code
- `interact`: raw query + clarify + profile + code
- `disamb_only`: disambiguated/full query + code (no profile)
- `e2e`: interact pipeline + code-to-flow
- `flow`: flow-only execution (requires reference `case_XXX.py` files under `simulator/assets/solutions/`)

## BYOA (Third-Party Agent Frameworks)

For external frameworks, we recommend a single benchmark setting: **E2E**.

You can submit either:
- **code track**: write CSV outputs to `case_xxx/solution/cand/`
- **flow track**: write CSV outputs to `case_xxx/solution/flow_cand/`

Public per-case inputs for participants:
- `data/case_xxx/query.md`
- `data/case_xxx/inputs/*.csv`

Local benchmark-side components:
- User simulator local API: `simulator.LocalUserSimulatorAPI`
- Flow machine-readable contract: `py2flow/flow.schema.json`

Then evaluate with:

```bash
python -m evaluate.batch --results-root @output/<your_framework>/e2e --candidate-kind auto
# or force one track:
python -m evaluate.batch --results-root @output/<your_framework>/e2e --candidate-kind code
python -m evaluate.batch --results-root @output/<your_framework>/e2e --candidate-kind flow
```

Full integration contract:
- `docs/BYOA_E2E.md`

### Reference Solutions for `flow` and User Simulator

`flow` mode and user simulator alignment both rely on benchmark reference solution files:

- Path: `simulator/assets/solutions/case_XXX.py`
- Reason not bundled in the public repo: potential data leakage risk
- Request access by email: `j1n9zhe@gmail.com`

## Usage

### Run a single case

```bash
python run.py --case 52 --run_mode disamb --model openai/gpt-5.2
```

### Run a range of cases

```bash
python run.py --case 5-8 --run_mode orig --model openai/gpt-5.2
```

### Run all cases

```bash
python run.py --run_mode orig --model openai/gpt-5.2
```

### Use defaults from settings

If you set `experiment.run_mode` in `config/experiment.yaml` and `llm.providers.<provider>.model`
in `config/llm.yaml`, you can omit
`--run_mode` and `--model`:

```bash
python run.py --case 52
```

### Notes on e2e

`e2e` can run directly. It automatically reuses existing `interact` artifacts when available, and
`interact` can reuse artifacts prepared during a previous `e2e` run.

## Recommended Workflow

1) Dry-run with one case:

```bash
./scripts/run_orig.sh --case 1 --model openai/gpt-5.2
```

2) Run a full mode for all cases:

```bash
./scripts/run_orig.sh --model openai/gpt-5.2
```

3) Evaluate the completed run:

```bash
python -m evaluate.batch --results-root @output/<model_info>/orig
```

4) Inspect the summary CSV:

```bash
@output/<model_info>/orig/evaluation_summary.csv
```

## Output

Results are saved under `@output/<model_info>/<run_mode>/<case_name>/` by default.

```
@output/<model_info>/<run_mode>/<case_name>/
├── rounds/             # per-round logs
└── solution/
    └── final_status.json
```

## Troubleshooting

- `API key not provided`:
  - Ensure `.env` exists at repo root and contains a valid key (for example `OPENROUTER_API_KEY=...`).
- `run_mode is empty`:
  - Pass `--run_mode` explicitly or set `experiment.run_mode` in `config/experiment.yaml`.
- `Reference solution not found` (usually in `flow` mode):
  - Prepare `simulator/assets/solutions/case_XXX.py` first. For access, contact `j1n9zhe@gmail.com`.
- `No candidate directory with CSV outputs found` during batch evaluation:
  - The run did not produce result CSV files under `solution/cand` or `solution/flow_cand` for that case.
