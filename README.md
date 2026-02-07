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

## Configuration

- `config/experiment.yaml` holds experiment/runtime defaults (tracked).
- `config/llm.yaml` holds provider/model routing defaults (tracked).
- Optional local overrides (gitignored):
  - `config/experiment.local.yaml`
  - `config/llm.local.yaml`
  - `config/settings.local.yaml` (global override)
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
- `flow`: flow-only execution (requires solution.py)

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

## Output

Results are saved under `@output/<model_info>/<run_mode>/<case_name>/` by default.

```
@output/<model_info>/<run_mode>/<case_name>/
├── rounds/             # per-round logs
└── solution/
    └── final_status.json
```
