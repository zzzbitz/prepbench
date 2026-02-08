# Contributing

Thanks for contributing to PrepBench.

## Development Setup

```bash
git clone <repository_url>
cd prepbench
pip install -r requirements.txt
cp .env.example .env
```

## Scope

- BYOA benchmark path (primary): `docs/BYOA_E2E.md`
- Paper reproduction path (secondary): `docs/REPRO.md`

Please keep these two tracks clearly separated in docs and code changes.

## Pull Request Checklist

Before opening a PR, verify:

```bash
python -m py_compile run.py examples/prep_agent/run_prepagent.py
./scripts/run_prepagent.sh --list 1
python -m evaluate.batch --help
```

## Coding and Documentation Rules

- Use English in code, comments, and markdown files.
- Prefer small, reviewable changes.
- Do not commit secrets or `.env`.
- Keep behavior changes explicit in PR descriptions.

## Reporting Issues

Include:
- command used
- full error output
- expected behavior
- minimal reproduction path
