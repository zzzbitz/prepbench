# Simulator Package

This package contains benchmark-provided simulator components. It is not part of the model-under-test agent set.

## Layout

- `user_simulator.py`: user simulator implementation (formerly `ClarifierAgent`).
- `prompts/`: simulator prompt configs and templates.
- `assets/solutions/`: reference solutions for each case (`case_XXX.py`).

## Compatibility

- `agents/clarifier_agent.py` is kept as a shim for backward compatibility.
- `core.case_assets.resolve_reference_solution_path` supports both:
  - `simulator/assets/solutions/case_XXX.py` (preferred)
  - `data/case_XXX/solution.py` (legacy fallback)
