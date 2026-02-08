# Simulator Package

This package contains benchmark-provided simulator components. It is not part of the model-under-test agent set.

## Layout

- `user_simulator.py`: user simulator implementation.
- `local_api.py`: local session-based interface (`LocalUserSimulatorAPI`) for BYOA integration.
- `prompts/`: simulator prompt configs and templates.
- `assets/solutions/`: reference solutions for each case (`case_XXX.py`).

## Notes

- Public repository keeps `assets/solutions/` as a placeholder only.
- Reference `case_XXX.py` files are distributed on request due to data leakage concerns.
- Contact: `j1n9zhe@gmail.com`.
