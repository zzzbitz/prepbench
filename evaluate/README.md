# Evaluator

Minimal evaluator for comparing a candidate output directory against a ground-truth (GT) directory using a GT-provided `config.json`.

## Scope
- Input: `gt_dir` (must contain `config.json`) and `cand_dir`.
- Output: a boolean pass/fail plus **only the first mismatch** detail.
- Fail-fast: evaluation stops at the first meaningful mismatch.
- No CLI, no batch scripts, no side effects (no printing).

## Public API

```python
from evaluate.core import evaluate

passed, first_error = evaluate(gt_dir, cand_dir, config_path=None)
```

Return value:
- `passed`: `True` if all files match.
- `first_error`: `None` on success, otherwise a dict describing the first mismatch.

`first_error` schema (example):
```json
{
  "error_type": "ROW_COUNT_MISMATCH",
  "message": "File output_01.csv: gold=100, cand=99",
  "file": "output_01.csv",
  "detail": {
    "...": "optional fields depending on error"
  }
}
```

## config.json
The GT directory must include `config.json` with this shape:

```json
{
  "files": {
    "output_01.csv": {
      "ignore_order": true,
      "key": ["id"],
      "columns": {
        "id": "number",
        "name": "text_norm",
        "created_at": "datetime"
      }
    }
  }
}
```

Notes:
- `key` is required and must be non-empty.
- `columns` defines types for **all** columns used in comparison.
- `ignore_order` affects **non-key** column comparison only.

## Matching rules (high-level)
- Files must exist on both sides and counts must match.
- Rows are aligned by key; key comparison is type-aware and tolerant for `number`.
- If a GT key has multiple candidate matches, this is an error (fail-fast).
- First mismatch returned could be one of:
  - `CONFIG_MISSING`, `FILE_COUNT_MISMATCH`, `MISSING_FILE`, `READ_ERROR`,
  - `MISSING_COLUMNS`, `ROW_COUNT_MISMATCH`, `NON_UNIQUE_KEY`, `KEY_MISMATCH`,
  - `VALUE_MISMATCH`, `ALIGNMENT_ERROR`, `EXCEPTION`.

For `KEY_MISMATCH`, `detail.reason` can be:
- `missing_key`
- `multiple_matches`
- `extra_keys`

## Dependencies
- `pandas`

## Examples

```python
from evaluate.core import evaluate

ok, err = evaluate("/path/to/gt", "/path/to/cand")
if ok:
    print("PASS")
else:
    print("FAIL", err)
```
