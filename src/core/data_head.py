from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class SchemaHead:
    columns: List[str]
    dtypes: Dict[str, str]
    head: List[Dict[str, Any]]


def read_schema_and_head(csv_path: Path, nrows: int = 5) -> SchemaHead:
    try:
        df = pd.read_csv(csv_path, nrows=nrows, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(csv_path, nrows=min(5, nrows))
        for c in df.columns:
            df[c] = df[c].astype(str)

    columns = df.columns.tolist()
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    head = df.head(nrows).to_dict(orient="records")
    return SchemaHead(columns=columns, dtypes=dtypes, head=head)

class DataHead:
    """
    Encapsulates logic for sampling input data.
    Currently implements the 'first 5 lines' strategy.

    Output format (per CSV file):
    {
      "columns": [col1, col2, ...],
      "rows": [
        [v11, v12, ...],
        [v21, v22, ...],
        ...  # Up to first 5 rows; values are strings; order aligns with columns.
      ]
    }
    """
    
    def get_preview(self, input_dir: str | Path) -> Dict[str, Any]:
        """
        Scan the input directory and return a lightweight profile for CSV files.
        Returns a dict keyed by filename with {columns, rows} or {error}.
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            return {}
            
        profile = {}
        for p in sorted(input_path.glob("*.csv")):
            try:
                sch = self._read_schema_and_head(p)
                # Convert head records into rows aligned with the columns order.
                rows: List[List[str]] = []
                for rec in sch.head:
                    row = [str(rec.get(col, "")) for col in sch.columns]
                    rows.append(row)
                profile[p.name] = {"columns": sch.columns, "rows": rows}
            except Exception as e:
                profile[p.name] = {"error": str(e)}
        return profile

    def _read_schema_and_head(self, csv_path: Path, nrows: int = 5) -> SchemaHead:
        return read_schema_and_head(csv_path, nrows=nrows)
