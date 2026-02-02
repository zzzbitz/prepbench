from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from py2flow.errors import FlowError
from py2flow.executor import DAGExecutor
from py2flow.executor import DebugConfig
from py2flow.ir import DAG


def exec_flow(
    data_path: str | Path,
    *,
    dump_nodes: set[str] | None = None,
    trace: bool = False,
    on_fail_dump: bool = False,
    validate_only: bool = False,
    explain: bool = False,
    debug_sample: int = 3,
) -> dict[str, object]:
    """
    Load flow.json under data_path, validate as a py2flow DAG, and execute with pandas.
    Paths inside the DAG are resolved relative to data_path (base_path).

    Note: flow.json only supports 11 kinds (input/project/filter/join/union/aggregate/dedup/sort/pivot/output/script)
    and CSV-only I/O.
    """
    data_path = Path(data_path)
    if not data_path.exists() or not data_path.is_dir():
        raise ValueError(f"Error: --data-path is invalid or does not exist: {data_path}")
    flow_file = data_path / "flow.json"
    if not flow_file.exists():
        raise FileNotFoundError(f"Error: {flow_file} does not exist")

    with flow_file.open("r", encoding="utf-8") as f:
        try:
            flow_dict = json.load(f)
        except json.JSONDecodeError as exc:
            hint = (
                "Your flow.json may embed a large Script.inline_code and contain invalid JSON escaping. "
                "Please keep inline_code short and properly escaped, or refactor logic into standard operators (project/filter/aggregate/...) to reduce script size."
            )
            raise json.JSONDecodeError(f"{exc.msg}. {hint}", exc.doc, exc.pos) from exc
    dag = DAG.from_dict(flow_dict)
    if validate_only:
        return {}
    if trace:
        logging.basicConfig(level=logging.INFO)
    if explain:
        dag.validate()
        order = DAGExecutor._topological_order(dag.nodes)
        for nid in order:
            node = dag.nodes[nid]
            print(f"node={nid} kind={node.kind.value} inputs={node.inputs} params={sorted((node.params or {}).keys())}")
        return {}
    executor = DAGExecutor(
        dag,
        base_path=data_path,
        debug=DebugConfig(dump_nodes=dump_nodes, trace=trace, on_fail_dump=on_fail_dump, sample_rows=debug_sample),
    )
    return executor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute py2flow flow.json under --data-path (11 kinds, CSV-only I/O)."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Directory path containing flow.json and related data files",
    )
    parser.add_argument(
        "--dump-nodes",
        default="",
        help="Comma-separated node ids to dump to data_path/flow_cand/@debug/<node>.csv",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Log per-node rows/columns and a small sample",
    )
    parser.add_argument(
        "--on-fail-dump",
        action="store_true",
        help="On failure, dump upstream inputs and params to data_path/flow_cand/@debug/@fail/<node>/",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only parse and validate flow.json; do not execute",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print node summary (kind/inputs/params keys) after validation; do not execute",
    )
    parser.add_argument(
        "--debug-sample",
        type=int,
        default=3,
        help="When --trace is set, number of rows to include in the sample (default: 3)",
    )

    args = parser.parse_args()
    dump_nodes = {x.strip() for x in str(args.dump_nodes).split(",") if x.strip()} or None
    try:
        exec_flow(
            Path(args.data_path),
            dump_nodes=dump_nodes,
            trace=bool(args.trace),
            on_fail_dump=bool(args.on_fail_dump),
            validate_only=bool(args.validate_only),
            explain=bool(args.explain),
            debug_sample=int(args.debug_sample),
        )
    except FlowError as exc:
        print(str(exc))
        help_text = getattr(exc, "help", None)
        if help_text:
            print(f"Help: {help_text}")
        raise SystemExit(1) from exc
