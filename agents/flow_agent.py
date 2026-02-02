from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from core.prompt_loader import load_prompt_yaml
from llm_connect.utils import create_llm_client_from_profile
from llm_connect.config import get_llm_params, get_model_name


class FlowAgent:
    """Agent that generates flow.json DAG via LLM."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or get_model_name()
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "prompts" / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.jinja_env.get_template("flow_agent.jinja2")

    def _collect_context(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Collect context for prompt rendering."""
        solution_text = session_state.get("solution_text") or ""
        if not isinstance(solution_text, str) or not solution_text.strip():
            raise ValueError("FlowAgent requires non-empty session_state['solution_text'].")

        return {
            "case_name": str(session_state.get("case_name") or ""),
            "solution_text": solution_text,
        }

    def _build_prompt(
        self, ctx: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None
    ) -> str:
        """Builds the prompt using the Jinja2 template."""
        cfg = load_prompt_yaml("flow_agent")

        return self.template.render(
            system_prompt_text=cfg.get("system", ""),
            guidelines_text=cfg.get("guidelines", ""),
            operator_reference=cfg.get("operator_cookbook", ""),
            exec_error_instructions=cfg.get("exec_error_instructions", ""),
            context=ctx,
            feedback=feedback,
        )

    def _extract_json(self, raw_response: str) -> tuple[Optional[dict], Optional[dict]]:
        """Extract JSON from LLM response.
        
        Returns:
            (flow_dict, parse_error) - one of them is None
        """
        # Try fenced JSON block first
        match = re.search(r"```json\s*\n(.*?)\n```", raw_response, re.DOTALL)
        json_text = match.group(1) if match else raw_response.strip()

        try:
            flow_dict = json.loads(json_text)
            if not isinstance(flow_dict, dict):
                return None, {
                    "type": "json_parse",
                    "message": "Parsed JSON is not an object",
                    "details": {"parsed_type": type(flow_dict).__name__},
                }
            return flow_dict, None
        except json.JSONDecodeError as e:
            return None, {
                "type": "json_parse",
                "message": str(e),
                "details": {
                    "line": e.lineno,
                    "column": e.colno,
                    "pos": e.pos,
                },
            }

    def _validate_dag(self, flow_dict: dict) -> Optional[dict]:
        """Validate flow_dict using py2flow IR.
        
        Returns:
            parse_error dict if validation failed, None if OK
        """
        from py2flow.ir import DAG
        from py2flow.errors import FlowValidationError

        try:
            DAG.from_dict(flow_dict)
            return None
        except FlowValidationError as e:
            return {
                "type": "validation",
                "message": str(e),
                "details": {
                    "error_code": getattr(e, "error_code", None),
                    "node_id": getattr(e, "node_id", None),
                    "step_kind": str(getattr(e, "step_kind", None)),
                    "field": getattr(e, "field", None),
                },
            }
        except Exception as e:
            return {
                "type": "validation",
                "message": str(e),
                "details": {},
            }

    def generate_flow(
        self,
        session_state: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None,
    ) -> tuple[dict, str, List[Dict[str, str]], Optional[dict]]:
        """Generate flow.json via LLM.

        Args:
            session_state: Context including task_dir and solution_text.
            feedback: Optional feedback from previous failed attempt.

        Returns:
            (flow_dict, raw_response, messages, parse_error)
            - flow_dict: parsed and validated flow dict, or empty dict on failure
            - raw_response: raw LLM output
            - messages: list of message dicts sent to LLM
            - parse_error: None on success, or error dict on parse/validation failure
        """
        ctx = self._collect_context(session_state)

        llm = create_llm_client_from_profile(self.model_name)
        if llm is None:
            raise RuntimeError("LLM configuration not found.")

        prompt_content = self._build_prompt(ctx, feedback)
        messages = [{"role": "user", "content": prompt_content}]

        params = get_llm_params("FlowAgent", "generate_flow") or {}
        # Use lower temperature for more stable JSON generation
        if "temperature" not in params:
            params["temperature"] = 0.3

        raw_response = llm.generate(messages, **params)

        # Step 1: Extract JSON
        flow_dict, parse_error = self._extract_json(raw_response)
        if parse_error:
            return {}, raw_response, messages, parse_error

        # Step 2: Validate DAG
        validation_error = self._validate_dag(flow_dict)
        if validation_error:
            return {}, raw_response, messages, validation_error

        return flow_dict, raw_response, messages, None
