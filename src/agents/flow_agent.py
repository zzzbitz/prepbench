from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from core.prompt_loader import load_prompt_yaml
from llm_connect.utils import create_llm_client_from_profile
from llm_connect.config import get_llm_params, get_model_name

_JSON_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)


class FlowAgent:
    """Agent that generates flow.json DAG via LLM."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        prompt_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        prompt_name: str = "flow_agent",
        template_name: str = "flow_agent.jinja2",
    ):
        self.model_name = model_name or get_model_name()
        self.prompt_dir = prompt_dir
        self.prompt_name = prompt_name
        # Setup Jinja2 environment
        tmpl_dir = template_dir or (Path(__file__).parent / "prompts" / "templates")
        self.jinja_env = Environment(
            loader=FileSystemLoader(tmpl_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.jinja_env.get_template(template_name)

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
        cfg = load_prompt_yaml(
            self.prompt_name,
            required_keys=("system", "core_guidelines", "operator_definitions"),
            prompt_dir=self.prompt_dir,
        )

        return self.template.render(
            system_prompt_text=cfg["system"],
            core_guidelines_text=cfg["core_guidelines"],
            operator_definitions_text=cfg["operator_definitions"],
            operator_reference=cfg.get("operator_cookbook", ""),
            exec_error_instructions=cfg.get("exec_error_instructions", ""),
            context=ctx,
            feedback=feedback,
        )

    def _extract_json(self, raw_response: str) -> tuple[Optional[dict], Optional[dict]]:
        """Extract JSON dict robustly from LLM response.

        Strategy:
        1) Parse whole response as JSON.
        2) Parse fenced blocks (```json ... ``` / ``` ... ```).
        3) Parse balanced-brace JSON object candidates.
        """
        text = (raw_response or "").strip()
        if not text:
            return None, {
                "type": "json_parse",
                "message": "Empty response",
                "details": {},
            }

        def _parse_dict(candidate: str) -> Optional[dict]:
            try:
                parsed = json.loads(candidate)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None

        def _score(obj: dict) -> int:
            # Prefer typical DAG-like payloads.
            score = 0
            if isinstance(obj.get("nodes"), list):
                score += 3
            if isinstance(obj.get("edges"), list):
                score += 2
            if isinstance(obj.get("steps"), list):
                score += 1
            return score

        # 1) Whole-text parse.
        whole = _parse_dict(text)
        if whole is not None:
            return whole, None

        # 2) Fenced blocks.
        fenced_candidates: list[dict] = []
        for match in _JSON_CODE_FENCE_RE.finditer(text):
            parsed = _parse_dict(match.group(1).strip())
            if parsed is not None:
                fenced_candidates.append(parsed)
        if fenced_candidates:
            fenced_candidates.sort(key=_score, reverse=True)
            return fenced_candidates[0], None

        # 3) Balanced-brace object extraction.
        brace_candidates: list[dict] = []
        i = 0
        while i < len(text):
            if text[i] != "{":
                i += 1
                continue
            start = i
            depth = 0
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if in_string and ch == "\\":
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        parsed = _parse_dict(text[start : j + 1])
                        if parsed is not None:
                            brace_candidates.append(parsed)
                        i = j
                        break
            i += 1

        if brace_candidates:
            brace_candidates.sort(key=_score, reverse=True)
            return brace_candidates[0], None

        return None, {
            "type": "json_parse",
            "message": "Failed to extract valid JSON object from model response",
            "details": {},
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
