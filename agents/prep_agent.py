from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from core.data_head import DataHead
from core.prompt_loader import load_prompt_yaml
from core.text_utils import extract_single_code_block
from llm_connect.config import get_llm_params
from llm_connect.utils import create_llm_client_from_profile


@dataclass(frozen=True)
class InteractAction:
    action_type: str  # "ask" | "done" | "code" | "invalid"
    content: str
    parse_error: Optional[str] = None


class PrepAgent:
    def __init__(self, model_name: str, data_head: Optional[DataHead] = None) -> None:
        self.model_name = model_name
        self.data_head = data_head or DataHead()
        template_dir = Path(__file__).parent / "prompts" / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)
        self.template = self.jinja_env.get_template("prep_agent.jinja2")

    def _collect_context(
        self,
        session_state: Dict[str, Any],
        *,
        qa_history: List[Dict[str, Any]],
        runtime_feedback: Optional[Dict[str, Any]],
        code_started: bool,
        max_questions: Optional[int],
        max_questions_per_ask: int,
        questions_used: int,
    ) -> Dict[str, Any]:
        # Prefer cached inputs_preview from session_state to avoid repeated IO.
        if "inputs_preview" in session_state:
            inputs_preview = session_state["inputs_preview"]
        else:
            tdir = Path(session_state["task_dir"])
            inputs_preview = self.data_head.get_preview(tdir / "inputs")
        
        ctx: Dict[str, Any] = {
            "task_dir": session_state["task_dir"],
            "query_md": session_state["query"],
            "inputs_preview": inputs_preview,
            "qa_history": qa_history,
            "runtime_feedback": runtime_feedback,
            "code_started": code_started,
            "max_questions": max_questions,
            "max_questions_per_ask": max_questions_per_ask,
            "questions_used": questions_used,
        }
        return ctx

    def _build_prompt(self, ctx: Dict[str, Any]) -> str:
        cfg = load_prompt_yaml("prep_agent")
        return self.template.render(
            system_prompt_text=cfg.get("system", ""),
            guidelines_text=cfg.get("guidelines", ""),
            context=ctx,
        )

    def _parse_action(self, raw: str) -> InteractAction:
        if not raw or not raw.strip():
            return InteractAction("invalid", "", parse_error="empty_response")

        text = raw.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        head = lines[0] if lines else ""
        # Be tolerant to models that add a short preamble before the action line.
        # If the first non-empty line isn't an action header, search for the first line that is.
        if head and not head.lower().startswith(("ask", "done", "code")):
            for i, ln in enumerate(lines):
                if ln.lower().startswith(("ask", "done", "code")):
                    lines = lines[i:]
                    head = lines[0]
                    break

        def _extract_after_prefix(prefixes: tuple[str, ...]) -> Optional[str]:
            for p in prefixes:
                if head.lower().startswith(p.lower()):
                    return head[len(p) :].strip()
            return None

        # Parse Ask: action
        ask_payload = _extract_after_prefix(("Ask:", "ASK:", "Ask ", "ASK ", "Ask(", "ASK("))
        if ask_payload is not None:
            # Note: We no longer strip trailing ")" to avoid removing valid content.
            # If question spans multiple lines, keep the rest.
            if len(lines) > 1:
                tail = "\n".join(lines[1:]).strip()
                if tail:
                    ask_payload = (ask_payload + "\n" + tail).strip()
            if not ask_payload:
                return InteractAction("invalid", "", parse_error="ask_missing_question")
            return InteractAction("ask", ask_payload)

        # Parse Done: action (clarify phase complete)
        done_payload = _extract_after_prefix(("Done:", "DONE:", "Done ", "DONE ", "Done.", "DONE."))
        if done_payload is not None or head.lower() == "done":
            return InteractAction("done", "")

        # Parse Code: action (kept for backward compatibility during transition)
        code_payload = _extract_after_prefix(("Code:", "CODE:", "Code ", "CODE ", "Code(", "CODE("))
        if code_payload is not None:
            if code_payload.endswith(")"):
                code_payload = code_payload[:-1].strip()
            body = text
            code_block = extract_single_code_block(body)
            if code_block:
                return InteractAction("code", code_block)
            # Fallback: treat everything after the first line as code.
            tail = "\n".join(text.splitlines()[1:]).strip()
            if tail:
                return InteractAction("code", tail)
            if code_payload:
                return InteractAction("code", code_payload)
            return InteractAction("invalid", "", parse_error="code_missing_body")

        # Fallback: if the model directly returned a python block, treat as code.
        code_block = extract_single_code_block(text)
        if code_block:
            return InteractAction("code", code_block)

        return InteractAction("invalid", "", parse_error="unrecognized_action")

    def generate_action(
        self,
        session_state: Dict[str, Any],
        *,
        qa_history: List[Dict[str, Any]],
        runtime_feedback: Optional[Dict[str, Any]],
        code_started: bool,
        max_questions: Optional[int],
        max_questions_per_ask: int,
        questions_used: int,
    ) -> tuple[InteractAction, str, List[Dict[str, str]]]:
        ctx = self._collect_context(
            session_state,
            qa_history=qa_history,
            runtime_feedback=runtime_feedback,
            code_started=code_started,
            max_questions=max_questions,
            max_questions_per_ask=max_questions_per_ask,
            questions_used=questions_used,
        )
        llm = create_llm_client_from_profile(self.model_name)
        if llm is None:
            raise RuntimeError("LLM configuration not found.")

        prompt_content = self._build_prompt(ctx)
        messages = [{"role": "user", "content": prompt_content}]

        params = get_llm_params("PrepAgent", "generate_action") or {}
        raw_response = llm.generate(messages, **(params or {}))

        action = self._parse_action(raw_response or "")
        return action, raw_response, messages
