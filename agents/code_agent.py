from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any
from jinja2 import Environment, FileSystemLoader

from core.utils import list_input_files
from core.data_head import DataHead


from core.utils.code import extract_single_solution_from_raw
from core.prompt_loader import load_prompt_yaml
from llm_connect.utils import create_llm_client_from_profile
from llm_connect.config import get_llm_params
from core.orchestration.mode_spec import get_mode_spec


class CodeAgent:
    def __init__(self, model_name: str, data_head: Optional[DataHead] = None):
        self.model_name = model_name
        self.data_head = data_head or DataHead()
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "prompts" / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)

    def _collect_context(self, session_state: Dict[str, Any]) -> Dict:
        mode = str(session_state.get("run_mode", ""))
        spec = get_mode_spec(mode)
        ctx = {
            "task_dir": session_state["task_dir"],
            "query_md": session_state["query"],
            "run_mode": mode,
            "allow_profile": spec.allow_profile,
            "allow_clarify": spec.allow_clarify,
            "allow_flow": spec.allow_flow,
        }

        tdir = Path(session_state["task_dir"])
        
        inputs = list_input_files(tdir)
        ctx["inputs"] = [p.name for p in inputs]
        
        inputs_preview = self.data_head.get_preview(tdir / "inputs")
        ctx["inputs_preview"] = inputs_preview
        
        # Include qa_history if present (from interact mode clarify phase)
        if "qa_history" in session_state:
            ctx["qa_history"] = session_state["qa_history"]

        if "profile_summary" in session_state:
            ctx["profile_summary"] = session_state["profile_summary"]

        if "cleanspec_appendix" in session_state:
            ctx["cleanspec_appendix"] = session_state["cleanspec_appendix"]

        return ctx

    def _build_prompt(self, ctx: Dict, feedback: Optional[Dict[str, Any]] = None) -> str:
        """Builds the prompt using the Jinja2 template."""
        prompt_name = "code_agent"
        cfg = load_prompt_yaml(prompt_name)
        template = self.jinja_env.get_template("code_agent.jinja2")

        # Render the template with all the context and feedback
        return template.render(
            system_prompt_text=cfg.get("system", ""),
            guidelines_text=cfg.get("guidelines", ""),
            exec_error_instructions=cfg.get("exec_error_instructions", ""),
            context=ctx,
            feedback=feedback
        )
    


    def generate_code(self, session_state: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> tuple[str, str, List[Dict[str, str]]]:
        """Generates a single solution using the templated prompt."""
        ctx = self._collect_context(session_state)
        llm = create_llm_client_from_profile(self.model_name)
        if llm is None:
            raise RuntimeError("LLM configuration not found.")

        # The prompt is now a single string rendered from the template
        prompt_content = self._build_prompt(ctx, feedback)
        
        # For compatibility with LLM client, wrap it in a user message
        messages = [
            {"role": "user", "content": prompt_content}
        ]

        params = get_llm_params("CodeAgent", "generate_code") or {}
        raw_response = llm.generate(messages, **(params or {}))
        solve_code, is_complete = extract_single_solution_from_raw(raw_response)
        
        if not solve_code:
            # Fallback: if extraction failed completely
            return "", raw_response, messages
            
        if not is_complete:
            # Log warning but return what we have (Orchestrator might retry or fail execution)
            print("[CodeAgent] Warning: Generated code has syntax errors (incomplete?).")

        return solve_code, raw_response, messages
