from __future__ import annotations

from typing import Any, Dict, Tuple
import re
from pathlib import Path

from core.prompt_loader import load_prompt_yaml
from llm_connect.utils import create_llm_client_from_profile
from llm_connect.config import get_llm_params
from core.utils.code import extract_code_from_response
from jinja2 import Environment, FileSystemLoader


class ProfileAgent:
    """ProfileAgent for two-round data profiling with LLM decision.
    
    Flow:
    1. generate_profile_code() - Generate initial profiling script
    2. decide_or_summarize() - After execution, decide to SUMMARY or CODE
    3. summarize() - If round 2 executed, generate final summary
    """

    def __init__(
        self,
        model_name: str,
        *,
        prompt_dir: Path | None = None,
        template_dir: Path | None = None,
        profile_prompt_name: str = "profile_agent",
        profile_template_name: str = "profile_agent.jinja2",
        decide_prompt_name: str = "profile_agent_decide",
        decide_template_name: str = "profile_agent_decide.jinja2",
        summarize_prompt_name: str = "profile_agent_summarize",
        summarize_template_name: str = "profile_agent_summarize.jinja2",
    ) -> None:
        self.model_name = model_name
        self._llm = None
        self.prompt_dir = prompt_dir
        tmpl_dir = template_dir or (Path(__file__).parent / "prompts" / "templates")
        self.profile_prompt_name = profile_prompt_name
        self.profile_template_name = profile_template_name
        self.decide_prompt_name = decide_prompt_name
        self.decide_template_name = decide_template_name
        self.summarize_prompt_name = summarize_prompt_name
        self.summarize_template_name = summarize_template_name
        self.jinja_env = Environment(
            loader=FileSystemLoader(tmpl_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _get_llm(self):
        if self._llm is None:
            self._llm = create_llm_client_from_profile(self.model_name)
            if self._llm is None:
                raise RuntimeError("LLM configuration not found.")
        return self._llm

    def _render_prompt(self, prompt_name: str, template_name: str, context: Dict[str, Any]) -> str:
        prompt_cfg = load_prompt_yaml(
            prompt_name,
            required_keys=("system", "guidelines"),
            prompt_dir=self.prompt_dir,
        )
        template = self.jinja_env.get_template(template_name)
        return template.render(
            system_prompt_text=prompt_cfg["system"],
            guidelines_text=prompt_cfg["guidelines"],
            context=context,
        )

    def generate_profile_code(
        self,
        session_state: Dict[str, Any],
    ) -> Tuple[str, str, list[dict[str, str]]]:
        """Generate data profiling code.
        
        Args:
            session_state: Contains task_dir, query, inputs, inputs_preview
            
        Returns:
            (code, raw_response, messages)
        """
        prompt_content = self._render_prompt(
            self.profile_prompt_name,
            self.profile_template_name,
            {
                "query_md": session_state.get("query", ""),
                "task_dir": session_state.get("task_dir", ""),
                "inputs": session_state.get("inputs", []),
                "inputs_preview": session_state.get("inputs_preview", {}),
            },
        )

        llm = self._get_llm()
        messages = [{"role": "user", "content": prompt_content}]
        params = get_llm_params("ProfileAgent", "generate_profile_code") or {}
        raw_response = llm.generate(messages, **(params or {}))
        code = extract_code_from_response(raw_response)
        return code, raw_response, messages

    def decide_or_summarize(
        self,
        session_state: Dict[str, Any],
        execution_result: Dict[str, Any],
        max_summary_chars: int = 4000,
    ) -> Tuple[str, str, str, list[dict[str, str]]]:
        """Decide after round 1: SUMMARY or CODE.
        
        Args:
            session_state: Task context
            execution_result: Round 1 execution result {ok, stdout, stderr}
            max_summary_chars: Maximum characters for summary
            
        Returns:
            (decision, content, raw_response, messages)
            - decision: "SUMMARY" or "CODE"
            - content: Summary text or new code
        """
        status = "Success" if execution_result.get("ok") else "Failed"
        stdout = (execution_result.get("stdout") or "")[:3000]  # Truncate
        stderr = execution_result.get("stderr") or ""
        
        stderr_section = ""
        if stderr:
            stderr_section = f"- Stderr:\n```\n{stderr[:1000]}\n```"
        
        prompt_content = self._render_prompt(
            self.decide_prompt_name,
            self.decide_template_name,
            {
                "query_md": session_state.get("query", ""),
                "task_dir": session_state.get("task_dir", ""),
                "inputs": session_state.get("inputs", []),
                "execution_status": status,
                "stdout": stdout,
                "stderr_section": stderr_section,
                "max_summary_chars": max_summary_chars,
            },
        )

        llm = self._get_llm()
        messages = [{"role": "user", "content": prompt_content}]
        params = get_llm_params("ProfileAgent", "decide_or_summarize") or {}
        raw_response = llm.generate(messages, **(params or {}))
        
        # Parse decision
        decision, content = self._parse_decision(raw_response)
        
        return decision, content, raw_response, messages

    def _parse_decision(self, raw_response: str) -> Tuple[str, str]:
        """Parse LLM response to extract decision and content.
        
        Strategy:
        1. Check the first non-empty line for explicit SUMMARY: or CODE: marker
        2. If CODE: found, extract code block after the marker
        3. Fallback: if no explicit marker, check for code blocks → CODE
        4. Default: treat as SUMMARY, stripping any code fences
        """
        text = raw_response.strip()
        lines = text.split("\n")
        
        # Find first non-empty line
        first_line = ""
        first_line_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                first_line = stripped
                first_line_idx = i
                break
        
        # Check for explicit SUMMARY: at start of first non-empty line
        if re.match(r"^SUMMARY:\s*", first_line, re.IGNORECASE):
            # Extract everything after SUMMARY: marker
            remaining_first_line = re.sub(r"^SUMMARY:\s*", "", first_line, flags=re.IGNORECASE)
            content_lines = [remaining_first_line] + lines[first_line_idx + 1:]
            content = "\n".join(content_lines).strip()
            # Clean up: remove any code fences from summary
            content = self._strip_code_fences(content)
            return "SUMMARY", content
        
        # Check for explicit CODE: at start of first non-empty line
        if re.match(r"^CODE:\s*", first_line, re.IGNORECASE):
            # Extract code block after CODE: marker
            remaining = "\n".join(lines[first_line_idx:])
            remaining = re.sub(r"^CODE:\s*", "", remaining, flags=re.IGNORECASE)
            code = extract_code_from_response(remaining)
            if code:
                return "CODE", code
            # If no code block found after CODE:, treat as invalid → fallback
        
        # Fallback: check if response contains a code block anywhere
        code = extract_code_from_response(text)
        if code:
            return "CODE", code
        
        # Default: treat entire response as summary, strip code fences
        return "SUMMARY", self._strip_code_fences(text)

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences from text."""
        # Remove fence markers but keep the inner content.
        text = re.sub(r"^```[\w-]*\s*$", "", text, flags=re.MULTILINE)
        text = text.replace("```", "")
        return text.strip("`").strip()

    def summarize(
        self,
        session_state: Dict[str, Any],
        round1_result: Dict[str, Any],
        round2_result: Dict[str, Any],
        max_summary_chars: int = 4000,
    ) -> Tuple[str, str, list[dict[str, str]]]:
        """Generate final summary after round 2.
        
        Args:
            session_state: Task context
            round1_result: Round 1 execution result
            round2_result: Round 2 execution result
            max_summary_chars: Maximum characters for summary
            
        Returns:
            (summary, raw_response, messages)
        """
        r1_status = "Success" if round1_result.get("ok") else "Failed"
        r1_stdout = (round1_result.get("stdout") or "")[:2000]
        
        r2_status = "Success" if round2_result.get("ok") else "Failed"
        r2_stdout = (round2_result.get("stdout") or "")[:2000]
        
        prompt_content = self._render_prompt(
            self.summarize_prompt_name,
            self.summarize_template_name,
            {
                "query_md": session_state.get("query", ""),
                "round1_status": r1_status,
                "round1_stdout": r1_stdout,
                "round2_status": r2_status,
                "round2_stdout": r2_stdout,
                "max_summary_chars": max_summary_chars,
            },
        )

        llm = self._get_llm()
        messages = [{"role": "user", "content": prompt_content}]
        params = get_llm_params("ProfileAgent", "summarize") or {}
        raw_response = llm.generate(messages, **(params or {}))
        
        # Clean up the summary
        summary = self._strip_code_fences(raw_response.strip())
        
        return summary, raw_response, messages
