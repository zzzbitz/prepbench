from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from core.prompt_loader import load_prompt_yaml
from core.schemas.user_simulator import UserSimulatorResponseSchema
from core.structured_outlines import resolve_outlines_credentials, structured_json, structured_outputs_enabled
from llm_connect.config import get_llm_params
from llm_connect.utils import create_llm_client_from_profile

logger = logging.getLogger(__name__)
PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class UserSimulatorAnswerItem:
    sub_question: str
    classification: str  # "hit" | "fallback" | "refuse_need_data" | "refuse_too_broad" | "refuse_illegal" | "refuse_irrelevant"
    source: str  # "lib" | "fallback" | "refuse"
    answer: str
    canonical_value: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    ref: Optional[str] = None  # slot_id if hit, else None


@dataclass(frozen=True)
class UserSimulatorResult:
    answers: List[UserSimulatorAnswerItem]
    raw_response: str
    messages: List[Dict[str, str]]
    parse_error: Optional[str] = None
    raw_attempts: Optional[List[str]] = None
    
    @property
    def combined_answer(self) -> str:
        parts = []
        for i, item in enumerate(self.answers, 1):
            if len(self.answers) == 1:
                parts.append(item.answer)
            else:
                parts.append(f"{i}. {item.answer}")
        return "\n".join(parts)
    
    @property
    def classification(self) -> str:
        return self.answers[0].classification if self.answers else "fallback"
    
    @property
    def source(self) -> str:
        return self.answers[0].source if self.answers else "refuse"
    
    @property
    def answer(self) -> str:
        return self.combined_answer
    
    @property
    def ref(self) -> Optional[str]:
        return self.answers[0].ref if self.answers else None

    @property
    def canonical_value(self) -> Optional[str]:
        return self.answers[0].canonical_value if self.answers else None

    @property
    def details(self) -> Optional[Dict[str, Any]]:
        return self.answers[0].details if self.answers else None


_JSON_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)

# "ref" is optional in schema; keep it as optional in parser validation as well.
_ANSWER_REQUIRED_FIELDS = ("sub_question", "classification", "source", "answer")
_MAX_FEEDBACK_CHARS = 2000
_ALLOWED_CLASSIFICATIONS = {
    "hit",
    "fallback",
    "refuse_need_data",
    "refuse_too_broad",
    "refuse_illegal",
    "refuse_irrelevant",
}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object: prefer objects with 'answers' array, handle nested braces correctly."""
    if not text:
        return None
    
    # First try direct parsing (most reliable when LLM returns pure JSON)
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    
    # Try to extract from code fences first (most common LLM format)
    for fence_match in _JSON_CODE_FENCE_RE.finditer(text):
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    
    # Fallback: find JSON objects by bracket matching
    candidates: list[Dict[str, Any]] = []
    i = 0
    while i < len(text) and len(candidates) < 5:
        if text[i] == '{':
            # Find matching closing brace
            depth = 0
            start = i
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                c = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if c == '\\' and in_string:
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        # Found matching brace
                        json_str = text[start:j+1]
                        try:
                            obj = json.loads(json_str)
                            if isinstance(obj, dict):
                                candidates.append(obj)
                        except Exception:
                            pass
                        i = j
                        break
        i += 1
    
    if not candidates:
        return None
    
    # Prefer objects with 'answers' key that is a list
    with_answers = [c for c in candidates if isinstance(c.get("answers"), list)]
    if with_answers:
        # Pick the one with the most answers
        return max(with_answers, key=lambda c: len(c.get("answers") or []))
    
    # Fallback to first valid dict
    return candidates[0]


def _sanitize_answer(answer: str) -> str:
    if not answer:
        return ""
    # Hard ban on code fences in the user-visible answer.
    answer = answer.replace("```", "").strip()
    return answer


def _normalize_classification(classification: str) -> str:
    value = (classification or "").strip()
    if value == "hit_amb_kb":
        return "hit"
    if value in {"fallback_flow", "fallback_solution"}:
        return "fallback"
    if value == "illegal":
        return "refuse_illegal"
    if value in _ALLOWED_CLASSIFICATIONS:
        return value
    return "fallback"

def _normalize_ref_for_classification(
    classification: str,
    ref: Optional[str],
    canonical_value: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if classification != "hit":
        if canonical_value == ref:
            canonical_value = None
        ref = None
    return ref, canonical_value


def _parse_single_answer(data: Dict[str, Any], sub_question: str = "") -> UserSimulatorAnswerItem:
    details = data.get("details") if isinstance(data.get("details"), dict) else None

    classification = _normalize_classification(str(data.get("classification") or "fallback"))
    source = data.get("source")
    if not source:
        source = _source_for_structured_classification(classification)

    ref = data.get("ref")
    if ref is None and isinstance(details, dict):
        ref = details.get("slot_id")

    canonical_value = data.get("canonical_value")
    if canonical_value is None:
        canonical_value = ref
    ref, canonical_value = _normalize_ref_for_classification(classification, ref, canonical_value)

    return UserSimulatorAnswerItem(
        sub_question=sub_question or str(data.get("sub_question", "")),
        classification=classification,
        source=str(source),
        answer=_sanitize_answer(str(data.get("answer") or "")),
        canonical_value=canonical_value,
        details=details,
        ref=ref,
    )


def _validate_parsed_answers(
    parsed: Optional[Dict[str, Any]],
    expected_sub_questions: List[str],
) -> Optional[str]:
    if not isinstance(parsed, dict):
        return "json_parse_failed"
    answers = parsed.get("answers")
    if not isinstance(answers, list):
        return "answers_missing_or_not_list"
    if not answers:
        return "answers_empty"
    if expected_sub_questions and len(answers) != len(expected_sub_questions):
        return f"answer_count_mismatch: expected={len(expected_sub_questions)} got={len(answers)}"
    for i, ans in enumerate(answers):
        if not isinstance(ans, dict):
            return f"answer_not_object: index={i}"
        missing = [k for k in _ANSWER_REQUIRED_FIELDS if k not in ans]
        if missing:
            return f"answer_missing_fields: index={i} missing={missing}"
        if not isinstance(ans.get("sub_question"), str):
            return f"sub_question_not_string: index={i}"
        if not isinstance(ans.get("classification"), str):
            return f"classification_not_string: index={i}"
        if not isinstance(ans.get("source"), str):
            return f"source_not_string: index={i}"
        if not isinstance(ans.get("answer"), str):
            return f"answer_not_string: index={i}"
        ref_val = ans.get("ref")
        if ref_val is not None and not isinstance(ref_val, str):
            return f"ref_not_string_or_null: index={i}"
        if expected_sub_questions and ans.get("sub_question") != expected_sub_questions[i]:
            return (
                "sub_question_mismatch: "
                f"index={i} expected={expected_sub_questions[i]!r} got={ans.get('sub_question')!r}"
            )
    return None


def _format_runtime_feedback(error: str, raw: str) -> str:
    snippet = raw or ""
    truncated = False
    if len(snippet) > _MAX_FEEDBACK_CHARS:
        snippet = snippet[:_MAX_FEEDBACK_CHARS]
        truncated = True
    suffix = "\n[truncated]" if truncated else ""
    return (
        "ParseError: " + error + "\n"
        "PreviousOutput:\n" + snippet + suffix
    )


def _source_for_structured_classification(classification: str) -> str:
    if classification == "hit":
        return "lib"
    if classification == "fallback":
        return "fallback"
    if classification in {"refuse_need_data", "refuse_too_broad", "refuse_illegal", "refuse_irrelevant"}:
        return "refuse"
    return "fallback"


def _build_structured_result(
    result: UserSimulatorResponseSchema,
    *,
    raw_response: str,
    messages: List[Dict[str, str]],
) -> UserSimulatorResult:
    items: List[UserSimulatorAnswerItem] = []
    for ans in result.answers:
        answer = _sanitize_answer(ans.answer or "")
        if not answer:
            answer = "I cannot answer that question."
            classification = "refuse_illegal"
            source = "refuse"
        else:
            classification = _normalize_classification(ans.classification)
            source = ans.source or _source_for_structured_classification(classification)
        ref = ans.ref or ans.canonical_value
        canonical_value = ans.canonical_value or ans.ref
        ref, canonical_value = _normalize_ref_for_classification(classification, ref, canonical_value)
        item = UserSimulatorAnswerItem(
            sub_question=ans.sub_question,
            classification=classification,
            source=source,
            answer=answer,
            canonical_value=canonical_value,
            details=ans.details,
            ref=ref,
        )
        items.append(item)
    return UserSimulatorResult(answers=items, raw_response=raw_response, messages=messages)


class UserSimulator:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name
        template_dir = PROMPT_DIR / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)
        self.template = self.jinja_env.get_template("clarify_agent.jinja2")

    def _build_prompt(self, ctx: Dict[str, Any]) -> str:
        cfg = load_prompt_yaml(
            "clarify_agent",
            required_keys=("system", "guidelines"),
            prompt_dir=PROMPT_DIR,
        )
        return self.template.render(
            system_prompt_text=cfg["system"],
            guidelines_text=cfg["guidelines"],
            context=ctx,
        )

    def answer(
        self,
        *,
        query_full_text: str,
        amb_kb_json: Optional[Dict[str, Any]],
        solution_text: str,
        question: str,
        expected_sub_questions: Optional[List[str]] = None,
        runtime_feedback: Optional[str] = None,
    ) -> UserSimulatorResult:
        ctx = {
            "query_full_text": query_full_text,
            "question": question,
            "amb_kb_json": amb_kb_json or {},
            "solution_text": solution_text,
            "expected_sub_questions": expected_sub_questions or [],
            "runtime_feedback": runtime_feedback or "",
        }
        prompt_content = self._build_prompt(ctx)
        messages = [{"role": "user", "content": prompt_content}]

        params = get_llm_params("UserSimulator", "answer") or {}
        raw = None

        if structured_outputs_enabled():
            model_name = self.model_name or ""
            base_url = None
            try:
                from llm_connect.config import get_model_name

                model_name = get_model_name(self.model_name, agent="user_simulator")
                api_key, base_url = resolve_outlines_credentials(agent="user_simulator")
                parsed = structured_json(
                    model_name=model_name,
                    prompt=prompt_content,
                    schema=UserSimulatorResponseSchema,
                    base_url=base_url,
                    api_key=api_key,
                    **(params or {}),
                )
                raw = parsed.model_dump_json()
                return _build_structured_result(parsed, raw_response=raw, messages=messages)
            except Exception as exc:
                logger.warning(
                    "Outlines structured fallback: mode=structured model=%s exc=%s base_url_empty=%s",
                    model_name,
                    type(exc).__name__,
                    not bool(base_url),
                )
                raw = None

        llm = create_llm_client_from_profile(self.model_name, agent="user_simulator")
        if llm is None:
            raise RuntimeError("LLM configuration not found.")

        raw = llm.generate(messages, **(params or {}))

        def _parse_answers(raw_text: str) -> tuple[Optional[UserSimulatorResult], Optional[str]]:
            parsed = _extract_json_object(raw_text or "")
            error = _validate_parsed_answers(parsed, expected_sub_questions or [])
            if error:
                return None, error
            items = []
            for ans_data in parsed.get("answers", []):
                if isinstance(ans_data, dict):
                    item = _parse_single_answer(ans_data)
                    if not item.answer:
                        item = UserSimulatorAnswerItem(
                            sub_question=item.sub_question,
                            classification="refuse_illegal" if item.classification == "fallback" else item.classification,
                            source="refuse" if item.source == "fallback" else item.source,
                            answer="I cannot answer that question.",
                            canonical_value=item.canonical_value,
                            details=item.details,
                            ref=item.ref,
                        )
                    items.append(item)
            if not items:
                return None, "answers_empty_after_parse"
            return (
                UserSimulatorResult(
                    answers=items,
                    raw_response=raw_text or "",
                    messages=messages,
                ),
                None,
            )

        result, error = _parse_answers(raw or "")
        if error:
            retry_feedback = _format_runtime_feedback(error, raw or "")
            ctx["runtime_feedback"] = retry_feedback
            retry_prompt = self._build_prompt(ctx)
            retry_messages = [{"role": "user", "content": retry_prompt}]
            raw_retry = llm.generate(retry_messages, **(params or {}))
            result, error = _parse_answers(raw_retry or "")
            if result:
                return UserSimulatorResult(
                    answers=result.answers,
                    raw_response=result.raw_response,
                    messages=result.messages,
                    parse_error=None,
                    raw_attempts=[raw or "", raw_retry or ""],
                )

            # Final fallback: synthesize refusal answers to preserve alignment.
            fallback_questions = expected_sub_questions or [question]
            answers = [
                UserSimulatorAnswerItem(
                    sub_question=q,
                    classification="refuse_illegal",
                    source="refuse",
                    answer="I cannot answer this because the user simulator response was invalid.",
                    canonical_value=None,
                    details=None,
                    ref=None,
                )
                for q in fallback_questions
            ]
            return UserSimulatorResult(
                answers=answers,
                raw_response=raw_retry or "",
                messages=retry_messages,
                parse_error=error,
                raw_attempts=[raw or "", raw_retry or ""],
            )

        return UserSimulatorResult(
            answers=result.answers,
            raw_response=result.raw_response,
            messages=result.messages,
            parse_error=None,
            raw_attempts=[raw or ""],
        )
