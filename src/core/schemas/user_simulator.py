from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from core.schemas.base import OAISchemaBase


class UserSimulatorAnswerItemSchema(OAISchemaBase):
    sub_question: str
    classification: Literal[
        "hit",
        "fallback",
        "refuse_need_data",
        "refuse_too_broad",
        "refuse_illegal",
        "refuse_irrelevant",
    ]
    source: Literal["lib", "fallback", "refuse"]
    answer: str
    ref: Optional[str] = None
    canonical_value: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class UserSimulatorResponseSchema(OAISchemaBase):
    answers: list[UserSimulatorAnswerItemSchema]
