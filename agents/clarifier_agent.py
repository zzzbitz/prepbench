"""Backward-compatible shim for the benchmark user simulator."""

from simulator.user_simulator import (
    ClarifierAgent,
    ClarifierAnswerItem,
    ClarifierResult,
    UserSimulator,
)

__all__ = [
    "UserSimulator",
    "ClarifierAgent",
    "ClarifierAnswerItem",
    "ClarifierResult",
]
