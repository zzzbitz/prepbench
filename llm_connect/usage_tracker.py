"""
Token usage tracker for LLM API calls.

Provides a context-aware tracker using contextvars for thread/process safety.
Each case run gets its own tracker that accumulates usage across phases.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class PhaseUsage:
    """Token usage statistics for a single phase."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    cost_usd: float = 0.0

    def add(self, prompt: int, completion: int, cost: float) -> None:
        """Record a single LLM call's usage."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.call_count += 1
        self.cost_usd += cost

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class UsageTracker:
    """
    Tracks token usage across multiple phases for a single case run.
    
    Usage:
        tracker = UsageTracker(model="openai/gpt-4o", prompt_price=0.000005, completion_price=0.000015)
        set_tracker(tracker)
        
        tracker.set_phase("clarify")
        # ... LLM calls happen, openrouter_client calls tracker.record() ...
        
        tracker.set_phase("code")
        # ... more LLM calls ...
        
        tracker.save(output_path / "token_usage.json")
        set_tracker(None)
    """
    
    model: str = ""
    phases: Dict[str, PhaseUsage] = field(default_factory=dict)
    current_phase: str = "default"
    prompt_price: float = 0.0  # $/token (already converted from $/1M)
    completion_price: float = 0.0
    unknown_usage_calls: int = 0  # Calls where usage data was missing

    def set_phase(self, phase: str) -> None:
        """Switch to a new phase. Creates the phase if it doesn't exist."""
        self.current_phase = phase
        if phase not in self.phases:
            self.phases[phase] = PhaseUsage()

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Record token usage from a single LLM call.
        
        Always increments call_count, even if tokens are 0.
        """
        if self.current_phase not in self.phases:
            self.phases[self.current_phase] = PhaseUsage()
        
        cost = prompt_tokens * self.prompt_price + completion_tokens * self.completion_price
        self.phases[self.current_phase].add(prompt_tokens, completion_tokens, cost)

    def record_unknown(self) -> None:
        """Record a call where usage data was not available."""
        self.unknown_usage_calls += 1
        # Still increment call_count with 0 tokens
        self.record(0, 0)

    def to_dict(self) -> dict:
        """
        Convert tracker to JSON-serializable dict.
        
        Always includes 'total' for schema consistency, even for single-phase runs.
        """
        # Calculate total across all phases
        total = PhaseUsage()
        for p in self.phases.values():
            total.prompt_tokens += p.prompt_tokens
            total.completion_tokens += p.completion_tokens
            total.total_tokens += p.total_tokens
            total.call_count += p.call_count
            total.cost_usd += p.cost_usd

        result = {"model": self.model}
        
        # Always include per-phase breakdown
        for name, usage in self.phases.items():
            result[name] = usage.to_dict()
        
        # Always include total for consistent schema
        result["total"] = total.to_dict()

        if self.unknown_usage_calls > 0:
            result["unknown_usage_calls"] = self.unknown_usage_calls

        return result

    def save(self, path: Path) -> None:
        """Write usage data to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


# Context variable for current tracker (None when not tracking)
_current_tracker: ContextVar[Optional[UsageTracker]] = ContextVar("usage_tracker", default=None)


def get_tracker() -> Optional[UsageTracker]:
    """Get the current tracker from context, or None if not set."""
    return _current_tracker.get()


def set_tracker(tracker: Optional[UsageTracker]) -> None:
    """Set the current tracker in context. Pass None to disable tracking."""
    _current_tracker.set(tracker)
