from __future__ import annotations

import os
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache

from config.config_loader import load_env_file, load_settings

@lru_cache(maxsize=1)
def load_settings_config() -> Dict[str, Any]:
    """Loads the unified settings configuration file."""
    load_env_file()
    return load_settings()

def _get_llm_root() -> Dict[str, Any]:
    data = load_settings_config()
    llm_root = data.get("llm", {})
    return llm_root if isinstance(llm_root, dict) else {}

def _get_top_level_agent_section(agent: str) -> Dict[str, Any]:
    data = load_settings_config()
    sect = data.get(agent)
    if isinstance(sect, dict):
        return sect
    return {}

def _get_effective_section_for_agent(agent: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns the effective config section for an agent.

    Goal: avoid repeating provider configuration across llm/clarifier while keeping
    agent-specific overrides explicit.

    Effective fields:
    - active_provider: agent override if present, else llm.active_provider
    - providers: merged dict (llm.providers overlaid by agent.providers if present)
    - default_params: kept as-is (params are handled by get_llm_params)
    - model: agent top-level model override (optional)
    """
    llm_root = _get_llm_root()
    if agent is None:
        return llm_root

    if agent == "clarifier":
        # Prefer dedicated top-level section; accept historical typo as alias.
        top = _get_top_level_agent_section("clarifier") or _get_top_level_agent_section("claifier")
        # Legacy nested support: llm.clarifier / llm.claifier
        legacy = (llm_root or {}).get("clarifier")
        if not (isinstance(legacy, dict) and legacy):
            legacy = (llm_root or {}).get("claifier")
        if not (isinstance(legacy, dict) and legacy):
            legacy = {}
        agent_sect = top or legacy
    else:
        agent_sect = {}

    if not isinstance(agent_sect, dict):
        agent_sect = {}

    base_providers = (llm_root or {}).get("providers") if isinstance(llm_root, dict) else {}
    agent_providers = agent_sect.get("providers")

    merged_providers: Dict[str, Any] = {}
    if isinstance(base_providers, dict):
        merged_providers.update(base_providers)
    if isinstance(agent_providers, dict):
        # Allow agent-specific override of provider details if explicitly provided.
        for k, v in agent_providers.items():
            if k in merged_providers and isinstance(merged_providers[k], dict) and isinstance(v, dict):
                merged = dict(merged_providers[k])
                merged.update(v)
                merged_providers[k] = merged
            else:
                merged_providers[k] = v

    active_provider = agent_sect.get("active_provider") or (llm_root or {}).get("active_provider")

    effective = dict(agent_sect)
    effective.setdefault("active_provider", active_provider)
    effective["providers"] = merged_providers
    return effective


def _get_section_for_agent(agent: Optional[str] = None) -> Dict[str, Any]:
    data = load_settings_config()
    llm_root = data.get("llm", {})
    if agent == "clarifier":
        # Top-level dedicated section: prefer `clarifier`, accept `claifier` as alias.
        top = data.get("clarifier")
        if not (isinstance(top, dict) and top):
            top = data.get("claifier")
        if isinstance(top, dict) and top:
            return top
        nested = (llm_root or {}).get("clarifier")
        if not (isinstance(nested, dict) and nested):
            nested = (llm_root or {}).get("claifier")
        if isinstance(nested, dict) and nested:
            return nested
        return {}
    return llm_root


def get_active_profile(agent: Optional[str] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
    sect = _get_effective_section_for_agent(agent)
    if agent == "clarifier":
        env_key = "LLM_CLARIFIER_ACTIVE_PROVIDER"
    else:
        env_key = "LLM_ACTIVE_PROVIDER"
    active = os.getenv(env_key, (sect or {}).get("active_provider"))
    profiles = (sect or {}).get("providers", {})

    if not active or active not in profiles:
        return None
    return active, profiles[active]

def get_model_names(agent: Optional[str] = None) -> list[str]:
    """
    Returns configured model name(s) for an agent.

    - agent is None (code/prep): allow list or string from provider config.
    - agent == "clarifier": must resolve to a single model (top-level `clarifier.model` or provider model string).
    """
    sect = _get_effective_section_for_agent(agent)

    def _from_value(v: Any) -> list[str]:
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
            return out
        return []

    # Agent-level override: allow `clarifier.model` as string only.
    if agent == "clarifier":
        top = (sect or {}).get("model")
        if isinstance(top, str) and top.strip():
            return [top.strip()]
        if isinstance(top, list):
            raise ValueError("Clarifier model must be a single string; lists are not allowed.")

    prof = get_active_profile(agent)
    if not prof:
        return []
    active_provider, profile = prof
    model_field = profile.get("model", "")

    models = _from_value(model_field)
    if agent == "clarifier":
        if len(models) != 1:
            raise ValueError(
                f"Clarifier model must be a single string (active_provider={active_provider})."
            )
    return models


def get_model_name(override: Optional[str] = None, agent: Optional[str] = None) -> str:
    if override and str(override).strip():
        return str(override).strip()

    if agent == "clarifier":
        env_var = "LLM_CLARIFIER_MODEL"
    else:
        env_var = "LLM_MODEL"
    env_model = os.getenv(env_var)
    if env_model and env_model.strip():
        return env_model.strip()

    models = get_model_names(agent)
    return models[0] if models else ""


def _has_top_level_clarifier() -> bool:
    data = load_settings_config()
    top = data.get("clarifier")
    if isinstance(top, dict) and top:
        return True
    top = data.get("claifier")
    return isinstance(top, dict) and bool(top)


def validate_clarifier_settings() -> None:
    """Fail-fast validation for Clarifier settings used by interact mode.

    Accepts either:
    - top-level `clarifier` (preferred; `claifier` is accepted as an alias), or
    - legacy `llm.clarifier` (fallback).

    Validates the effective provider selection and enforces a single-string model (no lists).
    """
    data = load_settings_config()

    # Clarifier must be explicitly configured (top-level preferred).
    has_top = isinstance(data.get("clarifier"), dict) and bool(data.get("clarifier"))
    has_top_alias = isinstance(data.get("claifier"), dict) and bool(data.get("claifier"))
    has_legacy = isinstance((_get_llm_root() or {}).get("clarifier"), dict) and bool((_get_llm_root() or {}).get("clarifier"))
    has_legacy_alias = isinstance((_get_llm_root() or {}).get("claifier"), dict) and bool((_get_llm_root() or {}).get("claifier"))
    if not (has_top or has_top_alias or has_legacy or has_legacy_alias):
        raise ValueError(
            "Clarifier section is missing (configure top-level `clarifier` (preferred) or legacy `llm.clarifier`; `claifier` is accepted as an alias)."
        )

    clar_section = _get_effective_section_for_agent("clarifier")
    active = os.getenv("LLM_CLARIFIER_ACTIVE_PROVIDER", (clar_section or {}).get("active_provider"))
    providers = (clar_section or {}).get("providers", {}) or {}
    if not active or active not in providers:
        raise ValueError("Clarifier active_provider is not configured or not present in providers.")

    # Model can be overridden by env; if not, validate config model type/value.
    env_model = os.getenv("LLM_CLARIFIER_MODEL")
    if env_model and env_model.strip():
        return

    top_model = (clar_section or {}).get("model")
    if isinstance(top_model, str) and top_model.strip():
        return

    model_field = (providers.get(active) or {}).get("model")
    if model_field is None:
        raise ValueError(f"Clarifier model is not configured for provider '{active}'.")
    if isinstance(model_field, list):
        raise ValueError(f"Clarifier model for provider '{active}' must be a single string (lists are not allowed).")
    if not isinstance(model_field, str) or not model_field.strip():
        raise ValueError(f"Clarifier model for provider '{active}' must be a non-empty string.")


def get_llm_params(agent_name: str, step_name: str) -> Dict[str, Any]:
    is_clarifier = agent_name.lower().startswith("clarifier")
    
    if is_clarifier:
        if _has_top_level_clarifier():
            sect = _get_section_for_agent("clarifier")
            params = (sect or {}).get("default_params")
            if isinstance(params, dict) and params:
                return params
            # If clarifier section exists but doesn't specify params, inherit from llm.
            root = _get_section_for_agent(None)
            return (root or {}).get("default_params") or {}
        root = _get_section_for_agent(None)
        return (root or {}).get("default_params") or {}
    
    root = _get_section_for_agent(None)
    return (root or {}).get("default_params") or {}
