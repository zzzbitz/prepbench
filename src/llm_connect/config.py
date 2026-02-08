from __future__ import annotations

import re
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache

from config.config_loader import get_env_value, load_env_file, load_settings


USER_SIMULATOR_AGENT = "user_simulator"


def _is_user_simulator_agent(agent: Optional[str]) -> bool:
    if not isinstance(agent, str):
        return False
    normalized = agent.strip().lower().replace("-", "_")
    return normalized in {"user_simulator", "usersimulator"}


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

    Goal: avoid repeating provider configuration across llm/user_simulator while keeping
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

    if _is_user_simulator_agent(agent):
        agent_sect = _get_top_level_agent_section(USER_SIMULATOR_AGENT)
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
    if _is_user_simulator_agent(agent):
        top = data.get(USER_SIMULATOR_AGENT)
        if isinstance(top, dict):
            return top
        return {}
    return llm_root


def get_active_profile(agent: Optional[str] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
    sect = _get_effective_section_for_agent(agent)
    if _is_user_simulator_agent(agent):
        env_key = "LLM_USER_SIMULATOR_ACTIVE_PROVIDER"
    else:
        env_key = "LLM_ACTIVE_PROVIDER"
    active = get_env_value(env_key, "").strip() or (sect or {}).get("active_provider")
    profiles = (sect or {}).get("providers", {})

    if not active or active not in profiles:
        return None
    return active, profiles[active]

def get_model_names(agent: Optional[str] = None) -> list[str]:
    """
    Returns configured model name(s) for an agent.

    - agent is None (code/prep): allow list or string from provider config.
    - agent == "user_simulator": must resolve to a single model (top-level `user_simulator.model` or provider model string).
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

    # Agent-level override: allow `user_simulator.model` as string only.
    if _is_user_simulator_agent(agent):
        top = (sect or {}).get("model")
        if isinstance(top, str) and top.strip():
            return [top.strip()]
        if isinstance(top, list):
            raise ValueError("User simulator model must be a single string; lists are not allowed.")

    prof = get_active_profile(agent)
    if not prof:
        return []
    active_provider, profile = prof
    model_field = profile.get("model", "")

    models = _from_value(model_field)
    if _is_user_simulator_agent(agent):
        if len(models) != 1:
            raise ValueError(
                f"User simulator model must be a single string (active_provider={active_provider})."
            )
    return models


def get_model_name(override: Optional[str] = None, agent: Optional[str] = None) -> str:
    if override and str(override).strip():
        return str(override).strip()

    if _is_user_simulator_agent(agent):
        env_var = "LLM_USER_SIMULATOR_MODEL"
    else:
        env_var = "LLM_MODEL"
    env_model = get_env_value(env_var, "")
    if env_model and env_model.strip():
        return env_model.strip()

    models = get_model_names(agent)
    return models[0] if models else ""


def validate_user_simulator_settings() -> None:
    """Fail-fast validation for user simulator settings used by interact mode."""
    data = load_settings_config()

    # user_simulator must be explicitly configured at top level.
    has_top = isinstance(data.get(USER_SIMULATOR_AGENT), dict) and bool(data.get(USER_SIMULATOR_AGENT))
    if not has_top:
        raise ValueError(
            "user_simulator section is missing (configure top-level `user_simulator`)."
        )

    sim_section = _get_effective_section_for_agent(USER_SIMULATOR_AGENT)
    active = get_env_value("LLM_USER_SIMULATOR_ACTIVE_PROVIDER", "").strip() or (sim_section or {}).get("active_provider")
    providers = (sim_section or {}).get("providers", {}) or {}
    if not active or active not in providers:
        raise ValueError("user_simulator active_provider is not configured or not present in providers.")

    # Model can be overridden by .env; if not, validate config model type/value.
    env_model = get_env_value("LLM_USER_SIMULATOR_MODEL", "")
    if env_model and env_model.strip():
        return

    top_model = (sim_section or {}).get("model")
    if isinstance(top_model, str) and top_model.strip():
        return

    model_field = (providers.get(active) or {}).get("model")
    if model_field is None:
        raise ValueError(f"user_simulator model is not configured for provider '{active}'.")
    if isinstance(model_field, list):
        raise ValueError(f"user_simulator model for provider '{active}' must be a single string (lists are not allowed).")
    if not isinstance(model_field, str) or not model_field.strip():
        raise ValueError(f"user_simulator model for provider '{active}' must be a non-empty string.")


def _camel_to_snake(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return ""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.replace("-", "_").lower()


def _alias_candidates(name: str) -> list[str]:
    raw = (name or "").strip()
    if not raw:
        return []
    snake = _camel_to_snake(raw)
    lower = raw.lower().replace("-", "_")

    candidates: list[str] = []
    for item in (raw, lower, snake):
        if item and item not in candidates:
            candidates.append(item)

    for item in (lower, snake):
        if item.endswith("_agent"):
            short = item[:-6]
            if short and short not in candidates:
                candidates.append(short)
        elif item.endswith("agent"):
            short = item[:-5]
            if short and short not in candidates:
                candidates.append(short)
    return candidates


def _extract_params_override(params_root: Any, agent_name: str, step_name: str) -> Dict[str, Any]:
    """Extract llm.params.<agent>.<step> overrides with light aliases."""
    if not isinstance(params_root, dict):
        return {}

    merged: Dict[str, Any] = {}
    agent_aliases = _alias_candidates(agent_name)
    step_aliases = _alias_candidates(step_name)
    if step_name and step_name not in step_aliases:
        step_aliases.append(step_name)

    # Support params.<agent>.default + params.<agent>.<step>
    for agent_key in agent_aliases:
        node = params_root.get(agent_key)
        if not isinstance(node, dict):
            continue
        default_node = node.get("default")
        if isinstance(default_node, dict):
            merged.update(default_node)
        for step_key in step_aliases:
            step_node = node.get(step_key)
            if isinstance(step_node, dict):
                merged.update(step_node)

    # Support params.<step> as global step-level fallback.
    for step_key in step_aliases:
        step_node = params_root.get(step_key)
        if isinstance(step_node, dict):
            merged.update(step_node)

    return merged


def get_llm_params(agent_name: str, step_name: str) -> Dict[str, Any]:
    normalized_agent = (agent_name or "").strip().lower().replace("-", "_")
    is_user_simulator = normalized_agent in {"user_simulator", "usersimulator"}

    llm_root = _get_section_for_agent(None)
    llm_default = (llm_root or {}).get("default_params") or {}
    llm_params_root = (llm_root or {}).get("params") or {}

    merged: Dict[str, Any] = {}
    if isinstance(llm_default, dict):
        merged.update(llm_default)

    if is_user_simulator:
        simulator_sect = _get_section_for_agent(USER_SIMULATOR_AGENT)
        simulator_default = (simulator_sect or {}).get("default_params") or {}
        if isinstance(simulator_default, dict) and simulator_default:
            merged.update(simulator_default)

        llm_override = _extract_params_override(llm_params_root, agent_name, step_name)
        merged.update(llm_override)

        simulator_params_root = (simulator_sect or {}).get("params") or {}
        if isinstance(simulator_params_root, dict):
            merged.update(_extract_params_override(simulator_params_root, agent_name, step_name))
        return merged

    merged.update(_extract_params_override(llm_params_root, agent_name, step_name))
    return merged
