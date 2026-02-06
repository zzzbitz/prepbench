from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, Optional

from .openrouter_client import OpenAICompatibleChatClient, OpenRouterLLMClient

ProviderFactory = Callable[[Dict[str, Any], str, Optional[str]], object]

_REGISTRY: Dict[str, ProviderFactory] = {}


def register_provider(provider_type: str, factory: ProviderFactory) -> None:
    if not isinstance(provider_type, str) or not provider_type.strip():
        raise ValueError("provider_type must be a non-empty string")
    if not callable(factory):
        raise TypeError("factory must be callable")
    _REGISTRY[provider_type.strip().lower()] = factory


def get_provider(provider_type: Optional[str]) -> Optional[ProviderFactory]:
    if not isinstance(provider_type, str) or not provider_type.strip():
        return None
    return _REGISTRY.get(provider_type.strip().lower())


def list_providers() -> list[str]:
    return sorted(_REGISTRY.keys())


def load_provider_factory(dotted_path: str) -> ProviderFactory:
    if not isinstance(dotted_path, str) or not dotted_path.strip():
        raise ValueError("provider_factory must be a non-empty string")
    if ":" not in dotted_path:
        raise ValueError("provider_factory must be in the format 'module:function'")

    module_path, attr = dotted_path.split(":", 1)
    module_path = module_path.strip()
    attr = attr.strip()
    if not module_path or not attr:
        raise ValueError("provider_factory must be in the format 'module:function'")

    module = importlib.import_module(module_path)
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise TypeError(f"provider_factory '{dotted_path}' is not callable")
    return factory


def _get_profile_str(profile: Dict[str, Any], key: str) -> str:
    value = profile.get(key)
    if isinstance(value, str):
        return value.strip()
    return ""


def _resolve_api_key(profile: Dict[str, Any], *, env_keys: list[str]) -> str:
    profile_key = _get_profile_str(profile, "api_key")
    if profile_key:
        return profile_key
    for env_key in env_keys:
        value = os.getenv(env_key) or ""
        if value.strip():
            return value.strip()
    return ""


def _resolve_base_url(profile: Dict[str, Any], *, default_url: str, env_keys: list[str]) -> str:
    profile_url = _get_profile_str(profile, "base_url")
    if profile_url:
        return profile_url
    for env_key in env_keys:
        value = os.getenv(env_key) or ""
        if value.strip():
            return value.strip()
    return default_url


def _resolve_retry_config(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    retry_cfg = profile.get("retry")
    if isinstance(retry_cfg, dict):
        return retry_cfg
    return None


def _openrouter_factory(profile: Dict[str, Any], model_name: str, agent: Optional[str] = None) -> object:
    api_key = _resolve_api_key(
        profile,
        env_keys=["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
    )
    base_url = _resolve_base_url(
        profile,
        default_url="https://openrouter.ai/api/v1",
        env_keys=["OPENROUTER_BASE_URL", "OPENAI_BASE_URL"],
    )
    http_referer = profile.get("http_referer")
    x_title = profile.get("x_title")
    return OpenRouterLLMClient(
        api_key=api_key,
        model_name=model_name,
        http_referer=http_referer if isinstance(http_referer, str) else None,
        x_title=x_title if isinstance(x_title, str) else None,
        base_url=base_url,
        retry_config=_resolve_retry_config(profile),
    )


def _openai_compatible_factory(profile: Dict[str, Any], model_name: str, agent: Optional[str] = None) -> object:
    api_key = _resolve_api_key(
        profile,
        env_keys=["OPENAI_API_KEY", "OPENROUTER_API_KEY"],
    )
    base_url = _resolve_base_url(
        profile,
        default_url="https://api.openai.com/v1",
        env_keys=["OPENAI_BASE_URL", "OPENROUTER_BASE_URL"],
    )
    http_referer = profile.get("http_referer")
    x_title = profile.get("x_title")
    return OpenAICompatibleChatClient(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        http_referer=http_referer if isinstance(http_referer, str) else None,
        x_title=x_title if isinstance(x_title, str) else None,
        retry_config=_resolve_retry_config(profile),
    )


register_provider("openrouter", _openrouter_factory)
register_provider("openai_compatible", _openai_compatible_factory)
register_provider("openai-compatible", _openai_compatible_factory)
