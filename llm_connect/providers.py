from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, Optional

from .openrouter_client import OpenRouterLLMClient

ProviderFactory = Callable[[Dict[str, Any], str, Optional[str]], object]

_REGISTRY: Dict[str, ProviderFactory] = {}


def register_provider(provider_type: str, factory: ProviderFactory) -> None:
    if not isinstance(provider_type, str) or not provider_type.strip():
        raise ValueError("provider_type must be a non-empty string")
    if not callable(factory):
        raise TypeError("factory must be callable")
    _REGISTRY[provider_type.strip()] = factory


def get_provider(provider_type: Optional[str]) -> Optional[ProviderFactory]:
    if not isinstance(provider_type, str) or not provider_type.strip():
        return None
    return _REGISTRY.get(provider_type.strip())


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


def _openrouter_factory(profile: Dict[str, Any], model_name: str, agent: Optional[str] = None) -> object:
    api_key = os.getenv("OPENROUTER_API_KEY") or ""
    http_referer = profile.get("http_referer")
    x_title = profile.get("x_title")
    return OpenRouterLLMClient(
        api_key=api_key,
        model_name=model_name,
        http_referer=http_referer if isinstance(http_referer, str) else None,
        x_title=x_title if isinstance(x_title, str) else None,
    )


register_provider("openrouter", _openrouter_factory)
