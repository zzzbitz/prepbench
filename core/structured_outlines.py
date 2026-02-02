from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Iterable, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_MODEL_CACHE_SIZE = 8
_GENERATOR_CACHE_SIZE = 128

_MODEL_CACHE: OrderedDict[tuple[str, str | None], tuple[str | None, Any]] = OrderedDict()
_JSON_GENERATOR_CACHE: OrderedDict[tuple[str, str | None, Type[T]], Any] = OrderedDict()
_CHOICE_GENERATOR_CACHE: OrderedDict[tuple[str, str | None, tuple[str, ...]], Any] = OrderedDict()


def structured_outputs_enabled() -> bool:
    return os.environ.get("ENABLE_OUTLINES", "0") == "1"


def resolve_outlines_credentials(agent: str | None = None) -> tuple[str | None, str | None]:
    api_key = None
    base_url = None

    try:
        from llm_connect.config import get_active_profile
    except Exception:
        get_active_profile = None

    if get_active_profile:
        profile = get_active_profile(agent)
        if profile:
            _, prof = profile
            api_key = prof.get("api_key") or api_key
            base_url = prof.get("base_url") or base_url

    env_api_key = (
        os.environ.get("OUTLINES_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    env_base_url = os.environ.get("OUTLINES_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        api_key = env_api_key
    if not base_url:
        base_url = env_base_url
    return api_key, base_url


def clear_outlines_caches() -> None:
    _MODEL_CACHE.clear()
    _JSON_GENERATOR_CACHE.clear()
    _CHOICE_GENERATOR_CACHE.clear()


def _lru_get(cache: OrderedDict, key: Any) -> Any:
    if key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def _lru_set(cache: OrderedDict, key: Any, value: Any, maxsize: int) -> None:
    if key in cache:
        cache.pop(key)
    cache[key] = value
    if len(cache) > maxsize:
        cache.popitem(last=False)


def _clear_generator_caches() -> None:
    _JSON_GENERATOR_CACHE.clear()
    _CHOICE_GENERATOR_CACHE.clear()


def _get_model(model_name: str, base_url: str | None, api_key: str | None):
    key = (model_name, base_url)
    cached = _lru_get(_MODEL_CACHE, key)
    if cached is not None:
        cached_key, cached_model = cached
        if api_key is None or api_key == cached_key:
            return cached_model

    from outlines import models

    model = models.openai(model_name, api_key=api_key, base_url=base_url)
    _lru_set(_MODEL_CACHE, key, (api_key, model), _MODEL_CACHE_SIZE)
    _clear_generator_caches()
    return model


def get_outlines_openai_model(
    model_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
):
    if api_key is None or base_url is None:
        resolved_key, resolved_url = resolve_outlines_credentials()
        api_key = api_key or resolved_key
        base_url = base_url or resolved_url

    if kwargs:
        from outlines import models

        return models.openai(model_name, api_key=api_key, base_url=base_url, **kwargs)
    return _get_model(model_name, base_url, api_key)


def _build_json_generator(
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    schema: Type[T],
):
    from outlines import generate

    model = _get_model(model_name, base_url, api_key)
    return generate.json(model, schema)


def _build_choice_generator(
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    choices: tuple[str, ...],
):
    from outlines import generate

    model = _get_model(model_name, base_url, api_key)
    return generate.choice(model, list(choices))


def get_json_generator(
    *,
    model_name: str,
    schema: Type[T],
    base_url: str | None = None,
    api_key: str | None = None,
):
    _get_model(model_name, base_url, api_key)
    key = (model_name, base_url, schema)
    cached = _lru_get(_JSON_GENERATOR_CACHE, key)
    if cached is not None:
        return cached
    generator = _build_json_generator(model_name, base_url, api_key, schema)
    _lru_set(_JSON_GENERATOR_CACHE, key, generator, _GENERATOR_CACHE_SIZE)
    return generator


def get_choice_generator(
    *,
    model_name: str,
    choices: Iterable[str],
    base_url: str | None = None,
    api_key: str | None = None,
):
    _get_model(model_name, base_url, api_key)
    key = (model_name, base_url, tuple(choices))
    cached = _lru_get(_CHOICE_GENERATOR_CACHE, key)
    if cached is not None:
        return cached
    generator = _build_choice_generator(model_name, base_url, api_key, key[2])
    _lru_set(_CHOICE_GENERATOR_CACHE, key, generator, _GENERATOR_CACHE_SIZE)
    return generator


def _coerce_schema_output(schema: Type[T], result: Any) -> T:
    if isinstance(result, schema):
        return result
    if isinstance(result, BaseModel):
        return schema.model_validate(result.model_dump())
    if isinstance(result, str):
        return schema.model_validate_json(result)
    return schema.model_validate(result)


def structured_json(
    *,
    model_name: str,
    prompt: str,
    schema: Type[T],
    base_url: str | None = None,
    api_key: str | None = None,
    **gen_kwargs: Any,
) -> T:
    if not structured_outputs_enabled():
        raise RuntimeError("Outlines disabled")
    if api_key is None or base_url is None:
        resolved_key, resolved_url = resolve_outlines_credentials()
        api_key = api_key or resolved_key
        base_url = base_url or resolved_url
    generator = get_json_generator(
        model_name=model_name,
        schema=schema,
        base_url=base_url,
        api_key=api_key,
    )
    result = generator(prompt, **gen_kwargs)
    return _coerce_schema_output(schema, result)


def structured_choice(
    *,
    model_name: str,
    prompt: str,
    choices: Iterable[str],
    base_url: str | None = None,
    api_key: str | None = None,
    **gen_kwargs: Any,
) -> str:
    if not structured_outputs_enabled():
        raise RuntimeError("Outlines disabled")
    if api_key is None or base_url is None:
        resolved_key, resolved_url = resolve_outlines_credentials()
        api_key = api_key or resolved_key
        base_url = base_url or resolved_url
    generator = get_choice_generator(
        model_name=model_name,
        choices=choices,
        base_url=base_url,
        api_key=api_key,
    )
    result = generator(prompt, **gen_kwargs)
    return str(result)
