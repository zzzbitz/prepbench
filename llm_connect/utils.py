from typing import Optional

from .config import get_active_profile, get_model_name
from .providers import get_provider, load_provider_factory


def create_llm_client_from_profile(model_name: Optional[str] = None, agent: Optional[str] = None) -> Optional[object]:
    """
    Create an LLM client based on the active provider configuration.

    Config routing:
    - When `agent == "clarifier"`, settings are resolved from the dedicated top-level `clarifier` section
      (fallback: `llm.clarifier`, then `llm`).
    - Otherwise, settings are resolved from `llm`.

    Model resolution order:
    1) `model_name` override (if provided)
    2) `.env` (`LLM_CLARIFIER_MODEL` or `LLM_MODEL`)
    3) provider config `model`

    Provider resolution:
    - If provider config has `provider_factory` (format "module:function"), it is used directly.
    - Otherwise, `type` is resolved via the built-in registry (e.g., "openrouter").
    """
    profile_info = get_active_profile(agent)
    if not profile_info:
        return None

    _, prof = profile_info

    final_model = get_model_name(model_name, agent=agent)
    if not final_model:
        raise RuntimeError(
            "Model name is not configured. "
            "Set it in .env (`LLM_MODEL` / `LLM_CLARIFIER_MODEL`) "
            "or `config/settings.yaml` under the active provider (`llm.providers.*.model` / `clarifier.model`)."
        )

    factory_path = prof.get("provider_factory") or prof.get("factory")
    if isinstance(factory_path, str) and factory_path.strip():
        factory = load_provider_factory(factory_path.strip())
        return factory(prof, final_model, agent)

    ptype = prof.get("type")
    factory = get_provider(ptype)
    if factory is not None:
        return factory(prof, final_model, agent)

    raise ValueError(
        f"Unknown provider type: {ptype}. "
        "Register a provider or set provider_factory='module:function' in settings."
    )
