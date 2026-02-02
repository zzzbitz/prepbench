# llm_connect

Utilities for creating LLM clients from configuration, routing providers, and tracking token usage.

## Scope

- Build an LLM client based on `config/settings.yaml` + environment variables
- Support multiple providers via a registry and dynamic `provider_factory`
- Provide a usage tracker that downstream clients can report into
- Offer a concrete OpenRouter client with retry logic

## Public API

From `llm_connect`:

- `create_llm_client_from_profile(model_name: str | None = None, agent: str | None = None) -> object | None`
- `register_provider(provider_type: str, factory: Callable)`
- `list_providers() -> list[str]`

From `llm_connect.config` (used by other modules):

- `get_active_profile(agent: str | None = None) -> (name, profile) | None`
- `get_model_name(override: str | None = None, agent: str | None = None) -> str`
- `get_llm_params(agent_name: str, step_name: str) -> dict`
- `validate_clarifier_settings() -> None`

## Provider Resolution

1) Resolve active provider from config or env:
   - `LLM_ACTIVE_PROVIDER` or `LLM_CLARIFIER_ACTIVE_PROVIDER`
2) Resolve model:
   - `model_name` override
   - `LLM_MODEL` / `LLM_CLARIFIER_MODEL`
   - provider `model` field
3) Resolve provider implementation:
   - `provider_factory` (format `module:function`) if present
   - otherwise `type` matched in registry (built-in: `openrouter`)

If no provider can be resolved, an error is raised with guidance.

## Configuration Examples

### Built-in OpenRouter

```yaml
llm:
  active_provider: openrouter
  providers:
    openrouter:
      type: openrouter
      model: openai/gpt-5.2
      http_referer: "https://example.com"
      x_title: "prepbench"
```

API keys must be provided via `.env` or environment variables:

```bash
cat << 'EOF' > .env
OPENROUTER_API_KEY=your-key
EOF
```

### Custom Provider (dynamic import)

```yaml
llm:
  active_provider: my_provider
  providers:
    my_provider:
      provider_factory: "my_pkg.my_provider:create_client"
      model: "your/model"
```

Factory signature:

```python
def create_client(profile: dict, model_name: str, agent: str | None = None) -> object:
    ...
```

Returned object must implement:

```python
generate(messages, temperature, max_tokens, timeout) -> str
```

## Usage Tracking

`llm_connect.usage_tracker` provides a context-aware tracker:

- `set_tracker(tracker)` / `get_tracker()`
- `UsageTracker.record(prompt_tokens, completion_tokens)` / `record_unknown()`

If a provider wants to report usage, it can call `get_tracker()` and record tokens.
`OpenRouterLLMClient` already does this.

## OpenRouter Client

`llm_connect.openrouter_client.OpenRouterLLMClient` implements:

- Retry with exponential backoff for transient HTTP failures
- Optional usage tracking via `usage_tracker`

## External Usage Notes

- `data_synthesis/synthesizer.py` currently instantiates `OpenRouterLLMClient` directly.
  If you want non-OpenRouter providers there, refactor it to use the factory
  in `llm_connect.utils.create_llm_client_from_profile` or a custom provider factory.
