
from dataclasses import dataclass
from typing import Optional

from .config_loader import get_env_value, load_env_file, load_settings


@dataclass(frozen=True)
class ProfileConfig:
    enabled: bool
    max_rows_per_file: int
    max_top_values: int
    max_unique_values: int
    max_columns: int
    max_column_highlights: int
    max_rounds: int              # Maximum code execution rounds for ProfileAgent
    max_summary_chars: int       # Maximum characters for profile summary

@dataclass
class ExperimentConfig:
    """
    Centralized configuration for the experiment.
    All runtime settings should be accessed through this class.
    """
    # Model settings
    model_name: str
    
    # Execution settings
    max_rounds_debug: int
    max_rounds_interact: int
    run_mode: str
    parallel_execution: bool
    jobs: Optional[int]
    timeout: int

    # Interact-mode settings
    question_ratio: Optional[float]
    max_questions_per_ask: int
    max_questions_cap: Optional[int]
    
    # Paths
    output_root_template: str

    # ProfileAgent settings (e2e mode)
    profile: ProfileConfig
    
    @staticmethod
    def _validate_positive(name: str, value: int) -> int:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    @classmethod
    def load_config(cls, *, model_name_override: Optional[str] = None) -> 'ExperimentConfig':
        """Load configuration from YAML with optional local overrides and .env."""
        # Ensure .env is loaded before reading config for .env-backed defaults
        load_env_file()

        data = load_settings()
        exp_data = data.get("experiment", {})
        profile_data = data.get("profile")
        if not isinstance(profile_data, dict):
            raise ValueError("profile config is missing or invalid.")
        llm_data = data.get("llm", {})
        
        # Resolve model name:
        # 1) explicit override
        # 2) .env LLM_MODEL
        # 3) provider config model (legacy)
        model_name = ""
        if model_name_override and str(model_name_override).strip():
            model_name = str(model_name_override).strip()
        else:
            env_model = get_env_value("LLM_MODEL", "").strip()
            if env_model:
                model_name = env_model
            else:
                active_provider = llm_data.get("active_provider")
                if active_provider:
                    provider_config = llm_data.get("providers", {}).get(active_provider, {})
                    model_name = provider_config.get("model", "")

                if isinstance(model_name, list):
                    model_name = next((m.strip() for m in model_name if isinstance(m, str) and m.strip()), "")
                elif isinstance(model_name, str) and "," in model_name:
                    model_name = model_name.split(",")[0].strip()

        if not model_name:
            raise ValueError("Model name not found. Provide --model or set LLM_MODEL in .env.")

        run_mode = exp_data.get("run_mode", "orig")
        if isinstance(run_mode, list):
            run_mode = run_mode[0]
        elif isinstance(run_mode, str) and "," in run_mode:
            run_mode = run_mode.split(",")[0].strip()

        question_ratio = exp_data.get("question_ratio", 1.0)
        if question_ratio is not None:
            question_ratio = cls._validate_positive("question_ratio", float(question_ratio))
        
        max_questions_per_ask = int(exp_data.get("max_questions_per_ask", 3))
        if max_questions_per_ask <= 0:
            max_questions_per_ask = 3
        
        max_questions_cap = exp_data.get("max_questions_cap")
        if max_questions_cap is not None:
            max_questions_cap = cls._validate_positive("max_questions_cap", int(max_questions_cap))

        profile_cfg = ProfileConfig(
            enabled=bool(profile_data.get("enabled", True)),
            max_rows_per_file=cls._validate_positive(
                "profile.max_rows_per_file", int(profile_data.get("max_rows_per_file", 2000))
            ),
            max_top_values=cls._validate_positive(
                "profile.max_top_values", int(profile_data.get("max_top_values", 5))
            ),
            max_unique_values=cls._validate_positive(
                "profile.max_unique_values", int(profile_data.get("max_unique_values", 50))
            ),
            max_columns=cls._validate_positive(
                "profile.max_columns", int(profile_data.get("max_columns", 80))
            ),
            max_column_highlights=cls._validate_positive(
                "profile.max_column_highlights", int(profile_data.get("max_column_highlights", 12))
            ),
            max_rounds=cls._validate_positive(
                "profile.max_rounds", int(profile_data.get("max_rounds", 2))
            ),
            max_summary_chars=cls._validate_positive(
                "profile.max_summary_chars", int(profile_data.get("max_summary_chars", 4000))
            ),
        )

        return cls(
            model_name=model_name,
            max_rounds_debug=cls._validate_positive("max_rounds_debug", int(exp_data.get("max_rounds_debug", 6))),
            max_rounds_interact=cls._validate_positive("max_rounds_interact", int(exp_data.get("max_rounds_interact", 8))),
            run_mode=run_mode,
            parallel_execution=exp_data.get("parallel_execution", True),
            jobs=cls._validate_positive("jobs", int(exp_data.get("jobs"))) if exp_data.get("jobs") is not None else None,
            timeout=cls._validate_positive("timeout", int(exp_data.get("timeout", 120))),
            question_ratio=question_ratio,
            max_questions_per_ask=max_questions_per_ask,
            max_questions_cap=max_questions_cap,
            output_root_template=exp_data.get("output_root_template", "@output/{model_info}/{run_mode}/{case_name}"),
            profile=profile_cfg,
        )
