#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, List
import os
from config.config_loader import load_env_file

load_env_file()

API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
HTTP_REFERER: Optional[str] = None
X_TITLE: Optional[str] = None

FULL = {
    "model": [
        "openai/gpt-5.2",
    ],
    "temperature": 0.7,
    "max_tokens": 10000,
    "timeout": 180,
}

AMB = {
    "model": "openai/gpt-5.2",
    "temperature": 0.7,
    "max_tokens": 15000,
    "timeout": 180,
}

def get_api_key() -> str:
    return API_KEY


def get_full_models() -> List[str]:
    models = FULL.get("model")
    if isinstance(models, str):
        models = [models]
    if not isinstance(models, list) or not models:
        raise ValueError("FULL['model'] must be a non-empty list[str]")
    cleaned: List[str] = []
    for m in models:
        if not isinstance(m, str) or not m.strip():
            raise ValueError("FULL['model'] entries must be non-empty strings")
        if m not in cleaned:
            cleaned.append(m)
    return cleaned


def get_full_default_model() -> str:
    return get_full_models()[0]
