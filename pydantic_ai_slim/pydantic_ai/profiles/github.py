from __future__ import annotations as _annotations

from . import ModelProfile
from .openai import openai_model_profile

__all__ = [
    'github_copilot_model_profile',
]


def github_copilot_model_profile(model_name: str) -> ModelProfile:
    """Return a ModelProfile for GitHub Copilot models.

    GitHub Copilot chat models use an OpenAI-compatible interface via LiteLLM,
    so we reuse the OpenAI model profile configuration.
    """
    return openai_model_profile(model_name)
