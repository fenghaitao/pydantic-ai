from __future__ import annotations as _annotations

from . import ModelProfile


def github_copilot_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a GitHub Copilot model.
    
    GitHub Copilot models are typically GPT-based models provided through GitHub's API.
    They support similar capabilities to OpenAI models including tools, JSON schema output,
    and JSON object output.
    """
    return ModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        supports_image_output=False,  # GitHub Copilot doesn't support image output
        default_structured_output_mode='tool',
    )