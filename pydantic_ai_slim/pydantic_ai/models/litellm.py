from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from typing_extensions import assert_never

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils
from .._parts_manager import ModelResponsePartsManager
from ..usage import RunUsage
from .._output import DEFAULT_OUTPUT_TOOL_NAME, OutputObjectDefinition
from .._run_context import RunContext
from .._thinking_part import split_content_into_text_and_thinking
from .._utils import now_utc as _now_utc, number_to_datetime
from ..exceptions import UserError
from ..messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..profiles import ModelProfile, ModelProfileSpec
from ..profiles.openai import openai_model_profile
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests

try:
    import litellm
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install `litellm` to use the LiteLLM model, '
        'you can install it with `pip install litellm`'
    ) from _import_error

__all__ = (
    'LiteLLMModel',
    'LiteLLMModelSettings',
)


class LiteLLMModelSettings(ModelSettings, total=False):
    """Settings used for a LiteLLM model request."""
    
    # ALL FIELDS MUST BE `litellm_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    
    litellm_api_base: str
    """Base URL for the LiteLLM API."""
    
    litellm_api_key: str
    """API key for the LiteLLM provider."""
    
    litellm_custom_llm_provider: str
    """Custom LLM provider identifier for LiteLLM."""


@dataclass(init=False)
class LiteLLMModel(Model):
    """A model that uses the LiteLLM API directly.
    
    This model uses LiteLLM's completion API directly, which properly handles
    OAuth2 authentication for providers like GitHub Copilot.
    """
    
    _model_name: str = field(repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a LiteLLM model.

        Args:
            model_name: The name of the model to use (e.g., 'github_copilot/gpt-4.1')
            profile: The model profile to use. Defaults to a profile picked based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name
        
        # Determine the appropriate profile based on model name
        if profile is None:
            # Use OpenAI profile for all LiteLLM models (including github_copilot/*)
            # We pass the raw model name; providers can adjust if needed.
            profile = openai_model_profile(model_name.split('/', 1)[1] if '/' in model_name else model_name)
        
        super().__init__(settings=settings, profile=profile)

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return 'litellm'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        
        litellm_messages = await self._map_messages(messages)
        tools = self._get_tools(model_request_parameters) if model_request_parameters.tool_defs else None
        
        # Prepare LiteLLM parameters
        kwargs = {
            'model': self._model_name,
            'messages': litellm_messages,
            'stream': False,
        }
        
        # Add GitHub Copilot specific headers if using github_copilot model
        if self._model_name.startswith('github_copilot/'):
            kwargs['extra_headers'] = {
                'Editor-Version': 'vscode/1.85.0',
                'Copilot-Integration-Id': 'vscode-chat',
            }
        
        # Add tools if present
        if tools:
            kwargs['tools'] = tools
            if not model_request_parameters.allow_text_output:
                kwargs['tool_choice'] = 'required'
            else:
                kwargs['tool_choice'] = 'auto'
        
        # Add model settings
        if model_settings:
            if temp := model_settings.get('temperature'):
                kwargs['temperature'] = temp
            if max_tokens := model_settings.get('max_tokens'):
                kwargs['max_tokens'] = max_tokens
            if top_p := model_settings.get('top_p'):
                kwargs['top_p'] = top_p
            if stop := model_settings.get('stop_sequences'):
                kwargs['stop'] = stop
        
        try:
            response = await litellm.acompletion(**kwargs)
            return self._process_response(response)
        except Exception as e:
            raise ModelHTTPError(status_code=500, model_name=self.model_name, body=str(e)) from e

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        
        litellm_messages = await self._map_messages(messages)
        tools = self._get_tools(model_request_parameters) if model_request_parameters.tool_defs else None
        
        # Prepare LiteLLM parameters for streaming
        kwargs = {
            'model': self._model_name,
            'messages': litellm_messages,
            'stream': True,
        }
        
        # Add GitHub Copilot specific headers if using github_copilot model
        if self._model_name.startswith('github_copilot/'):
            kwargs['extra_headers'] = {
                'Editor-Version': 'vscode/1.85.0',
                'Copilot-Integration-Id': 'vscode-chat',
            }
        
        # Add tools if present
        if tools:
            kwargs['tools'] = tools
            if not model_request_parameters.allow_text_output:
                kwargs['tool_choice'] = 'required'
            else:
                kwargs['tool_choice'] = 'auto'
        
        # Add model settings
        if model_settings:
            if temp := model_settings.get('temperature'):
                kwargs['temperature'] = temp
            if max_tokens := model_settings.get('max_tokens'):
                kwargs['max_tokens'] = max_tokens
            if top_p := model_settings.get('top_p'):
                kwargs['top_p'] = top_p
            if stop := model_settings.get('stop_sequences'):
                kwargs['stop'] = stop
        
        try:
            response = await litellm.acompletion(**kwargs)
            yield LiteLLMStreamedResponse(response, model_request_parameters, self.profile, self.model_name)
        except Exception as e:
            raise ModelHTTPError(status_code=500, model_name=self.model_name, body=str(e)) from e

    def _process_response(self, response: Any) -> ModelResponse:
        """Process a non-streamed response."""
        timestamp = _now_utc()
        items: list[ModelResponsePart] = []
        
        choice = response.choices[0]
        
        if choice.message.content:
            items.extend(
                split_content_into_text_and_thinking(choice.message.content, self.profile.thinking_tags)
            )
        
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                items.append(
                    ToolCallPart(
                        tool_name=tool_call.function.name,
                        args_json_str=tool_call.function.arguments,
                        tool_call_id=tool_call.id,
                    )
                )
        
        # Map finish reason
        finish_reason_map = {
            'stop': 'stop',
            'length': 'length',
            'tool_calls': 'tool_call',
            'content_filter': 'content_filter',
        }
        finish_reason = finish_reason_map.get(choice.finish_reason, 'error')
        
        # Create usage info
        usage_info = None
        if hasattr(response, 'usage') and response.usage:
            usage_info = RunUsage(
                input_tokens=getattr(response.usage, 'prompt_tokens', 0),
                output_tokens=getattr(response.usage, 'completion_tokens', 0),
            )
        
        return ModelResponse(
            parts=items,
            usage=usage_info,
            model_name=response.model,
            timestamp=timestamp,
            provider_response_id=getattr(response, 'id', None),
            provider_name='litellm',
            finish_reason=finish_reason,
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[dict[str, Any]]:
        """Convert tool definitions to LiteLLM format."""
        tools = []
        for tool_def in model_request_parameters.tool_defs.values():
            tools.append({
                'type': 'function',
                'function': {
                    'name': tool_def.name,
                    'description': tool_def.description or '',
                    'parameters': tool_def.parameters_json_schema,
                }
            })
        return tools

    async def _map_messages(self, messages: list[ModelMessage]) -> list[dict[str, Any]]:
        """Map pydantic_ai messages to LiteLLM format."""
        litellm_messages: list[dict[str, Any]] = []
        
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        litellm_messages.append({
                            'role': 'system',
                            'content': part.content,
                        })
                    elif isinstance(part, UserPromptPart):
                        content = part.content
                        if isinstance(content, str):
                            litellm_messages.append({
                                'role': 'user',
                                'content': content,
                            })
                        else:
                            # For multi-modal content, we'd need to handle it properly
                            # For now, just convert to string
                            text_content = []
                            for item in content:
                                if isinstance(item, str):
                                    text_content.append(item)
                                # TODO: Handle other content types like images
                            litellm_messages.append({
                                'role': 'user',
                                'content': '\n'.join(text_content),
                            })
                    elif isinstance(part, ToolReturnPart):
                        litellm_messages.append({
                            'role': 'tool',
                            'tool_call_id': part.tool_call_id,
                            'content': part.model_response_str(),
                        })
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            litellm_messages.append({
                                'role': 'user',
                                'content': part.model_response(),
                            })
                        else:
                            litellm_messages.append({
                                'role': 'tool',
                                'tool_call_id': part.tool_call_id,
                                'content': part.model_response(),
                            })
            elif isinstance(message, ModelResponse):
                content_parts = []
                tool_calls = []
                
                for part in message.parts:
                    if isinstance(part, TextPart):
                        content_parts.append(part.content)
                    elif isinstance(part, ThinkingPart):
                        start_tag, end_tag = self.profile.thinking_tags
                        content_parts.append(f'{start_tag}\n{part.content}\n{end_tag}')
                    elif isinstance(part, ToolCallPart):
                        tool_calls.append({
                            'id': part.tool_call_id,
                            'type': 'function',
                            'function': {
                                'name': part.tool_name,
                                'arguments': part.args_as_json_str(),
                            }
                        })
                    elif isinstance(part, (BuiltinToolCallPart, BuiltinToolReturnPart)):
                        # Skip builtin tool parts
                        pass
                    else:
                        assert_never(part)
                
                assistant_message = {'role': 'assistant'}
                if content_parts:
                    assistant_message['content'] = '\n\n'.join(content_parts)
                if tool_calls:
                    assistant_message['tool_calls'] = tool_calls
                
                litellm_messages.append(assistant_message)
        
        return litellm_messages


class LiteLLMStreamedResponse(StreamedResponse):
    """Streamed response for LiteLLM models."""
    
    def __init__(
        self,
        response: Any,
        model_request_parameters: ModelRequestParameters,
        profile: ModelProfile,
        model_name: str,
    ):
        self._response = response
        self._model_request_parameters = model_request_parameters
        self._profile = profile
        self._model_name = model_name
        self._timestamp = _now_utc()
        self._usage: RunUsage | None = None
        self._parts_manager = ModelResponsePartsManager()
        self._final_usage_sent = False

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """The provider name."""
        return 'litellm'

    @property
    def timestamp(self) -> datetime:
        """The timestamp when the response was created."""
        return self._timestamp

    @property
    def model_request_parameters(self) -> ModelRequestParameters:
        """The model request parameters."""
        return self._model_request_parameters

    def usage(self) -> RunUsage:
        """The usage information for this response."""
        return self._usage or RunUsage(input_tokens=0, output_tokens=0)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Get the event iterator for this streamed response."""
        async for chunk in self._response:
            async for event in self._process_chunk(chunk):
                yield event
        
        # Ensure we always provide usage information, even if it's zero
        if not self._final_usage_sent:
            # Don't yield usage event here - the base class handles it
            self._final_usage_sent = True

    async def _process_chunk(self, chunk: Any) -> AsyncIterator[ModelResponseStreamEvent]:
        """Process a streaming chunk from LiteLLM."""
        if not chunk.choices:
            return
        
        choice = chunk.choices[0]
        delta = choice.delta
        
        if delta.content:
            # Use _parts_manager to handle text deltas
            event = self._parts_manager.handle_text_delta(
                vendor_part_id=0,  # Use index 0 for text content
                content=delta.content
            )
            if event:
                yield event
        
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                # Use tool_call.id as vendor_part_id if available
                vendor_id = tool_call.id if hasattr(tool_call, 'id') and tool_call.id else None
                
                event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=vendor_id,
                    tool_name=tool_call.function.name if hasattr(tool_call.function, 'name') else None,
                    args=tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else None,
                    tool_call_id=vendor_id
                )
                if event:
                    yield event
        
        # Handle usage info if present
        if hasattr(chunk, 'usage') and chunk.usage:
            self._usage = RunUsage(
                input_tokens=getattr(chunk.usage, 'prompt_tokens', 0),
                output_tokens=getattr(chunk.usage, 'completion_tokens', 0),
            )