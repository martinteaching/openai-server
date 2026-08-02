from enum import Enum

from openai import BaseModel
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat_model import ChatModel

from pydantic import AliasChoices, Field


class ChatCompletionRequest(BaseModel):
    model: ChatModel | str
    messages: list[ChatCompletionMessageParam]
    max_completion_tokens: int | None = Field(None, validation_alias=AliasChoices('max_completion_tokens', 'max_tokens'))
    temperature: float | None


class Role(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
