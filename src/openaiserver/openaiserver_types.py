from enum import Enum

from openai import BaseModel
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat_model import ChatModel

from pydantic import AliasChoices, Field, field_validator


class ChatCompletionRequest(BaseModel):
    model: ChatModel | str
    messages: list[ChatCompletionMessageParam]
    max_completion_tokens: int | None = Field(None, validation_alias=AliasChoices('max_completion_tokens', 'max_tokens'))
    temperature: float | None = 1.0

    @field_validator('messages')
    @classmethod
    def materialiseContent(cls, messages: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
        for message in messages:
            if not isinstance(content := message.get('content'), str | None):
                message['content'] = list(content)  # ty: ignore[invalid-assignment]
        return messages


class Role(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
