from enum import Enum

from openai import BaseModel
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat_model import ChatModel


class ChatCompletionRequest(BaseModel):
    model: ChatModel | str
    messages: list[ChatCompletionMessageParam]
    max_tokens: int | None
    temperature: float | None


class Role(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
