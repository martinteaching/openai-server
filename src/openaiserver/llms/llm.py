import time
from abc import ABC

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openaiserver.openaiserver_types import ChatCompletionRequest


class LLM(ABC):

    def createCompletion(
        self, messages: list[ChatCompletionMessageParam], content: str, model: str
    ) -> ChatCompletion | None:
        promptTokens: int = sum(
            [len(str(message['content']).split()) for message in messages]
        )
        completionTokens: int = len(content.split())
        return ChatCompletion(
            id=f'chatcmpl-{int(time.time()*1000)}',
            object='chat.completion',
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role='assistant', content=content),
                    finish_reason='stop',
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=promptTokens,
                completion_tokens=completionTokens,
                total_tokens=promptTokens + completionTokens,
            ),
        )

    def getCompletion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletion | None: ...
