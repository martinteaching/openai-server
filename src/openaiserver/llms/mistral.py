from llama_cpp import (
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
)
from openai.types.chat import ChatCompletionMessageParam

from openaiserver.llms.llm import LLM


class Mistral(LLM):

    def formatMessages(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionRequestMessage]:
        return [
            (
                ChatCompletionRequestSystemMessage(
                    content=str(message['content']), role='system'
                )
                if message['role'] == 'system'
                else (
                    ChatCompletionRequestAssistantMessage(
                        content=str(message['content']), role='assistant'
                    )
                    if message['role'] == 'assistant'
                    else ChatCompletionRequestUserMessage(
                        content=str(message['content']), role='user'
                    )
                )
            )
            for message in messages
        ]
