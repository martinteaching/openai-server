import asyncio, configparser, functools, os
from typing import Iterator

from llama_cpp import (
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Llama,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from openaiserver.llms.llm import LLM
from openaiserver.openaiserver_types import ChatCompletionRequest
from openaiserver.llms.llm_quant import LLM_Quant


class Llama__3__Quant(LLM_Quant[Llama, ChatCompletionRequestMessage]):

    def __init__(self) -> None:
        self._config = configparser.ConfigParser(os.environ)
        self._config.read('config/quantized_models.ini')
        self._model: str = 'Llama__3_1__8B_Quant_Instruct'
        self._llm: Llama = self.getModel(self._config.get(self._model, 'PATH'))

    def getModel(self, path: str) -> Llama:
        return Llama(model_path=path, chat_format='llama-3', n_ctx=4096)

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

    async def getCompletion(self, request: ChatCompletionRequest) -> ChatCompletion | None:
        completion: (
            CreateChatCompletionResponse | Iterator[CreateChatCompletionStreamResponse]
        ) = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                self._llm.create_chat_completion,
                messages=self.formatMessages(request.messages),
                max_tokens=request.max_completion_tokens if request.max_completion_tokens else None,
                temperature=request.temperature if request.temperature else 0.7,
            ),
        )
        if (
            not isinstance(completion, Iterator)
            and completion['choices'][-1]['message']['content']
        ):
            return super().createOpenAIChatCompletion(
                request.messages,
                completion['choices'][-1]['message']['content'],
                self._model,
            )
        return None
