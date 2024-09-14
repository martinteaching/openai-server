import configparser, os
from typing import Iterator

from llama_cpp import (
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Llama,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from openaiserver.llms.llama_3_quant import Llama__3__Quant
from openaiserver.openaiserver_types import ChatCompletionRequest
from openaiserver.llms.llm import LLM
from openaiserver.llms.llm_quant import LLM_Quant


class Biomistral__7B_Quant(LLM_Quant[Llama, ChatCompletionRequestMessage]):

    def __init__(self) -> None:
        self._config = configparser.ConfigParser(os.environ)
        self._config.read('config/quantized_models.ini')
        self._model: str = 'Biomistral__7B_Quant'
        self._llm: Llama = self.getModel(self._config.get(self._model, 'PATH'))

    def getModel(self, path: str) -> Llama:
        return Llama(model_path=path, chat_format='llama-2', n_ctx=4096)

    def formatMessages(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionRequestMessage]:
        return Llama__3__Quant().formatMessages(messages)

    def getCompletion(self, request: ChatCompletionRequest) -> ChatCompletion | None:
        completion: (
            CreateChatCompletionResponse | Iterator[CreateChatCompletionStreamResponse]
        ) = self._llm.create_chat_completion(
            messages=self.formatMessages(request.messages),
            max_tokens=request.max_tokens if request.max_tokens else None,
            temperature=request.temperature if request.temperature else 0.7,
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
