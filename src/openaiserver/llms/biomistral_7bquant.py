import configparser, os
from typing import Iterator

from llama_cpp import (
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Llama,
)
from openai.types.chat import ChatCompletion


from openaiserver.llms.mistral import Mistral
from openaiserver.openaiserver_types import ChatCompletionRequest


class Biomistral__7B_Quant(Mistral):

    def __init__(self) -> None:
        self._config = configparser.ConfigParser(os.environ)
        self._config.read('config/quantized_models.ini')
        self._model: str = 'Biomistral__7B_Quant'
        self._llm: Llama = Llama(
            model_path=self._config.get(self._model, 'PATH'),
            chat_format='llama-2',
        )

    def getCompletion(self, request: ChatCompletionRequest) -> ChatCompletion | None:
        completion: (
            CreateChatCompletionResponse | Iterator[CreateChatCompletionStreamResponse]
        ) = self._llm.create_chat_completion(
            messages=super().formatMessages(request.messages),
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
