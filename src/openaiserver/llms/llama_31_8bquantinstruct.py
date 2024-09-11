import configparser, os
from typing import Iterator, cast

from llama_cpp import (
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Llama,
)
from openai.types.chat import ChatCompletion

from openaiserver.llms.llama3 import Llama3
from openaiserver.openaiserver_types import ChatCompletionRequest


class Llama__3_1__8B_Quant_Instruct(Llama3):

    def __init__(self) -> None:
        self.__config = configparser.ConfigParser(os.environ)
        self.__config.read('config/quantized_models.ini')
        self.__llm: Llama = Llama(
            model_path=self.__config.get('LLAMA__31__8B_QUANT_INSTRUCT', 'PATH'),
            chat_format='llama-3',
        )

    def getCompletion(self, request: ChatCompletionRequest) -> ChatCompletion | None:
        completion: (
            CreateChatCompletionResponse | Iterator[CreateChatCompletionStreamResponse]
        ) = self.__llm.create_chat_completion(
            messages=super().formatMessages(request.messages),
            max_tokens=request.max_tokens if request.max_tokens else None,
            temperature=request.temperature if request.temperature else 0.2,
        )
        if (
            not isinstance(completion, Iterator)
            and completion['choices'][-1]['message']['content']
        ):
            return super().createOpenAIChatCompletion(
                request.messages,
                completion['choices'][-1]['message']['content'],
                'Llama__3_1__8B_QUANT_Instruct',
            )
        return None
