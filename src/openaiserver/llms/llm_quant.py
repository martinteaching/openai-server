from abc import ABC, abstractmethod

from openai.types.chat import ChatCompletionMessageParam

from openaiserver.llms.llm import LLM


class LLM_Quant[T, U](LLM, ABC):

    @abstractmethod
    def getModel(self, path: str) -> T: ...

    @abstractmethod
    def formatMessages(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> list[U]: ...
