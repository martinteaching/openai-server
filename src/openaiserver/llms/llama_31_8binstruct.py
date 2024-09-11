from typing import Any

import torch
import transformers  # type: ignore
from transformers.pipelines import TextGenerationPipeline  # type: ignore
from openaiserver.llms.llama3 import Llama3
from openai.types.chat import ChatCompletion

from openaiserver.openaiserver_types import ChatCompletionRequest


class Llama__3_1__8B_Instruct(Llama3):

    def __init__(self) -> None:
        self.__pipeline: TextGenerationPipeline = transformers.pipeline(
            'text-generation',
            model='meta-llama/Meta-Llama-3.1-8B-Instruct',
            model_kwargs={'torch_dtype': torch.bfloat16},
            device_map='auto',
        )

    def getCompletion(self, request: ChatCompletionRequest) -> ChatCompletion | None:
        completion: Any = self.__pipeline(
            request.messages,
            max_new_tokens=request.max_tokens,
        )
        return super().createOpenAIChatCompletion(
            request.messages,
            completion[0]['generated_text'][-1]['content'],
            'Llama__3_1__8B_Instruct',
        )
