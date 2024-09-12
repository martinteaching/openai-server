from llama_cpp import Llama

from openaiserver.llms.llama_3_quant import Llama__3__Quant


class MedLlama__3__8B_Quant(Llama__3__Quant):

    def __init__(self) -> None:
        super().__init__()
        self._model: str = 'MedLlama__3__8B_Quant'
        self._llm: Llama = Llama(
            model_path=self._config.get(self._model, 'PATH'),
            chat_format='llama-3',
        )
