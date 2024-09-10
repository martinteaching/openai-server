from openaiserver.llms.llm import LLM
import logging, uvicorn

from fastapi import FastAPI, HTTPException
from openai.types.chat import ChatCompletion

from openaiserver.openaiserver_types import ChatCompletionRequest
from openaiserver.llms.llama_31_8bquantinstruct import Llama__3_1__8B_Quant_Instruct


class OpenAIServer:

    def __init__(self) -> None:
        self.__app: FastAPI = FastAPI()
        self.__logger: logging.Logger = logging.getLogger()
        self.__model: LLM = Llama__3_1__8B_Quant_Instruct()
        self.setupRoutes()

    def getApp(self) -> FastAPI:
        return self.__app

    def setupRoutes(self) -> None:

        @self.__app.post('/v1/chat/completions', response_model=ChatCompletion)
        async def chatCompletions(request: ChatCompletionRequest) -> ChatCompletion:
            try:
                self.__logger.debug(request.messages)
                completion: ChatCompletion | None = self.__model.getCompletion(request)
                if not completion:
                    raise Exception('no output')
                self.__logger.debug(completion)
                return completion
            except Exception as exception:
                self.__logger.error(str(exception))
                raise HTTPException(status_code=500, detail=str(exception))

    def run(self, host: str = '0.0.0.0', port: int = 8080) -> None:
        uvicorn.run(self.__app, host=host, port=port)
