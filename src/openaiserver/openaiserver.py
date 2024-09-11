import logging, uvicorn
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Generator

from fastapi import Depends, FastAPI, HTTPException
from openai.types.chat import ChatCompletion
from sqlalchemy.orm.session import Session

from openaiserver.openaiserver_types import ChatCompletionRequest
from openaiserver.llms.llm import LLM
from openaiserver.llms.llama_31_8bquantinstruct import Llama__3_1__8B_Quant_Instruct
from openaiserver.db.database import Database
from openaiserver.db import models


class OpenAIServer:

    def __init__(self) -> None:
        self.__logger: logging.Logger = logging.getLogger()
        self.__model: LLM = Llama__3_1__8B_Quant_Instruct()
        self.__database: Database = Database()

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            models.Base.metadata.create_all(bind=self.__database.getEngine())
            yield

        self.__app: FastAPI = FastAPI(lifespan=lifespan)
        self.setupRoutes()

    def getApp(self) -> FastAPI:
        return self.__app

    def setupRoutes(self) -> None:

        def getDB() -> Generator[Session, None, None]:
            db: Session = self.__database.getSessionFactory()()
            try:
                yield db
            finally:
                self.__logger.debug('closing db')
                db.close()

        @self.__app.post('/v1/chat/completions', response_model=ChatCompletion)
        async def chatCompletions(
            request: ChatCompletionRequest, db: Session = Depends(getDB)
        ) -> ChatCompletion | None:
            try:
                self.__logger.debug(request)
                completion: ChatCompletion | None
                cachedResponse: models.Cache | None = (
                    db.query(models.Cache)
                    .filter(models.Cache.prompt == request.messages[-1]['content'])
                    .filter(models.Cache.model == request.model)
                    .first()
                )
                if cachedResponse:
                    self.__logger.debug('using cache')
                    completion = self.__model.createOpenAIChatCompletion(
                        request.messages, cachedResponse.response, request.model
                    )
                else:
                    completion = self.__model.getCompletion(request)
                    if not completion:
                        raise Exception('no llm output')
                    cacheEntry: models.Cache = models.Cache(
                        prompt=request.messages[-1]['content'],
                        response=completion.choices[0].message.content,
                        model=request.model,
                    )
                    db.add(cacheEntry)
                    db.commit()
                self.__logger.debug(completion)
                return completion
            except Exception as exception:
                self.__logger.error(str(exception))
                raise HTTPException(status_code=500, detail=str(exception))

    def run(self, host: str = '0.0.0.0', port: int = 8080) -> None:
        uvicorn.run(self.__app, host=host, port=port)
