import asyncio

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from openaiserver.llms.llm import LLM
from openaiserver.openaiserver_types import ChatCompletionRequest


class Claude__Haiku(LLM):

    def __init__(self, concurrency: int = 3, timeout: float = 120) -> None:
        self._model: str = 'haiku'
        self.__semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency)
        self.__timeout: float = timeout

    async def __run_claude_code_print(self, input: str) -> str:
        async with self.__semaphore:
            process: asyncio.subprocess.Process = await asyncio.create_subprocess_exec(
                'claude', '--model', self._model, '-p', input, stdout=asyncio.subprocess.PIPE
            )
            try:
                async with asyncio.timeout(self.__timeout):
                    return (await process.communicate())[0].decode().strip()
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    async def __create_chat_completion(self, messages: list[ChatCompletionMessageParam]) -> str:
        return await self.__run_claude_code_print(
            "\n".join(f"{message['role'].capitalize()}: {message['content']}" for message in messages)
        )

    async def getCompletion(self, request: ChatCompletionRequest) -> ChatCompletion | None:
        return super().createOpenAIChatCompletion(
            request.messages,
            await self.__create_chat_completion(request.messages),
            self._model,
        )
