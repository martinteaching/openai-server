import asyncio, mimetypes, os, tempfile, urllib.request

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from openaiserver.llms.llm import LLM
from openaiserver.openaiserver_types import ChatCompletionRequest


class Claude__Haiku(LLM):

    def __init__(self, concurrency: int = 3, timeout: float = 120, allowedTools: str = 'Read') -> None:
        self._model: str = 'haiku'
        self.__semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency)
        self.__timeout: float = timeout
        self.__allowedTools: str = allowedTools

    async def __run_claude_code_print(self, input: str, directory: str) -> str:
        async with self.__semaphore:
            process: asyncio.subprocess.Process = await asyncio.create_subprocess_exec(
                'claude', '--model', self._model, '--add-dir', directory, '--allowedTools', self.__allowedTools, '-p', input,
                stdout=asyncio.subprocess.PIPE,
            )
            try:
                async with asyncio.timeout(self.__timeout):
                    return (await process.communicate())[0].decode().strip()
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    @staticmethod
    def __get_text(message: ChatCompletionMessageParam) -> str:
        return (
            content
            if isinstance(content := message.get('content'), str)
            else ' '.join(part['text'] for part in content or [] if part['type'] == 'text')
        )

    @staticmethod
    def __get_image_url(message: ChatCompletionMessageParam) -> str | None:
        return (
            None
            if isinstance(content := message.get('content'), str)
            else next((part['image_url']['url'] for part in content or [] if part['type'] == 'image_url'), None)
        )

    @staticmethod
    def __write_image(url: str, path: str) -> str:
        with urllib.request.urlopen(url) as response:
            imagePath: str = path + (mimetypes.guess_extension(response.headers.get_content_type()) or '')
            with open(imagePath, 'wb') as file:
                file.write(response.read())
        return imagePath

    async def __create_chat_completion(self, messages: list[ChatCompletionMessageParam]) -> str:
        imageUrl: str | None = Claude__Haiku.__get_image_url(messages[-1])
        with tempfile.TemporaryDirectory() as directory:
            return await self.__run_claude_code_print(
                (
                    'Read this image before responding: '
                    + await asyncio.to_thread(Claude__Haiku.__write_image, imageUrl, os.path.join(directory, 'image'))
                    + '\n\n'
                    if imageUrl
                    else ''
                )
                + "\n".join(
                    f"{message['role'].capitalize()}: {Claude__Haiku.__get_text(message)}" for message in messages
                ),
                directory,
            )

    async def getCompletion(self, request: ChatCompletionRequest) -> ChatCompletion | None:
        return super().createOpenAIChatCompletion(
            request.messages,
            await self.__create_chat_completion(request.messages),
            self._model,
        )
