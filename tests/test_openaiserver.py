import pytest  # type: ignore
import time, random
from typing import Any, Generator
from dotenv import load_dotenv

from fastapi.testclient import TestClient

from openaiserver.openaiserver import OpenAIServer


@pytest.fixture(scope='session', autouse=True)
def load_env() -> None:
    load_dotenv()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    server: OpenAIServer = OpenAIServer()
    with TestClient(server.getApp()) as client:
        yield client


def test_chatCompletions(client: TestClient) -> None:
    response = client.post(
        url='/v1/chat/completions',
        json={
            'model': 'llama_31_8binstruct',
            'messages': [
                {'content': 'You are a helpful assistant.', 'role': 'system'},
                {
                    'content': 'what is an llm?',
                    'role': 'user',
                },
            ],
            'max_tokens': 1024,
            'temperature': 0.7,
        },
    )
    assert response.status_code == 200
    assert 'choices' in response.json()


def test_cache(client: TestClient) -> None:
    content: dict[str, Any] = {
        'messages': [
            {'content': 'You are a helpful assistant.', 'role': 'system'},
            {
                'content': 'what is an llm?',
                'role': 'user',
            },
        ],
        'max_tokens': 1024,
        'temperature': 0.7,
    }
    model: str = str(random.random())
    start: float = time.time()
    client.post(
        url='/v1/chat/completions',
        json=content | {'model': model},
    )
    end: float = time.time()
    elapsed: float = end - start
    start = time.time()
    client.post(
        url='/v1/chat/completions',
        json=content | {'model': model},
    )
    assert (time.time() - start) < elapsed
