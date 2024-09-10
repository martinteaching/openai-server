import pytest  # type: ignore
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from openaiserver.openaiserver import OpenAIServer


@pytest.fixture(scope='session', autouse=True)
def load_env() -> None:
    load_dotenv()


@pytest.fixture
def client() -> TestClient:
    server: OpenAIServer = OpenAIServer()
    return TestClient(server.getApp())


def test_chatCompletions(client: TestClient) -> None:
    response = client.post(
        url='/v1/chat/completions',
        json={
            'model': '',
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
