import pytest  # type: ignore
import time, random
from typing import Any, Callable, Generator
from dotenv import load_dotenv

from fastapi.testclient import TestClient
from sqlalchemy import Engine, StaticPool, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

from openaiserver.openaiserver import OpenAIServer
from openaiserver.db import models


@pytest.fixture(scope='session', autouse=True)
def load_env() -> None:
    load_dotenv()


@pytest.fixture(scope='function')
def getDB() -> Callable[[], Generator[Session, None, None]]:
    engine: Engine = create_engine(
        'sqlite:///:memory:',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )
    TestSessionFactory: Callable[[], Session] = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    models.Base.metadata.create_all(bind=engine)

    def override_getDB() -> Generator[Session, None, None]:
        try:
            db: Session = TestSessionFactory()
            yield db
        finally:
            db.close()

    return override_getDB


@pytest.fixture(scope='function')
def client(
    getDB: Callable[[], Generator[Session, None, None]]
) -> Generator[TestClient, None, None]:
    server: OpenAIServer = OpenAIServer()
    server.getApp().dependency_overrides[server.getDB()] = getDB
    with TestClient(server.getApp()) as client:
        yield client
    server.getApp().dependency_overrides.clear()


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
