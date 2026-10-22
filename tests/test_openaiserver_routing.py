from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest  # type: ignore
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion

from openaiserver.openaiserver import OpenAIServer


@pytest.fixture(scope='function')
def models() -> Generator[list[MagicMock], None, None]:
    completion: ChatCompletion = ChatCompletion.model_validate(
        {
            'id': 'foo',
            'object': 'chat.completion',
            'created': 0,
            'model': 'bar',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'baz'}, 'finish_reason': 'stop'}],
        }
    )
    instances: list[MagicMock] = [
        MagicMock(getCompletion=AsyncMock(return_value=completion)),
        MagicMock(getCompletion=AsyncMock(return_value=completion)),
    ]
    with patch('openaiserver.openaiserver.Claude__Haiku', side_effect=instances) as claudeHaiku, patch(
        'openaiserver.openaiserver.ConfigParser'
    ) as configParser:
        configParser.return_value.getboolean.return_value = False
        yield instances
    assert 'mcp__claude_ai_Gmail__get_thread' in claudeHaiku.call_args_list[1].args[2]


def request(model: str) -> int:
    with TestClient(OpenAIServer().getApp()) as client:
        return client.post(
            url='/v1/chat/completions', json={'model': model, 'messages': [{'role': 'user', 'content': 'qux'}]}
        ).status_code


def test_chatCompletions_gmailModel_routesToGmailClaude(models: list[MagicMock]) -> None:
    assert request('claude-gmail') == 200
    models[1].getCompletion.assert_awaited_once()
    models[0].getCompletion.assert_not_awaited()


def test_chatCompletions_otherModel_routesToDefaultClaude(models: list[MagicMock]) -> None:
    assert request('foo') == 200
    models[0].getCompletion.assert_awaited_once()
    models[1].getCompletion.assert_not_awaited()
