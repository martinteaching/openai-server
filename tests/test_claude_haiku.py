import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from openaiserver.llms.claude_haiku import Claude__Haiku
from openaiserver.openaiserver_types import ChatCompletionRequest


def test_getCompletion_allowedTools_passedToClaude() -> None:
    process: MagicMock = MagicMock(returncode=0, communicate=AsyncMock(return_value=(b'bar', None)))
    with patch(
        'openaiserver.llms.claude_haiku.asyncio.create_subprocess_exec', AsyncMock(return_value=process)
    ) as createSubprocessExec:
        asyncio.run(
            Claude__Haiku(1, 1, 'foo').getCompletion(
                ChatCompletionRequest.model_validate({'model': 'baz', 'messages': [{'role': 'user', 'content': 'qux'}]})
            )
        )
    assert createSubprocessExec.call_args.args[createSubprocessExec.call_args.args.index('--allowedTools') + 1] == 'foo'


def test_getCompletion_defaultTools_readOnly() -> None:
    process: MagicMock = MagicMock(returncode=0, communicate=AsyncMock(return_value=(b'bar', None)))
    with patch(
        'openaiserver.llms.claude_haiku.asyncio.create_subprocess_exec', AsyncMock(return_value=process)
    ) as createSubprocessExec:
        asyncio.run(
            Claude__Haiku().getCompletion(
                ChatCompletionRequest.model_validate({'model': 'baz', 'messages': [{'role': 'user', 'content': 'qux'}]})
            )
        )
    assert createSubprocessExec.call_args.args[createSubprocessExec.call_args.args.index('--allowedTools') + 1] == 'Read'
