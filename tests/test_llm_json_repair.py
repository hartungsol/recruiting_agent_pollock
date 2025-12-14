import pytest

from recruiting_agent_pollock.models.llm_client import LLMClient, LLMResponse, Message


@pytest.mark.asyncio
async def test_chat_with_json_repairs_single_quotes_and_trailing_commas() -> None:
    client = LLMClient()

    async def fake_chat(messages: list[Message], temperature: float = 0.7, max_tokens=None, **kwargs):
        return LLMResponse(
            content="{'a': 1, 'b': 'x',}",
            finish_reason="stop",
            model="test",
        )

    # Monkeypatch instance method
    client.chat = fake_chat  # type: ignore[assignment]

    data = await client.chat_with_json(messages=[Message(role="user", content="hi")])
    assert data == {"a": 1, "b": "x"}


@pytest.mark.asyncio
async def test_chat_with_json_repairs_unquoted_keys_and_fenced_json() -> None:
    client = LLMClient()

    async def fake_chat(messages: list[Message], temperature: float = 0.7, max_tokens=None, **kwargs):
        return LLMResponse(
            content="""```json
            {a: 1, b: true, c: null,}
            ```""",
            finish_reason="stop",
            model="test",
        )

    client.chat = fake_chat  # type: ignore[assignment]

    data = await client.chat_with_json(messages=[Message(role="user", content="hi")])
    assert data == {"a": 1, "b": True, "c": None}
