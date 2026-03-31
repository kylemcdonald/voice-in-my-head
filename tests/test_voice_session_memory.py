import pytest

import voice_session


@pytest.mark.asyncio
async def test_convert_experience_to_memory_returns_valid_response(monkeypatch):
    session = voice_session.VoiceSession("test-session")
    session._strings.update({
        "convert_experience_to_memory_prompt": "prompt {entire_transcript}",
        "convert_experience_to_memory_system": "system {goals_prompt}",
        "convert_experience_to_memory_backup": "backup memory",
    })

    async def fake_chatgpt(prompt, system=None, backup=None, max_tokens=1024):
        assert prompt == "prompt We heard anything can happen next in the room"
        assert system == "system "
        assert backup == "backup memory"
        return (
            'In reflecting on what just transpired, one phrase stayed with me: '
            '"anything can happen next". Are you ready? Repeat after me: '
            '"anything can happen next".'
        )

    monkeypatch.setattr(session, "_chatgpt", fake_chatgpt)

    result = await session.convert_experience_to_memory(
        "We heard anything can happen next in the room"
    )

    assert '"anything can happen next"' in result


@pytest.mark.asyncio
async def test_convert_experience_to_memory_rejects_placeholder_response(monkeypatch):
    session = voice_session.VoiceSession("test-session")
    session._strings.update({
        "convert_experience_to_memory_prompt": "prompt {entire_transcript}",
        "convert_experience_to_memory_system": "system {goals_prompt}",
        "convert_experience_to_memory_backup": "backup memory",
    })

    async def fake_chatgpt(prompt, system=None, backup=None, max_tokens=1024):
        return (
            'One phrase stood out to me, "[insert phrase here]". '
            'Repeat after me: "[insert phrase here]".'
        )

    monkeypatch.setattr(session, "_chatgpt", fake_chatgpt)

    result = await session.convert_experience_to_memory(
        "We heard anything can happen next in the room"
    )

    assert result == "backup memory"
