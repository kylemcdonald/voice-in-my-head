import pytest

import voice_session


@pytest.mark.asyncio
async def test_handle_listen_followup_prompts_for_incomplete_response(monkeypatch):
    session = voice_session.VoiceSession("test-session")
    session._last_spoken_text = "What do you want your inner voice to sound like?"
    session._strings["continue_incomplete_response"] = "Sorry, continue."

    speak_calls = []

    async def fake_classify(transcript):
        assert transcript == "Hmm, let me think about that."
        return "continue"

    async def fake_listen(mode=None, max_duration=None):
        assert mode == "short"
        assert max_duration is None
        return "I want it to be calmer."

    async def fake_speak(text, use_cache=True, remember_last_spoken=True):
        speak_calls.append((text, use_cache, remember_last_spoken))

    monkeypatch.setattr(session, "_classify_listen_response", fake_classify)
    monkeypatch.setattr(session, "listen", fake_listen)
    monkeypatch.setattr(session, "speak", fake_speak)

    result = await session._handle_listen_followup(
        "Hmm, let me think about that.",
        mode="short",
        max_duration=None,
    )

    assert result == "Hmm, let me think about that. I want it to be calmer."
    assert speak_calls == [("Sorry, continue.", True, False)]
    assert session._last_spoken_text == "What do you want your inner voice to sound like?"


@pytest.mark.asyncio
async def test_handle_listen_followup_repeats_last_question(monkeypatch):
    session = voice_session.VoiceSession("test-session")
    session._last_spoken_text = "Could you tell me more?"

    speak_calls = []

    async def fake_classify(transcript):
        assert transcript == "Sorry?"
        return "repeat"

    async def fake_listen(mode=None, max_duration=None):
        assert mode == "default"
        assert max_duration is None
        return "Here is my real answer."

    async def fake_speak(text, use_cache=True, remember_last_spoken=True):
        speak_calls.append((text, use_cache, remember_last_spoken))

    monkeypatch.setattr(session, "_classify_listen_response", fake_classify)
    monkeypatch.setattr(session, "listen", fake_listen)
    monkeypatch.setattr(session, "speak", fake_speak)

    result = await session._handle_listen_followup(
        "Sorry?",
        mode="default",
        max_duration=None,
    )

    assert result == "Here is my real answer."
    assert speak_calls == [("Could you tell me more?", True, True)]
