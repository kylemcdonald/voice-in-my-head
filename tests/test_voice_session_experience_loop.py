import asyncio
import time

import pytest

import voice_session


@pytest.mark.asyncio
async def test_experience_loop_starts_speak_timeout_during_continuous_speech(monkeypatch):
    session = voice_session.VoiceSession("test-session")
    session._start_time = time.time()
    session._onboarding_end_time = session._start_time

    wait_calls = []
    spoken = []

    async def fake_listen(mode=None, max_duration=None):
        return "visitor keeps talking"

    async def fake_wait_for_silence(*, silence_duration, max_wait):
        wait_calls.append((silence_duration, max_wait, session._is_speaking))
        if len(wait_calls) == 1:
            return True

        session._shutdown = True
        return False

    async def fake_respond_to_overheard(overheard):
        assert overheard == "visitor keeps talking"
        session._is_speaking = True
        session._silence_start_time = None
        return "interrupt anyway"

    async def fake_speak(text):
        spoken.append(text)

    monkeypatch.setattr(session, "listen", fake_listen)
    monkeypatch.setattr(session, "_wait_for_silence", fake_wait_for_silence)
    monkeypatch.setattr(session, "respond_to_overheard", fake_respond_to_overheard)
    monkeypatch.setattr(session, "speak", fake_speak)

    await asyncio.wait_for(session.experience_loop("goals"), timeout=0.5)

    assert wait_calls == [
        (
            voice_session.WAIT_DURATION_SECONDS,
            voice_session.MAX_TURN_TIME_SECONDS - voice_session.TURN_TIME_SECONDS,
            False,
        ),
        (voice_session.WAIT_DURATION_SECONDS, 30.0, True),
    ]
    assert spoken == []
