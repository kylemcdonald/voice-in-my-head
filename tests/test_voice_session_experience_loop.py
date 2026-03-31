import asyncio
import pytest

import voice_session


@pytest.mark.asyncio
async def test_experience_loop_speaks_immediately_once_force_interrupt_deadline_is_reached(monkeypatch):
    session = voice_session.VoiceSession("test-session")
    now = 1_000.0

    def fake_time():
        return now

    monkeypatch.setattr(voice_session.time, "time", fake_time)

    session._start_time = fake_time()
    session._onboarding_end_time = session._start_time

    wait_calls = []
    spoken = []

    async def fake_listen(mode=None, max_duration=None):
        nonlocal now
        now += max_duration
        return "visitor keeps talking"

    async def fake_wait_for_silence(*, silence_duration, max_wait):
        nonlocal now
        wait_calls.append((silence_duration, max_wait, session._is_speaking))
        now += max_wait
        return False

    async def fake_respond_to_overheard(overheard):
        nonlocal now
        assert overheard == "visitor keeps talking"
        session._is_speaking = True
        session._silence_start_time = None
        now += 0.5
        return "interrupt anyway"

    async def fake_speak(text):
        spoken.append(text)
        session._shutdown = True

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
        )
    ]
    assert spoken == ["interrupt anyway"]


@pytest.mark.asyncio
async def test_experience_loop_reuses_remaining_interrupt_budget_before_speaking(monkeypatch):
    session = voice_session.VoiceSession("test-session")
    now = 2_000.0

    def fake_time():
        return now

    monkeypatch.setattr(voice_session.time, "time", fake_time)

    session._start_time = fake_time()
    session._onboarding_end_time = session._start_time

    wait_calls = []
    spoken = []

    async def fake_listen(mode=None, max_duration=None):
        nonlocal now
        now += max_duration
        return "visitor pauses briefly"

    async def fake_wait_for_silence(*, silence_duration, max_wait):
        nonlocal now
        wait_calls.append((silence_duration, max_wait, session._is_speaking))
        if len(wait_calls) == 1:
            now += 5.0
            return True

        now += max_wait
        return False

    async def fake_respond_to_overheard(overheard):
        nonlocal now
        assert overheard == "visitor pauses briefly"
        session._is_speaking = True
        session._silence_start_time = None
        now += 1.0
        return "speak after remaining budget"

    async def fake_speak(text):
        spoken.append(text)
        session._shutdown = True

    monkeypatch.setattr(session, "listen", fake_listen)
    monkeypatch.setattr(session, "_wait_for_silence", fake_wait_for_silence)
    monkeypatch.setattr(session, "respond_to_overheard", fake_respond_to_overheard)
    monkeypatch.setattr(session, "speak", fake_speak)

    await asyncio.wait_for(session.experience_loop("goals"), timeout=0.5)

    assert wait_calls[0] == (
        voice_session.WAIT_DURATION_SECONDS,
        voice_session.MAX_TURN_TIME_SECONDS - voice_session.TURN_TIME_SECONDS,
        False,
    )
    assert wait_calls[1][0] == voice_session.WAIT_DURATION_SECONDS
    assert wait_calls[1][1] == pytest.approx(34.0)
    assert wait_calls[1][2] is True
    assert spoken == ["speak after remaining budget"]
