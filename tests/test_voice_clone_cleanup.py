import asyncio
import time
from types import SimpleNamespace

import pytest

import voice_session


class FakeVoices:
    def __init__(self, voices=()):
        self.deleted = []
        self._voices = list(voices)

    async def delete(self, voice_id):
        self.deleted.append(voice_id)

    async def get_all(self, show_legacy=False):
        return SimpleNamespace(voices=self._voices)


def fake_elevenlabs(voices=()):
    return SimpleNamespace(voices=FakeVoices(voices))


def voice(voice_id, description, age_seconds, category="cloned"):
    return SimpleNamespace(
        voice_id=voice_id,
        name=voice_id,
        category=category,
        description=description,
        created_at_unix=int(time.time() - age_seconds) if age_seconds is not None else None,
    )


@pytest.mark.asyncio
async def test_delete_cloned_voice_deletes_once():
    session = voice_session.VoiceSession("test-session")
    session._elevenlabs = fake_elevenlabs()
    session._cloned_voice_id = "clone-1"

    await session.delete_cloned_voice()
    await session.delete_cloned_voice()

    assert session._elevenlabs.voices.deleted == ["clone-1"]


@pytest.mark.asyncio
async def test_delete_cloned_voice_waits_for_clone_in_progress():
    session = voice_session.VoiceSession("test-session")
    session._elevenlabs = fake_elevenlabs()

    async def finish_clone():
        await asyncio.sleep(0.05)
        session._cloned_voice_id = "clone-2"

    session._cloning_task = asyncio.create_task(finish_clone())

    await session.delete_cloned_voice()

    assert session._elevenlabs.voices.deleted == ["clone-2"]


@pytest.mark.asyncio
async def test_close_mid_script_cancels_script_and_deletes_clone():
    session = voice_session.VoiceSession("test-session")
    session._elevenlabs = fake_elevenlabs()
    session._cloned_voice_id = "clone-3"
    script_cancelled = asyncio.Event()

    async def endless_script():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            script_cancelled.set()
            raise

    session.start = endless_script
    await session._handle_app_message({"message": "start"})
    await asyncio.sleep(0)

    await session.close()

    assert script_cancelled.is_set()
    assert session._elevenlabs.voices.deleted == ["clone-3"]


@pytest.mark.asyncio
async def test_script_end_deletes_clone():
    session = voice_session.VoiceSession("test-session")
    session._elevenlabs = fake_elevenlabs()

    async def short_script():
        session._cloned_voice_id = "clone-4"

    session.start = short_script
    await session._handle_app_message({"message": "start"})
    await session._script_task

    assert session._elevenlabs.voices.deleted == ["clone-4"]


@pytest.mark.asyncio
async def test_webrtc_failure_closes_websocket():
    session = voice_session.VoiceSession("test-session")
    closed = []

    async def close_ws():
        closed.append(True)

    session.set_close_callback(close_ws)
    await session._on_webrtc_state_change("connected")
    await session._on_webrtc_state_change("failed")

    assert closed == [True]


@pytest.mark.asyncio
async def test_stale_sweep_only_deletes_old_session_clones():
    stale = voice_session.STALE_CLONE_SECONDS + 60
    elevenlabs = fake_elevenlabs([
        voice("old-session", "vimh cloned voice", stale),
        voice("old-legacy-session", "vimh", stale),
        voice("live-session", "vimh cloned voice", 600),
        voice("named-clone", "", stale),
        voice("no-date", "vimh cloned voice", None),
        voice("premade", "vimh", stale, category="premade"),
    ])

    deleted = await voice_session.delete_stale_voice_clones(elevenlabs)

    assert deleted == 2
    assert elevenlabs.voices.deleted == ["old-session", "old-legacy-session"]


@pytest.mark.asyncio
async def test_script_row_passes_empty_variable_as_argument():
    received = []

    class Session:
        _shutdown = False

        async def convert_experience_to_memory(self, transcript):
            received.append(transcript)

    reader = voice_session.AsyncScriptReader.__new__(voice_session.AsyncScriptReader)
    reader.session = Session()
    reader.memory = {"entire_transcript": ""}

    await reader.run_row({"function": "convert_experience_to_memory", "input": "entire_transcript", "output": ""})

    assert received == [""]
