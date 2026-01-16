"""
Voice session manager - async replacement for VoiceInMyHead class.

Manages a complete voice interaction session:
- WebRTC audio connection
- AssemblyAI transcription (with configurable endpointing)
- ElevenLabs TTS
- OpenAI ChatGPT conversation
- Script execution
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import time
import wave
from pathlib import Path
from typing import Callable, Optional, Awaitable, List, Dict, Any

import aiofiles
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from elevenlabs import AsyncElevenLabs, VoiceSettings

from webrtc_handler import WebRTCSession
from assemblyai_stream import AssemblyAIStream, TranscriptMessage
from audio_tracks import (
    AudioOutputTrack,
    AudioInputHandler,
    decode_mp3_to_pcm,
    SAMPLE_RATE,
    CHANNELS,
    SAMPLE_WIDTH,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration from environment
TOTAL_TIME_MINUTES = int(os.getenv("TOTAL_TIME_MINUTES", "25"))
TURN_TIME_SECONDS = int(os.getenv("TURN_TIME_SECONDS", "50"))
WAIT_DURATION_SECONDS = float(os.getenv("WAIT_DURATION_SECONDS", "2"))
MAX_TURN_TIME_SECONDS = int(os.getenv("MAX_TURN_TIME_SECONDS", "90"))

# Listen mode parameters for AssemblyAI turn detection
# See: https://www.assemblyai.com/docs/universal-streaming/turn-detection
LISTEN_PARAMS = {
    "short": {
        "end_of_turn_confidence": 0.4,
        "min_silence_confident_ms": 160,
        "max_turn_silence_ms": 400,
    },
    "long": {
        "end_of_turn_confidence": 0.7,
        "min_silence_confident_ms": 800,
        "max_turn_silence_ms": 3600,
    },
    "default": {
        "end_of_turn_confidence": 0.4,
        "min_silence_confident_ms": 400,
        "max_turn_silence_ms": 1280,
    },
}


class AsyncSrtWriter:
    """Async version of SrtWriter for transcript logging."""

    def __init__(self, session_id: str):
        self.index = 0
        self.start: Optional[float] = None
        self.fn = f"transcripts/{session_id}.srt"

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        import math
        dec, whole = math.modf(seconds)
        dec = round(dec * 1000)
        m, s = divmod(int(whole), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{dec:03d}"

    async def write(self, begin: float, end: float, text: str) -> None:
        if self.start is None:
            self.start = begin

        begin_stamp = self._seconds_to_srt_time(begin - self.start)
        end_stamp = self._seconds_to_srt_time(end - self.start)

        os.makedirs(os.path.dirname(self.fn), exist_ok=True)

        async with aiofiles.open(self.fn, "a") as f:
            await f.write(f"{self.index}\n")
            await f.write(f"{begin_stamp} --> {end_stamp}\n")
            await f.write(f"{text}\n")
            await f.write("\n")
            self.index += 1


class AsyncScriptReader:
    """Async script reader and executor."""

    def __init__(self, script_fn: str, language: str, session: "VoiceSession"):
        self.session = session
        self.language = language
        self.memory: Dict[str, Any] = {}
        self.rows: List[Dict[str, str]] = []
        self._load_script(script_fn)

    def _load_script(self, script_fn: str) -> None:
        """Load and preprocess the script."""
        import csv

        with open(script_fn) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Preprocess: combine consecutive speak rows, handle localization
        processed = []
        prev_speak = None

        for row in rows:
            # Handle localization
            if self.language != "en" and self.language in row and row[self.language]:
                row["input"] = row[self.language]

            if row["function"] == "speak":
                if prev_speak is None:
                    prev_speak = row
                else:
                    prev_speak["input"] = f"{prev_speak['input']} {row['input']}"
            else:
                if prev_speak:
                    processed.append(prev_speak)
                    prev_speak = None
                processed.append(row)

        if prev_speak:
            processed.append(prev_speak)

        self.rows = processed
        logger.info(f"Loaded script with {len(self.rows)} rows")

    async def get_memory(self, name: str) -> Any:
        """Get a variable from memory, awaiting if it's a task."""
        value = self.memory.get(name)
        if asyncio.isfuture(value) or asyncio.iscoroutine(value):
            self.memory[name] = await value
        elif isinstance(value, asyncio.Task):
            self.memory[name] = await value
        return self.memory.get(name)

    async def run_row(self, row: Dict[str, str]) -> None:
        """Execute a single script row."""
        raw_function = row.get("function", "")
        raw_input = row.get("input", "")
        output = row.get("output", "")

        if not raw_function:
            return

        # Handle async prefix
        is_async = raw_function.startswith("async ")
        function_name = raw_function[6:] if is_async else raw_function

        # Get the method
        method = getattr(self.session, function_name, None)
        if not method:
            logger.warning(f"Unknown function: {function_name}")
            return

        # Process input
        input_val = raw_input

        # Try to convert to number
        try:
            input_val = float(raw_input)
        except (ValueError, TypeError):
            pass

        # Look up variable reference
        if isinstance(input_val, str) and input_val in self.memory:
            input_val = await self.get_memory(input_val)

        # Replace {variable} placeholders
        if isinstance(input_val, str):
            for key in self.memory:
                if f"{{{key}}}" in input_val:
                    val = await self.get_memory(key)
                    input_val = input_val.replace(f"{{{key}}}", str(val))

        # Execute
        logger.info(f"Script: {raw_function}({raw_input[:50] if raw_input else ''})")

        if is_async:
            # Run as background task
            if input_val:
                task = asyncio.create_task(method(input_val))
            else:
                task = asyncio.create_task(method())
            if output:
                self.memory[output] = task
        else:
            # Run synchronously
            if input_val:
                result = await method(input_val) if asyncio.iscoroutinefunction(method) else method(input_val)
            else:
                result = await method() if asyncio.iscoroutinefunction(method) else method()

            if output:
                self.memory[output] = result

    async def run(self) -> None:
        """Execute the entire script."""
        for row in self.rows:
            if self.session._shutdown:
                break
            try:
                await self.run_row(row)
            except Exception as e:
                logger.error(f"Script error: {e}")


class VoiceSession:
    """
    Main session class managing a complete voice interaction.

    Replaces VoiceInMyHead with an async-first design.
    """

    def __init__(self, session_id: str, config: Optional[Dict] = None):
        self.session_id = session_id
        self.config = config or {}

        # API clients
        self._openai: Optional[AsyncOpenAI] = None
        self._elevenlabs: Optional[AsyncElevenLabs] = None

        # WebRTC and signaling
        self._webrtc: Optional[WebRTCSession] = None
        self._send_message: Optional[Callable[[str], Awaitable[None]]] = None

        # AssemblyAI transcription
        self._transcription: Optional[AssemblyAIStream] = None

        # Session state
        self._started = False
        self._shutdown = False
        self._start_time: Optional[float] = None

        # Voice settings
        self._default_voice = self.config.get("voice", "Mimi")
        self._voice_list: List[Any] = []
        self._name_to_voice: Dict[str, str] = {}
        self._cloned_voice_id: Optional[str] = None

        # Transcription handling
        self._speech_queue: asyncio.Queue[str] = asyncio.Queue()
        self._last_utterance_time: Optional[float] = None

        # Speech state tracking for wait-for-silence logic
        self._is_speaking = False
        self._silence_start_time: Optional[float] = None
        self._utterance_ended = asyncio.Event()  # Set when AssemblyAI detects end of turn

        # Recording for voice cloning
        self._recording = False
        self._recorded_audio = bytearray()
        self._cloning_task: Optional[asyncio.Task] = None

        # Conversation state
        self._messages: List[Dict[str, str]] = []
        self._collecting_messages = False
        self._collected_messages: List[Dict[str, str]] = []

        # SRT writer
        self._srt_writer: Optional[AsyncSrtWriter] = None

        # Script reader
        self._script_reader: Optional[AsyncScriptReader] = None

        # Goals prompt for experience loop
        self._goals_prompt = ""

        # ChatGPT localized strings
        self._strings: Dict[str, str] = {}
        self._language = self.config.get("language", "en")

    def set_send_callback(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Set the WebSocket send callback."""
        self._send_message = callback

    async def _send_app_message(self, data: Dict) -> None:
        """Send an app message to the client."""
        if self._send_message:
            msg = {"type": "app-message", "data": data}
            logger.info(f"Sending app message: {data}")
            await self._send_message(json.dumps(msg))

    async def start_webrtc(self) -> None:
        """Initialize and start WebRTC connection."""
        self._webrtc = WebRTCSession(self.session_id)
        self._webrtc.set_send_callback(self._send_message)

        # Set up audio track callback
        self._webrtc.webrtc._on_audio_track = self._on_audio_track

        await self._webrtc.start()

    async def _on_audio_track(self, track) -> None:
        """Handle incoming audio track from browser."""
        logger.info("Audio track received from browser")

        # Start AssemblyAI transcription connection
        await self._start_transcription()

        # Connect input handler to transcription stream
        if self._webrtc and self._webrtc.webrtc.input_handler:
            self._webrtc.webrtc.input_handler.set_transcription_stream(self._transcription)

    async def _start_transcription(self) -> None:
        """Initialize AssemblyAI streaming connection."""
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            logger.error("ASSEMBLYAI_API_KEY not set")
            return

        # Configurable endpointing thresholds (less sensitive defaults)
        end_of_turn_confidence = float(os.getenv("END_OF_TURN_CONFIDENCE", "0.5"))
        min_silence_confident_ms = int(os.getenv("MIN_SILENCE_CONFIDENT_MS", "600"))
        max_turn_silence_ms = int(os.getenv("MAX_TURN_SILENCE_MS", "1500"))

        self._transcription = AssemblyAIStream(
            api_key=api_key,
            sample_rate=SAMPLE_RATE,
            end_of_turn_confidence=end_of_turn_confidence,
            min_silence_confident_ms=min_silence_confident_ms,
            max_turn_silence_ms=max_turn_silence_ms,
            on_transcript=self._on_transcript,
            on_utterance_end=self._on_utterance_end,
            # Note: on_speech_started available but not used for UI - we detect
            # speech start from transcript reception which is more reliable.
        )

        await self._transcription.connect()
        logger.info("AssemblyAI connected")

    async def _on_transcript(self, msg: TranscriptMessage) -> None:
        """Handle transcript from AssemblyAI."""
        text = msg.text.strip()
        if not text:
            return

        # Mark as speaking when we receive ANY transcript text (partial or final)
        # This triggers the UI indicator immediately when speech is detected
        if not self._is_speaking:
            logger.info("Visitor started speaking (transcript received)")
            self._is_speaking = True
            self._silence_start_time = None
            await self._send_app_message({"event": "user-speaking", "speaking": True})

        # Skip further processing for interim results
        if not msg.is_final:
            return

        logger.info(f"Transcript: {text}")

        # Add to speech queue
        await self._speech_queue.put(text)
        self._last_utterance_time = time.time()

        # Log to SRT
        if self._srt_writer:
            await self._srt_writer.write(
                time.time() - msg.duration,
                time.time(),
                f"(User)\n{text}"
            )

        # Send to client
        await self._send_app_message({
            "event": "chat-msg",
            "message": text,
            "user": "user",
        })

        # Add to conversation
        self._messages.append({"role": "user", "content": text})
        if self._collecting_messages:
            self._collected_messages.append({"role": "user", "content": text})

    async def _on_utterance_end(self) -> None:
        """Handle utterance end (silence detected)."""
        if not self._is_speaking:
            # Already not speaking, ignore duplicate event
            return
        logger.info("Visitor finished speaking")
        self._is_speaking = False
        self._silence_start_time = time.time()
        self._utterance_ended.set()  # Signal listen() that turn ended
        # Notify client to hide speaking indicator
        await self._send_app_message({"event": "user-speaking", "speaking": False})

    async def _on_speech_started(self) -> None:
        """
        Handle speech started (visitor began talking).

        NOTE: This callback is no longer used for UI indication because AssemblyAI's
        SpeechStarted event is too sensitive (triggers on breathing, background noise).
        Speech detection is now done in _on_transcript when actual text is received.
        This method is kept for potential future use or debugging.
        """
        # Log for debugging only - don't update UI state
        logger.debug("AssemblyAI SpeechStarted event (ignored for UI)")

    async def _wait_for_silence(self, silence_duration: float, max_wait: float) -> bool:
        """
        Wait for continuous silence before interjecting.

        Args:
            silence_duration: Required silence duration in seconds
            max_wait: Maximum time to wait in seconds

        Returns:
            True if silence was achieved, False if max_wait was reached
        """
        start_time = time.time()
        logger.info(f"Waiting for {silence_duration}s of silence (max {max_wait}s)")

        while not self._shutdown:
            elapsed = time.time() - start_time

            if elapsed >= max_wait:
                logger.info(
                    f"MAX_TURN_TIME reached ({max_wait + TURN_TIME_SECONDS}s total), "
                    "interjecting anyway"
                )
                return False

            if not self._is_speaking and self._silence_start_time is not None:
                silence_elapsed = time.time() - self._silence_start_time
                if silence_elapsed >= silence_duration:
                    logger.info(f"WAIT_DURATION reached ({silence_duration}s of silence)")
                    return True

            await asyncio.sleep(0.1)

        return False

    async def handle_signaling_message(self, message: str) -> None:
        """Handle incoming WebSocket signaling message."""
        if self._webrtc:
            await self._webrtc.handle_message(message)

        # Also handle app messages
        try:
            data = json.loads(message)
            if data.get("type") == "app-message":
                await self._handle_app_message(data.get("data", {}))
        except json.JSONDecodeError:
            pass

    async def _handle_app_message(self, data: Dict) -> None:
        """Handle app messages from client."""
        msg = data.get("message", "")

        if msg == "start":
            await self.start()
        elif msg == "end":
            await self.end()

    async def start(self) -> None:
        """Start the voice session (called when user taps Start)."""
        if self._started:
            return

        self._started = True
        self._start_time = time.time()
        self._shutdown = False

        logger.info(f"Starting session {self.session_id}")

        # Initialize clients
        self._openai = AsyncOpenAI()
        self._elevenlabs = AsyncElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

        # Load voice list
        await self._load_voices()

        # Load ChatGPT strings
        await self._load_chatgpt_strings()

        # Initialize SRT writer
        self._srt_writer = AsyncSrtWriter(self.session_id)

        # Load and run script
        script_name = self.config.get("script", "script.csv")
        script_path = f"scripts/{script_name}"

        self._script_reader = AsyncScriptReader(script_path, self._language, self)
        await self._script_reader.run()

    async def _load_voices(self) -> None:
        """Load available voices from ElevenLabs."""
        try:
            response = await self._elevenlabs.voices.get_all(show_legacy=True)
            self._voice_list = response.voices

            for voice in self._voice_list:
                self._name_to_voice[voice.name] = voice.voice_id
                # Partial name lookup
                short_name = voice.name.split(" - ")[0].split()[0] if " - " in voice.name else voice.name.split()[0]
                if short_name not in self._name_to_voice:
                    self._name_to_voice[short_name] = voice.voice_id

            logger.info(f"Loaded {len(self._voice_list)} voices")
        except Exception as e:
            logger.error(f"Failed to load voices: {e}")

    async def _load_chatgpt_strings(self) -> None:
        """Load localized strings for ChatGPT prompts."""
        import csv

        try:
            with open("scripts/chatgpt.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row.get("Key", "")
                    value = row.get(self._language, row.get("en", ""))
                    self._strings[key] = value
        except Exception as e:
            logger.error(f"Failed to load ChatGPT strings: {e}")

    async def end(self) -> None:
        """End the voice session."""
        logger.info(f"Ending session {self.session_id}")
        self._shutdown = True

    async def close(self) -> None:
        """Clean up all resources."""
        self._shutdown = True

        if self._transcription:
            await self._transcription.close()
            self._transcription = None

        if self._webrtc:
            await self._webrtc.close()
            self._webrtc = None

        logger.info(f"Session {self.session_id} closed")

    # ========== Script Functions ==========

    def set_voice(self, voice: str) -> None:
        """Set the default voice for TTS."""
        self._default_voice = voice
        logger.info(f"Voice set to: {voice}")

    async def speak(self, text: str, use_cache: bool = True) -> None:
        """Speak text using ElevenLabs TTS."""
        text = text.replace("'", "'")  # Fix apostrophe issues

        logger.info(f"Speaking: {text[:50]}...")

        # Send to client chat
        await self._send_app_message({
            "event": "chat-msg",
            "message": text,
            "user": "voice",
        })

        # Get voice ID
        voice = self._default_voice
        voice_id = voice
        if voice in self._name_to_voice:
            voice_id = self._name_to_voice[voice]

        # Check cache
        cache_hash = hashlib.sha256((voice + text).encode()).hexdigest()
        cache_path = Path(f"cache/{voice}/{cache_hash}.mp3")

        speak_start = time.time()

        if use_cache and cache_path.exists():
            # Read from cache
            async with aiofiles.open(cache_path, "rb") as f:
                mp3_data = await f.read()
        else:
            # Generate from ElevenLabs
            mp3_chunks = []
            async for chunk in self._elevenlabs.text_to_speech.stream(
                text=text,
                voice_id=voice_id,
                model_id="eleven_turbo_v2_5",
                optimize_streaming_latency=3,
            ):
                mp3_chunks.append(chunk)

            mp3_data = b"".join(mp3_chunks)

            # Save to cache
            if use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(cache_path, "wb") as f:
                    await f.write(mp3_data)

        # Decode MP3 to PCM
        pcm_data = decode_mp3_to_pcm(mp3_data)

        # Send to WebRTC output track
        if self._webrtc and self._webrtc.webrtc.output_track:
            # Split into chunks and queue
            chunk_size = SAMPLE_RATE * SAMPLE_WIDTH // 10  # 100ms chunks
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                await self._webrtc.webrtc.output_track.add_audio(chunk)

            # Wait for playback to complete (approximate)
            duration = len(pcm_data) / (SAMPLE_RATE * SAMPLE_WIDTH)
            await asyncio.sleep(duration)

        speak_end = time.time()

        # Log to SRT
        if self._srt_writer:
            await self._srt_writer.write(speak_start, speak_end, f"(Voice)\n{text}")

        # Add to conversation
        self._messages.append({"role": "assistant", "content": text})
        if self._collecting_messages:
            self._collected_messages.append({"role": "assistant", "content": text})

    async def play_sound(self, name: str) -> None:
        """Play a sound effect."""
        sound_path = Path(f"sound/{name}.wav")
        if not sound_path.exists():
            logger.warning(f"Sound not found: {name}")
            return

        # Read WAV file
        with wave.open(str(sound_path)) as f:
            audio = f.readframes(-1)

        # Send to output track
        if self._webrtc and self._webrtc.webrtc.output_track:
            await self._webrtc.webrtc.output_track.add_audio(audio)

            # Wait for playback
            duration = len(audio) / (SAMPLE_RATE * SAMPLE_WIDTH)
            await asyncio.sleep(duration)

    async def listen(
        self,
        mode: str = None,
        max_duration: Optional[float] = None,
    ) -> str:
        """
        Listen for user speech.

        Args:
            mode: Listen mode - "short", "long", or None for default.
                  Controls AssemblyAI turn detection sensitivity.
                  - short: Aggressive turn detection for brief responses
                  - long: Conservative turn detection for longer responses
                  - default/None: Balanced settings
            max_duration: If specified, listen for this fixed duration in seconds
                         (used by experience_loop). Ignores turn detection.

        Returns:
            Transcribed text
        """
        # Only update endpointing for turn-detection mode (scripted calls)
        # For max_duration mode (experience_loop), keep existing parameters to avoid reconnect
        if max_duration is None:
            params = LISTEN_PARAMS.get(mode, LISTEN_PARAMS["default"])
            if self._transcription:
                await self._transcription.update_endpointing(**params)

        # Clear utterance ended event and speech queue
        self._utterance_ended.clear()
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Only play sound for turn-detection mode (not fixed duration)
        if max_duration is None:
            await self.play_sound("listen-begin")

        start_time = time.time()
        transcripts = []

        while not self._shutdown:
            if max_duration is not None:
                # Fixed duration mode - listen for specified time
                if time.time() - start_time >= max_duration:
                    break
            else:
                # Turn detection mode - end when AssemblyAI signals end of turn
                if self._utterance_ended.is_set() and transcripts:
                    # Drain any remaining transcripts from queue
                    while not self._speech_queue.empty():
                        try:
                            text = self._speech_queue.get_nowait()
                            transcripts.append(text)
                        except asyncio.QueueEmpty:
                            break
                    break

            # Try to get transcript
            try:
                text = await asyncio.wait_for(self._speech_queue.get(), timeout=0.1)
                transcripts.append(text)
            except asyncio.TimeoutError:
                pass

        if max_duration is None:
            await self.play_sound("listen-end")

        result = " ".join(transcripts)
        logger.info(f"Listened: {result[:100]}...")
        return result

    async def wait(self, seconds: float) -> None:
        """Wait for specified duration."""
        await asyncio.sleep(seconds)

    # ========== Recording Functions ==========

    def start_recording(self) -> None:
        """Start recording audio for voice cloning."""
        if self._webrtc and self._webrtc.webrtc.input_handler:
            self._webrtc.webrtc.input_handler.start_recording()
        self._recording = True

    def stop_recording(self) -> None:
        """Stop recording audio."""
        if self._webrtc and self._webrtc.webrtc.input_handler:
            self._recorded_audio = self._webrtc.webrtc.input_handler.stop_recording()
        self._recording = False

    async def clone_voice(self) -> str:
        """Clone voice from recorded audio."""
        if not self._recorded_audio:
            logger.warning("No audio recorded for cloning")
            return ""

        logger.info(f"Cloning voice from {len(self._recorded_audio)} bytes")

        # Save as WAV temporarily
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            with wave.open(wav_file, "wb") as w:
                w.setnchannels(CHANNELS)
                w.setsampwidth(SAMPLE_WIDTH)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(bytes(self._recorded_audio))
            wav_path = wav_file.name

        # Convert to MP3 for ElevenLabs
        mp3_path = wav_path.replace(".wav", ".mp3")
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-b:a", "192k", mp3_path
        ], capture_output=True)

        # Clone with ElevenLabs IVC (Instant Voice Cloning)
        from datetime import datetime
        voice_name = datetime.now().isoformat()

        with open(mp3_path, "rb") as f:
            voice = await self._elevenlabs.voices.ivc.create(
                name=voice_name,
                files=[f],
                description="vimh cloned voice",
            )

        logger.info(f"Voice cloned successfully: {voice.voice_id}")

        # Set cloned voice ID immediately so stop_cloning can return it
        # even if subsequent operations fail
        self._cloned_voice_id = voice.voice_id

        # Set voice settings (API may vary by SDK version) - non-critical
        try:
            await self._elevenlabs.voices.settings.update(
                voice_id=voice.voice_id,
                settings=VoiceSettings(
                    stability=0.4,
                    similarity_boost=1.0,
                    style=0.4,
                    use_speaker_boost=True,
                ),
            )
            logger.info("Voice settings updated successfully")
        except Exception as e:
            logger.warning(f"Could not update voice settings (non-critical): {e}")

        # Cleanup temp files
        try:
            os.unlink(wav_path)
            os.unlink(mp3_path)
        except Exception as e:
            logger.warning(f"Could not cleanup temp files: {e}")

        return voice.voice_id

    def start_cloning(self) -> None:
        """Start async voice cloning task."""
        self._cloning_task = asyncio.create_task(self._do_clone())

    async def _do_clone(self) -> None:
        """Internal cloning task with error handling."""
        try:
            await self.clone_voice()
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            self._cloned_voice_id = ""  # Empty string signals failure

    async def stop_cloning(self) -> str:
        """Wait for cloning to complete and return voice ID."""
        # Wait for cloning with timeout
        timeout = 60  # 60 second timeout
        start = time.time()
        while self._cloned_voice_id is None:
            if time.time() - start > timeout:
                logger.error("Voice cloning timed out")
                return ""
            await asyncio.sleep(0.1)
        return self._cloned_voice_id

    # ========== Message Collection ==========

    def start_collecting_messages(self) -> None:
        """Start collecting conversation messages."""
        self._collecting_messages = True
        self._collected_messages = []

    def get_collected_messages(self) -> List[Dict[str, str]]:
        """Get collected messages and stop collecting."""
        messages = self._collected_messages.copy()
        self._collecting_messages = False
        self._collected_messages = []
        return messages

    # ========== ChatGPT Functions ==========

    async def _chatgpt(
        self,
        prompt: str,
        system: Optional[str] = None,
        backup: Optional[str] = None,
    ) -> str:
        """Call ChatGPT with prompt and optional system message."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"ChatGPT error: {e}")
            return backup or ""

    def _build_dialog(
        self,
        messages: List[Dict[str, str]],
        mapping: Optional[Dict[str, str]] = None,
    ) -> str:
        """Build dialog string from messages."""
        if mapping is None:
            mapping = {"user": "Patient", "assistant": "Therapist"}

        lines = []
        for m in messages:
            role = mapping.get(m["role"], m["role"])
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    async def convert_response_to_name(self, response: str) -> str:
        """Extract name from response."""
        if not response:
            return self._strings.get("convert_response_to_name_backup", "friend")

        return await self._chatgpt(
            self._strings.get("convert_response_to_name_prompt", "").format(response=response),
            system=self._strings.get("convert_response_to_name_system"),
            backup=self._strings.get("convert_response_to_name_backup", "friend"),
        )

    async def convert_existing_to_summary(self, messages: List[Dict[str, str]]) -> str:
        """Summarize existing voice description."""
        dialog = self._build_dialog(messages)
        return await self._chatgpt(
            self._strings.get("convert_existing_to_summary_prompt", "").format(dialog=dialog),
            system=self._strings.get("convert_existing_to_summary_system"),
            backup=self._strings.get("convert_existing_to_summary_backup", ""),
        )

    async def convert_goals_to_summary(self, messages: List[Dict[str, str]]) -> str:
        """Summarize user goals."""
        dialog = self._build_dialog(messages)
        return await self._chatgpt(
            self._strings.get("convert_goals_to_summary_prompt", "").format(dialog=dialog),
            system=self._strings.get("convert_goals_to_summary_system"),
            backup=self._strings.get("convert_goals_to_summary_backup", ""),
        )

    async def convert_goals_to_summary_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert goals to a prompt for the experience loop."""
        dialog = self._build_dialog(messages)
        return await self._chatgpt(
            self._strings.get("convert_goals_to_summary_prompt_prompt", "").format(dialog=dialog),
            system=self._strings.get("convert_goals_to_summary_prompt_system"),
            backup=self._strings.get("convert_goals_to_summary_prompt_backup", ""),
        )

    async def respond_to_overheard(self, overheard: str) -> str:
        """Generate response to overheard conversation."""
        import random

        if not overheard:
            kinds = self._strings.get("respond_to_overheard_kinds", "question,observation").split(",")
            verbs = self._strings.get("respond_to_overheard_verbs", "ask,say").split(",")
            kind = random.choice(kinds)
            verb = random.choice(verbs)

            return await self._chatgpt(
                self._strings.get("respond_to_overheard_empty_prompt", "").format(kind=kind, verb=verb),
                system=self._strings.get("respond_to_overheard_system", "").format(goals_prompt=self._goals_prompt),
                backup=self._strings.get("respond_to_overheard_backup", ""),
            )

        return await self._chatgpt(
            self._strings.get("respond_to_overheard_prompt", "").format(overheard=overheard),
            system=self._strings.get("respond_to_overheard_system", "").format(goals_prompt=self._goals_prompt),
            backup=self._strings.get("respond_to_overheard_backup", ""),
        )

    async def convert_experience_to_memory(self, transcript: str) -> str:
        """Convert experience transcript to memory."""
        return await self._chatgpt(
            self._strings.get("convert_experience_to_memory_prompt", "").format(entire_transcript=transcript),
            system=self._strings.get("convert_experience_to_memory_system", "").format(goals_prompt=self._goals_prompt),
            backup=self._strings.get("convert_experience_to_memory_backup", ""),
        )

    async def experience_loop(self, goals_prompt: str) -> str:
        """
        Main experience loop - listen and respond periodically.

        Behavior:
        1. Listen for TURN_TIME_SECONDS
        2. After turn time, wait for WAIT_DURATION_SECONDS of silence
        3. Generate ChatGPT response
        4. Wait for silence again before speaking (to avoid interrupting)
        5. Speak the response

        Args:
            goals_prompt: The prompt describing voice goals

        Returns:
            Complete transcript of the experience
        """
        self._goals_prompt = goals_prompt
        max_total_time = TOTAL_TIME_MINUTES * 60
        turn_time = TURN_TIME_SECONDS
        wait_duration = WAIT_DURATION_SECONDS
        max_turn_time = MAX_TURN_TIME_SECONDS

        entire_transcript = []

        # Use "short" mode turn detection for responsive silence detection
        params = LISTEN_PARAMS["short"]
        if self._transcription:
            await self._transcription.update_endpointing(**params)

        while not self._shutdown:
            elapsed = time.time() - self._start_time
            if elapsed > max_total_time:
                break

            try:
                # Reset speech state at start of turn
                self._is_speaking = False
                self._silence_start_time = time.time()

                logger.info(f"Starting new turn (listening for {turn_time}s)")
                overheard = await self.listen(max_duration=turn_time)
                entire_transcript.append(overheard)

                if self._shutdown:
                    break

                # Phase 2: Wait for silence before generating response
                logger.info(f"TURN_TIME reached ({turn_time}s), entering wait-for-silence phase")

                # Calculate remaining time for wait phase
                remaining_for_silence = max_turn_time - turn_time

                if remaining_for_silence > 0:
                    await self._wait_for_silence(
                        silence_duration=wait_duration,
                        max_wait=remaining_for_silence
                    )

                    # Collect any additional transcripts during wait phase
                    additional = []
                    while not self._speech_queue.empty():
                        try:
                            text = self._speech_queue.get_nowait()
                            additional.append(text)
                        except asyncio.QueueEmpty:
                            break

                    if additional:
                        additional_text = " ".join(additional)
                        entire_transcript.append(additional_text)
                        overheard = f"{overheard} {additional_text}".strip()

                if self._shutdown:
                    break

                # Phase 3: Generate response (ChatGPT call)
                logger.info("Generating response via ChatGPT")
                response = await self.respond_to_overheard(overheard)

                if self._shutdown:
                    break

                # Phase 4: Wait for silence again before speaking
                # This avoids interrupting if user started talking during ChatGPT call
                if self._is_speaking:
                    logger.info("User started talking during ChatGPT call, waiting for silence before speaking")
                    # Collect any new transcripts while waiting
                    while self._is_speaking and not self._shutdown:
                        await asyncio.sleep(0.1)
                    # Wait for the required silence duration
                    await self._wait_for_silence(
                        silence_duration=wait_duration,
                        max_wait=30.0  # Max 30s wait, then speak anyway
                    )
                    # Collect additional transcripts
                    while not self._speech_queue.empty():
                        try:
                            text = self._speech_queue.get_nowait()
                            entire_transcript.append(text)
                        except asyncio.QueueEmpty:
                            break

                if self._shutdown:
                    break

                # Phase 5: Speak response
                logger.info("Speaking response")
                await self.speak(response)

            except Exception as e:
                logger.error(f"Experience loop error: {e}")

        return " ".join(entire_transcript)
