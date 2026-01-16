"""
AssemblyAI WebSocket client for real-time streaming transcription.

Uses AssemblyAI's Universal-Streaming model with configurable semantic + acoustic endpointing.
"""

import asyncio
import json
import logging
import os
from typing import Callable, Optional, Awaitable
from dataclasses import dataclass

import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


@dataclass
class TranscriptMessage:
    """Represents a transcription result from AssemblyAI."""
    text: str
    is_final: bool
    speech_final: bool  # True when end of turn (silence detected)
    confidence: float
    start: float  # Start time in seconds
    duration: float  # Duration in seconds


class AssemblyAIStream:
    """
    Manages a WebSocket connection to AssemblyAI's streaming API.

    Sends raw audio and receives real-time transcriptions.
    Uses AssemblyAI's configurable endpointing to determine when the user
    has stopped speaking.
    """

    ASSEMBLYAI_WS_URL = "wss://streaming.assemblyai.com/v3/ws"

    def __init__(
        self,
        api_key: str,
        sample_rate: int = 48000,
        end_of_turn_confidence: float = 0.5,
        min_silence_confident_ms: int = 600,
        max_turn_silence_ms: int = 1500,
        on_transcript: Optional[Callable[[TranscriptMessage], Awaitable[None]]] = None,
        on_utterance_end: Optional[Callable[[], Awaitable[None]]] = None,
        on_speech_started: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        """
        Initialize AssemblyAIStream.

        Args:
            api_key: AssemblyAI API key
            sample_rate: Audio sample rate in Hz
            end_of_turn_confidence: Confidence threshold for end of turn (0.0-1.0)
            min_silence_confident_ms: Min silence (ms) when confidence threshold met
            max_turn_silence_ms: Max silence (ms) before forcing end of turn
            on_transcript: Callback for transcript messages
            on_utterance_end: Callback when utterance ends (user stopped speaking)
            on_speech_started: Callback when speech starts (user began speaking)
        """
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.end_of_turn_confidence = end_of_turn_confidence
        self.min_silence_confident_ms = min_silence_confident_ms
        self.max_turn_silence_ms = max_turn_silence_ms
        self.on_transcript = on_transcript
        self.on_utterance_end = on_utterance_end
        self.on_speech_started = on_speech_started

        self._ws: Optional[ClientConnection] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = asyncio.Event()
        self._closing = False
        self._session_id: Optional[str] = None
        self._last_turn_order: int = -1  # Track turn order for speech_started detection
        self._has_speech_in_turn: bool = False  # Track if we've seen speech in current turn

        # Audio buffering - AssemblyAI requires 50-1000ms chunks
        # At 48kHz, 16-bit mono: 50ms = 4800 bytes, we'll target ~100ms = 9600 bytes
        self._audio_buffer = bytearray()
        self._min_chunk_bytes = 4800  # 50ms at 48kHz, 16-bit mono
        self._bytes_actually_sent = 0  # Track actual bytes sent over websocket

    @property
    def is_connected(self) -> bool:
        """Returns True if connected to AssemblyAI."""
        return self._ws is not None and self._ws.close_code is None

    def _build_url(self) -> str:
        """Build the AssemblyAI WebSocket URL with query parameters."""
        params = [
            f"sample_rate={self.sample_rate}",
            f"encoding=pcm_s16le",
            f"end_of_turn_confidence_threshold={self.end_of_turn_confidence}",
            f"min_end_of_turn_silence_when_confident={self.min_silence_confident_ms}",
            f"max_turn_silence={self.max_turn_silence_ms}",
        ]
        return f"{self.ASSEMBLYAI_WS_URL}?{'&'.join(params)}"

    async def connect(self) -> None:
        """Connect to AssemblyAI's streaming API."""
        if self.is_connected:
            logger.warning("Already connected to AssemblyAI")
            return

        url = self._build_url()
        headers = {"Authorization": self.api_key}

        logger.info(f"Connecting to AssemblyAI: {url}")

        try:
            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
            )
            self._closing = False

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Wait for Begin message to confirm connection
            # The _receive_loop will set _connected when Begin is received
            logger.info("Waiting for AssemblyAI session to begin...")
        except Exception as e:
            logger.error(f"Failed to connect to AssemblyAI: {e}")
            raise

    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send raw audio data to AssemblyAI.

        Buffers audio to meet AssemblyAI's 50-1000ms chunk requirement.

        Args:
            audio_data: Raw PCM audio bytes (16-bit signed, little-endian)
        """
        if not self.is_connected:
            return

        # Buffer the audio
        self._audio_buffer.extend(audio_data)

        # Send when we have enough data (50ms minimum)
        if len(self._audio_buffer) >= self._min_chunk_bytes:
            try:
                chunk = bytes(self._audio_buffer)
                await self._ws.send(chunk)
                self._bytes_actually_sent += len(chunk)
                self._audio_buffer.clear()
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed while sending audio")
            except Exception as e:
                logger.error(f"Error sending audio to AssemblyAI: {e}")

    async def _receive_loop(self) -> None:
        """Receive and process messages from AssemblyAI."""
        try:
            async for message in self._ws:
                if self._closing:
                    break

                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from AssemblyAI: {message}")
                except Exception as e:
                    logger.error(f"Error handling AssemblyAI message: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            if not self._closing:
                logger.warning(f"AssemblyAI connection closed: {e}")
        except Exception as e:
            if not self._closing:
                logger.error(f"Error in AssemblyAI receive loop: {e}")

    async def _handle_message(self, data: dict) -> None:
        """Handle a parsed message from AssemblyAI."""
        msg_type = data.get("type", "")

        if msg_type == "Begin":
            # Session started
            self._session_id = data.get("id")
            expires_at = data.get("expires_at")
            logger.info(f"AssemblyAI session started: {self._session_id}, expires_at={expires_at}")
            self._connected.set()

        elif msg_type == "Turn":
            # Transcription result
            await self._handle_turn(data)

        elif msg_type == "Termination":
            # Session ended
            audio_duration = data.get("audio_duration_seconds", 0)
            session_duration = data.get("session_duration_seconds", 0)
            logger.info(f"AssemblyAI session ended: audio={audio_duration}s, session={session_duration}s")

        elif msg_type == "Error":
            error_code = data.get("code")
            error_msg = data.get("message", "Unknown error")
            logger.error(f"AssemblyAI error [{error_code}]: {error_msg}")

        else:
            logger.debug(f"Unknown AssemblyAI message type: {msg_type}")

    async def _handle_turn(self, data: dict) -> None:
        """Handle a Turn message from AssemblyAI."""
        turn_order = data.get("turn_order", 0)
        end_of_turn = data.get("end_of_turn", False)
        transcript = data.get("transcript", "").strip()
        confidence = data.get("end_of_turn_confidence", 0.0)
        words = data.get("words", [])

        # Detect speech started (new turn with content)
        if turn_order > self._last_turn_order and transcript:
            self._last_turn_order = turn_order
            self._has_speech_in_turn = False

        # Trigger speech_started on first transcript of a turn
        if transcript and not self._has_speech_in_turn:
            self._has_speech_in_turn = True
            if self.on_speech_started:
                logger.debug("Speech started (new transcript in turn)")
                await self.on_speech_started()

        # Calculate timing from words
        start = 0.0
        duration = 0.0
        if words:
            start = words[0].get("start", 0) / 1000.0  # Convert ms to seconds
            end = words[-1].get("end", 0) / 1000.0
            duration = end - start

        # Create transcript message
        if transcript:
            msg = TranscriptMessage(
                text=transcript,
                is_final=end_of_turn,
                speech_final=end_of_turn,
                confidence=confidence,
                start=start,
                duration=duration,
            )

            logger.debug(f"Transcript: {msg}")

            if self.on_transcript:
                await self.on_transcript(msg)

        # Handle end of turn (utterance end)
        if end_of_turn:
            logger.debug("End of turn detected")
            self._has_speech_in_turn = False
            if self.on_utterance_end:
                await self.on_utterance_end()

    async def close(self) -> None:
        """Close the connection to AssemblyAI."""
        self._closing = True
        self._connected.clear()
        self._audio_buffer.clear()

        if self._ws:
            try:
                # Send terminate message
                await self._ws.send(json.dumps({"type": "Terminate"}))
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing AssemblyAI connection: {e}")
            finally:
                self._ws = None

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        logger.info("Disconnected from AssemblyAI")

    async def wait_connected(self, timeout: float = 10.0) -> bool:
        """Wait for connection to be established."""
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def update_endpointing(
        self,
        end_of_turn_confidence: float,
        min_silence_confident_ms: int,
        max_turn_silence_ms: int,
    ) -> None:
        """
        Reconnect with new endpointing parameters if they differ from current.

        Args:
            end_of_turn_confidence: Confidence threshold for end of turn (0.0-1.0)
            min_silence_confident_ms: Min silence (ms) when confidence threshold met
            max_turn_silence_ms: Max silence (ms) before forcing end of turn
        """
        # Skip if parameters haven't changed
        if (self.end_of_turn_confidence == end_of_turn_confidence and
            self.min_silence_confident_ms == min_silence_confident_ms and
            self.max_turn_silence_ms == max_turn_silence_ms):
            return

        logger.info(f"Updating endpointing: confidence={end_of_turn_confidence}, "
                    f"min_silence={min_silence_confident_ms}ms, max_silence={max_turn_silence_ms}ms")

        self.end_of_turn_confidence = end_of_turn_confidence
        self.min_silence_confident_ms = min_silence_confident_ms
        self.max_turn_silence_ms = max_turn_silence_ms

        # Close existing connection and reconnect
        if self.is_connected:
            await self.close()
        await self.connect()
        await self.wait_connected()


async def test_assemblyai():
    """Simple test of the AssemblyAI stream."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("ASSEMBLYAI_API_KEY not set")
        return

    async def on_transcript(msg: TranscriptMessage):
        print(f"[{'FINAL' if msg.is_final else 'partial'}] {msg.text}")

    async def on_utterance_end():
        print("--- End of turn ---")

    async def on_speech_started():
        print(">>> Speech started <<<")

    stream = AssemblyAIStream(
        api_key=api_key,
        sample_rate=48000,
        on_transcript=on_transcript,
        on_utterance_end=on_utterance_end,
        on_speech_started=on_speech_started,
    )

    await stream.connect()
    connected = await stream.wait_connected()
    if connected:
        print("Connected! (Send Ctrl+C to stop)")
    else:
        print("Failed to connect within timeout")
        return

    # Keep alive for testing
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await stream.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(test_assemblyai())
