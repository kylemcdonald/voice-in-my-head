"""
Direct Deepgram WebSocket client for real-time transcription.

Replaces Daily.co's Deepgram integration with direct streaming.
Uses Deepgram's utterance detection instead of Silero VAD.
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
    """Represents a transcription result from Deepgram."""
    text: str
    is_final: bool
    speech_final: bool  # True when utterance is complete (silence detected)
    confidence: float
    start: float  # Start time in seconds
    duration: float  # Duration in seconds


class DeepgramStream:
    """
    Manages a WebSocket connection to Deepgram's streaming API.

    Sends raw audio and receives real-time transcriptions.
    Uses Deepgram's utterance detection (utterance_end_ms) to determine
    when the user has stopped speaking, replacing local VAD.
    """

    DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"

    def __init__(
        self,
        api_key: str,
        language: str = "en",
        sample_rate: int = 48000,
        channels: int = 1,
        utterance_end_ms: int = 1500,
        on_transcript: Optional[Callable[[TranscriptMessage], Awaitable[None]]] = None,
        on_utterance_end: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        """
        Initialize DeepgramStream.

        Args:
            api_key: Deepgram API key
            language: Language code (e.g., "en", "nl")
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            utterance_end_ms: Silence duration (ms) to trigger utterance end
            on_transcript: Callback for transcript messages
            on_utterance_end: Callback when utterance ends (user stopped speaking)
        """
        self.api_key = api_key
        self.language = language
        self.sample_rate = sample_rate
        self.channels = channels
        self.utterance_end_ms = utterance_end_ms
        self.on_transcript = on_transcript
        self.on_utterance_end = on_utterance_end

        self._ws: Optional[ClientConnection] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = asyncio.Event()
        self._closing = False

    @property
    def is_connected(self) -> bool:
        """Returns True if connected to Deepgram."""
        return self._ws is not None and self._ws.close_code is None

    def _build_url(self) -> str:
        """Build the Deepgram WebSocket URL with query parameters."""
        params = [
            f"encoding=linear16",
            f"sample_rate={self.sample_rate}",
            f"channels={self.channels}",
            f"language={self.language}",
            f"punctuate=true",
            f"interim_results=true",  # Get partial transcripts
            f"utterance_end_ms={self.utterance_end_ms}",
            f"vad_events=true",  # Get VAD events
        ]
        return f"{self.DEEPGRAM_WS_URL}?{'&'.join(params)}"

    async def connect(self) -> None:
        """Connect to Deepgram's streaming API."""
        if self.is_connected:
            logger.warning("Already connected to Deepgram")
            return

        url = self._build_url()
        headers = {"Authorization": f"Token {self.api_key}"}

        logger.info(f"Connecting to Deepgram: {url}")

        try:
            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
            )
            self._connected.set()
            self._closing = False

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            logger.info("Connected to Deepgram")
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            raise

    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send raw audio data to Deepgram.

        Args:
            audio_data: Raw PCM audio bytes (16-bit signed, little-endian)
        """
        if not self.is_connected:
            logger.warning("Cannot send audio: not connected to Deepgram")
            return

        try:
            await self._ws.send(audio_data)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed while sending audio")
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}")

    async def _receive_loop(self) -> None:
        """Receive and process messages from Deepgram."""
        try:
            async for message in self._ws:
                if self._closing:
                    break

                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Deepgram: {message}")
                except Exception as e:
                    logger.error(f"Error handling Deepgram message: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            if not self._closing:
                logger.warning(f"Deepgram connection closed: {e}")
        except Exception as e:
            if not self._closing:
                logger.error(f"Error in Deepgram receive loop: {e}")

    async def _handle_message(self, data: dict) -> None:
        """Handle a parsed message from Deepgram."""
        msg_type = data.get("type", "")

        if msg_type == "Results":
            # Transcription result
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])

            if alternatives:
                alt = alternatives[0]
                transcript = alt.get("transcript", "")

                if transcript:
                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)

                    msg = TranscriptMessage(
                        text=transcript,
                        is_final=is_final,
                        speech_final=speech_final,
                        confidence=alt.get("confidence", 0.0),
                        start=data.get("start", 0.0),
                        duration=data.get("duration", 0.0),
                    )

                    logger.debug(f"Transcript: {msg}")

                    if self.on_transcript:
                        await self.on_transcript(msg)

        elif msg_type == "UtteranceEnd":
            # User stopped speaking (based on utterance_end_ms)
            logger.debug("Utterance end detected")
            if self.on_utterance_end:
                await self.on_utterance_end()

        elif msg_type == "SpeechStarted":
            logger.debug("Speech started")

        elif msg_type == "Metadata":
            logger.debug(f"Deepgram metadata: {data}")

        elif msg_type == "Error":
            logger.error(f"Deepgram error: {data}")

    async def close(self) -> None:
        """Close the connection to Deepgram."""
        self._closing = True
        self._connected.clear()

        if self._ws:
            try:
                # Send close message
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing Deepgram connection: {e}")
            finally:
                self._ws = None

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        logger.info("Disconnected from Deepgram")

    async def wait_connected(self, timeout: float = 10.0) -> bool:
        """Wait for connection to be established."""
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


async def test_deepgram():
    """Simple test of the Deepgram stream."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("DEEPGRAM_API_KEY not set")
        return

    async def on_transcript(msg: TranscriptMessage):
        print(f"[{'FINAL' if msg.is_final else 'partial'}] {msg.text}")

    async def on_utterance_end():
        print("--- Utterance ended ---")

    stream = DeepgramStream(
        api_key=api_key,
        language="en",
        on_transcript=on_transcript,
        on_utterance_end=on_utterance_end,
    )

    await stream.connect()
    print("Connected! (Send Ctrl+C to stop)")

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
    asyncio.run(test_deepgram())
