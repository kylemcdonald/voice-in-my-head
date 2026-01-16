"""
Audio track implementations for aiortc WebRTC.

Provides:
- AudioOutputTrack: Sends TTS audio to browser
- MP3 decoder using PyAV (replaces streamp3)
"""

import asyncio
import fractions
import io
import logging
import time
from typing import AsyncGenerator, Optional

import av
import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame

logger = logging.getLogger(__name__)

# Audio format constants
SAMPLE_RATE = 48000  # WebRTC native sample rate
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
FRAME_DURATION = 0.02  # 20ms frames (standard for WebRTC)
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION)


class AudioOutputTrack(MediaStreamTrack):
    """
    Custom audio track that sends queued audio to the browser via WebRTC.

    Used for:
    - Playing TTS audio from ElevenLabs
    - Playing sound effects

    The track generates silence when no audio is queued, maintaining
    the RTP stream continuity.
    """

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._start_time: Optional[float] = None
        self._timestamp = 0
        self._buffer = bytearray()

        # Pre-generate silence frame for efficiency
        self._silence = bytes(SAMPLES_PER_FRAME * SAMPLE_WIDTH)

    async def recv(self) -> AudioFrame:
        """
        Called by aiortc to get the next audio frame.

        Returns audio from queue if available, otherwise silence.
        """
        if self._start_time is None:
            self._start_time = time.time()

        # Try to get audio from queue without blocking
        bytes_needed = SAMPLES_PER_FRAME * SAMPLE_WIDTH

        # Fill buffer from queue
        while len(self._buffer) < bytes_needed:
            try:
                chunk = self._queue.get_nowait()
                self._buffer.extend(chunk)
            except asyncio.QueueEmpty:
                break

        # Extract frame from buffer or use silence
        if len(self._buffer) >= bytes_needed:
            frame_data = bytes(self._buffer[:bytes_needed])
            self._buffer = self._buffer[bytes_needed:]
        else:
            frame_data = self._silence

        # Create AudioFrame
        samples = np.frombuffer(frame_data, dtype=np.int16)
        frame = AudioFrame.from_ndarray(
            samples.reshape(1, -1),  # (channels, samples)
            format="s16",
            layout="mono",
        )
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE)

        self._timestamp += SAMPLES_PER_FRAME

        # Maintain timing - sleep to match real-time
        elapsed = time.time() - self._start_time
        expected = self._timestamp / SAMPLE_RATE
        sleep_time = expected - elapsed - 0.005  # Small buffer
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

        return frame

    async def add_audio(self, audio_data: bytes) -> None:
        """
        Add audio data to the output queue.

        Args:
            audio_data: Raw PCM audio (16-bit signed, little-endian, mono, 48kHz)
        """
        await self._queue.put(audio_data)

    def add_audio_sync(self, audio_data: bytes) -> None:
        """
        Add audio data synchronously (for use from sync code).

        Args:
            audio_data: Raw PCM audio bytes
        """
        self._queue.put_nowait(audio_data)

    @property
    def queue_size(self) -> int:
        """Returns the current queue size in bytes (approximate)."""
        return self._queue.qsize() * SAMPLES_PER_FRAME * SAMPLE_WIDTH + len(self._buffer)

    def clear_queue(self) -> None:
        """Clear all queued audio."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buffer.clear()


def decode_mp3_to_pcm(
    mp3_data: bytes,
    target_sample_rate: int = SAMPLE_RATE,
    target_channels: int = CHANNELS,
) -> bytes:
    """
    Decode MP3 data to raw PCM using PyAV.

    Args:
        mp3_data: Raw MP3 bytes
        target_sample_rate: Target sample rate (default 48kHz)
        target_channels: Target number of channels (default mono)

    Returns:
        Raw PCM bytes (16-bit signed, little-endian)
    """
    # Create in-memory container
    container = av.open(io.BytesIO(mp3_data), format="mp3")

    # Set up resampler
    resampler = av.AudioResampler(
        format="s16",
        layout="mono" if target_channels == 1 else "stereo",
        rate=target_sample_rate,
    )

    pcm_chunks = []
    for frame in container.decode(audio=0):
        # Resample frame
        resampled = resampler.resample(frame)
        for r_frame in resampled:
            # Convert to bytes
            pcm_chunks.append(r_frame.to_ndarray().tobytes())

    container.close()
    return b"".join(pcm_chunks)


async def decode_mp3_stream(
    mp3_generator: AsyncGenerator[bytes, None],
    target_sample_rate: int = SAMPLE_RATE,
    target_channels: int = CHANNELS,
) -> AsyncGenerator[bytes, None]:
    """
    Decode streaming MP3 data to PCM chunks.

    This is trickier because MP3 is a frame-based codec.
    We accumulate data until we have complete frames.

    Args:
        mp3_generator: Async generator yielding MP3 chunks
        target_sample_rate: Target sample rate
        target_channels: Target channels

    Yields:
        PCM audio chunks
    """
    # Accumulate MP3 data
    mp3_buffer = bytearray()

    # Set up resampler
    resampler = av.AudioResampler(
        format="s16",
        layout="mono" if target_channels == 1 else "stereo",
        rate=target_sample_rate,
    )

    async for chunk in mp3_generator:
        mp3_buffer.extend(chunk)

        # Try to decode accumulated data
        # MP3 frames are ~1152 samples, but we need complete frames
        # For streaming, we accumulate and decode periodically
        if len(mp3_buffer) >= 4096:  # Decode when we have enough data
            try:
                container = av.open(io.BytesIO(bytes(mp3_buffer)), format="mp3")
                decoded_samples = 0

                for frame in container.decode(audio=0):
                    resampled = resampler.resample(frame)
                    for r_frame in resampled:
                        pcm = r_frame.to_ndarray().tobytes()
                        decoded_samples += len(pcm) // SAMPLE_WIDTH
                        yield pcm

                container.close()

                # Clear buffer after successful decode
                # Note: This simple approach may lose some data at boundaries
                # A more robust implementation would track exact byte positions
                mp3_buffer.clear()

            except av.AVError as e:
                # Not enough data for complete frames, continue accumulating
                logger.debug(f"Incomplete MP3 data, continuing: {e}")

    # Decode remaining data
    if mp3_buffer:
        try:
            container = av.open(io.BytesIO(bytes(mp3_buffer)), format="mp3")
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                for r_frame in resampled:
                    yield r_frame.to_ndarray().tobytes()
            container.close()
        except av.AVError as e:
            logger.warning(f"Could not decode remaining MP3 data: {e}")


class AudioInputHandler:
    """
    Handles incoming audio from browser WebRTC connection.

    Routes audio to:
    - Transcription service (AssemblyAI) for real-time transcription
    - Recording buffer for voice cloning
    """

    def __init__(self):
        self._recording = False
        self._recorded_audio = bytearray()
        self._transcription_stream = None
        self._track = None
        self._receive_task: Optional[asyncio.Task] = None

    def set_transcription_stream(self, stream) -> None:
        """Set the transcription stream to send audio to."""
        logger.info(f"AudioInputHandler: Transcription stream set, is_connected={stream.is_connected if stream else False}")
        self._transcription_stream = stream

    def start_recording(self) -> None:
        """Start recording incoming audio for voice cloning."""
        self._recording = True
        self._recorded_audio.clear()
        logger.info("Started recording audio")

    def stop_recording(self) -> bytes:
        """Stop recording and return recorded audio."""
        self._recording = False
        audio = bytes(self._recorded_audio)
        logger.info(f"Stopped recording, captured {len(audio)} bytes")
        return audio

    async def handle_track(self, track: MediaStreamTrack) -> None:
        """
        Handle an incoming audio track from WebRTC.

        Args:
            track: The aiortc MediaStreamTrack
        """
        logger.info(f"AudioInputHandler: Starting to handle track {track.kind}, id={track.id}")
        self._track = track
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("AudioInputHandler: Receive loop task created")

    async def _receive_loop(self) -> None:
        """Continuously receive frames from the track and route them."""
        logger.info("AudioInputHandler: Receive loop started, waiting for first frame...")
        frame_count = 0
        bytes_sent = 0
        try:
            while True:
                frame = await self._track.recv()

                # Convert frame to raw PCM bytes
                arr = frame.to_ndarray()

                if frame_count == 0:
                    logger.info(f"AudioInputHandler: First frame received! format={frame.format.name}, samples={frame.samples}, rate={frame.sample_rate}, array_shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")

                # aiortc returns float32 audio normalized to [-1.0, 1.0] for s16 format
                # Transcription services expect int16 PCM, so convert if needed
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    # Convert float to int16
                    arr = (arr * 32767).astype(np.int16)
                elif arr.dtype != np.int16:
                    arr = arr.astype(np.int16)

                # Flatten the array
                arr = arr.flatten()

                # Handle stereo -> mono conversion
                # If we have twice as many samples as expected, it's stereo interleaved
                expected_samples = frame.samples
                if len(arr) == expected_samples * 2:
                    # Stereo: average left and right channels
                    # Interleaved format: [L0, R0, L1, R1, ...]
                    arr = ((arr[0::2].astype(np.int32) + arr[1::2].astype(np.int32)) // 2).astype(np.int16)
                    if frame_count == 0:
                        logger.info(f"AudioInputHandler: Converted stereo to mono, new shape={arr.shape}")

                pcm_data = arr.tobytes()
                frame_count += 1

                # Route to transcription service
                if self._transcription_stream and self._transcription_stream.is_connected:
                    await self._transcription_stream.send_audio(pcm_data)
                    bytes_sent += len(pcm_data)

                # Log progress periodically (every ~5 seconds at 50fps)
                if frame_count % 250 == 0:
                    tx_status = f"connected={self._transcription_stream.is_connected}" if self._transcription_stream else "no stream"
                    logger.info(f"Audio receive: {frame_count} frames, {bytes_sent/1024:.1f}KB sent to transcription ({tx_status})")

                # Route to recording buffer
                if self._recording:
                    self._recorded_audio.extend(pcm_data)

        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__
            if "MediaStreamTrack ended" not in err_str and err_str:
                logger.error(f"Error in audio receive loop ({err_type}): {e}")
            elif not err_str:
                logger.debug(f"Audio receive loop ended ({err_type})")

    async def stop(self) -> None:
        """Stop handling the track."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None


async def test_mp3_decode():
    """Test MP3 decoding."""
    import os
    from pathlib import Path

    # Look for a cached MP3 file
    cache_dir = Path("cache")
    if cache_dir.exists():
        for mp3_file in cache_dir.rglob("*.mp3"):
            print(f"Testing with: {mp3_file}")
            with open(mp3_file, "rb") as f:
                mp3_data = f.read()

            pcm_data = decode_mp3_to_pcm(mp3_data)
            print(f"Decoded {len(mp3_data)} bytes MP3 -> {len(pcm_data)} bytes PCM")
            print(f"Duration: {len(pcm_data) / (SAMPLE_RATE * SAMPLE_WIDTH):.2f}s")
            return

    print("No cached MP3 files found to test")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(test_mp3_decode())
