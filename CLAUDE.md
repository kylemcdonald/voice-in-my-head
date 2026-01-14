# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice In My Head (VIMH) is a real-time voice interaction system using:
- **aiortc** for native WebRTC audio streaming (replaced Daily.co)
- **Deepgram** for real-time speech-to-text (direct WebSocket streaming)
- **ElevenLabs** for text-to-speech with voice cloning
- **OpenAI GPT-5.2** for conversational AI

Users connect via browser WebRTC to interact with an AI voice assistant that can clone their voice and provide personalized responses.

## Commands

### Development Server
```sh
# Activate environment
source .venv/bin/activate

# Run server
python server_async.py

# With Caddy for HTTPS (needed for WebRTC in browser)
caddy run  # In another terminal
```

### Environment Setup
```sh
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

### CSS Build
```sh
npm run buildcss    # Watch mode for Tailwind CSS
```

## Architecture

### Core Components

**`server_async.py`** - aiohttp web server:
- Serves static files
- Handles WebSocket signaling for WebRTC
- Creates sessions via `/spin-up-session` endpoint

**`voice_session.py`** - Main session class:
- Manages WebRTC connection via `webrtc_handler.py`
- Streams audio to Deepgram for transcription
- Plays TTS audio from ElevenLabs
- Executes CSV scripts with `AsyncScriptReader`
- Handles ChatGPT conversations

**`webrtc_handler.py`** - WebRTC signaling:
- Creates RTCPeerConnection with aiortc
- Handles SDP offer/answer exchange
- Manages ICE candidate exchange
- Routes audio tracks

**`deepgram_stream.py`** - Transcription:
- Direct WebSocket connection to Deepgram
- Uses utterance detection instead of local VAD
- Returns `TranscriptMessage` objects

**`audio_tracks.py`** - Audio handling:
- `AudioOutputTrack` - Sends TTS audio to browser
- `AudioInputHandler` - Receives audio from browser
- `decode_mp3_to_pcm()` - MP3 decoding using PyAV

### Audio Pipeline

1. Browser mic → WebRTC → aiortc → `AudioInputHandler`
2. Handler streams audio to Deepgram WebSocket
3. Deepgram returns transcripts with utterance detection
4. GPT generates response
5. ElevenLabs TTS → MP3 → PCM → `AudioOutputTrack` → WebRTC → browser

### Audio Format
- Sample rate: 48kHz (WebRTC native)
- Channels: 1 (mono)
- Bit depth: 16-bit signed PCM

### Script Format
Scripts in `scripts/*.csv` with columns: `function`, `output`, `input`, and optional language columns.
- Functions map to methods on `VoiceSession` class
- `async ` prefix runs function as background task
- `{variable}` syntax for variable substitution in `speak` input

### Key Environment Variables
```
ELEVENLABS_API_KEY=...
OPENAI_API_KEY=...
DEEPGRAM_API_KEY=...
TURN_TIME_SECONDS=50
TOTAL_TIME_MINUTES=25
LOCATION=...
```

### Signaling Protocol (WebSocket)
- `{type: "offer", sdp: "..."}` - Server sends SDP offer
- `{type: "answer", sdp: "..."}` - Client sends SDP answer
- `{type: "ice-candidate", ...}` - ICE candidate exchange
- `{type: "app-message", data: {...}}` - Chat messages, start/stop

### Sound Files
Audio files in `sound/` must be: mono, 48kHz, 16-bit WAV.

## Deprecated Files

Old Daily.co-based implementation is in `_deprecated/`:
- `vimh_daily.py` - Old main class using Daily SDK
- `server_flask.py` - Old Flask server
- `daily_helpers.py` - Daily.co API helpers
- `vad.py` - Silero VAD (replaced by Deepgram utterance detection)
