# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice In My Head (VIMH) is a real-time voice interaction system using:
- **aiortc** for native WebRTC audio streaming (replaced Daily.co)
- **AssemblyAI** for real-time speech-to-text (with configurable endpointing)
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
- Streams audio to AssemblyAI for transcription
- Plays TTS audio from ElevenLabs
- Executes CSV scripts with `AsyncScriptReader`
- Handles ChatGPT conversations

**`webrtc_handler.py`** - WebRTC signaling:
- Creates RTCPeerConnection with aiortc
- Handles SDP offer/answer exchange
- Manages ICE candidate exchange
- Routes audio tracks

**`assemblyai_stream.py`** - Transcription:
- Direct WebSocket connection to AssemblyAI Universal-Streaming
- Configurable endpointing thresholds for less sensitive VAD
- Returns `TranscriptMessage` objects
- Provides `on_transcript`, `on_utterance_end`, and `on_speech_started` callbacks

**`audio_tracks.py`** - Audio handling:
- `AudioOutputTrack` - Sends TTS audio to browser
- `AudioInputHandler` - Receives audio from browser
- `decode_mp3_to_pcm()` - MP3 decoding using PyAV

### Audio Pipeline

1. Browser mic → WebRTC → aiortc → `AudioInputHandler`
2. Handler streams audio to AssemblyAI WebSocket
3. AssemblyAI returns transcripts with configurable end-of-turn detection
4. GPT generates response
5. ElevenLabs TTS → MP3 → PCM → `AudioOutputTrack` → WebRTC → browser

### Audio Format
- Sample rate: 48kHz (WebRTC native)
- Channels: 1 (mono)
- Bit depth: 16-bit signed PCM

### Art Installation Structure

The entire piece is an art installation with two main phases:

1. **Interview/Onboarding** (~5-10 minutes): Guided conversation at the start where the system learns about the visitor, clones their voice, etc. Driven by CSV scripts in `scripts/`.

2. **Experience Loop** (until `TOTAL_TIME_MINUTES` runs out): The core of the piece. The system overhears conversations and periodically interjects with responses.

### Experience Loop Timing

The `experience_loop()` in `voice_session.py` controls when interjections happen:
1. **Listen phase**: Listen for `TURN_TIME_SECONDS` (default 50s)
2. **Wait-for-silence phase**: After turn time, wait for `WAIT_DURATION_SECONDS` (default 2s) of continuous silence before interjecting
3. **Max timeout**: If `MAX_TURN_TIME_SECONDS` (default 90s) total elapsed without finding silence, interject anyway
4. **Respond**: Generate and speak a response, then repeat

This avoids interrupting visitors mid-conversation by waiting for natural pauses.

### AssemblyAI Speech Events

`assemblyai_stream.py` provides three callbacks:
- `on_transcript(TranscriptMessage)` - Called when transcription is ready
- `on_utterance_end()` - Called when end of turn detected (configurable sensitivity)
- `on_speech_started()` - Called when speech begins (user started speaking)

These are used to track `_is_speaking` state in `VoiceSession` for the wait-for-silence logic.

### AssemblyAI Endpointing Configuration

AssemblyAI's endpointing is configurable via environment variables:
- `END_OF_TURN_CONFIDENCE` - Confidence threshold for end of turn (0.0-1.0, default 0.5)
- `MIN_SILENCE_CONFIDENT_MS` - Min silence (ms) when confidence met (default 600)
- `MAX_TURN_SILENCE_MS` - Max silence (ms) before forcing end of turn (default 1500)

Higher values = less sensitive (fewer false positives, better for noisy environments).

### Browser UI Indicators

The frontend (`static/run.html`) shows a 🗣️ emoji overlay when the visitor is speaking. This is controlled via app-messages:
```javascript
// Server sends: {type: "app-message", data: {event: "user-speaking", speaking: true/false}}
```

### Script Format
Scripts in `scripts/*.csv` with columns: `function`, `output`, `input`, and optional language columns.
- Functions map to methods on `VoiceSession` class
- `async ` prefix runs function as background task
- `{variable}` syntax for variable substitution in `speak` input

### Key Environment Variables
```
ELEVENLABS_API_KEY=...
OPENAI_API_KEY=...
ASSEMBLYAI_API_KEY=...
TURN_TIME_SECONDS=50           # Initial listening duration before looking for silence
WAIT_DURATION_SECONDS=2        # Silence duration needed before interjecting
MAX_TURN_TIME_SECONDS=90       # Max wait time, interject anyway after this
TOTAL_TIME_MINUTES=25          # Total session duration
LOCATION=...

# AssemblyAI endpointing (optional, less-sensitive defaults)
END_OF_TURN_CONFIDENCE=0.5     # Confidence threshold (0.0-1.0)
MIN_SILENCE_CONFIDENT_MS=600   # Min silence when confident (ms)
MAX_TURN_SILENCE_MS=1500       # Max silence before end of turn (ms)
```

### Signaling Protocol (WebSocket)
- `{type: "offer", sdp: "..."}` - Server sends SDP offer
- `{type: "answer", sdp: "..."}` - Client sends SDP answer
- `{type: "ice-candidate", ...}` - ICE candidate exchange
- `{type: "app-message", data: {...}}` - Chat messages, start/stop

### Sound Files
Audio files in `sound/` must be: mono, 48kHz, 16-bit WAV.

## Deployment (Digital Ocean)

### System Dependencies
**Important**: These must be installed on the server:
```sh
apt-get update
apt-get install -y ffmpeg git curl
```

**ffmpeg is required** for voice cloning (converts WAV to MP3 for ElevenLabs API).

### Install uv and Caddy
```sh
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Caddy
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
apt-get update && apt-get install -y caddy
```

### Clone and Setup
```sh
cd /opt
git clone -b nodaily https://github.com/kylemcdonald/voice-in-my-head.git vimh
cd vimh
uv sync
```

### Configure Caddy (`/etc/caddy/Caddyfile`)
```
your-domain.example.com {
    reverse_proxy localhost:8000
}
```

### Systemd Service (`/etc/systemd/system/vimh.service`)
```ini
[Unit]
Description=Voice In My Head
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/vimh
Environment="PATH=/usr/bin:/usr/local/bin:/root/.local/bin:/bin"
ExecStart=/root/.local/bin/uv run python server_async.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Start Services
```sh
systemctl daemon-reload
systemctl enable vimh
systemctl start vimh
systemctl restart caddy
```

### Recommended Server Specs
- **4 simultaneous sessions**: 2 vCPU, 4GB RAM
- **8 simultaneous sessions**: 4 vCPU, 8GB RAM

## Deprecated Files

Old Daily.co-based implementation is in `_deprecated/`:
- `vimh_daily.py` - Old main class using Daily SDK
- `server_flask.py` - Old Flask server
- `daily_helpers.py` - Daily.co API helpers
- `vad.py` - Silero VAD (replaced by AssemblyAI endpointing)

