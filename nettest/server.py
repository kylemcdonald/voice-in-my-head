"""
Minimal network test for running VIMH from a remote venue.

Serves a single mobile page that checks, from the visitor's network:
- HTTPS/WebSocket round trip to this server
- WebRTC audio over direct UDP (the production setup) and via TURN (UDP/TCP/TLS)
- The full voice pipeline: mic -> AssemblyAI -> OpenAI -> ElevenLabs -> speaker

Run from the repo root:  python nettest/server.py
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import ssl
import sys
import time
import uuid
from pathlib import Path

import aiohttp
import numpy as np
from aiohttp import web
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from dotenv import load_dotenv
from elevenlabs import AsyncElevenLabs
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assemblyai_stream import AssemblyAIStream, TranscriptMessage  # noqa: E402
from audio_tracks import AudioOutputTrack, decode_mp3_to_pcm, SAMPLE_RATE, SAMPLE_WIDTH  # noqa: E402
import llm  # noqa: E402

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nettest")

ACCESS_KEY = os.environ["NETTEST_ACCESS_KEY"]
TURN_HOST = os.environ["NETTEST_TURN_HOST"]
TURN_SECRET = os.environ["NETTEST_TURN_SECRET"]
SERVER_LABEL = os.environ.get("NETTEST_SERVER_LABEL", "Frankfurt (DigitalOcean fra1)")
PORT = int(os.environ.get("NETTEST_PORT", "8000"))
# Matilda, the default voice used by VoiceSession
VOICE_ID = os.environ.get("NETTEST_VOICE_ID", "XrExE9yKIg1WjnnlVkGX")
# Same model and TTS settings as the live session in voice_session.py
TTS_MODEL = "eleven_turbo_v2_5"

STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# No STUN on the server: the droplet has a public IP, and aiortc's STUN queries from the
# private VPC interface time out, which delays every answer by ~5s.
SERVER_ICE = RTCConfiguration(iceServers=[])
SSL_CONTEXT = ssl.create_default_context()

API_HOSTS = {
    "anthropic": "api.anthropic.com",
    "openai": "api.openai.com",
    "assemblyai": "streaming.assemblyai.com",
    "elevenlabs": "api.elevenlabs.io",
}


def ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def turn_credentials(run_id: str, ttl: int = 3600) -> dict:
    """Time-limited TURN credentials for coturn's use-auth-secret mode."""
    username = f"{int(time.time()) + ttl}:{run_id}"
    digest = hmac.new(TURN_SECRET.encode(), username.encode(), hashlib.sha1).digest()
    return {"username": username, "credential": base64.b64encode(digest).decode()}


def client_ip(request: web.Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.remote or "")


async def lookup_isp(ip: str) -> dict:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"https://ipinfo.io/{ip}/json") as resp:
                data = await resp.json(content_type=None)
                return {k: data.get(k) for k in ("org", "city", "region", "country")}
    except Exception as e:
        return {"error": str(e)}


async def probe_tls(host: str) -> dict:
    """Measure TCP connect and TLS handshake time from this server to an API host."""
    loop = asyncio.get_running_loop()
    try:
        t0 = loop.time()
        infos = await loop.getaddrinfo(host, 443, type=0)
        ip = infos[0][4][0]
        t1 = loop.time()
        reader, writer = await asyncio.wait_for(asyncio.open_connection(ip, 443), 5)
        t2 = loop.time()
        writer.close()
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, 443, ssl=SSL_CONTEXT, server_hostname=host), 5
        )
        t3 = loop.time()
        writer.close()
        # The second connection includes its own TCP handshake; report TLS on top of TCP.
        return {"dns_ms": ms(t1 - t0), "tcp_ms": ms(t2 - t1), "tls_ms": max(0, ms(t3 - t2) - ms(t2 - t1))}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def probe_apis() -> dict:
    names = list(API_HOSTS)
    results = await asyncio.gather(*(probe_tls(API_HOSTS[n]) for n in names))
    return dict(zip(names, results))


def short_error(e: Exception) -> str:
    """Keep API errors short enough to read on a phone screenshot."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            return f"{detail.get('status') or detail.get('code') or ''}: {detail.get('message', '')}"[:200]
    return f"{type(e).__name__}: {e}"[:200]


def pcm_from_frame(frame) -> np.ndarray:
    """Convert an aiortc audio frame to mono int16 (same handling as AudioInputHandler)."""
    arr = frame.to_ndarray()
    if arr.dtype in (np.float32, np.float64):
        arr = (arr * 32767).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    arr = arr.flatten()
    if len(arr) == frame.samples * 2:
        arr = ((arr[0::2].astype(np.int32) + arr[1::2].astype(np.int32)) // 2).astype(np.int16)
    return arr


class PeerTest:
    """One server-side peer connection: echoes data channel pings, measures mic audio, plays TTS."""

    def __init__(self, name: str, send):
        self.name = name
        self.send = send
        self.pc = RTCPeerConnection(configuration=SERVER_ICE)
        self.output = AudioOutputTrack()
        self.frames = 0
        self.peak = 0
        self.stt: AssemblyAIStream | None = None
        self._recv_task = None
        self.pc.on("datachannel", self._on_datachannel)
        self.pc.on("track", self._on_track)

    def _on_datachannel(self, channel):
        @channel.on("message")
        def on_message(message):
            channel.send(message)

    def _on_track(self, track):
        if track.kind == "audio":
            self._recv_task = asyncio.create_task(self._recv_loop(track))

    async def _recv_loop(self, track):
        try:
            while True:
                frame = await track.recv()
                pcm = pcm_from_frame(frame)
                self.frames += 1
                self.peak = max(self.peak, int(np.abs(pcm.astype(np.int32)).max()))
                if self.stt and self.stt.is_connected:
                    await self.stt.send_audio(pcm.tobytes())
        except Exception:
            pass

    async def answer(self, sdp: str) -> str:
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        self.pc.addTrack(self.output)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription.sdp

    def stats(self) -> dict:
        peak_dbfs = round(20 * np.log10(self.peak / 32768), 1) if self.peak > 0 else None
        return {"frames": self.frames, "peak_dbfs": peak_dbfs, "state": self.pc.connectionState}

    async def close(self):
        if self.stt:
            await self.stt.close()
        if self._recv_task:
            self._recv_task.cancel()
        await self.pc.close()


class Pipeline:
    """Mic -> AssemblyAI -> OpenAI -> ElevenLabs -> WebRTC, timing each stage."""

    def __init__(self, peer: PeerTest, send, openai: AsyncOpenAI, elevenlabs: AsyncElevenLabs):
        self.peer = peer
        self.send = send
        self.openai = openai
        self.llm = llm.from_env()
        self.elevenlabs = elevenlabs
        self.final_text = ""
        self.turn_done = asyncio.Event()

    async def on_transcript(self, msg: TranscriptMessage):
        await self.send({"type": "transcript", "text": msg.text, "final": msg.is_final})
        if msg.is_final and msg.text:
            self.final_text = msg.text
            self.turn_done.set()

    async def run(self, listen_timeout: float = 20.0):
        timings = {}
        stt = AssemblyAIStream(
            api_key=os.environ["ASSEMBLYAI_API_KEY"],
            sample_rate=SAMPLE_RATE,
            on_transcript=self.on_transcript,
        )
        t0 = time.monotonic()
        try:
            await stt.connect()
            if not await stt.wait_connected(10):
                raise RuntimeError("no Begin message")
        except Exception as e:
            await self.send({"type": "pipeline_error", "stage": "assemblyai", "fatal": True, "error": short_error(e)})
            await stt.close()
            return
        timings["stt_connect_ms"] = ms(time.monotonic() - t0)
        self.peer.stt = stt
        await self.send({"type": "stt_ready", **timings})

        try:
            await asyncio.wait_for(self.turn_done.wait(), listen_timeout)
            heard = True
        except asyncio.TimeoutError:
            heard = False
        self.peer.stt = None
        await stt.close()

        t_turn = time.monotonic()
        if heard:
            try:
                reply = await self.llm.complete(
                    self.final_text,
                    system="You are a friendly voice on a sound check. Reply to what the person said in one short sentence, under 15 words.",
                )
                reply = reply.strip() or "I heard you, but the language model gave an empty reply."
            except Exception as e:
                await self.send({"type": "pipeline_error", "stage": "llm", "error": short_error(e)})
                reply = "I could not reach the language model, but audio playback works."
            timings["llm_ms"] = ms(time.monotonic() - t_turn)
        else:
            reply = "I did not hear anything, but if you can hear me, playback works."

        t_tts = time.monotonic()
        try:
            chunks = []
            async for chunk in self.elevenlabs.text_to_speech.stream(
                text=reply,
                voice_id=VOICE_ID,
                model_id=TTS_MODEL,
                optimize_streaming_latency=3,
            ):
                if not chunks:
                    timings["tts_first_byte_ms"] = ms(time.monotonic() - t_tts)
                chunks.append(chunk)
            timings["tts_total_ms"] = ms(time.monotonic() - t_tts)
            timings["tts_provider"] = "ElevenLabs"
            pcm = decode_mp3_to_pcm(b"".join(chunks))
        except Exception as e:
            await self.send({"type": "pipeline_error", "stage": "elevenlabs", "error": short_error(e)})
            # Fall back to OpenAI TTS so the playback path is still tested.
            t_tts = time.monotonic()
            try:
                speech = await self.openai.audio.speech.create(
                    model="gpt-4o-mini-tts", voice="alloy", input=reply, response_format="mp3"
                )
                timings["tts_total_ms"] = ms(time.monotonic() - t_tts)
                timings["tts_provider"] = "OpenAI (fallback)"
                pcm = decode_mp3_to_pcm(speech.content)
            except Exception as e2:
                await self.send({"type": "pipeline_error", "stage": "openai-tts", "fatal": True, "error": short_error(e2)})
                return

        chunk_size = SAMPLE_RATE * SAMPLE_WIDTH // 10
        for i in range(0, len(pcm), chunk_size):
            await self.peer.output.add_audio(pcm[i:i + chunk_size])
        timings["server_turn_to_audio_ms"] = ms(time.monotonic() - t_turn)
        audio_ms = ms(len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH))
        await self.send({
            "type": "reply_playing",
            "heard_speech": heard,
            "transcript": self.final_text,
            "reply": reply,
            "audio_ms": audio_ms,
            **timings,
        })


async def index(request: web.Request) -> web.Response:
    return web.FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


async def websocket(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    if request.query.get("k") != ACCESS_KEY:
        await ws.send_json({"type": "denied"})
        await ws.close()
        return ws

    run_id = uuid.uuid4().hex[:6].upper()
    ip = client_ip(request)
    log = {"run_id": run_id, "client_ip": ip, "started": time.time(), "events": []}
    peers: dict[str, PeerTest] = {}
    openai = AsyncOpenAI()
    elevenlabs = AsyncElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    tasks = set()
    logger.info(f"[{run_id}] connected from {ip}")

    async def send(msg: dict):
        log["events"].append({"t": round(time.time() - log["started"], 3), **msg})
        if not ws.closed:
            await ws.send_json(msg)

    def spawn(coro):
        task = asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def send_isp():
        await send({"type": "isp", **(await lookup_isp(ip))})

    async def send_api_probe():
        await send({"type": "api_probe", "results": await probe_apis()})

    creds = turn_credentials(run_id)
    await send({
        "type": "hello",
        "run_id": run_id,
        "client_ip": ip,
        "server": SERVER_LABEL,
        "turn": {"host": TURN_HOST, **creds},
    })
    spawn(send_isp())
    spawn(send_api_probe())

    try:
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(message.data)
            kind = data.get("type")
            if kind == "ping":
                await ws.send_json({"type": "pong", "t": data["t"]})
            elif kind == "offer":
                name = data["test"]
                peer = PeerTest(name, send)
                peers[name] = peer
                sdp = await peer.answer(data["sdp"])
                await send({"type": "answer", "test": name, "sdp": sdp})
            elif kind == "server_stats":
                peer = peers.get(data["test"])
                await send({"type": "server_stats", "test": data["test"], **(peer.stats() if peer else {})})
            elif kind == "close_peer":
                peer = peers.pop(data["test"], None)
                if peer:
                    await peer.close()
            elif kind == "pipeline_start":
                peer = peers.get(data["test"])
                if peer:
                    spawn(Pipeline(peer, send, openai, elevenlabs).run())
            elif kind == "client_log":
                log["events"].append({"t": round(time.time() - log["started"], 3), **data})
            elif kind == "result":
                log["client_result"] = data["result"]
                (RESULTS_DIR / f"{run_id}.json").write_text(json.dumps(log, indent=2))
                logger.info(f"[{run_id}] saved result")
    finally:
        for task in list(tasks):
            task.cancel()
        for peer in peers.values():
            await peer.close()
        if "client_result" not in log:
            (RESULTS_DIR / f"{run_id}-incomplete.json").write_text(json.dumps(log, indent=2))
        logger.info(f"[{run_id}] disconnected")
    return ws


def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket)
    web.run_app(app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
