"""
WebRTC handler for aiortc peer connections.

Manages:
- RTCPeerConnection setup and lifecycle
- SDP offer/answer exchange
- ICE candidate handling
- Audio track management
"""

import asyncio
import json
import logging
from typing import Callable, Optional, Awaitable
from dataclasses import dataclass

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay

from audio_tracks import AudioOutputTrack, AudioInputHandler

logger = logging.getLogger(__name__)

# STUN servers for NAT traversal
ICE_SERVERS = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ]
)


@dataclass
class SignalingMessage:
    """Represents a WebSocket signaling message."""
    type: str
    data: dict

    @classmethod
    def from_json(cls, json_str: str) -> "SignalingMessage":
        """Parse from JSON string."""
        obj = json.loads(json_str)
        return cls(type=obj.get("type", ""), data=obj)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({"type": self.type, **self.data})


class WebRTCHandler:
    """
    Handles a WebRTC peer connection for a single session.

    Manages the full lifecycle:
    - Creates RTCPeerConnection
    - Handles signaling (offer/answer/ICE)
    - Manages audio tracks
    """

    def __init__(
        self,
        on_audio_track: Optional[Callable[[any], Awaitable[None]]] = None,
        on_connection_state_change: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        """
        Initialize WebRTCHandler.

        Args:
            on_audio_track: Callback when remote audio track is received
            on_connection_state_change: Callback when connection state changes
        """
        self._pc: Optional[RTCPeerConnection] = None
        self._output_track: Optional[AudioOutputTrack] = None
        self._input_handler: Optional[AudioInputHandler] = None
        self._relay = MediaRelay()

        self._on_audio_track = on_audio_track
        self._on_connection_state_change = on_connection_state_change

        self._ice_candidates: list[RTCIceCandidate] = []
        self._pending_ice: list[dict] = []
        self._remote_description_set = False

    @property
    def connection_state(self) -> str:
        """Returns the current connection state."""
        if self._pc:
            return self._pc.connectionState
        return "closed"

    @property
    def output_track(self) -> Optional[AudioOutputTrack]:
        """Returns the output audio track."""
        return self._output_track

    @property
    def input_handler(self) -> Optional[AudioInputHandler]:
        """Returns the input audio handler."""
        return self._input_handler

    async def create_offer(self) -> dict:
        """
        Create a new peer connection and generate an SDP offer.

        Returns:
            SDP offer as dict with 'type' and 'sdp' keys
        """
        # Create peer connection with ICE servers
        self._pc = RTCPeerConnection(configuration=ICE_SERVERS)

        # Set up event handlers
        self._pc.on("connectionstatechange", self._on_connection_state_changed)
        self._pc.on("icecandidate", self._on_ice_candidate)
        self._pc.on("track", self._on_track)

        # Create output track for TTS audio
        self._output_track = AudioOutputTrack()
        self._pc.addTrack(self._output_track)

        # Create input handler for receiving audio
        self._input_handler = AudioInputHandler()

        # Create offer
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        logger.info("Created SDP offer")

        return {
            "type": "offer",
            "sdp": self._pc.localDescription.sdp,
        }

    async def handle_answer(self, sdp: str) -> None:
        """
        Handle SDP answer from the client.

        Args:
            sdp: The SDP answer string
        """
        if not self._pc:
            raise RuntimeError("No peer connection")

        answer = RTCSessionDescription(sdp=sdp, type="answer")
        await self._pc.setRemoteDescription(answer)
        self._remote_description_set = True

        logger.info("Set remote description from answer")

        # Process any pending ICE candidates
        for candidate_data in self._pending_ice:
            await self._add_ice_candidate(candidate_data)
        self._pending_ice.clear()

    async def handle_ice_candidate(self, candidate_data: dict) -> None:
        """
        Handle ICE candidate from the client.

        Args:
            candidate_data: ICE candidate dict with 'candidate', 'sdpMid', 'sdpMLineIndex'
        """
        if self._remote_description_set:
            await self._add_ice_candidate(candidate_data)
        else:
            # Queue candidate until remote description is set
            self._pending_ice.append(candidate_data)

    async def _add_ice_candidate(self, candidate_data: dict) -> None:
        """Add an ICE candidate to the peer connection."""
        if not self._pc:
            return

        candidate_str = candidate_data.get("candidate", "")
        if not candidate_str:
            return

        try:
            candidate = RTCIceCandidate(
                component=1,
                foundation="",
                ip="",
                port=0,
                priority=0,
                protocol="udp",
                type="host",
                sdpMid=candidate_data.get("sdpMid"),
                sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
            )
            # Parse the candidate string
            # aiortc expects the full candidate string
            await self._pc.addIceCandidate(candidate)
            logger.debug(f"Added ICE candidate: {candidate_str[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to add ICE candidate: {e}")

    def _on_connection_state_changed(self) -> None:
        """Handle connection state changes."""
        state = self._pc.connectionState
        logger.info(f"Connection state changed: {state}")

        if self._on_connection_state_change:
            asyncio.create_task(self._on_connection_state_change(state))

    def _on_ice_candidate(self, candidate: RTCIceCandidate) -> None:
        """Handle new local ICE candidate."""
        if candidate:
            self._ice_candidates.append(candidate)
            logger.debug(f"New ICE candidate: {candidate}")

    async def _on_track(self, track) -> None:
        """Handle incoming track from peer."""
        logger.info(f"Received track: {track.kind}")

        if track.kind == "audio":
            # Route to input handler
            if self._input_handler:
                await self._input_handler.handle_track(track)

            # Notify callback
            if self._on_audio_track:
                await self._on_audio_track(track)

    def get_ice_candidates(self) -> list[dict]:
        """
        Get all gathered ICE candidates.

        Returns:
            List of ICE candidate dicts
        """
        candidates = []
        for c in self._ice_candidates:
            candidates.append({
                "candidate": str(c),
                "sdpMid": c.sdpMid,
                "sdpMLineIndex": c.sdpMLineIndex,
            })
        return candidates

    async def close(self) -> None:
        """Close the peer connection and clean up."""
        if self._input_handler:
            await self._input_handler.stop()
            self._input_handler = None

        if self._pc:
            await self._pc.close()
            self._pc = None

        self._output_track = None
        logger.info("WebRTC connection closed")


class WebRTCSession:
    """
    High-level session manager combining WebRTC and signaling.

    Handles the full signaling flow over WebSocket.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.webrtc = WebRTCHandler(
            on_connection_state_change=self._on_state_change,
        )
        self._send_message: Optional[Callable[[str], Awaitable[None]]] = None
        self._connected = asyncio.Event()

    def set_send_callback(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Set the callback for sending WebSocket messages."""
        self._send_message = callback

    async def _on_state_change(self, state: str) -> None:
        """Handle connection state changes."""
        if state == "connected":
            self._connected.set()
        elif state in ("failed", "closed"):
            self._connected.clear()

    async def start(self) -> None:
        """Start the session by creating and sending an offer."""
        offer = await self.webrtc.create_offer()

        if self._send_message:
            await self._send_message(json.dumps(offer))

        # Send ICE candidates after a short delay
        await asyncio.sleep(0.5)
        for candidate in self.webrtc.get_ice_candidates():
            if self._send_message:
                msg = {"type": "ice-candidate", **candidate}
                await self._send_message(json.dumps(msg))

    async def handle_message(self, message: str) -> None:
        """
        Handle a signaling message from the WebSocket.

        Args:
            message: JSON string message
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "answer":
                await self.webrtc.handle_answer(data.get("sdp", ""))

            elif msg_type == "ice-candidate":
                await self.webrtc.handle_ice_candidate(data)

            elif msg_type == "app-message":
                # Forward to application layer
                logger.debug(f"App message: {data}")

            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def wait_connected(self, timeout: float = 30.0) -> bool:
        """Wait for WebRTC connection to be established."""
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def close(self) -> None:
        """Close the session."""
        await self.webrtc.close()


async def test_webrtc():
    """Simple test of WebRTC handler."""
    logging.basicConfig(level=logging.DEBUG)

    handler = WebRTCHandler()
    offer = await handler.create_offer()

    print("SDP Offer:")
    print(offer["sdp"][:500])
    print("...")

    print(f"\nICE candidates: {len(handler.get_ice_candidates())}")

    await handler.close()
    print("Test complete")


if __name__ == "__main__":
    asyncio.run(test_webrtc())
