# Voice In My Head: venue network requirements

## What the piece does on the network

Each visitor wears headphones connected to a phone. A web page on the phone holds a live, two-way voice conversation with our server in Frankfurt, Germany, for about 25 minutes. Several visitors can be connected at once.

- The web page and its control channel use ordinary HTTPS and secure WebSockets.
- The voice audio is WebRTC: encrypted, real-time audio, ideally over UDP.
- The phones only talk to our two servers. Speech recognition, voice synthesis and the language model are called from Frankfurt, so the venue network does not need access to any other service.

## Firewall rules

| Destination | Protocol / port | Purpose | Needed? |
|---|---|---|---|
| App server | TCP 443 (HTTPS, WebSocket) | Web page and control channel | **Required** |
| Relay server | UDP 443 | Voice audio (preferred path) | **Required** |
| Relay server | TCP 443 (TURN over TLS) | Voice audio fallback if UDP is blocked | **Required** |
| Relay server | UDP and TCP 3478 | Standard TURN port | Optional |
| App server | UDP 49152–65535 | Direct voice audio, no relay | Optional |
| stun.l.google.com | UDP 19302 | Address discovery for direct audio | Optional |

The relay is a standard TURN server (coturn). It sits in the same Frankfurt data center as the app server, so using it adds no noticeable delay.

Current test servers (production addresses will be confirmed before opening):

- App server: `vimh-test.iyoiyo.studio` (164.92.185.10)
- Relay server: `vimh-turn.iyoiyo.studio` (46.101.124.6)

## Please also make sure

- **No TLS/SSL inspection** for these two servers. Visitors' phones won't trust a re-signed certificate, and the page will fail to load.
- **No VoIP or application-control blocking** for these two servers. If your firewall classifies or throttles real-time voice traffic, please allowlist both addresses.
- **Long-lived connections are allowed.** One WebSocket and one audio stream stay open for 30 minutes or more per visitor. Please don't time out idle or long-running connections in under an hour.
- **No captive portal** on the network the phones use, or the phones are pre-authenticated. A login page that appears mid-session drops the audio.
- **Phones can reach the internet directly**, not only through an HTTP proxy. WebRTC audio can't pass through an explicit web proxy.
- **DNS resolves both hostnames** and doesn't filter them.
- Client isolation on the Wi-Fi is fine; phones never need to talk to each other.

## Capacity and quality

- **Bandwidth:** under 100 kbit/s each way per visitor. 1 Mbit/s each way is enough for 10 simultaneous visitors.
- **Latency to Frankfurt:** under 200 ms round trip. From Dubai we measured about 125 ms.
- **Packet loss:** under 2%. **Jitter:** under 30 ms.
- **Wi-Fi:** strong 5 GHz coverage in the whole visitor area. Audio is sensitive to drop-outs and roaming between access points more than to raw speed.

## How to check

Open the test link we send you on the venue network, on the phones that will be used in the installation, and tap **Start**. It takes about a minute. Send us a screenshot of the result. A good result says at least one path works and "voice round trip confirmed". Ideally TURN UDP 443 is green.

What we found on a Dubai network so far (Emirates Internet, three devices): UDP to most ports is blocked, including UDP 3478. UDP 443 and TCP 443 work well, with 0% loss and about 125 ms round trip. So the relay on port 443 is the path we expect to use.
