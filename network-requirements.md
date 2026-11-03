# Voice In My Head: venue network requirements

## What the piece does on the network

Each visitor wears headphones connected to a phone. A web page on the phone holds a live, two-way voice conversation with our server in Frankfurt, Germany, for about 25 minutes. Several visitors can be connected at once.

- The web page and its control channel use ordinary HTTPS and secure WebSockets.
- The voice audio is WebRTC: encrypted, real-time audio sent over UDP directly between the phone and our server.
- The phones only talk to our server. Speech recognition, voice synthesis and the language model are called from Frankfurt, so the venue network does not need access to any other service.

## Our server

- Hostname: `vimh-fra.iyoiyo.studio`
- IP address: `164.92.185.10` (Frankfurt, Germany)

## Firewall rules

All traffic starts from the phones (outbound). Please allow these destinations and their replies:

| Destination | Protocol / port | Purpose | Needed? |
|---|---|---|---|
| 164.92.185.10 | TCP 443 (HTTPS, WebSocket) | Web page and control channel | **Required** |
| 164.92.185.10 | **UDP 49152–65535** | Voice audio | **Required** |
| stun.l.google.com, stun1.l.google.com | UDP 19302 | Address discovery (helps the connection start) | Optional |

The phone's own UDP source port can be any port. Each visitor's audio stream uses one UDP port on our server, picked from the range above.

## Please also make sure

- **UDP is not blocked or throttled** for our server. If your firewall, ISP policy or application control blocks or throttles VoIP or "unknown UDP" traffic, please allowlist 164.92.185.10. This is the most important item: without UDP the audio can't connect.
- **No TLS/SSL inspection** for our server. Visitors' phones won't trust a re-signed certificate, and the page will fail to load.
- **Long-lived connections are allowed.** One WebSocket and one audio stream stay open for 30 minutes or more per visitor. Please don't time out idle or long-running TCP or UDP flows in under an hour, and keep UDP NAT mappings alive for at least 30 seconds between packets (audio flows continuously, so this is normally fine).
- **No captive portal** on the network the phones use, or the phones are pre-authenticated. A login page that appears mid-session drops the audio.
- **Phones can reach the internet directly**, not only through an HTTP proxy. WebRTC audio can't pass through an explicit web proxy.
- **DNS resolves `vimh-fra.iyoiyo.studio`** and doesn't filter it.
- Client isolation on the Wi-Fi is fine; phones never need to talk to each other.

## Capacity and quality

- **Bandwidth:** under 100 kbit/s each way per visitor. 1 Mbit/s each way is enough for 10 simultaneous visitors.
- **Latency to Frankfurt:** under 200 ms round trip. From Dubai we measured about 125 ms.
- **Packet loss:** under 2%. **Jitter:** under 30 ms.
- **Wi-Fi:** strong 5 GHz coverage in the whole visitor area. Audio is sensitive to drop-outs and roaming between access points more than to raw speed.

## How to check

Open the test link we send you on the venue network, on the phones that will be used in the installation, and tap **Start**. It takes about a minute. Send us a screenshot of the result.

The line that matters is **Direct UDP**: it must show a green check. That is the path the installation uses. The TURN rows test a relay we are not using at the moment, so they don't need to pass.

On a Dubai network we tested earlier (Emirates Internet), Direct UDP was blocked. UDP to our server's ports was dropped, while TCP 443 and UDP 443 worked. That is the kind of policy that needs an exception for our server.
