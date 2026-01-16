# Voice In My Head

## eSIM setup

1. Remove any previous eSIM.
2. Sign up for a new eSIM using [Airalo](https://airalo.com/) or [Nomad](https://www.getnomad.app/).
3. Install the eSIM, and turn on Data Roaming.
4. Check the [latency to cloud servers](https://cloudpingtest.com/digital_ocean).

## Server Setup

Create a cloud server on Digital Ocean. For 4 simultaneous sessions, 2 vCPU and 4GB RAM is sufficient. After creating the machine, add the IP address to the appropriate DNS record.

### Install System Dependencies

```sh
sudo apt update
sudo apt upgrade -y
sudo apt install -y ffmpeg git curl  # ffmpeg is REQUIRED for voice cloning
```

**Important**: `ffmpeg` is required for voice cloning - it converts recorded audio to MP3 for the ElevenLabs API.

### Install uv (Python package manager)

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### Install Caddy (HTTPS reverse proxy)

```sh
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy
```

### Clone and Install

```sh
cd /opt
sudo git clone -b nodaily https://github.com/kylemcdonald/voice-in-my-head.git vimh
cd vimh
sudo uv sync
```

### Configure Environment

Create `/opt/vimh/.env`:

```
ELEVENLABS_API_KEY=...
OPENAI_API_KEY=...
ASSEMBLYAI_API_KEY=...
TURN_TIME_SECONDS=50
TOTAL_TIME_MINUTES=25
LOCATION=...
```

### Configure Caddy

Edit `/etc/caddy/Caddyfile`:

```
your-domain.example.com {
    reverse_proxy localhost:8000
}
```

### Create Systemd Service

Create `/etc/systemd/system/vimh.service`:

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
sudo systemctl daemon-reload
sudo systemctl enable vimh
sudo systemctl start vimh
sudo systemctl restart caddy
```

Caddy will automatically obtain an SSL certificate from Let's Encrypt.

## Setting up the iPhones

* Setup the iPhones without any Apple account
* Go to Settings > Notifications
    * Display As: Count
    * Siri Suggestions: Off
    * App-by-app: Off
    * Emergency Alerts: Off
* Get Airalo e-SIM for region
* Settings > Cellular Data Options > Data Roaming: On
* Home screen: remove Calendar stack, move bottom bar to lower row
* Enable AirDrop and share `phone/background.png` picture
* Under photo sharing at bottom left, set as wallpaper and pinch to resize
* Safari > Microphone > Allow
* Lock to portrait mode
* After connecting the AirPods, turn Automatic Ear Detection: Off
* Disable [NameDrop](https://support.apple.com/guide/personal-safety/secure-airdrop-and-namedrop-ips7d84d2cdc/)

### Setting up Guided Access

[Apple Reference](https://support.apple.com/en-us/HT202612)

* Open iOS page and "Add Shortcut to Home Screen"
* Go to Settings > Accessibility, then turn on Guided Access.
* Tap Passcode Settings, then tap Set Guided Access Passcode.
* Open the app from home screen.
* Triple-click the Home button.
* Set options:
    * Side Button ON
    * Volume Buttons ON
    * Motion OFF
    * Software Keyboards ON
    * Touch ON
    * Dictionary OFF
    * Time Limit OFF
* Tap Guided Access, then tap Start.

## Local Development

```sh
# Install dependencies
uv sync

# Run server
uv run python server_async.py

# For HTTPS (required for WebRTC microphone access)
# Run Caddy in another terminal:
caddy run
```

## Setting the Duration

The duration of the experience is controlled in the .env file.

## Notes on sound design

Sounds should match the WebRTC audio stream:

* 1 channel (mono)
* 48kHz (WebRTC native sample rate)
* 16-bit depth

They should also always fade out quickly, or sometimes they can create a lingering noise.

`helpers/prepare-sound.sh` will help prepare sounds for this format.

## Notes on the script

Each row of the script has a function, input and output.

The function is the name of a function instead the `VoiceInMyHead` class. The input is the input to that function, and the output is where the output is saved.

When you save output to a variable, you can reference that as an input in later rows.

If a variable is referenced in a `speak` line, it should be surrounded by {curly braces}. (This is because the `speak` lines get preprocessed and combined.)
