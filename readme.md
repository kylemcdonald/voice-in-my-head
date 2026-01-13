# Voice In My Head

## eSIM setup

1. Remove any previous eSIM.
2. Sign up for a new eSIM using [Airalo](https://airalo.com/) or [Nomad](https://www.getnomad.app/).
3. Install the eSIM, and turn on Data Roaming.
4. Check the [latency to cloud servers](https://cloudpingtest.com/digital_ocean).

## Setup

Create a cloud server. If installing on Digital Ocean, make sure to enable [the agent with advanced metrics](https://docs.digitalocean.com/products/monitoring/how-to/install-agent/).

For 4 users, 8 CPUs and 16 GB RAM is recommended. After creating the machine, add the IP address to the appropriate DNS record.

Prep the packages:

```sh
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential # needed for streamp3 package
sudo apt install -y libmp3lame-dev # needed for elevenlabs
sudo apt install -y ffmpeg # for processing elevenlabs input
```

Install Anaconda:

```sh
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b
$HOME/anaconda3/bin/conda init
source ~/.bashrc
rm Anaconda3-2024.06-1-Linux-x86_64.sh
```

Clone the repo:

```sh
git clone https://github.com/kylemcdonald/voice-in-my-head.git
cd voice-in-my-head
```

Create the environment:

```sh
conda create -y -n vimh python=3.9
conda activate vimh
conda install -y -c conda-forge libstdcxx-ng # needed for daily-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Setup nginx:

```sh
# first, edit .nginx to represent the desired subdomain
sudo apt install -y nginx
sudo ufw allow 'Nginx Full'
sudo cp .nginx /etc/nginx/sites-available/vimh.iyoiyo.studio
sudo ln -s /etc/nginx/sites-available/vimh.iyoiyo.studio /etc/nginx/sites-enabled/
```

Setup certbot:

```sh
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
sudo certbot --nginx
```

Install nvm, Node, and Tailwind:

```sh
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install 16
npm install -D tailwindcss
npx tailwindcss init
npm run buildcss
```

Fill out the .env file with the appropriate keys. Make sure that the Deepgram API key has the "Member" role.

```
ELEVENLABS_API_KEY=...
OPENAI_API_KEY=...
DAILY_API_KEY=...
DEEPGRAM_API_KEY=...
TURN_TIME_SECONDS=50
TOTAL_TIME_MINUTES=25
ROOM_EXPIRE_MINUTES=35
LOCATION=...
```

Install the service:

```sh
bash install-service.sh
```

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

## Testing

Run server with Flask autoreloading:

```sh
flask --app server.py --debug run
```

Run the server with gunicorn:

```sh
gunicorn -w 4 server:app
```

Shortcut for running gunicorn:

```sh
./run.sh
```

## Setting the Duration

The duration of the experience is controlled in the .env file.

## Notes on sound design

Sounds should match the audio stream:

* 1 channels (mono)
* 44.1kHz
* 16-bit depth

Note they might get slightly glitched by the compression and streaming algorithms.

They should also always fade out quickly, or sometimes they can create a lingering noise.

`helpers/prepare-sound.sh` will help prepare sounds for this format.

## Notes on the script

Each row of the script has a function, input and output.

The function is the name of a function instead the `VoiceInMyHead` class. The input is the input to that function, and the output is where the output is saved.

When you save output to a variable, you can reference that as an input in later rows.

If a variable is referenced in a `speak` line, it should be surrounded by {curly braces}. (This is because the `speak` lines get preprocessed and combined.)
