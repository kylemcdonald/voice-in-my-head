#!/bin/bash
# Provision the network test app on a fresh Ubuntu 24.04 droplet (run as root).
# Expects the repo subset in /opt/vimh-nettest and a .env there with API keys and NETTEST_* settings.
# Usage: APP_HOST=test.example.com bash setup-app.sh
set -euo pipefail
: "${APP_HOST:?}"

export DEBIAN_FRONTEND=noninteractive
if ! command -v caddy >/dev/null; then
  apt-get update -q
  apt-get install -y -q debian-keyring debian-archive-keyring apt-transport-https curl gpg ffmpeg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q
  apt-get install -y -q caddy
fi
command -v uv >/dev/null || [ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh

cd /opt/vimh-nettest
/root/.local/bin/uv sync --no-install-project

cat > /etc/caddy/Caddyfile <<EOF
$APP_HOST {
    reverse_proxy localhost:8000
}
EOF
systemctl reload caddy || systemctl restart caddy

cat > /etc/systemd/system/vimh-nettest.service <<EOF
[Unit]
Description=VIMH network test
After=network.target

[Service]
WorkingDirectory=/opt/vimh-nettest
ExecStart=/root/.local/bin/uv run --no-sync python nettest/server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable vimh-nettest
systemctl restart vimh-nettest
sleep 2
systemctl --no-pager status vimh-nettest | head -5
