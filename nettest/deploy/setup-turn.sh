#!/bin/bash
# Provision a coturn TURN server on a fresh Ubuntu 24.04 droplet.
# Usage: TURN_HOST=turn.example.com TURN_SECRET=... bash setup-turn.sh
# Listens on 3478 (UDP/TCP) and 443 (UDP/TCP/TLS) so it can get through restrictive networks.
set -euo pipefail
: "${TURN_HOST:?}" "${TURN_SECRET:?}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q coturn certbot

if [ ! -d "/etc/letsencrypt/live/$TURN_HOST" ]; then
  systemctl stop coturn || true
  certbot certonly --standalone --non-interactive --agree-tos --register-unsafely-without-email -d "$TURN_HOST"
fi

mkdir -p /etc/letsencrypt/renewal-hooks/deploy
cat > /etc/letsencrypt/renewal-hooks/deploy/coturn.sh <<EOF
#!/bin/bash
install -d -o turnserver -g turnserver /etc/coturn/certs
install -m 640 -o turnserver -g turnserver /etc/letsencrypt/live/$TURN_HOST/fullchain.pem /etc/coturn/certs/fullchain.pem
install -m 640 -o turnserver -g turnserver /etc/letsencrypt/live/$TURN_HOST/privkey.pem /etc/coturn/certs/privkey.pem
systemctl restart coturn
EOF
chmod +x /etc/letsencrypt/renewal-hooks/deploy/coturn.sh
# certbot standalone needs port 80 free during renewal; coturn doesn't use it.

cat > /etc/turnserver.conf <<EOF
listening-port=3478
tls-listening-port=443
fingerprint
use-auth-secret
static-auth-secret=$TURN_SECRET
realm=$TURN_HOST
server-name=$TURN_HOST
cert=/etc/coturn/certs/fullchain.pem
pkey=/etc/coturn/certs/privkey.pem
no-tlsv1
no-tlsv1_1
no-cli
no-multicast-peers
denied-peer-ip=0.0.0.0-0.255.255.255
denied-peer-ip=10.0.0.0-10.255.255.255
denied-peer-ip=100.64.0.0-100.127.255.255
denied-peer-ip=127.0.0.0-127.255.255.255
denied-peer-ip=169.254.0.0-169.254.255.255
denied-peer-ip=172.16.0.0-172.31.255.255
denied-peer-ip=192.168.0.0-192.168.255.255
min-port=49152
max-port=65535
total-quota=200
syslog
simple-log
verbose
EOF

# Allow binding to port 443 as the turnserver user.
mkdir -p /etc/systemd/system/coturn.service.d
cat > /etc/systemd/system/coturn.service.d/override.conf <<EOF
[Service]
AmbientCapabilities=CAP_NET_BIND_SERVICE
EOF
[ -f /etc/default/coturn ] && sed -i 's/^#\?TURNSERVER_ENABLED=.*/TURNSERVER_ENABLED=1/' /etc/default/coturn

systemctl daemon-reload
/etc/letsencrypt/renewal-hooks/deploy/coturn.sh
systemctl enable coturn
sleep 1
systemctl --no-pager status coturn | head -5
ss -lntup | grep turnserver
