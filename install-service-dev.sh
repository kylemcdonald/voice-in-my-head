SERVICE_ID=vimh-dev
SERVICE_NAME="Voice In My Head (Dev)"

USER=$(whoami)
SERVICES_DIR=/etc/systemd/system/

cat >$SERVICES_DIR/$SERVICE_ID.service <<EOL
[Unit]
Description=$SERVICE_NAME
Wants=network-online.target
After=network-online.target
[Service]
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/run-dev.sh
User=$USER
Restart=always
[Install]
WantedBy=multi-user.target
EOL

systemctl daemon-reload

systemctl enable $SERVICE_ID
systemctl start $SERVICE_ID