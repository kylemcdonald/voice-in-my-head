SERVICE_ID=vimh
SERVICE_NAME="Voice In My Head"

USER=$(whoami)
SERVICES_DIR=/etc/systemd/system/

cat >$SERVICES_DIR/$SERVICE_ID.service <<EOL
[Unit]
Description=$SERVICE_NAME
Wants=network-online.target
After=network-online.target
[Service]
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/run.sh
User=$USER
Restart=always
[Install]
WantedBy=multi-user.target
EOL

systemctl daemon-reload

systemctl enable $SERVICE_ID
systemctl start $SERVICE_ID