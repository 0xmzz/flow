#!/bin/bash
# vmflow installer. Run INSIDE the Linux VM:
#   curl -s http://<host>:8756/vm/setup.sh | sudo bash -s -- <FLOW_TOKEN>
set -e
TOKEN="$1"
[ -z "$TOKEN" ] && { echo "usage: curl -s http://<host>:8756/vm/setup.sh | sudo bash -s -- <token>"; exit 1; }
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }

# Find the Flow host from FLOW_HOST or the current default gateway.
GW=$(ip route | awk '/^default/ {print $3; exit}')
HOST_URL=""
OVERRIDE="${FLOW_HOST:-auto}"
if [ "$OVERRIDE" != "auto" ]; then
  HOST_URL="${OVERRIDE%/}"
fi
for c in "$GW"; do
  [ -n "$HOST_URL" ] && break
  [ -z "$c" ] && continue
  if curl -s -m 3 "http://$c:8756/health" | grep -q '"ok"'; then
    HOST_URL="http://$c:8756"; break
  fi
done
[ -z "$HOST_URL" ] && { echo "ERROR: cannot reach Flow on the host. Is the VM network up (try: sudo dhclient) and Flow running?"; exit 1; }
echo "Flow host: $HOST_URL"

export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null || true
apt-get install -y -q python3-evdev ydotool curl >/dev/null

mkdir -p /opt/vmflow
curl -s "$HOST_URL/vm/daemon.py" -o /opt/vmflow/daemon.py
curl -s "$HOST_URL/vm/netwatch.sh" -o /opt/vmflow/netwatch.sh
chmod +x /opt/vmflow/netwatch.sh

cat > /etc/vmflow.env <<EOF
FLOW_HOST=$OVERRIDE
FLOW_TOKEN=$TOKEN
YDOTOOL_SOCKET=/run/ydotoold.sock
EOF
chmod 600 /etc/vmflow.env

cat > /etc/systemd/system/ydotoold.service <<'EOF'
[Unit]
Description=ydotool daemon (uinput typing)
[Service]
ExecStart=/usr/bin/ydotoold --socket-path=/run/ydotoold.sock --socket-own=0:0
Restart=always
RestartSec=2
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/vmflow.service <<'EOF'
[Unit]
Description=Flow VM bridge - hold RIGHT CTRL to dictate via the host GPU
After=ydotoold.service network.target
Requires=ydotoold.service
[Service]
EnvironmentFile=/etc/vmflow.env
ExecStart=/usr/bin/python3 /opt/vmflow/daemon.py
Restart=always
RestartSec=2
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/flow-netwatch.service <<'EOF'
[Unit]
Description=Heal guest DHCP after the Hyper-V Default Switch renumbers
[Service]
Type=oneshot
EnvironmentFile=-/etc/vmflow.env
ExecStart=/opt/vmflow/netwatch.sh
EOF

cat > /etc/systemd/system/flow-netwatch.timer <<'EOF'
[Unit]
Description=Check the guest can still reach the Hyper-V host
[Timer]
OnBootSec=30s
OnUnitActiveSec=60s
[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now ydotoold.service vmflow.service >/dev/null 2>&1
systemctl enable --now flow-netwatch.timer >/dev/null 2>&1
sleep 1
systemctl --no-pager -l status vmflow.service | tail -4
echo ""
echo "vmflow installed. Hold RIGHT CTRL anywhere in this VM to dictate."
