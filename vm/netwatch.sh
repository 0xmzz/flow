#!/bin/sh
# Heal the guest's network after the Hyper-V Default Switch renumbers.
#
# The Default Switch is a NAT switch that Microsoft re-subnets on every host
# reboot (this guest has seen 172.24 -> 172.25 -> 172.23 -> 172.27). A guest
# holding a lease from the previous subnet goes completely dark: no route to
# the host, so the Flow bridge stops and nothing on the host side can reach in
# to fix it. Renewing DHCP is all that's needed — the daemon discovers the host
# as the default gateway at call time, so any new subnet just works.
#
# Installed by vm/setup.sh as a systemd timer, run once a minute.

CONN="${FLOW_NM_CONNECTION:-flow-dhcp}"

gateway() {
    ip route | awk '/^default/ {print $2; exit}'
}

GW=$(gateway)

# Reachable gateway = healthy. Cheap check, no dependency on the Flow server
# being up (beast sleeps; that must not trigger a network rebuild).
if [ -n "$GW" ] && ping -c 1 -W 2 "$GW" >/dev/null 2>&1; then
    exit 0
fi

logger -t flow-netwatch "gateway '${GW:-none}' unreachable; renewing DHCP"

if command -v nmcli >/dev/null 2>&1 && nmcli -t -f NAME con show 2>/dev/null | grep -qx "$CONN"; then
    nmcli con down "$CONN" >/dev/null 2>&1
    nmcli con up "$CONN" >/dev/null 2>&1
elif command -v dhclient >/dev/null 2>&1; then
    IFACE=$(ip -o link show | awk -F': ' '$2 !~ /lo/ {print $2; exit}')
    dhclient -r "$IFACE" >/dev/null 2>&1
    dhclient "$IFACE" >/dev/null 2>&1
else
    logger -t flow-netwatch "no nmcli connection '$CONN' and no dhclient; cannot heal"
    exit 1
fi

NEW=$(gateway)
if [ -n "$NEW" ] && ping -c 1 -W 2 "$NEW" >/dev/null 2>&1; then
    logger -t flow-netwatch "recovered; gateway now $NEW"
else
    logger -t flow-netwatch "renew did not restore a reachable gateway"
fi
