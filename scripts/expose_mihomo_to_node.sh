#!/usr/bin/env bash
# expose_mihomo_to_node.sh — push this machine's mihomo (127.0.0.1:7897) to a compute node.
#
# WHY THIS DIRECTION
#   The rented GPU nodes are publicly reachable; this machine (cpu2) is behind NAT
#   (192.168.122.0/24) and cannot be reached from outside. So the tunnel is opened
#   *from here to the node* with `ssh -R`, which binds the node's loopback:7897 and
#   forwards it to this machine's mihomo. The node then uses
#   `http://127.0.0.1:7897` and needs no credentials for anything.
#
# USAGE (on cpu2)
#   expose_mihomo_to_node.sh <node-host> <node-port> [--user root] [--key ~/.ssh/id_x] [--alias NAME]
#   expose_mihomo_to_node.sh --list          # show installed tunnels
#   expose_mihomo_to_node.sh --remove <alias>
#
# NOTES
#   * unit name: pi-tunnel-<alias>.service  (same naming as the convention)
#   * Restart=always + ServerAliveInterval keeps it alive across network blips;
#   * the first connection records the node's host key (accept-new) — verify the
#     fingerprint against the provider's console if you care about TOFU.
set -uo pipefail

NODE=""; NPORT=22; NUSER=root; KEY=""; ALIAS=""; ACTION="install"

case "${1:-}" in
  --list)
    ACTION=list ;;
  --remove)
    ACTION=remove; ALIAS=${2:?alias required} ;;
  *)
    NODE=${1:?usage: expose_mihomo_to_node.sh <node-host> <node-port> [--user U] [--key K] [--alias A]}
    NPORT=${2:?port required}; shift 2 || true
    while [ $# -gt 0 ]; do
      case "$1" in
        --user) NUSER=$2; shift ;;
        --key)  KEY=$2; shift ;;
        --alias) ALIAS=$2; shift ;;
      esac
      shift
    done ;;
esac
[ -n "$ALIAS" ] || ALIAS=$(printf '%s' "$NODE" | tr '.' '-')
UNIT=/etc/systemd/system/pi-tunnel-${ALIAS}.service

if [ "$ACTION" = "list" ]; then
  for f in /etc/systemd/system/pi-tunnel-*.service; do
    [ -e "$f" ] || continue
    svc=$(basename "$f" .service)
    printf '%-34s %-8s %s\n' "$svc" "$(systemctl is-active "$svc")" "$(grep -oE '\-L [0-9.:]+' "$f" | head -1)"
  done
  exit 0
fi

if [ "$ACTION" = "remove" ]; then
  systemctl disable --now "pi-tunnel-${ALIAS}.service" 2>/dev/null
  rm -f "$UNIT" && systemctl daemon-reload
  echo "removed pi-tunnel-${ALIAS}.service"
  exit 0
fi

# sanity: local mihomo must be alive, otherwise the tunnel forwards nothing
code=$(curl -s -o /dev/null -m 8 -x http://127.0.0.1:7897 -w '%{http_code}' https://github.com || echo 000)
echo "local mihomo check: github via 127.0.0.1:7897 -> $code"
[ "$code" = "200" ] || { echo "WARNING: local mihomo is not serving 200s; the tunnel will still be installed"; }

keyopt=""
[ -n "$KEY" ] && keyopt="-i $KEY"

cat > "$UNIT" <<EOF
[Unit]
Description=Egress tunnel: push local mihomo 7897 -> node $NODE (pi tunnel convention)
After=network-online.target
Wants=network-online.target

[Service]
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\
  -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new $keyopt \\
  -p $NPORT -R 127.0.0.1:7897:127.0.0.1:7897 $NUSER@$NODE
Restart=always
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now "pi-tunnel-${ALIAS}.service"
sleep 4
systemctl is-active "pi-tunnel-${ALIAS}.service" || { systemctl status "pi-tunnel-${ALIAS}.service" --no-pager | tail -12; exit 1; }

echo
echo "installed pi-tunnel-${ALIAS}.service -> $NODE:$NPORT (node's 127.0.0.1:7897 now serves this mihomo)"
echo "on the node run:  bash scripts/node_egress_setup.sh verify      # expect github/pypi/pytorch 200"
echo "tip: add the node to ~/.ssh/config and to the burst notes for traceability"