#!/usr/bin/env bash
# node_egress_setup.sh — egress for a fresh compute node, following the server convention.
#
# CONVENTION (see Server/SERVER_SETUP_CONVENTION.md + Coding/vps-operations.md §8.2)
#   * the node must have a working egress for: pip (download.pytorch.org), git
#     (github.com) and optionally huggingface.co;
#   * when direct access is blocked, a **tunnel to an upstream proxy** is installed
#     as a systemd unit with Restart=always, and the proxy variables live in
#     `/root/.pi/env` **with a scheme** (`http://127.0.0.1:7897`) — a bare
#     `127.0.0.1:7897` makes pi's Node side die with ERR_INVALID_URL;
#   * `NO_PROXY` whitelists the directly-reachable domains so that traffic that
#     does not need the tunnel never enters it.
#
# TWO PATTERNS (auto-detected, both are idempotent)
#   tunnel : node's 127.0.0.1:7897 -> upstream's mihomo. Either the upstream pushes
#            its proxy here (`expose_mihomo_to_node.sh` on the upstream, ssh -R),
#            or this node pulls it (`--mode tunnel --upstream`, ssh -L).
#   socks  : node runs `ssh -D 127.0.0.1:1080 <upstream>` and exports
#            ALL_PROXY=socks5h://… — no proxy needed on the upstream, just SSH.
#
# USAGE
#   node_egress_setup.sh probe                       # what is reachable directly?
#   node_egress_setup.sh auto   [--upstream U]       # probe, then install the best pattern
#   node_egress_setup.sh tunnel --upstream root@1.2.3.4 -p 22 [--key /root/.ssh/id_x]
#   node_egress_setup.sh socks  --upstream root@1.2.3.4 -p 13212 [--key ...]
#   node_egress_setup.sh verify                      # re-test through the configured path
#   node_egress_setup.sh status
set -uo pipefail

MODE=${1:-auto}; shift || true
UPSTREAM=""; UPORT=22; KEY=""; ALIAS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --upstream) UPSTREAM=$2; shift ;;
    -p|--port)  UPORT=$2; shift ;;
    --key)      KEY=$2; shift ;;
    --alias)    ALIAS=$2; shift ;;
  esac
  shift
done
[ -n "$ALIAS" ] || ALIAS=$(hostname)
UNIT=/etc/systemd/system/pi-tunnel-${ALIAS}.service
ENVF=/root/.pi/env
DOMAINS=(https://github.com https://pypi.org/simple/ https://download.pytorch.org/whl/ https://huggingface.co)

log() { printf '\n=== %s ===\n' "$*"; }

http_code() {   # $1 = url  ->  3-digit code, never empty (curl exits non-zero on some TLS oddities)
  local c
  c=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$1" 2>/dev/null)
  printf '%s' "${c:-000}"
}

probe_direct() {
  log "direct reachability"
  local ok=1
  for u in "${DOMAINS[@]}"; do
    c=$(http_code "$u")
    printf '  %-38s %s\n' "$u" "$c"
    case "$u" in *huggingface*) ;; *) [ "$c" = "200" ] || ok=0 ;; esac
  done
  return $((1-ok))          # 0 = all required domains reachable directly
}

write_env() {   # $1 = proxy url or "" for direct
  mkdir -p /root/.pi
  {
    echo "# managed by node_egress_setup.sh ($(date -Iseconds))"
    if [ -n "$1" ]; then
      echo "export HTTP_PROXY=$1"
      echo "export HTTPS_PROXY=$1"
      echo "export http_proxy=$1"
      echo "export https_proxy=$1"
    fi
    echo "export NO_PROXY=localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.aliyuncs.com"
    echo "export no_proxy=\$NO_PROXY"
  } > "$ENVF"
  chmod 600 "$ENVF"
  echo "wrote $ENVF:"
  sed -n '1,12p' "$ENVF"
}

install_tunnel() {
  [ -n "$UPSTREAM" ] || { echo "ERROR: --upstream required for tunnel mode" >&2; return 2; }
  local host=${UPSTREAM#*@}
  local keyopt=""
  [ -n "$KEY" ] && keyopt="-i $KEY"
  log "installing reverse-pull tunnel: node 127.0.0.1:7897 -> $UPSTREAM mihomo"
  cat > "$UNIT" <<EOF
[Unit]
Description=Egress tunnel: local 7897 -> $UPSTREAM mihomo (host convention)
After=network-online.target
Wants=network-online.target

[Service]
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\
  -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new $keyopt \\
  -p $UPORT -L 127.0.0.1:7897:127.0.0.1:7897 $UPSTREAM
Restart=always
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload && systemctl enable --now "pi-tunnel-${ALIAS}.service"
  sleep 3
  systemctl is-active "pi-tunnel-${ALIAS}.service"
  write_env "http://127.0.0.1:7897"
}

install_socks() {
  [ -n "$UPSTREAM" ] || { echo "ERROR: --upstream required for socks mode" >&2; return 2; }
  local keyopt=""
  [ -n "$KEY" ] && keyopt="-i $KEY"
  log "installing dynamic SOCKS tunnel: local 1080 -> $UPSTREAM egress"
  cat > "$UNIT" <<EOF
[Unit]
Description=Egress SOCKS: local 1080 -> $UPSTREAM (host convention)
After=network-online.target
Wants=network-online.target

[Service]
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\
  -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new $keyopt \\
  -p $UPORT -D 127.0.0.1:1080 $UPSTREAM
Restart=always
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload && systemctl enable --now "pi-tunnel-${ALIAS}.service"
  sleep 3
  systemctl is-active "pi-tunnel-${ALIAS}.service"
  write_env "socks5h://127.0.0.1:1080"
}

verify() {
  log "verify through the configured path"
  set -a; # shellcheck disable=SC1090
  [ -f "$ENVF" ] && . "$ENVF"; set +a
  for u in "${DOMAINS[@]}"; do
    printf '  %-38s %s\n' "$u" "$(http_code "$u")"
  done
  echo "  (github/pypi/pytorch must be 200; huggingface 200/000 both acceptable — the tokenizer cache is shipped with the data)"
}

main() {
  case "$MODE" in
    probe)  probe_direct; exit 0 ;;
    auto)
      if probe_direct; then
        log "direct egress is sufficient — no proxy installed"
        write_env ""
        verify
      else
        [ -n "$UPSTREAM" ] || { echo "direct egress incomplete and no --upstream given"; exit 3; }
        install_tunnel
        verify
      fi ;;
    tunnel) install_tunnel; verify ;;
    socks)  install_socks; verify ;;
    verify) verify ;;
    status)
      systemctl status "pi-tunnel-${ALIAS}.service" --no-pager | head -8
      ss -ltnp | grep -E '7897|1080' || echo "no tunnel ports listening"
      cat "$ENVF" 2>/dev/null ;;
    *) echo "unknown mode: $MODE" >&2; exit 2 ;;
  esac
}
main