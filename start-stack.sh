#!/usr/bin/env bash
# Boot-safe launcher for the dubbing stack. Run by dubbing.service at boot,
# and safe to run by hand at any time — it is idempotent.
#
# `docker compose up -d` alone is not enough at boot on this box:
#
#   1. EXPOSE_BIND=${TAILSCALE_IP} binds postgres/redis/minio to the tailscale
#      address. docker.service is ordered only after network-online.target and
#      has no dependency on tailscaled.service, so docker can start before
#      tailscale0 holds that IP — every bind then fails and the containers are
#      left half-started (this is how the stack broke previously).
#   2. The UI port can be claimed by another project on this machine, which
#      fails the whole `up` with "port is already allocated".
#
# So: wait for the bind address to exist, then start; if a published port is
# genuinely taken by something else, move the UI to the next free port and
# record that choice in .env so the URL stays stable across reboots.

set -uo pipefail

PROJECT_DIR="/home/ubuntu/dubbing"
cd "$PROJECT_DIR" || exit 1

log() { echo "[dubbing-stack] $*"; }

env_get() {
	# Last assignment wins, mirroring how docker compose reads .env.
	grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2- |
		tr -d "\"'" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

# ── 1. Wait for the tailscale bind address ────────────────────────────────────
# Empty EXPOSE_BIND/TAILSCALE_IP means container-internal only, nothing to wait
# for. Otherwise block until the IP is actually on an interface.
BIND_IP="$(env_get TAILSCALE_IP)"

if [ -n "$BIND_IP" ]; then
	for i in $(seq 1 90); do
		if ip -4 addr show 2>/dev/null | grep -qw "$BIND_IP"; then
			log "bind address $BIND_IP is up (waited $((i - 1))s)"
			break
		fi
		if [ "$i" -eq 90 ]; then
			# Don't hard-fail: the UI binds to UI_BIND, not EXPOSE_BIND, so the
			# app itself may still be usable. Let compose report what breaks.
			log "WARNING: $BIND_IP never appeared after 90s; starting anyway"
		fi
		sleep 1
	done
fi

# ── 2. Start, relocating the UI port if it is already taken ───────────────────
port_is_free() {
	! ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"
}

next_free_port() {
	local p=$1
	while [ "$p" -lt 65535 ]; do
		port_is_free "$p" && { echo "$p"; return 0; }
		p=$((p + 1))
	done
	return 1
}

for attempt in 1 2 3; do
	out="$(docker compose up -d --remove-orphans 2>&1)"
	rc=$?
	echo "$out"

	if [ $rc -eq 0 ]; then
		log "stack up on port $(env_get UI_PORT)"
		exit 0
	fi

	# Only a port collision is worth retrying on a different port; anything
	# else (bad image, bad env, disk full) must surface as a real failure.
	if ! grep -qiE 'port is already allocated|address already in use|bind for .* failed' <<<"$out"; then
		log "ERROR: compose failed for a reason unrelated to ports; not retrying"
		exit $rc
	fi

	cur="$(env_get UI_PORT)"
	cur="${cur:-8621}"
	new="$(next_free_port $((cur + 1)))" || { log "ERROR: no free port available"; exit 1; }

	log "port $cur is taken by another project; moving UI to $new"
	if grep -qE '^UI_PORT=' .env; then
		sed -i -E "s|^UI_PORT=.*|UI_PORT=$new|" .env
	else
		printf '\nUI_PORT=%s\n' "$new" >>.env
	fi
done

log "ERROR: stack did not come up after 3 attempts"
exit 1
