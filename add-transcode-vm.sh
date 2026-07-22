#!/usr/bin/env bash
#
# add-transcode-vm.sh — turn a fresh VM into a transcode worker.
#
# RUN THIS ON THE VM YOU ARE ADDING, not on the coordinator.
#
#   scp -r docker-compose.remote.yml remote.env Dockerfile requirements.txt app \
#          add-transcode-vm.sh  user@vm:~/dubbing/
#   ssh user@vm
#   cd ~/dubbing && ./add-transcode-vm.sh --tailscale-key tskey-auth-XXXX
#
# There is no agent, no System ID and no registration server. A worker is just a
# Celery process that reaches IN to the coordinator over Tailscale, so a VM
# behind NAT with no public IP works. It appears in `celery inspect` within a
# few seconds of starting — that is the whole "heartbeat".
#
# Secrets are NEVER baked into this file. The Tailscale key is passed as an
# argument (or prompted for) and used once; the coordinator's credentials live
# in remote.env, which should be chmod 600 and never committed.

set -euo pipefail

TS_KEY=""
QUEUES="${QUEUES:-media}"
NAME="${WORKER_NAME:-$(hostname -s)}"
CONCURRENCY="${WORKER_CONCURRENCY:-}"

usage() {
  cat <<'USAGE'
Usage: ./add-transcode-vm.sh [options]

  --tailscale-key KEY   Tailscale auth key (tskey-auth-...). Prompted if omitted.
                        Use a SINGLE-USE, EPHEMERAL, PRE-AUTHORIZED key.
  --queues LIST         Queues to serve (default: media)
                          media  ffmpeg: transcode, thumbnails, shots, mix
                          ai     whisper/diarization/translation (needs ~6GB RAM,
                                 downloads ~4GB of models on first use)
  --name NAME           Worker name in `celery inspect` (default: hostname)
  --concurrency N       Parallel tasks. Defaults to PHYSICAL cores.
  --skip-tailscale      Already on the tailnet; don't touch it.
  -h, --help            This.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tailscale-key) TS_KEY="$2"; shift 2 ;;
    --queues)        QUEUES="$2"; shift 2 ;;
    --name)          NAME="$2"; shift 2 ;;
    --concurrency)   CONCURRENCY="$2"; shift 2 ;;
    --skip-tailscale) TS_KEY="SKIP"; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f remote.env ]] || die "remote.env not found. Copy it from the coordinator (it holds the credentials)."
[[ -f docker-compose.remote.yml ]] || die "docker-compose.remote.yml not found. Copy it from the coordinator."
[[ -d app ]] || die "app/ not found. Copy the whole directory from the coordinator."

# shellcheck disable=SC1091
set -a; . ./remote.env; set +a
: "${COORDINATOR:?COORDINATOR missing from remote.env}"

# ── 1. tailscale ──────────────────────────────────────────────────────────
say "1/5  Tailscale"
if [[ "$TS_KEY" == "SKIP" ]]; then
  ok "skipped by request"
elif command -v tailscale >/dev/null && tailscale status >/dev/null 2>&1; then
  ok "already connected as $(tailscale status --self --peers=false 2>/dev/null | awk '{print $2}' | head -1)"
else
  if [[ -z "$TS_KEY" ]]; then
    # -s: never echo a credential to the terminal or the shell history.
    read -rsp "  Tailscale auth key (input hidden): " TS_KEY; echo
  fi
  [[ -n "$TS_KEY" ]] || die "no auth key given"
  command -v tailscale >/dev/null || {
    echo "  installing tailscale…"
    curl -fsSL https://tailscale.com/install.sh | sh >/dev/null
  }
  sudo tailscale up --auth-key="$TS_KEY" >/dev/null
  unset TS_KEY   # out of the environment as soon as it is spent
  ok "joined the tailnet"
fi

# ── 2. can we actually see the coordinator? ───────────────────────────────
say "2/5  Reaching the coordinator at $COORDINATOR"
for probe in "${REDIS_PORT:-6479}:redis" "${PG_PORT:-5532}:postgres" "${MINIO_PORT:-9100}:minio"; do
  port="${probe%%:*}"; svc="${probe##*:}"
  if timeout 5 bash -c "</dev/tcp/$COORDINATOR/$port" 2>/dev/null; then
    ok "$svc reachable on $port"
  else
    die "cannot reach $svc at $COORDINATOR:$port
       · is this VM on the same tailnet?  tailscale status
       · is the coordinator up?           docker compose ps
       · does it publish on tailscale?    EXPOSE_BIND in its .env"
  fi
done

# ── 3. docker ─────────────────────────────────────────────────────────────
say "3/5  Docker"
if command -v docker >/dev/null; then
  ok "already installed"
else
  echo "  installing docker…"
  curl -fsSL https://get.docker.com | sh >/dev/null
  sudo usermod -aG docker "$USER" || true
  warn "you were added to the 'docker' group — log out and back in if the next step fails"
fi
docker compose version >/dev/null 2>&1 || die "docker compose v2 not available"

# ── 4. size it to the machine ─────────────────────────────────────────────
say "4/5  Sizing"
if [[ -z "$CONCURRENCY" ]]; then
  # PHYSICAL cores, not threads. Hyperthreads do not help ffmpeg much and
  # oversubscribing makes every job slower — this bit us on the coordinator.
  # UNIQUE core ids — `grep -c` counts one line per LOGICAL cpu, which reports
  # 24 on a box with 16 physical cores. That is the exact "24 cores" trap this
  # project already documented for the coordinator; the script fell for it too.
  CONCURRENCY="$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l || nproc)"
  CONCURRENCY="$(printf '%s' "$CONCURRENCY" | tr -dc '0-9')"
  [[ -n "$CONCURRENCY" && "$CONCURRENCY" -gt 0 ]] || CONCURRENCY=2
fi
MEM_GB="$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)"
ok "$CONCURRENCY worker(s), ${MEM_GB}GB RAM, queues: $QUEUES"
if [[ "$QUEUES" == *ai* && "$MEM_GB" -lt 6 ]]; then
  warn "the 'ai' queue wants ~6GB+; this VM has ${MEM_GB}GB. Consider --queues media."
fi

# ── 5. start ──────────────────────────────────────────────────────────────
say "5/5  Starting the worker"
WORKER_NAME="$NAME" WORKER_CONCURRENCY="$CONCURRENCY" QUEUES="$QUEUES" \
  docker compose -f docker-compose.remote.yml --env-file remote.env up -d --build

echo
echo "  waiting for it to register…"
for _ in $(seq 1 20); do
  if docker compose -f docker-compose.remote.yml logs 2>/dev/null | grep -q "celery@\|ready\."; then
    ok "worker is up"
    break
  fi
  sleep 2
done

cat <<EOF

  Done. This VM now serves the '$QUEUES' queue.

  Confirm from the coordinator:
    docker compose exec worker-ai celery -A app.tasks.celery inspect active_queues

  Watch it here:
    docker compose -f docker-compose.remote.yml logs -f

  Remove it:
    docker compose -f docker-compose.remote.yml down
EOF
