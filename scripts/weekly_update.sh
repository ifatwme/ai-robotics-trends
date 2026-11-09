#!/usr/bin/env bash
# Server-side weekly refresh: pull latest code, rebuild the pipeline image, collect + analyse, re-render ./site.
# The live nginx container serves ./site read-only, so the new dashboard is visible immediately.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && . ./.env && set +a
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
log "git pull"; git pull --ff-only --quiet || log "git pull failed — continuing with local checkout"
mkdir -p data out site cache "${HISTORY_HOST_DIR:-state/history}"
chown -R "${PIPELINE_UID:-10001}:${PIPELINE_GID:-10001}" data out site cache "${HISTORY_HOST_DIR:-state/history}"
log "build pipeline image"; docker compose --profile pipeline build pipeline
log "run pipeline";         docker compose run --rm pipeline collect all
log "done"
