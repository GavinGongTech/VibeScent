#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Ensure bun is on PATH (WSL non-interactive shells don't load ~/.bashrc) ──
export PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"

# ── Load environment variables from .env ─────────────────────────────────────
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
  echo "✓ Loaded environment from .env"
else
  echo "⚠  No .env file found — NVIDIA_API_KEY may be missing"
fi

# ── Kill any stale processes on our ports ─────────────────────────────────────
echo "→ Clearing ports 8000 8001 3000..."
for port in 8000 8001 3000; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 1

_CLEANED_UP=0

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
  [ "$_CLEANED_UP" -eq 1 ] && return; _CLEANED_UP=1
  echo ""
  echo "→ Shutting down VibeScent..."
  kill "${ML_PID:-}" "${SCRAPER_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
  wait "${ML_PID:-}" "${SCRAPER_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
  sleep 2
  kill -9 "${ML_PID:-}" "${SCRAPER_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
  echo "✓ All services stopped."
}
trap cleanup EXIT INT TERM

# ── ML backend (port 8000) ───────────────────────────────────────────────────
echo ""
echo "┌─────────────────────────────────────────────┐"
echo "│  VibeScent — Starting Services              │"
echo "└─────────────────────────────────────────────┘"
echo ""
echo "▶ [1/3] Starting ML backend   → http://localhost:8000"
uv run uvicorn "vibescents.backend_app:create_configured_app" \
  --factory \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level warning \
  2>&1 | stdbuf -oL sed 's/^/  \x1b[36m[backend]\x1b[0m /' &
ML_PID=$!

# Wait for backend (up to 90s — MiniLM warm-up on first run)
for i in $(seq 1 90); do
  if curl -sf http://localhost:8000/healthz > /dev/null 2>&1; then
    echo "  ✓ Backend ready  (${i}s)"
    break
  fi
  if ! kill -0 "$ML_PID" 2>/dev/null; then
    echo "  ✗ Backend crashed — check logs above"
    exit 1
  fi
  sleep 1
done

# ── Scraper API (port 8001) ────────────────────────────────────────────────
echo "▶ [2/3] Starting Scraper API  → http://localhost:8001"
uv run uvicorn vibescents.scraper_app:app --host 0.0.0.0 --port 8001 \
  --log-level warning \
  2>&1 | stdbuf -oL sed 's/^/  \x1b[33m[scraper]\x1b[0m /' &
SCRAPER_PID=$!

for i in $(seq 1 15); do
  if curl -sf http://localhost:8001/healthz > /dev/null 2>&1; then
    echo "  ✓ Scraper ready  (${i}s)"
    break
  fi
  if ! kill -0 "$SCRAPER_PID" 2>/dev/null; then
    echo "  ✗ Scraper crashed — check logs above"
    exit 1
  fi
  sleep 1
done

# ── Next.js frontend (port 3000) ─────────────────────────────────────────────
echo "▶ [3/3] Starting Frontend     → http://localhost:3000"
node_modules/.bin/next dev 2>&1 | stdbuf -oL sed 's/^/  \x1b[35m[frontend]\x1b[0m /' &
FRONTEND_PID=$!

# ── All services ready ────────────────────────────────────────────────────────
echo ""
echo "┌─────────────────────────────────────────────┐"
echo "│  ✓ VibeScent is running!                    │"
echo "│                                             │"
echo "│  App      →  http://localhost:3000          │"
echo "│  Backend  →  http://localhost:8000          │"
echo "│  Scraper  →  http://localhost:8001          │"
echo "│                                             │"
echo "│  Press Ctrl+C to stop all services.        │"
echo "└─────────────────────────────────────────────┘"
echo ""

# ── Monitor — exit if any service crashes ────────────────────────────────────
while true; do
  if ! kill -0 "$ML_PID" 2>/dev/null; then
    echo "  ✗ [backend] exited unexpectedly"
    exit 1
  fi
  if ! kill -0 "$SCRAPER_PID" 2>/dev/null; then
    echo "  ✗ [scraper] exited unexpectedly"
    exit 1
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "  ✗ [frontend] exited unexpectedly"
    exit 1
  fi
  sleep 5
done
