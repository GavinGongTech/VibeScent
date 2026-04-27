#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

_CLEANED_UP=0

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
  [ "$_CLEANED_UP" -eq 1 ] && return; _CLEANED_UP=1
  echo ""
  echo "Shutting down..."
  kill "$ML_PID" "$SCRAPER_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$ML_PID" "$SCRAPER_PID" "$FRONTEND_PID" 2>/dev/null || true
  sleep 3
  kill -9 "$ML_PID" "$SCRAPER_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── ML backend (port 8000) ───────────────────────────────────────────────────
echo "Starting ML backend on http://localhost:8000 ..."
uv run uvicorn "vibescents.backend_app:create_configured_app" \
  --factory \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  2>&1 | stdbuf -oL sed 's/^/[backend] /' &
ML_PID=$!

# Wait for backend to be ready (up to 30s — corpus load takes a few seconds)
echo "Waiting for backend to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/healthz > /dev/null 2>&1; then
    echo "Backend ready."
    break
  fi
  if ! kill -0 "$ML_PID" 2>/dev/null; then
    echo "ERROR: Backend process died. Check logs above."
    exit 1
  fi
  sleep 1
done

# ── Scraper API (port 8001) ────────────────────────────────────────────────
echo "Starting Scraper API on http://localhost:8001 ..."
uv run uvicorn vibescents.scraper_app:app --host 0.0.0.0 --port 8001 \
  --log-level info \
  2>&1 | stdbuf -oL sed 's/^/[scraper] /' &
SCRAPER_PID=$!

# Wait for scraper to be ready (up to 15s)
echo "Waiting for scraper to be ready..."
for i in $(seq 1 15); do
  if curl -sf http://localhost:8001/healthz > /dev/null 2>&1; then
    echo "Scraper ready."
    break
  fi
  if ! kill -0 "$SCRAPER_PID" 2>/dev/null; then
    echo "ERROR: Scraper process died. Check logs above."
    exit 1
  fi
  sleep 1
done

# ── Next.js frontend (port 3000) ─────────────────────────────────────────────
echo "Starting frontend on http://localhost:3000 ..."
bun run dev:web 2>&1 | stdbuf -oL sed 's/^/[frontend] /' &
FRONTEND_PID=$!

# ── Ready ────────────────────────────────────────────────────────────────────
echo ""
echo "  VibeScent is running"
echo "  Frontend → http://localhost:3000"
echo "  Backend  → http://localhost:8000"
echo "  Press Ctrl+C to stop."
echo ""

# Monitor all three processes and detect crashes
while true; do
  if ! kill -0 "$ML_PID" 2>/dev/null; then
    echo "ERROR: backend has exited unexpectedly"
    exit 1
  fi
  if ! kill -0 "$SCRAPER_PID" 2>/dev/null; then
    echo "ERROR: scraper has exited unexpectedly"
    exit 1
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "ERROR: frontend has exited unexpectedly"
    exit 1
  fi
  sleep 5
done
