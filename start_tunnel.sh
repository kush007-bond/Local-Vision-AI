#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " LocalVisionAI — Remote Access via Cloudflare Tunnel"
echo "============================================================"
echo

# ── 1. Build frontend ────────────────────────────────────────
echo "[1/3] Building frontend..."
(cd "$(dirname "$0")/frontend" && npm run build)

# ── 2. Start LocalVisionAI API server in background ─────────
echo
echo "[2/3] Starting LocalVisionAI server on http://localhost:8765 ..."
uvicorn localvisionai.api.server:app --host 0.0.0.0 --port 8765 &
SERVER_PID=$!
sleep 2   # give server time to bind

# Cleanup on exit
cleanup() {
    echo
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 3. Start Cloudflare tunnel ───────────────────────────────
echo
echo "[3/3] Starting Cloudflare tunnel..."
echo
echo " The public URL will appear below (look for 'trycloudflare.com')."
echo " Share that URL with the remote device — no port forwarding needed."
echo
echo " Camera feed stays on the remote browser."
echo " AI inference runs here, results stream back via WebSocket."
echo
echo " Press Ctrl+C to stop everything."
echo "============================================================"
echo

cloudflared tunnel --url http://localhost:8765
