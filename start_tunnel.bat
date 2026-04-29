@echo off
setlocal

echo ============================================================
echo  LocalVisionAI — Remote Access via Cloudflare Tunnel
echo ============================================================
echo.

REM ── 0. Kill any stale uvicorn ────────────────────────────────
taskkill /F /IM uvicorn.exe >nul 2>&1
timeout /t 1 /nobreak >nul

REM ── 1. Build frontend ────────────────────────────────────────
echo [1/3] Building frontend...
cd /d "%~dp0frontend"
call npm run build
if errorlevel 1 (
    echo ERROR: Frontend build failed.
    exit /b 1
)
cd /d "%~dp0"

REM ── 2. Start server in background (same env as this shell) ───
echo.
echo [2/3] Starting server...
start /b uvicorn localvisionai.api.server:app --host 127.0.0.1 --port 8765 > server.log 2>&1

REM Wait until /health responds (up to 30s)
echo Waiting for server to be ready...
set /a tries=0
:health_loop
timeout /t 2 /nobreak >nul
curl -s http://localhost:8765/health >nul 2>&1
if not errorlevel 1 goto server_ready
set /a tries+=1
if %tries% lss 15 goto health_loop
echo ERROR: Server did not start. Check server.log for details.
type server.log
exit /b 1

:server_ready
echo Server is ready.

REM ── 3. Start Cloudflare tunnel ───────────────────────────────
echo.
echo [3/3] Starting Cloudflare tunnel...
echo.
echo  Open the trycloudflare.com URL below on the remote device.
echo  Press Ctrl+C to stop.
echo ============================================================
echo.

cloudflared tunnel --url http://localhost:8765

REM Cleanup
echo.
echo Stopping server...
taskkill /F /IM uvicorn.exe >nul 2>&1
