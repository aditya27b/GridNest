#!/bin/bash
# start.sh — Launches both servers with one command
# ML Backend: port 8001 | GridNest 3D Viewer: port 8000

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Energy Intelligence × GridNest — Hackathon Demo   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "▶ Starting ML Backend on http://localhost:8001 ..."
cd "$ROOT"
uvicorn backend.main:app --port 8001 --host 0.0.0.0 &
ML_PID=$!

sleep 2

echo "▶ Starting GridNest 3D Viewer on http://localhost:8000 ..."
cd "$ROOT/frontend"
uvicorn server.api:app --port 8000 --host 0.0.0.0 &
GN_PID=$!

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ML API Docs  →  http://localhost:8001/docs"
echo "  3D Viewer    →  http://localhost:8000"
echo "  Health check →  http://localhost:8000/api/health"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop both servers."

# Wait and trap Ctrl+C to kill both
trap "kill $ML_PID $GN_PID 2>/dev/null; echo 'Stopped.'" INT TERM
wait $ML_PID $GN_PID
