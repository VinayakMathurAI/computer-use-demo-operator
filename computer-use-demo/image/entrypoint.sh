#!/bin/bash
set -e

./start_all.sh
./novnc_startup.sh

# Run FastAPI server instead of streamlit
cd /home/computeruse && python -m computer_use_demo.server > /tmp/server_logs.txt 2>&1 &

python http_server.py > /tmp/http_server_logs.txt 2>&1 &

echo "✨ Computer Use Demo is ready!"
echo "➡️  Open http://localhost:8080 in your browser to begin"

# Keep the container running
tail -f /dev/null