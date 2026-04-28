#!/bin/bash
pkill -f uvicorn || true
sleep 2
rm -f /tmp/vibescent.log
bash start.sh > /tmp/vibescent.log 2>&1 &
