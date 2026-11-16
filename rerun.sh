#!/usr/bin/env bash
# Stage 3: the best search, end to end, on the pilot's contracts.
cd "$(dirname "$0")"
PY=python3
$PY best.py 2>&1 | grep -v -e pynvml -e "Loading weights" || exit 1
curl -s -X POST "http://127.0.0.1:8797/start?mode=long&minutes=300" > /dev/null
until curl -s -m 3 http://127.0.0.1:8081/health | grep -q ok; do sleep 5; done
for m in new5 new10; do $PY bench.py run $m 2>&1 | grep -v pynvml; done
$PY bench.py score 2>&1 | grep -v pynvml
curl -s -X POST http://127.0.0.1:8797/stop > /dev/null
echo done
