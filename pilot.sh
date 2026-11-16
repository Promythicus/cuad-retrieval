#!/usr/bin/env bash
# The pilot: index, start the model through its launcher, ask three ways, score, stop the model.
cd "$(dirname "$0")"
PY=python3
$PY bench.py index 2>&1 | grep -v pynvml || exit 1
curl -s -X POST "http://127.0.0.1:8797/start?mode=long&minutes=300" > /dev/null
until curl -s -m 3 http://127.0.0.1:8081/health | grep -q ok; do sleep 5; done
echo "model up: $(cat ~/models/qwen38/running)"
for m in rag5 rag10 full; do $PY bench.py run $m 2>&1 | grep -v pynvml; done
$PY bench.py score 2>&1 | grep -v pynvml
curl -s -X POST http://127.0.0.1:8797/stop > /dev/null
echo done
