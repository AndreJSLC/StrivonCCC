#!/bin/bash
# Gate every trained model as it appears. Stops when models/STOP exists.
# env OPPS: opponents (default: the hand-built heuristic), PAIRS (default 50),
#     PATTERN: which models to gate (default *), TAG: marker suffix so several loops can coexist
cd /root/strivon
OPPS=${OPPS:-"models/level4_heuristic.onnx"}
PAIRS=${PAIRS:-50}
PATTERN=${PATTERN:-*}
TAG=${TAG:-gated}
while true; do
  for t in models/$PATTERN.trained; do
    [ -e "$t" ] || continue
    n=$(basename "$t" .trained)
    [ -f "models/$n.$TAG" ] && continue
    /root/venv/bin/python gate/gate.py "models/$n.onnx" $OPPS --pairs "$PAIRS" --workers "${W:-20}" \
        --out gate/results.jsonl > "logs/$n.$TAG.log" 2>&1
    touch "models/$n.$TAG"
    echo "$(date +%T) $TAG $n" >> logs/queue.log
    grep -h "vs\|CPU" "logs/$n.$TAG.log" >> logs/queue.log
  done
  [ -f "models/STOP_$TAG" ] && break
  sleep 10
done
