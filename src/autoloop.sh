#!/bin/bash
# loop for a configurable net named <tag><i>.onnx; replay buffer of the last 5 generations.
# usage: autoloop.sh <tag> <ch> <blocks> <start_iter> <end_iter> <workers> <games_per_worker> <sims>
# generation i self-plays with <tag><i-1>.onnx and writes data/<tag>h<i>_*.npz
cd /root/strivon/src
tag=$1; ch=$2; bl=$3; w=$6; g=$7; s=$8
for i in $(seq "$4" "$5"); do
  p=$((i - 1))
  /root/venv/bin/python selfplay.py "${tag}${p}.onnx" "data/${tag}h${i}" "$w" "$g" "$s" >> "${tag}loop.log" 2>&1
  globs=()
  for j in $(seq $((i - 4)) "$i"); do globs+=("data/${tag}h${j}_*.npz"); done
  /root/venv/bin/python train2.py --heur /root/strivon/models/level4_heuristic.onnx --out "${tag}${i}.onnx" --ch "$ch" --blocks "$bl" \
    --epochs 5 --init "${tag}${p}.onnx.pt" "${globs[@]}" "data/h2[0-9]_*.npz" >> "${tag}loop.log" 2>&1
  DUEL_WORKERS="$w" /root/venv/bin/python duel.py "${tag}${i}.onnx" "${tag}${p}.onnx" 30 2>&1 | tail -1 >> results.txt
done
