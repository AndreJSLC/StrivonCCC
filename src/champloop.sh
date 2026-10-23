#!/bin/bash
# AlphaGo-Zero-style gated loop: self-play always uses the current champion; a candidate replaces it only if it wins.
# usage: champloop.sh <champ_name> <ch> <blocks> <start_iter> <end_iter> <workers> <games> <sims> <lr>
cd /root/strivon/src
champ=$1; ch=$2; bl=$3; w=$6; g=$7; s=$8; lr=$9
echo "$champ" > champ.txt
for i in $(seq "$4" "$5"); do
  /root/venv/bin/python selfplay.py "${champ}.onnx" "data/bbh${i}" "$w" "$g" "$s" >> champloop.log 2>&1
  globs=()
  for j in $(seq $((i - ${WIN:-2})) "$i"); do globs+=("data/bbh${j}_*.npz"); done
  cand="cc${i}"
  /root/venv/bin/python train2.py --heur /root/strivon/models/level4_heuristic.onnx --out "${cand}.onnx" --ch "$ch" --blocks "$bl" \
    --epochs "${EPOCHS:-2}" --lr "$lr" --init "${champ}.onnx.pt" "${globs[@]}" >> champloop.log 2>&1
  res=$(DUEL_SEED=$((i * 100000)) DUEL_WORKERS="$w" /root/venv/bin/python duel.py "${cand}.onnx" "${champ}.onnx" 40 2>&1 | tail -1)
  echo "$res" >> results.txt
  wins=$(echo "$res" | sed -n 's/.*: \([0-9]*\)\/\([0-9]*\) =.*as cyan \([0-9]*\)\/\([0-9]*\),.*/\1 \2 \3 \4/p')
  set -- $wins
  # gate: overall >= 55% AND the candidate must not collapse as cyan (>= 45% of its cyan games)
  if [ -n "$1" ] && [ $(( $1 * 100 / $2 )) -ge 55 ] && [ $(( $3 * 100 / $4 )) -ge 45 ]; then
    champ="$cand"; echo "$champ" > champ.txt; echo "NEW CHAMPION $champ ($res)" >> results.txt
  fi
done
