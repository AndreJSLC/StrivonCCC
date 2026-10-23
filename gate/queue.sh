#!/bin/bash
# Train every variant in a spec file on the GPU, one after another.
# usage: queue.sh specs.txt      spec line: NAME ARGS...   (# comments allowed)
# env DATA: space-separated data globs (default data/g1_*.npz); left unquoted on purpose
cd /root/strivon
DATA=${DATA:-data/g1_*.npz}
mkdir -p models logs
while read -r name args; do
  [ -z "$name" ] && continue
  [[ "$name" == \#* ]] && continue
  [ -f "models/$name.trained" ] && continue
  echo "$(date +%T) train $name" >> logs/queue.log
  if /root/venv/bin/python gate/arch.py $args --out "models/$name.onnx" $DATA > "logs/$name.train.log" 2>&1; then
    touch "models/$name.trained"
    echo "$(date +%T) done  $name: $(grep saved logs/$name.train.log)" >> logs/queue.log
  else
    echo "$(date +%T) FAIL  $name" >> logs/queue.log
  fi
done < "$1"
echo "$(date +%T) queue $1 finished" >> logs/queue.log
