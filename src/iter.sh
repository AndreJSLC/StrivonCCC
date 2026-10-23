#!/bin/bash
# one AlphaZero-style iteration on the pod: wait for self-play files, train from the previous net, duel.
# usage: iter.sh <data_prefix> <n_files> <init_pt> <out_name> <ref_onnx> [extra data globs...]
set -u
cd /root/strivon/src
prefix=$1; nfiles=$2; init=$3; out=$4; ref=$5; shift 5
until [ "$(ls data/${prefix}_*.npz 2>/dev/null | wc -l)" -ge "$nfiles" ]; do sleep 10; done
/root/venv/bin/python train2.py --heur /root/strivon/models/level4_heuristic.onnx --out "$out" --ch 48 --blocks 4 \
  --epochs 8 --init "$init" "data/${prefix}_*.npz" "$@"
DUEL_WORKERS=30 /root/venv/bin/python duel.py "$out" "$ref" 50
