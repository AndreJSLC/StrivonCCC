#!/bin/bash
# GPU-batched champion self-play farm. Rounds of P processes x C concurrent games
# (G games per process) at S sims, the network on the pod's GPU and the referee search on the CPUs.
# Uses candidates/champion.onnx as it is at each round's start. Output: sp/c<md5-6>_r<round>_<proc>.npz
# (selfplay format + q), the same names the CPU farm writes. Stop: touch sp/STOP.
cd /root/strivon
quota=$(awk '{ if ($1 == "max") print 0; else print int($1 / $2) }' /sys/fs/cgroup/cpu.max 2>/dev/null)
if [ -z "$quota" ] && [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
  quota=$(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) ))
fi
P=${P:-${quota:-12}}; [ "$P" -lt 1 ] && P=12
G=${G:-24}; C=${C:-24}; S=${S:-300}
mkdir -p sp logs
r=0
while [ ! -f sp/STOP ]; do
  r=$((r + 1))
  tag=$(md5sum candidates/champion.onnx | cut -c1-6)
  /root/venv_gpu/bin/python gate/gpu_selfplay.py candidates/champion.onnx "sp/c${tag}_r${r}" "$P" "$G" "$C" "$S" "$((RANDOM * 100 + r))" >> logs/gpufarm.log 2>&1
  echo "$(date +%T) round $r done: champion $tag, $P procs x $G games at $S sims" >> logs/gpufarm.log
done
