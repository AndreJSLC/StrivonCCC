#!/bin/bash
# Continuous champion self-play. Rounds of W workers x G games at S sims, each
# round using whatever candidates/champion.onnx is at its start, so a champion swap needs only a
# file copy. Output: sp/c<md5-6>_r<round>_<worker>.npz (selfplay format + q). Stop: touch sp/STOP.
cd /root/strivon
quota=$(awk '{ if ($1 == "max") print 0; else print int($1 / $2) }' /sys/fs/cgroup/cpu.max 2>/dev/null)
if [ -z "$quota" ] && [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then  # cgroup v1
  quota=$(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) ))
fi
W=${W:-${quota:-12}}; [ "$W" -lt 1 ] && W=12
G=${G:-30}; S=${S:-300}
mkdir -p sp logs
r=0
while [ ! -f sp/STOP ]; do
  r=$((r + 1))
  tag=$(md5sum candidates/champion.onnx | cut -c1-6)
  /root/venv/bin/python gate/selfplay.py candidates/champion.onnx "sp/c${tag}_r${r}" "$W" "$G" "$S" "$((RANDOM * 100 + r))" >> logs/spfarm.log 2>&1
  echo "$(date +%T) round $r done: champion $tag, $W workers x $G games at $S sims" >> logs/spfarm.log
done
