#!/bin/bash
# Pod setup: unpacks the code bundle (engine, models, gate scripts) and the self-play data.
set -e
echo "nproc $(nproc)  cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
free -g | head -2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
mkdir -p /root/strivon/candidates && cd /root/strivon
tar -xzf /root/strivon_bundle.tgz
[ -f /root/data_g1.tgz ] && tar -xzf /root/data_g1.tgz -C /root/strivon && echo "data files: $(ls data | wc -l)"
cp -f /root/*.onnx /root/strivon/candidates/ 2>/dev/null || true
python -m venv --system-site-packages /root/venv
/root/venv/bin/pip -q install numpy onnx onnxruntime psutil msgpack 2>&1 | grep -v -i "notice" || true
/root/venv/bin/pip -q install --no-deps -e src/env 2>&1 | grep -v -i "notice" || true
/root/venv/bin/python -c "import onnx, onnxruntime, torch, colony_one; print('ok onnx', onnx.__version__, 'ort', onnxruntime.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python3 -c "import os; print('affinity', len(os.sched_getaffinity(0)))"
ls -la /root/strivon /root/strivon/candidates
