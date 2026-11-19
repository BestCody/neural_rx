"""Time the receivers on a rented A100 with Modal: runs time_fused.py there and brings the results back.

    modal run experiments/20261114_tensorrt_latency/modal_a100.py [--gpu A100-40GB]
    -> results/fused_a100.json, results/fused_a100_samples.npz, results/time_fused_a100.log

The container holds this folder's scripts, the kernels' source and the exported files of build/ (export.py); the
engines are built there for its GPU, with the same TensorRT version as on the project's server. One A100 (40 GB) with
4 cores is about 2.1 dollars per hour at Modal's list prices of November 2026; the function is stopped after 30
minutes at the latest and is not repeated if it fails.
"""
from pathlib import Path
import subprocess
import sys

import modal

HERE = Path(__file__).resolve().parent
REMOTE = Path("/root/latency")
SCRIPTS = ("time_trt.py", "time_fused.py", "filter_kernels.py", "fitted_filter.cu")
EXPORTED = ("large_k2.onnx", "large_k3.onnx", "large_k8.onnx", "proposed_cgnn.onnx", "reference.npz",
            "filter_tables.npz")

app = modal.App("nrx-latency")
# The local files are only read where the app is launched; inside the container this file sits elsewhere.
image = (modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
         .pip_install("tensorrt-cu12==10.3.0", "cuda-python==12.4.0", "numpy==1.26.4")
         .env({"PYTHONUNBUFFERED": "1"}))
for name in SCRIPTS:
    image = image.add_local_file(HERE/name, str(REMOTE/name))
for name in EXPORTED:
    image = image.add_local_file(HERE/"build"/name, str(REMOTE/"build"/name))


@app.function(image=image, gpu="A100-40GB", cpu=4., memory=16384, timeout=1800)
def measure():
    machine = subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; "
                             "grep -m1 'model name' /proc/cpuinfo", shell=True, capture_output=True, text=True).stdout
    run = subprocess.run([sys.executable, "-u", "time_fused.py", "--tag", "a100"], cwd=REMOTE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    files = {path.name: path.read_bytes() for path in (REMOTE/"results").glob("fused_a100*")}
    files["time_fused_a100.log"] = (machine+run.stdout).encode()
    return run.returncode, files


@app.local_entrypoint()
def main(gpu: str = "A100-40GB"):
    code, files = measure.with_options(gpu=gpu).remote()
    (HERE/"results").mkdir(exist_ok=True)
    for name, data in files.items():
        (HERE/"results"/name).write_bytes(data)
    print(files["time_fused_a100.log"].decode()[-5000:])
    if code:
        raise SystemExit(f"time_fused.py ended with code {code}; see results/time_fused_a100.log")
