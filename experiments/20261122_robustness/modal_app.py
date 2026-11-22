"""Run this folder's evaluate.py on rented GPUs with Modal, one container per (test, condition, range of sequences).

    modal run experiments/20261122_robustness/modal_app.py --smoke                     # one sequence, three SNR points
    modal run --detach experiments/20261122_robustness/modal_app.py --tests tdd_dddsu,cfo --parts 2
    modal volume get nrx-robustness results experiments/20261122_robustness/           # fetch the results

The container holds a copy of src/, config/, the three checkpoints the receivers load, the power-control experiment
and this folder, at the same places as in the repository; the results go to the volume nrx-robustness. A sequence whose
result exists there is not decoded again, so an interrupted run is continued by starting it again.

Cost: one L40S with 4 cores and 16 GiB is about 2.2 dollars per hour at Modal's list prices of November 2026; each
container stops after --hours (default 3).
"""
import json
from pathlib import Path
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
REMOTE = Path("/root/neural_rx")
FOLDER = "experiments/20261122_robustness"
CHECKPOINTS = ("models/pilot_memory_no_teacher_20261118/checkpoint_006000",
               "models/neural_rx_k2_20261117/checkpoint_011000", "models/pretrained")
INPUTS = ("experiments/20261019_learned_blend_joint_training_6000/inputs",)
CONDITIONS = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE       # only read where the app is launched

app = modal.App("nrx-robustness")
image = (modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04", add_python="3.11")
         .apt_install("build-essential")
         .add_local_file(ROOT/"requirements-lock.txt", "/root/requirements-lock.txt", copy=True)
         .run_commands("pip install -c /root/requirements-lock.txt sionna==0.18.0 'tensorflow[and-cuda]==2.15.1' "
                       "numpy==1.26.4 matplotlib==3.10.9")
         .env({"PYTHONPATH": str(REMOTE/"src"), "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONUNBUFFERED": "1",
               "CUDA_HOME": "/usr/local/cuda", "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/usr/local/cuda"})
         .add_local_dir(ROOT/"src", str(REMOTE/"src"), ignore=["**/__pycache__"])
         .add_local_dir(ROOT/"config", str(REMOTE/"config")))
for path in CHECKPOINTS+INPUTS:
    image = image.add_local_dir(ROOT/path, str(REMOTE/path))
image = (image.add_local_dir(ROOT/"experiments/20261112_early_retransmission",
                             str(REMOTE/"experiments/20261112_early_retransmission"),
                             ignore=["results", "logs", "**/__pycache__"])
         .add_local_dir(ROOT/FOLDER, str(REMOTE/FOLDER), ignore=["results", "logs", "**/__pycache__", "modal_app.py"]))
results = modal.Volume.from_name("nrx-robustness", create_if_missing=True)


@app.function(image=image, gpu="L40S", cpu=4., memory=16384, timeout=6*3600, volumes={"/results": results})
def evaluate(test: str, condition: str, first: int, last: int, hours: float, points: str = "", receivers: str = ""):
    if test == "harq":
        command = ["python", "-u", str(REMOTE/FOLDER/"harq.py"), "--condition", condition, "--output-dir", "/results/results"]
    else:
        command = ["python", "-u", str(REMOTE/FOLDER/"evaluate.py"), "--test", test, "--condition", condition,
                   "--output-dir", "/results/results"]
        command += ["--points", *points.split(",")] if points else []
        command += ["--receivers", *receivers.split(",")] if receivers else []
    started = time.monotonic()
    command += ["--first-episode", str(first), "--last-episode", str(last)]
    log = Path("/results/logs")/f"{test}_{condition}_{first:02d}_{last:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as out:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, cwd=str(REMOTE))
        while process.poll() is None:            # results are committed to the volume every two minutes
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass
            results.commit()
            if time.monotonic()-started > 3600*hours and process.poll() is None:
                process.terminate()
                process.wait()
                print(f"{test} {condition} {first}-{last}: time limit reached", flush=True)
    results.commit()
    tail = log.read_text().splitlines()[-3:]
    return {"test": test, "condition": condition, "first": first, "last": last, "returncode": process.returncode,
            "hours": round((time.monotonic()-started)/3600, 3), "log_tail": tail}


@app.local_entrypoint()
def main(tests: str = "tdd_dddsu,cfo", parts: int = 2, episodes: int = 32, hours: float = 3., smoke: str = ""):
    """--tests harq runs harq.py on the 16 test sequences (16-31) of each condition; --smoke tdd or harq runs one
    sequence of natural_umi."""
    if smoke:
        calls = [("tdd_dddsu", "natural_umi", 0, 1, 0.5, "0,8,16", "") if smoke == "tdd" else
                 ("harq", "natural_umi", 20, 21, 0.5, "", "")]   # 13 errors, at all checks, 2 missed, 4 false alarms
    else:
        calls = []
        for test in tests.split(","):
            first, last = (16, 32) if test == "harq" else (0, episodes)
            size = -(-(last-first)//parts)
            calls += [(test, condition, start, min(start+size, last), hours, "", "")
                      for condition in CONDITIONS for start in range(first, last, size)]
    print(json.dumps({"containers": len(calls)}), flush=True)
    for report in evaluate.starmap(calls):
        print(json.dumps(report), flush=True)
