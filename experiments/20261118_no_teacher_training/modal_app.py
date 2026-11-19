"""Run this folder's training (run.py) on two rented GPUs with Modal.

    modal run --detach experiments/20261118_no_teacher_training/modal_app.py --hours 2.2
    modal run experiments/20261118_no_teacher_training/modal_app.py --hours 0.2 --name trial      # short trial
    ... --gpu L40S:2 --execution '["--gpu", "0", ...]'      # other GPUs; another split of the work (see EXECUTION)

The container holds a copy of src/, config/, the released weights, stage 1's checkpoint and this folder, at the same
places as in the repository. The development sequences come from the volume nrx-dev-episodes (uploaded once with
`modal volume put nrx-dev-episodes experiments/20261113_retrained_k2_baseline/episodes/development /development`) and
are copied to the container's disk while the training starts, because the benchmark locks its files. The model directory, the development results
and the logs are copied to the volume nrx-run (under <name>/) every two minutes, and copied back when the function
starts again after an interruption, so the training resumes from its latest checkpoint.

--hours is the limit on the function's total running time over all restarts; it is how the cost is bounded (two H100,
8 cores and 64 GiB are about 8.8 dollars per hour at Modal's list prices of November 2026). When it is used up, the
training is interrupted and the last state is stored.

The container runs Python 3.11, because Modal no longer builds images with the repository's Python 3.10; the pinned
versions of TensorFlow, Sionna and NumPy are the same.

    modal volume get nrx-run run/results/best.json -        # best development check so far
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import modal

HERE = Path(__file__).resolve().parent
RUN = "20261118_no_teacher_training"
MODEL = "pilot_memory_no_teacher_20261118"
STAGE1 = "models/neural_rx_k2_20261117/checkpoint_011000"
CACHE = "experiments/20261113_retrained_k2_baseline/episodes/development"
REMOTE = Path("/root/neural_rx")
SYNC_SECONDS = 120
# The split of the work in the container: the learner alone on GPU 0 (no second gradient process), data generation
# alone on GPU 1 with several channel processes and whole sequences per rendering batch. Measured on two H100 without
# the multi-process service: 4.0 s per update with the server's split (second gradient process and data generation
# sharing GPU 1), 3.1 s with this one; in both the learner waited for data while the data processes took turns.
# The data processes get fixed shares of the data GPU's memory (a fifth for the renderer, the rest split between the
# channel processes, as on the server): without limits they grew until one of them could not allocate any more.
EXECUTION = ["--gpu", "0", "--data-gpu", "1", "--replica-gpu", "none", "--detector-batch-tbs", "16",
             "--descriptor-chunk-tbs", "32", "--render-batch-tbs", "32", "--prefetch-examples", "8",
             "--prefetch-memory-mib", "16384", "--channel-workers", "3", "--preparation-gpu-memory-mib", "72000"]
# The repository is read only where the app is launched; inside the container this file sits elsewhere and the image is
# not rebuilt, so the local paths below are not looked at there.
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE

app = modal.App("nrx-no-teacher")
image = (modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04", add_python="3.11")
         .apt_install("build-essential")
         .add_local_file(ROOT/"requirements-lock.txt", "/root/requirements-lock.txt", copy=True)
         .run_commands("pip install -c /root/requirements-lock.txt sionna==0.18.0 'tensorflow[and-cuda]==2.15.1' "
                       "numpy==1.26.4 matplotlib==3.10.9")
         .env({"PYTHONPATH": str(REMOTE/"src"), "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONUNBUFFERED": "1",
               "CUDA_HOME": "/usr/local/cuda", "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/usr/local/cuda"})
         .add_local_dir(ROOT/"src", str(REMOTE/"src"), ignore=["**/__pycache__"])
         .add_local_dir(ROOT/"config", str(REMOTE/"config"))
         .add_local_dir(ROOT/"models/pretrained", str(REMOTE/"models/pretrained"))
         .add_local_dir(ROOT/STAGE1, str(REMOTE/STAGE1))
         .add_local_dir(ROOT/"experiments"/RUN, str(REMOTE/"experiments"/RUN),
                        ignore=["results", "logs", "**/__pycache__", "modal_app.py"]))
outputs = modal.Volume.from_name("nrx-run", create_if_missing=True)
episodes = modal.Volume.from_name("nrx-dev-episodes", create_if_missing=True)


def mirror(source, target):
    """Copy what changed in `source` to `target` and remove there what is gone. Checkpoint directories are published
    whole by the trainer, so they are copied once, under a temporary name first."""
    if not source.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    names = set()
    for item in source.iterdir():
        if item.name.startswith(".") or item.name == "run.lock":
            continue
        names.add(item.name)
        goal, temporary = target/item.name, target/f".tmp-{item.name}"
        if item.is_dir() and item.name.startswith("checkpoint_"):
            if not goal.exists():
                shutil.rmtree(temporary, ignore_errors=True)
                shutil.copytree(item, temporary)
                temporary.rename(goal)
        elif item.is_dir():
            mirror(item, goal)
        elif not goal.exists() or goal.stat().st_size != item.stat().st_size or goal.stat().st_mtime < item.stat().st_mtime:
            shutil.copy2(item, temporary)
            temporary.replace(goal)
    for item in target.iterdir():
        if item.name not in names:
            shutil.rmtree(item) if item.is_dir() else item.unlink()


@app.function(image=image, gpu="H100:2", cpu=8., memory=65536, timeout=6*3600,
              volumes={"/outputs": outputs, "/episodes": episodes},
              retries=modal.Retries(max_retries=2, backoff_coefficient=1., initial_delay=10.))
def train(hours: float, name: str = "run", execution: str = ""):
    started = time.monotonic()
    stored = Path("/outputs")/name
    pairs = [(REMOTE/"models"/MODEL, stored/"model"), (REMOTE/"experiments"/RUN/"results", stored/"results"),
             (REMOTE/"experiments"/RUN/"logs", stored/"logs")]
    budget = stored/"seconds.json"
    used = json.loads(budget.read_text())["seconds"] if budget.is_file() else 0.
    if used >= 3600*hours:
        print(f"time limit already used: {used/3600:.2f} h of {hours} h", flush=True)
        return
    for local, remote in pairs:                          # state of an earlier, interrupted start
        mirror(remote, local)
    cache = REMOTE/CACHE

    def fetch():
        """The development sequences are first needed at the first development check, so they are copied to the
        container's disk while the training already runs."""
        temporary = cache.with_name(cache.name+".copying")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        files = [path for path in Path("/episodes/development").iterdir() if path.suffix != ".lock"]
        with ThreadPoolExecutor(32) as pool:
            list(pool.map(lambda path: shutil.copy2(path, temporary/path.name), files))
        temporary.rename(cache)
        print(f"{len(files)} development sequences copied after {time.monotonic()-started:.0f} s", flush=True)

    if not cache.is_dir():
        threading.Thread(target=fetch, daemon=True).start()
    print(f"started; {used/3600:.2f} h of {hours} h used before this start", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], check=True)
    (REMOTE/"experiments"/RUN/"logs").mkdir(parents=True, exist_ok=True)
    log = (REMOTE/"experiments"/RUN/"logs"/"train.log").open("a")
    environment = dict(os.environ, NRX_EXECUTION=execution or json.dumps(EXECUTION))
    # NVIDIA's multi-process service, as on the project's server: processes that share a GPU run at the same time
    # instead of taking turns.
    service = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps/pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps/log"}
    for directory in service.values():
        os.makedirs(directory, exist_ok=True)
    identity = Path(service["CUDA_MPS_PIPE_DIRECTORY"])/"nvidia-cuda-mps-control.pid"
    running = identity.is_file() and Path(f"/proc/{identity.read_text().strip()}").exists()   # from an earlier attempt
    if running or (shutil.which("nvidia-cuda-mps-control") and subprocess.run(
            ["nvidia-cuda-mps-control", "-d"], env=dict(os.environ, **service)).returncode == 0):
        environment.update(service)
        print("multi-process service", "already running" if running else "started", flush=True)
    else:
        print("multi-process service not available; processes on one GPU take turns", flush=True)
    process = subprocess.Popen([sys.executable, "-u", str(REMOTE/"experiments"/RUN/"run.py")], cwd=REMOTE,
                               stdout=log, stderr=subprocess.STDOUT, env=environment)
    lock = threading.Lock()

    def store():
        with lock:
            for local, remote in pairs:
                mirror(local, remote)
            budget.write_text(json.dumps({"seconds": used+time.monotonic()-started}))
            outputs.commit()

    def load(previous):
        """CPU cores used by each Python process since the last call, and the GPUs' utilization."""
        ticks, now = {}, time.monotonic()
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    fields = (entry/"stat").read_text().rsplit(")", 1)[1].split()
                    if b"python" in (entry/"cmdline").read_bytes():
                        ticks[int(entry.name)] = int(fields[11])+int(fields[12])
                except OSError:
                    pass
        cores = sorted(((ticks[pid]-previous[0].get(pid, ticks[pid]))/os.sysconf("SC_CLK_TCK")/(now-previous[1])
                        for pid in ticks), reverse=True)
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().replace("\n", " | ")
        return (ticks, now), f"CPU cores per process {[round(c, 2) for c in cores if c > .02]}, GPUs {gpus}"

    progress = REMOTE/"models"/MODEL/"progress.json"
    limit_reached = False
    sample, last = ({}, time.monotonic()), (0, time.monotonic())
    while process.poll() is None:
        try:
            process.wait(SYNC_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        store()
        update = json.loads(progress.read_text())["update"] if progress.is_file() else 0
        best = REMOTE/"experiments"/RUN/"results/best.json"
        checks = json.loads(best.read_text())["development"] if best.is_file() else {}
        sample, usage = load(sample)
        rate = (time.monotonic()-last[1])/(update-last[0]) if update > last[0] else float("nan")
        last = (update, time.monotonic())
        print(f"{(used+time.monotonic()-started)/3600:.2f} h used, update {update}, {rate:.2f} s per update, "
              f"development checks {checks}; {usage}", flush=True)
        if used+time.monotonic()-started >= 3600*hours and process.poll() is None:
            limit_reached = True
            process.send_signal(signal.SIGINT)           # the trainer keeps its latest checkpoint
            try:
                process.wait(180)
            except subprocess.TimeoutExpired:
                process.kill()
    log.close()
    store()
    tail = (REMOTE/"experiments"/RUN/"logs"/"train.log").read_text()[-3000:]
    if limit_reached:
        print("time limit reached; training interrupted and stored", flush=True)
    elif process.returncode:
        print(tail, flush=True)
        raise RuntimeError(f"training ended with code {process.returncode}")


@app.local_entrypoint()
def main(hours: float, name: str = "run", gpu: str = "H100:2", execution: str = ""):
    train.with_options(gpu=gpu).remote(hours, name, execution)
