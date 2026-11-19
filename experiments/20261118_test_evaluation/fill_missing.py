"""Decode the test sequences that are still missing after the processes of run.sh have ended (a process that runs out
of GPU memory ends its whole chain). Waits for the other test_eval_* tmux sessions, then runs evaluate.py on the
missing sequences only, two processes per GPU, and repeats until none is missing.

    tmux new-session -d -s test_eval_zz "python fill_missing.py"
"""
import os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SETTINGS = ("sweep", "sweep_high", "power_control")
CONDITIONS = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
RECEIVERS = ("pilot_memory", "pilot_memory_emptied")
ME = "test_eval_zz"


def others_running():
    sessions = subprocess.run(["tmux", "ls"], capture_output=True, text=True).stdout.splitlines()
    return [s.split(":")[0] for s in sessions if s.startswith("test_eval_") and not s.startswith(ME)]


def missing():
    jobs = []
    for setting in SETTINGS:
        for condition in CONDITIONS:
            for episode in range(32):
                name = f"episode_{episode:05d}.npz"
                if not all((HERE/"results"/setting/condition/r/name).is_file() for r in RECEIVERS):
                    jobs.append((setting, condition, episode))
    return jobs


def main():
    while others_running():
        time.sleep(30)
    env = dict(os.environ)
    pid = ROOT/".mps/pipe/nvidia-cuda-mps-control.pid"
    if pid.is_file() and Path(f"/proc/{pid.read_text().strip()}").exists():
        env.update(CUDA_MPS_PIPE_DIRECTORY=str(ROOT/".mps/pipe"), CUDA_MPS_LOG_DIRECTORY=str(ROOT/".mps/log"))
    for attempt in range(8):
        jobs = missing()
        print(time.strftime("%H:%M:%S"), "missing sequences:", len(jobs), flush=True)
        if not jobs:
            return
        slots = [0, 0, 1, 1]                                      # two processes per GPU
        running = []
        for index, (setting, condition, episode) in enumerate(jobs):
            while len(running) >= len(slots):
                running = [(p, g) for p, g in running if p.poll() is None]
                time.sleep(5)
            gpu = next(g for g in slots if [x for _, x in running].count(g) < slots.count(g))
            running.append((subprocess.Popen([sys.executable, str(HERE/"evaluate.py"), "--gpu", str(gpu), "--setting",
                setting, "--condition", condition, "--first-episode", str(episode), "--last-episode", str(episode+1),
                "--receivers", *RECEIVERS], env=env, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), gpu))
        for process, _ in running:
            process.wait()
    print(time.strftime("%H:%M:%S"), "still missing after all attempts:", len(missing()), flush=True)


if __name__ == "__main__":
    main()
