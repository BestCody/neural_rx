"""Re-decode the K-Best baselines on TBs with one scheduled UE; all other saved results stay as they are."""
import os
import sys
from pathlib import Path

first, last, gpu = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402

from neural_rx.benchmark import tbler_sweep as sweep  # noqa: E402
from neural_rx.benchmark.runtime import validate_episode  # noqa: E402

CONDITION = "schedule_gap_reentry"
METHODS = ("baseline_perf_csi_kbest", "baseline_lmmse_kbest")
sweep.configure_runtime()
protocol, plan, _, bank, anchors = sweep.sweep_context()
covariance = sweep.load_covariance("covariance", CONDITION)
receivers = {m: sweep.make_receiver(m, protocol, covariance) for m in METHODS}
for index in range(first, last):
    for directory in ("sweep", "sweep_high"):
        name = f"{directory}/{CONDITION}/%s/episode_{index:05d}.npz"
        targets = {m: Path("results/kbest_one_stream")/(name % m) for m in METHODS}
        if all(path.is_file() for path in targets.values()):
            continue
        saved = {m: dict(np.load(Path("results")/(name % m), allow_pickle=False)) for m in METHODS}
        snr = saved[METHODS[0]]["snr_db"].tolist()
        job = dict(sweep.sweep_job(plan, anchors, CONDITION), snr_db=snr)
        episode = validate_episode(bank.generate(job, index), protocol, job)
        channel = bank.last_channel
        tbs = [tb for tb in range(sweep.SCORING_START, 64) if episode.active[tb].sum() == 1]
        for method in METHODS:
            if not np.array_equal(saved[method]["fingerprint"], episode.fingerprint):
                raise RuntimeError(f"{name % method}: regenerated episode differs")
            perfect = method == "baseline_perf_csi_kbest"
            errors, disagreement = sweep.decode_episode(method, receivers[method], episode,
                                                        channel if perfect else None, only_tbs=tbs)
            saved[method]["errors"][:, tbs] = errors[:, tbs]
            saved[method]["crc_disagreement"][:, tbs] = disagreement[:, tbs]
            sweep._atomic_savez(targets[method], **saved[method])
            print(index, directory, method, "TBs", len(tbs), "failures", int(errors[:, tbs].sum()), flush=True)
