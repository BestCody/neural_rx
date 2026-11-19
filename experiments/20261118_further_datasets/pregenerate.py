"""Generate sequences of the listed conditions into an episode cache, in parallel with the benchmark run that uses it.

The set-up is that of neural_rx.benchmark.frozen.run_experiment for the confirmation phase, so the cache bindings are
identical; each sequence is written under its lock and skipped if it exists, in whichever process reaches it first.

    pregenerate.py --gpu 0 --episode-cache episodes/v2 --anchors <anchors.json> --jobs 7 5 3 [--benchmark-protocol ...]
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--episode-cache", type=Path, required=True)
parser.add_argument("--anchors", type=Path, required=True)
parser.add_argument("--benchmark-protocol", type=Path, default=ROOT/"config/temporal_benchmark_v2.json")
parser.add_argument("--jobs", type=int, nargs="+", required=True, help="positions in the plan's list of conditions")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sionna as sn
import tensorflow as tf
from neural_rx.backbone import E2E_Model, Parameters
from neural_rx.benchmark.episodes import CachedScenarioBank, scenario_identity
from neural_rx.benchmark.frozen import load_ray_traced_banks, protocol_and_plan
from neural_rx.benchmark.generalization_protocol import verify_payload
from neural_rx.benchmark.scenarios import TemporalBenchmarkScenarioBank

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(20260930)
sn.config.seed = 20260930
artifact = json.loads(args.anchors.read_text())
verify_payload(artifact, field="manifest_sha256")
protocol, plan = protocol_and_plan("confirmation", 132, None, artifact["conditions"], protocol_path=args.benchmark_protocol)
rt_banks = load_ray_traced_banks(protocol, plan)
identity = scenario_identity(ROOT/"config/nrx_large.cfg", protocol)
p = Parameters("nrx_large.cfg", training=False, num_tx_eval=2, n_size_bwp_override=132)
scenarios = TemporalBenchmarkScenarioBank(p, E2E_Model(p, training=False, mcs_arr_eval_idx=0), tf, rt_banks=rt_banks)
provider = CachedScenarioBank(scenarios, args.episode_cache, protocol, identity)
for position in args.jobs:
    job = plan["jobs"][position]
    if job["provider"] == "ray_traced":
        raise SystemExit("ray-traced sequences have their own cache; not generated here")
    made = 0
    for index in reversed(range(int(job["episode_count"]))):       # the benchmark run goes forward
        if not provider.path_for(job, index).is_file():
            provider.generate(job, index)
            made += 1
    print(json.dumps({"condition": job["condition"], "generated": made}), flush=True)
