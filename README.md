# Pilot-History Neural Receiver

One deployed memory-receiver architecture:

```
Current LS pilots + seven previous pilot records per UE
  -> learned channel filter and correlated uncertainty
  -> CGNN initializer and two processing rounds
  -> eight Sionna LDPC iterations
  -> physical symbol-pair correction and learned per-bit blend
  -> ten further LDPC iterations, retaining decoder state
  -> payload bits and CRC.
```

| Role | Path |
| --- | --- |
| Selected trained receiver | `models/learned_blend_joint_20261019/checkpoint_005000/` |
| Latest checkpoint for continuing the same campaign | `models/learned_blend_joint_20261019/checkpoint_006000/` |
| Initialization used by that campaign | `models/learned_blend/initial_receiver.weights.h5` |
| Frozen Cold K2/K3/K8 controls | `models/pretrained/nrx_large_weights` |

The checkpoints in the retained campaign are training milestones of this one
architecture. The model folders and experiment archives of retired receivers
were removed. The initializer retains the inherited trained detector; it is
not a randomly initialized network.

### Versioned artifacts

The files above are committed, together with the campaign's metadata
(`run.json`, `progress.json`, `completion.json`, `latest.json`) and the
experiment archive described under [Results and reproduction](#results-and-reproduction).
To keep the repository small, the following stay local and are not in git:

- intermediate milestones `checkpoint_000000`–`checkpoint_004000`
- the per-update training history `models/learned_blend_joint_20261019/history.jsonl` (82 MB)
- the experiment training log `experiments/20261019_learned_blend_joint_training_6000/run.log` (82 MB)
- the RAM-backed `episodes` archive, which is regenerated from the retained plans and seeds

`retained_artifacts.json` records SHA-256 hashes for the committed weights.

## Setup

### Requirements

- Linux with an NVIDIA GPU. The decoder-correction ops are CUDA-only; the
  receiver raises an error on load when no GPU is visible. Results were
  produced on 2× NVIDIA A40 (48 GB, compute capability 8.6).
- An NVIDIA driver that supports CUDA 12 (tested with 550.163).
- The CUDA toolkit, for `nvcc` (tested with CUDA 12.6), and `g++`.
- Python 3.10 (pinned in `.python-version`).

### Environment

```bash
conda create -n neural_rx python=3.10
conda activate neural_rx
pip install -r requirements.txt   # pinned through requirements-lock.txt
pip install -e ".[dev]"            # this project, plus pytest and ruff
```

TensorFlow installs its own CUDA 12.2 runtime and cuDNN through pip
(`tensorflow[and-cuda]`), so only the driver and the compiler toolchain come
from the system. The editable install (`pip install -e .`) makes the
`neural_rx` package importable and installs the `nrx-*` commands below. Keep it
editable: configs, models and datasets are read from this checkout
(`neural_rx.paths.ROOT`).

### CUDA toolkit and native ops

The ops in `src/neural_rx/receiver/native_ops/` are compiled
automatically the first time the receiver loads. The build uses:

- `nvcc` from `PATH`, or else `$CUDA_HOME/bin/nvcc` (default `/usr/local/cuda`)
- headers and `libcudart` from `$CUDA_HOME/include` and `$CUDA_HOME/lib64`
- the GPU architectures of the visible devices

The compiled library is cached in `~/.cache/neural_rx_native/` (or
`$XDG_CACHE_HOME/neural_rx_native/`), keyed by the op sources, compiler,
TensorFlow version and GPU architectures. If the build fails, see `build.log`
in that cache directory. To use a toolkit outside `/usr/local/cuda`:

```bash
export CUDA_HOME=/path/to/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

### Ray-tracing environment

Sionna RT needs a separate environment because it uses a different NumPy and
Python version; see `docs/sionna_rt_evaluation.txt`. It is only needed to
export new channel banks. The exported Munich banks in `datasets/rt_munich/`
are read by the benchmark in the `neural_rx` environment.

```bash
conda create -n sionna_rt python=3.11
conda activate sionna_rt
pip install -r requirements-rt.txt
```

### Tests

From the repository root, in the `neural_rx` environment:

```bash
pytest                                  # full suite, about 6 minutes on an A40
pytest tests/test_artifact_io.py        # a single module
ruff check .                            # lint
```

Several tests run TensorFlow on the GPU and compile the native ops on first
use. Training is deterministic for a given recipe seed
(`tf.config.experimental.enable_op_determinism()`), but bit-exact results are
only expected on the same GPU model and CUDA/cuDNN versions.

## Code

```
src/neural_rx/
  backbone/                Original CGNN (cgnn.py), Parameters, E2E model, backbone loading (core.py).
  receiver/                Pilot-history memory receiver:
    model.py                 Complete receiver composition.
    channel_filter.py        Learned channel estimate and uncertainty.
    pilot_history.py         Per-UE GPU history and lifecycle management.
    pilot_layout.py          Pilot positions, neighborhoods, and grid mappings.
    decoder_correction.py    Physical scoring and learned bit-prediction blend.
    native_ops/              CUDA operations and their training gradients.
    benchmark_adapter.py     Stateful evaluation adapter.
    training*.py             Joint training, fresh data, batching, and workers.
    checkpoints.py           Atomic checkpoint recovery.
  decoding.py              Shared Sionna decoder and bit mapping (memory and Cold receivers).
  benchmark/               Shared accuracy/latency evaluation, protocol, scoring, smoke test.
  data/                    Statistical (temporal.py) and shared-geometry (geometry.py) generators.
  rt/                      Ray-traced channel export (export.py) and benchmark integration.
  provenance/              Artifact hashing, atomic writes, and source-closure records.
  verification/            Independent reference calculations used by checks.
  cli/                     Entry points behind the nrx-* commands.
  paths.py                 Repository root for configs, models, and datasets.
tests/                     Correctness tests and shared fixtures.
scripts/                   Paper-figure scripts (plot_*.py).

config/memory_receiver.json        Current receiver dimensions and memory policy.
config/memory_training.json        Joint training recipe.
config/temporal_benchmark_v2.json  Default evaluation scenarios and scoring (v1 plus ray-traced Munich).
config/temporal_benchmark_v1.json  Statistical-channel scenarios; used by training and the 20261019 campaign.
config/nrx_large.cfg               Radio setup and original backbone settings.
config/rt_munich.json              Munich ray-tracing setup for exporting channel banks.
```

## Commands

Run from the repository root, in the `neural_rx` environment:

```bash
nrx-benchmark --help        # shared accuracy/latency benchmark
nrx-campaign --help         # training campaign with development evaluation and selection
nrx-train --help            # joint training loop only
nrx-smoke --help            # quick adapter smoke test
```

`nrx-campaign` runs training with periodic development evaluation,
checkpoint selection, final confirmation, and a latency comparison.
`nrx-train` provides the training loop without that orchestration.
`nrx-train-backbone` retains the original backbone trainer.

To evaluate the selected model, pass

```
--experiment-checkpoint models/learned_blend_joint_20261019/checkpoint_005000
```

to the shared benchmark. Accuracy phases also require calibration anchors.
The completed campaign's anchors are in
`experiments/20261019_learned_blend_joint_training_6000/inputs/anchors.json`.
They cover the v1 statistical conditions only; evaluate against them with
`--benchmark-protocol config/temporal_benchmark_v1.json`.

The default v2 protocol adds `ray_traced_munich` as a report condition: it is
evaluated and reported in calibration and confirmation, but it is excluded from
the core TBLER, the statistical tests, development, and training. Calibration
and confirmation read the independently seeded banks declared in v2 from
`datasets/rt_munich/`, and cache their episodes in `datasets/rt_munich/episodes`.

## Results and reproduction

`experiments/20261019_learned_blend_joint_training_6000/` retains the launch
configuration, actual source snapshot, development results, confirmation
results, and trained-weight latency results. `retained_artifacts.json`
identifies these artifacts and their roles.

The selected model's primary confirmation TBLER was 15.846%, versus 16.504%
for frozen Cold K8 and 17.004% with identical weights reset every TB. Memory
benefit passed the declared statistical criterion; K8 superiority did not.
Measured p95 latency was 42.846 ms versus Cold K3's 44.018 ms. The shared timer
includes memory, detection, decoding, and transfers, and excludes initial
LS estimation and interpolation.

### Figures

Both scripts read only committed files and run in seconds without a GPU:

```bash
python scripts/plot_receiver_diagrams.py   # misc/receiver_architecture.*, misc/evaluation_setup.*
python scripts/plot_paper_results.py       # misc/confirmation_tbler.*, misc/latency.*, misc/development_tbler.*
```

`plot_paper_results.py` verifies each result's `result_sha256` and
`protocol_sha256` seal before plotting, and writes the plotted values to
`misc/paper_results.csv` and the input result hashes to
`misc/paper_results_manifest.json`. Figures are written as PNG and PDF.

| Reported number | Source | Figure |
| --- | --- | --- |
| TBLER 15.846% / 17.004% / 16.504% (persistent / reset / Cold K8, core pooled) | `results/confirmation_005000/results.json` → `summary` | `confirmation_tbler` |
| p95 latency 42.846 ms vs Cold K3 44.018 ms | `results/latency_005000/results.json` → `receivers` | `latency` |
| Checkpoint selection (update 5000) | `results/development_*/results.json`, `results/selection.json` | `development_tbler` |

Paths in the table are relative to
`experiments/20261019_learned_blend_joint_training_6000/`.

Published checkpoint metadata still uses the internal filename
`cnn_autoencoder_finalist.json`. The recipe protocol and deterministic seed
namespace also retain their saved identifiers. These are serialization
identifiers, not alternative models; changing them would alter artifact
identity or training data. Live files use the names listed above.

Recorded results belong to their saved source snapshots. Renamed live source
has a new fingerprint and does not silently inherit cached result validity.
Some episode archives use RAM-backed storage and do not survive a reboot;
their source, plans, seeds, and anchors are retained for regeneration.
