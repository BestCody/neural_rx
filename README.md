# Pilot-History Neural Receiver

One deployed memory-receiver architecture:

```
Current LS pilots + the LS pilots of up to seven previous TBs per UE (the pilot memory)
  -> channel filter whose weights are fitted per UE on the memory by least squares (no trained weights)
  -> CGNN initializer and two processing rounds
  -> twenty Sionna LDPC iterations
  -> payload bits and CRC;
the LS estimate of the TB at its pilots becomes the UE's newest memory entry.
```

The filter (`src/neural_rx/receiver/channel_filter.py`) predicts each pilot's LS estimate from its neighbours in the
same TB and from the newest entries, with weights fitted on the three newest entries, chooses on the current TB how
many entries still fit (0 to 4), and combines the prediction with the pilot's own estimate by their error variances.

Training has two stages on the same sequences; only the network is trained. The loss is the binary cross-entropy of
the LLRs against the transmitted bits plus the original NRX's channel-estimate term. Stage 1 also used the soft bits of
the shipped eight-iteration network as a teacher (`distillation_weight` 0.5); stage 2 does not (the default, 0).

| Role | Path |
| --- | --- |
| Stage 1, neural_rx with two iterations (run `experiments/20261117_neural_rx_training`) | `models/neural_rx_k2_20261117/` |
| Stage 2, the trained receiver: the network behind the memory and filter, from stage 1's best checkpoint (run `experiments/20261118_no_teacher_training`) | `models/pilot_memory_no_teacher_20261118/` |
| Initialization of stage 1 | `models/pretrained/nrx_large_weights` |
| Frozen Cold K2/K3/K8 controls | `models/pretrained/nrx_large_weights` |

The released Large NRX weights are the starting point of stage 1; nothing is trained from random initialization.

### Versioned artifacts

Model folders hold the checkpoints at every development check (`checkpoint_NNNNNN/`), the run metadata (`run.json`,
`progress.json`, `latest.json`) and the per-update history (`history.jsonl`, large, not meant for git). The selected
checkpoints are `models/neural_rx_k2_20261117/checkpoint_011000` (stage 1) and
`models/pilot_memory_no_teacher_20261118/checkpoint_006000` (stage 2, the proposed receiver);
`src/neural_rx/benchmark/tbler_sweep.py` points to the latter.

## Setup

### Requirements

- Linux with an NVIDIA GPU. The memory buffer op is CUDA-only; the
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
pytest                                  # full suite, about 4 minutes on an A40
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
    channel_filter.py        Learned channel filter over the current pilots and the memory.
    pilot_history.py         Per-UE GPU history and lifecycle management.
    pilot_layout.py          Pilot positions, neighborhoods, and grid mappings.
    native_ops/              CUDA operation of the memory buffer.
    benchmark_adapter.py     Stateful evaluation adapter.
    training*.py             Training (both stages), fresh data, batching, and workers.
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
scripts/                   Paper numbers, their checks, tables, and figures (nrx_paper_*.py, make_nrx_paper_tables.py,
                           plot_nrx_paper_figures.py).

config/memory_receiver.json        Current receiver dimensions and memory policy.
config/memory_training.json        Training recipe (both stages).
config/temporal_benchmark_v2.json  Default evaluation scenarios and scoring (v1 plus ray-traced Munich).
config/temporal_benchmark_v1.json  Statistical-channel scenarios; used by the development checks and the test sequences.
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

To evaluate the selected model with the shared benchmark, pass

```
--experiment-checkpoint models/pilot_memory_no_teacher_20261118/checkpoint_006000
```

Accuracy phases also require calibration anchors:
`experiments/20261019_learned_blend_joint_training_6000/inputs/anchors.json` for the v1 statistical conditions
(with `--benchmark-protocol config/temporal_benchmark_v1.json`), and
`experiments/20261021_package_layout_validation/results/v2_calibration/anchors.json` for the default v2 protocol.

The default v2 protocol adds `ray_traced_munich` as a report condition: it is
evaluated and reported in calibration and confirmation, but it is excluded from
the core TBLER, the statistical tests, development, and training. Calibration
and confirmation read the independently seeded banks declared in v2 from
`datasets/rt_munich/`, and cache their episodes in `datasets/rt_munich/episodes`.

## Results and reproduction

All results are simulations with one training seed. Required SNR is the SNR ($E_b/N_0$) at a TBLER of 10% on 32 test
sequences of 64 slots per condition, 2x4 MU-MIMO, 16-QAM (MCS 14), 132 PRBs; PC = closed-loop power control.

| Receiver | UMi | Channel change | Scheduling gaps | UMi, PC | Channel change, PC | Scheduling gaps, PC |
| --- | --- | --- | --- | --- | --- | --- |
| Proposed NRX | 7.05 | 7.02 | 8.12 | 1.74 | 2.16 | 2.48 |
| Proposed NRX, memory emptied before every TB | 7.75 | 7.72 | 8.90 | 2.54 | 2.90 | 3.11 |
| Large NRX retrained, 2 iterations (stage 1) | 7.56 | 7.63 | 8.93 | 2.45 | 2.77 | 2.99 |
| Large NRX released, 2 iterations | 8.07 | 8.09 | 9.58 | 2.63 | 3.02 | 3.27 |
| Large NRX released, 3 iterations | 7.33 | 7.50 | 8.85 | 2.25 | 2.60 | 2.84 |
| Large NRX released, 8 iterations | 6.85 | 6.89 | 7.95 | 1.77 | 2.10 | 2.31 |
| LMMSE + K-Best | 6.86 | 6.62 | 8.30 | 1.89 | 2.18 | 2.68 |
| Perfect CSI + K-Best | 6.59 | 6.27 | 6.53 | 1.44 | 1.75 | 1.69 |

Median TensorRT 10.3 FP16 inference time on an NVIDIA A100 (export as in the code release of Wiesmayr et al., network
only, 2,000 runs; `experiments/20261114_tensorrt_latency/results/fused_a100.json`, measured on a rented GPU with
`modal_a100.py`): Large NRX with 2 / 3 / 8 iterations 0.79 / 1.07 / 2.46 ms, proposed NRX 0.81 ms. Including the
computation for the next slot (fit and the memory's part of the predictions, 0.11 ms), all GPU work per slot is
0.91 ms; with that computation running at the same time on a second stream, the slot takes 0.88 ms. So the proposed
NRX meets the 1 ms budget either way, and the Large NRX with 3 iterations does not. On the server's A40
(`results/fused_a40.json`): 1.41 / 1.89 / 4.26 ms, proposed 1.44 ms, 1.52 ms including the fit. The channel filter
runs as CUDA kernels (`fitted_filter.cu`, launched by `filter_kernels.py`): before the slot filter_gram (normal
equations, added up from slices of cells), filter_solve (Gauss-Jordan, in one warp's registers up to 31 unknowns) and
filter_history; on the slot filter_predict and filter_expand (0.02 ms). `time_fused.py` checks the kernels against
the TensorFlow receiver and also times every receiver as a CUDA graph, without the host's time to launch the kernels.
`make_a100_bundle.sh` and `run_a100.sh` repeat the measurement on another machine.

On two further datasets at a fixed SNR per condition (17 cases: TDL-B100, TDL-C300, UMi, UMa, channel change,
scheduling gaps, closely spaced LOS UEs, near-far UEs, and, on one dataset, ray-traced Munich channels), the memory
lowers TBLER in 16 cases and leaves it unchanged in one. The proposed NRX is ahead of the Large NRX with 8 iterations
in 6 of the 17 cases (TDL-B100, TDL-C300 and UMa on both datasets) and behind in 10. TBLER over all cases of the first
/ second dataset: proposed NRX 17.1% / 14.1%, memory emptied 22.7% / 20.0%, Large NRX with 8 iterations 16.6% / 14.1%.

| Step | Folder | Run | Main output |
| --- | --- | --- | --- |
| Stage 1: Large NRX retrained with 2 iterations | `experiments/20261117_neural_rx_training` | `./run.sh start` (runs until `./run.sh stop`) | `results/best.json`, `models/neural_rx_k2_20261117/` |
| Stage 2: proposed receiver on top of stage 1 | `experiments/20261118_no_teacher_training` | `./run.sh start` (runs until `./run.sh stop`); its first 2,048 updates were made on rented GPUs with `modal_app.py` and brought here with `bring_home.sh` | `results/best.json`, `models/pilot_memory_no_teacher_20261118/` |
| Test sequences, with and without power control | `experiments/20261118_test_evaluation` | `RECEIVERS="pilot_memory pilot_memory_emptied" ./run.sh` (the stage-1 results are kept), `python fill_missing.py` if a process ran out of GPU memory, then `python summarize.py`, `python doppler.py` | `results/summary.json`, `results/doppler.json` |
| Ablation and classical comparisons on the test sequences (filter from the current TB only; memory and filter with K-Best; classical LMMSE over several slots with K-Best) | `experiments/20261118_test_evaluation` | `python slot_correlation.py` (statistics across slots), `RECEIVERS="pilot_memory_current_tb filter_kbest wiener_kbest wiener_slot_kbest" ./run_extra.sh`, then `python current_tb.py`, `python filter_kbest_summary.py`, `python wiener_check.py` | `results/current_tb.json`, `results/filter_kbest.json`, `results/wiener_check.json` |
| Power control with a random phase jump at every power step | `experiments/20261118_test_evaluation` | `RECEIVERS="pilot_memory pilot_memory_emptied neural_rx_k2_trained" SETTINGS=power_control_phase ./run_extra.sh` (phase jumps in `experiments/20261112_early_retransmission/phase_jumps.py`), `python phase_steps.py --condition <condition>` (the jumps of all pool UEs), then `python phase_jumps_summary.py` | `results/phase_jumps.json` |
| Latency | `experiments/20261114_tensorrt_latency` | `python export.py` (neural_rx env), then on an A100 `modal run modal_a100.py` (or `./make_a100_bundle.sh` and `run_a100.sh` there); on the server `~/venvs/tensorrt/bin/python time_fused.py --tag a40` | `results/fused_a100.json` (paper), `results/fused_a40.json` |
| Network decisions: fallback (including the fallbacks a real-time BS could run: LS + LMMSE and the stateless two-iteration NRXs) and early retransmission | `experiments/20261118_network_decisions` | `./run.sh`, then `./run.sh analyze`; for the stateless retrained NRX, `./run_stateless.sh` (its evidence, checked against its stored decoding), then `python compare_receivers.py` | `results/fallback.json`, `results/controller.json`, `results/compare_receivers.json` |
| 64-QAM (MCS 19) and QPSK (MCS 9), like Wiesmayr et al.: calibration, both training stages, test evaluation | `experiments/20261121_modulations` | `./calibrate.sh nrx_large_64qam.cfg 0` and `./calibrate.sh nrx_large_qpsk.cfg 1`, then `./run.sh start` (development sets in `/dev/shm`; per modulation stage 1 and 2, each ending after three development checks without a new best or at 20,000 updates, then `evaluate.py` for the six NRXs and `summarize.py`; the classical receivers of both modulations last). State on 2026-11-23: 64-QAM trained (stage 1 best at 28,000 of 30,000 updates under the earlier limit, stage 2 at 8,000) and its NRXs evaluated; QPSK training | `results/<label>/summary.json`, `models/<label>_k2_20261121/`, `models/<label>_pilot_memory_20261121/` |
| Robustness with power control: TDD uplink slots (DDDSU), a carrier frequency offset of +-0.1 ppm, HARQ retransmissions with Chase combining for the early-retransmission controller, when the channel filter falls back from its memory (stored TBs used per TB after a channel change or a phase jump), and a screening of LS + LMMSE against the proposed NRX under impairments outside training (Doppler, frequency offset, phase noise) | `experiments/20261122_robustness` | on Modal, `modal run --detach experiments/20261122_robustness/modal_app.py` with `--tests tdd_dddsu,cfo`, `--tests harq` (after `python harq_inputs.py`), `--kprobe power_control:controlled_topology_change,power_control_phase:natural_umi --episodes 16`, or `--screen none:0,doppler:600,doppler:1000,cfo:0.5,cfo:1,phase_noise:0.05,phase_noise:0.15 --episodes 4`; then `modal volume get nrx-robustness results experiments/20261122_robustness/` and `python summarize.py`, `python harq_summary.py`, `python fallback_k_summary.py` | `results/summary.json`, `results/harq_summary.json`, `results/fallback_k_summary.json`, `results/screen/` |
| Two further datasets at fixed SNRs | `experiments/20261118_further_datasets` | `./run.sh` | `results/v2/results.json`, `results/seed86/results.json` |
| Paper numbers, tables, checks, figures (every result the text states is in a figure or table; the tables are written from the result files and the recheck compares every cell with its own recomputation) | `scripts/` | `python scripts/make_nrx_paper_tables.py` (`--check`: tables up to date), `python scripts/nrx_paper_numbers.py --check paper/main.tex --allowed 0.25 0.05 2.14 0.01 38.901 5.1 4.8 10.3`, `python scripts/nrx_paper_recheck.py`, `python scripts/plot_nrx_paper_figures.py` | `paper/tables/`, `paper/numbers.json`, `paper/figures/` |
| IEEE version of the paper (IEEEtran conference template from CTAN, six pages): no Network Decisions section, QPSK and 64-QAM with a summary figure | `paper_ieee/` | `python scripts/plot_nrx_ieee_figures.py`, `make -C paper_ieee pdf`, `NRX_PAPER=paper_ieee python scripts/nrx_paper_recheck.py`, `python scripts/nrx_paper_numbers.py --check paper_ieee/main.tex --allowed 0.25 0.05 2.14 0.01 38.901 5.1 4.8 10.3` | `paper_ieee/build/main.pdf`, `paper_ieee/build/overleaf_ieee.zip` |

Both training stages use both GPUs, so only one runs at a time; each has a development check every 1,000 updates on
the development set in `experiments/20261113_retrained_k2_baseline/episodes/development`.

Stored results of other experiments that the steps above read: the released Large NRX and the classical receivers on
the same test sequences (`experiments/20261104_tbler_sweep`, and with power control
`experiments/20261112_early_retransmission/results/receivers_pc`, which also provides `power_control.py`), and the
calibration anchors named under [Commands](#commands). Of the experiments of earlier receiver designs, only what is
still read remains: these two folders with their validation record (`experiments/20261021_current_code_reproduction`),
the two anchor files, and the development set.

Published checkpoint metadata still uses the internal filename
`cnn_autoencoder_finalist.json`. The recipe protocol and deterministic seed
namespace also retain their saved identifiers. These are serialization
identifiers, not alternative models; changing them would alter artifact
identity or training data. Live files use the names listed above.

Recorded results belong to their saved source snapshots. Renamed live source
has a new fingerprint and does not silently inherit cached result validity.
Some episode archives use RAM-backed storage and do not survive a reboot;
their source, plans, seeds, and anchors are retained for regeneration.
