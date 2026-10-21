PILOT-HISTORY NEURAL RECEIVER

One deployed memory-receiver architecture:
  Current LS pilots + seven previous pilot records per UE
    -> learned channel filter and correlated uncertainty
    -> CGNN initializer and two processing rounds
    -> eight Sionna LDPC iterations
    -> physical symbol-pair correction and learned per-bit blend
    -> ten further LDPC iterations, retaining decoder state
    -> payload bits and CRC.

Selected trained receiver:
  models/learned_blend_joint_20261019/checkpoint_005000/
Latest checkpoint for continuing the same campaign:
  models/learned_blend_joint_20261019/checkpoint_006000/
Initialization used by that campaign:
  models/learned_blend/initial_receiver.weights.h5
Frozen Cold K2/K3/K8 controls:
  models/pretrained/nrx_large_weights

The checkpoints in the retained campaign are training milestones of this one
architecture. The model folders and experiment archives of retired receivers
were removed. The initializer retains the inherited trained detector; it is
not a randomly initialized network.

CODE

scripts/pilot_history_receiver/
  model.py                 Complete receiver composition.
  channel_filter.py        Learned channel estimate and uncertainty.
  pilot_history.py         Per-UE GPU history and lifecycle management.
  pilot_layout.py          Pilot positions, neighborhoods, and grid mappings.
  decoder_correction.py    Physical scoring and learned bit-prediction blend.
  native_ops/              CUDA operations and their training gradients.
  benchmark_adapter.py     Stateful evaluation adapter.
  training*.py             Joint training, fresh data, batching, and workers.
  checkpoints.py           Atomic checkpoint recovery.

scripts/receiver_decoding.py       Shared Sionna decoder and bit mapping.
utils/neural_rx.py                 Original CGNN backbone.
scripts/temporal_benchmark_*.py    Shared accuracy/latency evaluation.
scripts/temporal_training_data.py Statistical evolving-channel generator.
scripts/geometry_mumimo_data.py   Shared-geometry UMi/UMa generator.
scripts/rt_*.py                   Ray-traced channel integration.
scripts/verification/             Independent reference calculations for checks.
scripts/test_*.py                 Correctness tests.

config/memory_receiver.json       Current receiver dimensions and memory policy.
config/memory_training.json       Joint training recipe.
config/temporal_benchmark_v1.json  Declared evaluation scenarios and scoring.
config/nrx_large.cfg              Radio setup and original backbone settings.
config/rt_munich.json             Optional Munich ray-tracing setup.

COMMANDS (from the repository root, in the neural_rx environment)

  python scripts/run_temporal_benchmark.py --help
  python scripts/run_memory_training.py --help
  python scripts/train_memory_receiver.py --help

run_memory_training.py runs training with periodic development evaluation,
checkpoint selection, final confirmation, and a latency comparison.
train_memory_receiver.py provides the training loop without that orchestration.
train_neural_rx.py retains the original backbone trainer.

To evaluate the selected model, pass
  --experiment-checkpoint models/learned_blend_joint_20261019/checkpoint_005000
to the shared benchmark. Accuracy phases also require calibration anchors.
The completed campaign's anchors are in
  experiments/20261019_learned_blend_joint_training_6000/inputs/anchors.json

RESULTS AND REPRODUCTION

experiments/20261019_learned_blend_joint_training_6000/ retains the launch
configuration, actual source snapshot, training log, development results,
confirmation results, and trained-weight latency results.
retained_artifacts.json identifies these artifacts and their roles.

The selected model's primary confirmation TBLER was 15.846%, versus 16.504%
for frozen Cold K8 and 17.004% with identical weights reset every TB. Memory
benefit passed the declared statistical criterion; K8 superiority did not.
Measured p95 latency was 42.846 ms versus Cold K3's 44.018 ms. The shared timer
includes memory, detection, decoding, and transfers, and excludes initial
LS estimation and interpolation.

Published checkpoint metadata still uses the internal filename
cnn_autoencoder_finalist.json. The recipe protocol and deterministic seed
namespace also retain their saved identifiers. These are serialization
identifiers, not alternative models; changing them would alter artifact
identity or training data. Live files use the names listed above.

Recorded results belong to their saved source snapshots. Renamed live source
has a new fingerprint and does not silently inherit cached result validity.
Some episode archives use RAM-backed storage and do not survive a reboot;
their source, plans, seeds, and anchors are retained for regeneration.
