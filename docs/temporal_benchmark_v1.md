# Temporal Neural RX benchmark v1

## Purpose

Every receiver idea uses one benchmark contract. Architecture-specific code may
decode a batch, but it cannot choose data, seeds, metrics, windows, statistics,
or acceptance rules.

The benchmark tests four claims:

1. Temporal K=2 approaches a training-matched Cold K=8.
2. Persistent memory improves the same K=2 model with memory reset.
3. Memory remains stable and recovers after a channel change or scheduling gap.
4. The improvement justifies measured latency and per-UE storage.

## Baselines

Historical Cold K=8 preserves the NVIDIA comparison. Matched Cold K=8 must use
the same training examples, distribution, and recipe seeds as the candidate.
A historical checkpoint cannot substitute for that matched accuracy baseline.
Cold K=3 uses one frozen checkpoint because it is timed but not scored.

Perfect-CSI K-Best is an algorithmic reference. It is not a capacity bound or
a guaranteed oracle.

## Scenario bank

Core ranking conditions are natural UMi, natural UMa, a controlled topology
change, and scheduling gap with re-entry. TDL-B100 and TDL-C300 are separate
reporting conditions so one profile cannot hide failure on the other.

Close-LOS and near-far conditions are stress tests. They never contribute to
the core mean score or select a winner by themselves.

Each geometry condition is stratified by LOS state, speed, and angular
separation. These labels are scoring metadata and cannot enter the receiver.

## SNR calibration

Only matched Cold K=8 is used to calibrate operating points. The simulation
control is Eb/N0 even though internal schema fields retain the shorter `snr`
name. For each non-stress condition, the measured curve must bracket 10%
TBLER. A sealed artifact freezes the nearest interior anchor and its adjacent
1 dB points. Candidate evaluation cannot begin without this artifact.

SNR@10% is reported only when the measured curve brackets 10%. Screening uses
paired excess TBLER at the frozen anchor because it remains defined even when a
candidate has an error floor.

## Stages

| Stage | Conditions | Episodes | TBs | SNRs | Training |
| --- | --- | ---: | ---: | ---: | ---: |
| Screen | Four core | 16 | 16 | Anchor | 500 updates |
| Development | Four core | 64 | 64 | Three | 6,000 updates |
| Confirmation | All eight | 64, then 128/256 if needed | 64 | Anchor | Three independent 6,000-update runs |

Screen and development exclude historical and stress-only reporting. The
development stage adds the two neighboring SNRs only after an idea passes the
anchor-only screen.

Confirmation starts with 64 episodes per condition. It stops when the 95%
confidence-interval half-width is at most one percentage point for both the
worst core K=8 gap and memory gain. Otherwise, it advances to the predeclared
128-episode and then 256-episode plan. The maximum is a cap, not an automatic
workload.

Each stage uses disjoint seeds. Confirmation remains locked until the design,
checkpoint-selection rule, and analysis are frozen. Looking at confirmation
and then changing the receiver consumes that confirmation split.

## Required methods

- Candidate K=2 with persistent memory.
- The same checkpoint with memory reset every TB.
- Training-matched Cold K=8 as the accuracy target.
- Frozen Cold K=3 for latency only.
- Historical Cold K=8 for final reporting only.
- Perfect-CSI K-Best for final reporting only.

Cold K=3 is not retrained per seed or decoded across accuracy episodes.

## Metrics

The primary gap is candidate TBLER minus matched Cold K=8 TBLER. Conditions
receive equal weight so a frequent easy condition cannot hide a hard one. The
worst core-condition gap is always reported.

The primary score is measured only at the calibrated anchor; the lower and
upper points support threshold interpolation and never get averaged into the
primary score. Memory gain is reset TBLER minus persistent TBLER. Late drift
compares TBs 49--64 against TBs 17--32. Adaptation reports immediate excess
errors, recovery time, post-change excess-error area, and the first four TBs
after UE re-entry.

Episode outage is the fraction of channel realizations for which more than
half of active UE transport blocks fail. It exposes the all-or-nothing failures
that aggregate TBLER can hide.

End-to-end per-TB latency includes accelerator synchronization. It is measured
in a separate fixed test with 16 warm-up TBs and 128 timed TBs at the natural
UMi anchor. Candidate K=2 and frozen Cold K=3 use batch size one. Ranking uses
candidate p95 latency normalized by frozen Cold K=3. Cold K=8 is an accuracy
target and is not timed.

Confidence intervals resample whole episodes. Confirmation additionally
resamples training seeds. TBs from one channel history are never counted as
independent experimental units.

## Selection

Correctness, causality, provenance, and reproducibility are hard gates. Passing
candidates form an accuracy-latency Pareto frontier. The default winner has the
smallest worst-condition K=8 gap on that frontier, followed by mean K=8 gap,
memory gain, and normalized p95 latency.

## Registration

Before implementation, every candidate records one structural change, its
hypothesis, the fixed metric expected to improve, the minimum useful change,
the compute limit, and the stop rule. Results from incompatible benchmark or
training protocols cannot share a leaderboard.

Each trained seed maps to an explicit checkpoint checksum. Model-specific
training code may differ between candidates and baselines, but paired models
must use identical training data, example counts, update counts, recipe seeds,
and checkpoint-selection rules.

Checkpoints registered under the former K=2 latency-only protocol can be
rebound without retraining only through `rebind_temporal_benchmark_k3.py`.
The script requires the exact predecessor checksum, preserves checkpoint
hashes, replaces only the latency reference, and writes a sealed migration
record. It also rebinds the latency-independent K=8 SNR calibration so the new
stage plan is internally consistent.

## Execution model

The shared runner owns episode generation, method pairing, reset timing, CRC
checks, and latency timing. An architecture adapter starts and clears state,
decodes single or batched TBs, synchronizes its accelerator, and verifies
immutable artifacts.

Candidate adapters receive only the current received grid, current LS channel,
noise variance, active mask, UE identities, and slot number. Scenario labels,
future TBs, payload truth, change events, and perfect channels are unavailable.
Only the K-Best adapter receives the perfect channel.

Accuracy evaluation batches eight independent episodes. TBs remain sequential
inside each episode, so no receiver can access future memory. The batched and
single-episode adapters must produce the same decoded bits and CRC decisions in
the smoke test. Soft logits may differ slightly because GPU kernels accumulate
floating-point values in a different order; the smoke records that difference.

Matched Cold K=8 accuracy is evaluated once for each stage and training seed.
Cold K=3 uses one frozen checkpoint for the latency microbenchmark.
Historical Cold K=8 and perfect-CSI K-Best do not depend on training seed, so
they use a separate sealed cache that is produced once for confirmation.
Candidate code evaluates only the persistent and reset paths. Composition
fails unless all caches contain identical Eb/N0 values, episode fingerprints,
active masks, UE identities, and channel-change events.

The reference cache stores matched Cold K=8 accuracy and frozen Cold K=3
latency. This avoids decoding Cold K=3 over the full accuracy bank or timing a
Cold K=8 receiver that is used only as the quality target.

The scenario bank is implemented in
[`temporal_benchmark_scenarios.py`](../scripts/temporal_benchmark_scenarios.py).
It provides independent TDL, shared-topology UMi/UMa, a midpoint topology
change at TB9 or TB33 depending on stage length, scheduling gaps with physical
UE re-entry, close-LOS geometry, and controlled 6/12 dB near--far imbalance.

## Workflow

1. Register the candidate and required references with
   `register_temporal_benchmark_candidate.py`.
2. Run matched Cold K=8 calibration with `run_temporal_benchmark.py` in
   `calibration` mode.
3. Seal the operating points with `calibrate_temporal_benchmark.py`.
4. Create an immutable stage plan with `prepare_temporal_benchmark.py`.
5. Run `reference` mode once per required training seed.
6. For confirmation, run `static-reference` once for historical K=8 and
   perfect-CSI K-Best.
7. Run `candidate` mode for each idea and seed.
8. Join each candidate with its matched cache and, for confirmation, its
   static cache using
   `compose_temporal_benchmark_run.py`.
9. Build one or more paired leaderboards with
   `analyze_temporal_benchmark.py`.

The artifact-map JSON passed to the runner maps each registered SHA-256 digest
to its local file path. Every path is rehashed before an adapter is created.

The production adapter factories are:

- `temporal_benchmark_adapters:make_v18_segment_safe_adapter` for v18.
- `temporal_benchmark_adapters:make_cold_nrx_adapter` for all Cold Neural RX
  references.
- `temporal_benchmark_adapters:make_perfect_csi_kbest_adapter` for the oracle.

Every manifest registers the selected radio `.cfg` as a support artifact. A
v18 manifest also registers `temporal_receiver_v18_segment_safe.json` and
`segment_safe_pd_l8_threshold.json`. The adapters reject an unregistered or
changed runtime configuration before decoding.

Run the two-TB adapter smoke test before creating a reference cache:

```bash
PYTHONPATH=scripts:. python scripts/smoke_temporal_benchmark_adapters.py
```

The smoke uses registered local artifacts and the real decoders, but its
two-TB output is a wiring check rather than benchmark evidence.

## Implemented safeguards

- Disjoint calibration, screen, development, and confirmation episode seeds.
- Fixed training seeds and equal training examples per stage.
- Confirmation lock requiring a separately sealed design authorization.
- Atomic TDL profile reporting and equal core-condition weighting.
- Whole-episode bootstrap intervals, with training-seed resampling at final
  confirmation.
- Exact cross-candidate episode and cached-reference equality checks.
- Seed-independent caching for historical K=8 and K-Best.
- Batched accuracy with single-episode latency measurement.
- Precision-based confirmation stopping at 64, 128, or 256 episodes.
- Scenario occurrence checks for topology changes, gaps, and UE transitions.
- Hard rejection of inactive-UE errors, CRC disagreement, non-finite output,
  changed checkpoints, unplanned operating points, and wrong K values.
