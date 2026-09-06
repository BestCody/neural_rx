# Temporal receiver v18

## Current architecture

V18 is the strongest retained temporal receiver. It combines three parts:

1. PD-L8 filters each UE's current and previous seven DMRS observations.
2. A K=2 CGNN student is distilled from the frozen Cold K=8 receiver.
3. Coded Joint-List Refinement exchanges detector and LDPC information.

Memory contains channel-pilot evidence only. It is routed by persistent UE
identity, discarded after re-entry or an external reset, and reset when causal
DMRS disagreement exceeds the calibrated threshold. Previous payload bits and
hidden detection state are not stored across transport blocks.

The receiver configuration is frozen in
[`temporal_receiver_v18_segment_safe.json`](../config/temporal_receiver_v18_segment_safe.json).
The calibrated reset rule is stored in
[`segment_safe_pd_l8_threshold.json`](../config/segment_safe_pd_l8_threshold.json).

## Retained evidence

On the historical known-comparison data, optimized exact-CJLR PD-L8 K=2 needed
2.039 dB for 10% TBLER, while Cold K=8 needed 2.402 dB. The measured threshold
gap was -0.363 dB with a 95% interval from -0.409 to -0.320 dB.

At 2.25 dB, persistent PD-L8 reduced TBLER by 7.450 percentage points compared
with resetting the same receiver every TB. The paired 95% interval was 5.301 to
9.794 points. CJLR reduced TBLER by 11.802 points compared with ordinary BP20
on the same distilled PD-L8 K=2 model, with an interval of 9.208 to 14.509.

These results establish a strong in-distribution receiver and a measurable
memory contribution. They do not establish broad generalization. The old TDL-D
and TDL-E test produced pathological neural-decoder saturation, so it was not a
valid general deployment benchmark. V18 remains a candidate until it passes
the systematic UMi/UMa, change, re-entry, near-far, and close-LOS benchmark.

## Current Benchmark v1 result

The V3 Segment-Safe screen reached 9.93% core TBLER, compared with 9.38% for
matched Cold K=8. Resetting the same candidate every TB produced 10.14%, so the
measured memory gain was 0.21 percentage points and its confidence interval
included zero.

The exact audit found five candidate errors not made by K=8. Four also occurred
when V18 memory was reset every TB, so current-TB K=2 processing explains 80%
of the observed differential cells. The four scheduling failures affect one
continuously active UE; immediate re-entry is not the cause.

Three scheduling failures combine a weak UE near -6.9 dB, approximately 9 dB
near-far separation, and mean spatial overlap from 0.431 to 0.446. V3 training
covered the power deficit but not the shared array geometry. Its scalar power
augmentation cannot change normalized channel overlap.

Memory still reduced wrong payload bits across the four scheduling failures
from 2,962 to 1,436. It improved the estimates without moving those TBs across
the final decoder boundary. The one memory-specific topology failure remains a
secondary cross-UE state-isolation issue.

K=2 and K=8 already receive the same PD-L8 estimate. K=8 still has six extra
joint message-passing rounds for separating current payloads. The next design
or training change must target current-TB weak-user separation rather than a
longer memory horizon or another reset rule.

V4 was rejected. On the same episode bank it introduced 16 candidate failures,
repaired none, and increased the paired K=8 gap from 0.54 to 1.83 percentage
points. V3 therefore remains the active checkpoint and training recipe.

Old and aligned scheduling screens are not a valid paired training comparison:
a generator change altered the schedule RNG stream under the same protocol
JSON. Future comparisons must cache exact episode tensors and bind the
generator revision before attributing small changes to training.

## Source map

- `pilot_domain_channel.py`: pilot extraction and interpolation primitives.
- `pilot_domain_memory.py`: causal PD-L8 state, UE routing, and reset logic.
- `pilot_domain_channel_v19.py`: exact newest-only streamed estimation.
- `pilot_domain_memory_v19.py`: active streamed online PD-L8 memory.
- `temporal_receiver_core.py`: shared K=2 and tensor-conversion utilities.
- `cjlr_receiver.py`: exact coded joint-refinement primitives.
- `cjlr_optimized_receiver.py`: optimized exact-list implementation.
- `cjlr_final_runtime.py`: immutable paired v18 decode runtime.
- `train_temporal_benchmark.py`: matched K=8 and aligned K=2 training.

All new designs must use the shared benchmark in
[`temporal_benchmark_v1.md`](temporal_benchmark_v1.md).
