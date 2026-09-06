# V18 training correction pass

## Decision

Correct the V18 training contract before changing its memory architecture.
Segment-Safe V18 inference, PD-L8, K=2 CGNN depth, CJLR, and the benchmark
remain unchanged.

The corrected training pipeline must train and seal the exact matched Cold K=8
reference first. The K=2 student then learns from that checkpoint while using
the same Segment-Safe memory behavior used at evaluation.

No long training is authorized. The first candidate receives exactly 500
updates and must pass the existing paired screen.

## Implementation

The correction is implemented as the sole supported training protocol. The V1
conclusion is preserved in `experiment_history.md`; its obsolete code and raw
artifacts are not retained.

- Segment-Safe training and evaluation share `SegmentSafePilotBank`.
- Sparse tracks accept arbitrary opaque UE IDs.
- The v2 scheduler draws an effective pool of three through eight UEs.
- `temporal_training_curriculum.py` selects the five event-relative strata.
- Matched K=8 writes a sealed episode manifest and checkpoint summary.
- Candidate training requires that summary and replays the manifest exactly.
- Registration rejects any teacher, checksum, or data-fingerprint mismatch.
- `orchestrate_v18_aligned_training.py` enforces teacher-first execution.

## Verified defects

### The student learned from the wrong K=8 target

Candidate training constructs a fresh K=8 model from
`weights/nrx_large_weights` and freezes it. In parallel, the benchmark trains a
different matched Cold K=8 model for 500 updates. Evaluation compares V18 with
the second model, not with the teacher used for distillation.

The model checksums prove that these are different:

- Candidate's frozen teacher: `2c7512fa...f658e43`.
- Evaluated matched K=8: `23026846...99a7e`.

The student was therefore optimized to imitate one target and scored against
another. A distillation experiment cannot establish K=2 versus matched K=8
until the teacher and evaluated reference are the same sealed checkpoint.

### Training and evaluation used different memory transitions

Training calls the legacy `build_pd_l8_sequence`. The corrected evaluator calls
`OnlineSegmentSafePilotDomainMemory`.

Legacy PD-L8 can use history at ages where Segment-Safe selects current-only
probation, and it can retain an early boundary that Segment-Safe rejects. This
changes the channel tensor supplied to StateInit even though the neural weights
are unchanged.

### Training could not create physical-UE replacement

Training fixes the UE pool at two, equal to the receiver's two scheduled
positions. The dynamic scheduler can only reorder IDs 0 and 1. It cannot replace
one scheduled UE with ID 2 or 3.

The schedule benchmark uses a four-UE pool. Its most difficult descriptive
state is a continuing UE whose co-scheduled interference partner changes. That
state is absent from the existing training distribution.

This does not mean training a separate model for each UE. UE IDs remain opaque
keys for memory routing. All UEs use the same shared K=2 weights.

### Training targets absolute time rather than receiver state

Each 16-TB sequence updates on TBs 1, 2, 5, 9, and 16. Those positions provide
age coverage only when the same UEs stay active. With dynamic scheduling, they
do not guarantee coverage of:

- A newly observed UE.
- A returning UE.
- A continuing UE with a replacement partner.
- Segment-Safe probation.
- A detected channel boundary.
- Mature valid memory for both active UEs.

The training report counts aggregate inactive cells and reordered TBs, but it
does not count how many optimizer updates supervise each of these states.

## Corrected dependency order

The two training jobs can no longer run in parallel because the student depends
on the final matched teacher.

1. Generate the sealed training episode manifest.
2. Train matched Cold K=8 for exactly 500 updates.
3. Seal its checkpoint, file hash, and model checksum.
4. Load that exact checkpoint as the candidate's frozen teacher.
5. Train Segment-Safe V18 K=2 for exactly 500 updates.
6. Verify the teacher remains unchanged.
7. Register both checkpoints and run paired evaluation.

The candidate summary must record `teacher_checkpoint_sha256`. Registration
must reject the candidate unless that value equals the checkpoint hash in the
matched Cold K=8 manifest.

## Corrected data contract

### Shared receiver, arbitrary identities

The receiver still processes at most two simultaneous UEs. A training episode
draws a larger pool of stable opaque identities and schedules zero, one, or two
of them into the two receiver positions.

- Draw physical-pool size independently from 3 through 8.
- Randomize identities, replacement order, inactivity, and re-entry.
- Never pass the numeric identity value into the neural network.
- Use identity only to retrieve the correct pilot-memory track.
- Use the same neural weights for every scheduled UE.

This teaches permutation and scheduling invariance rather than memorizing UE
types.

### Training channels

Keep training separate from benchmark realizations and seeds. Use the declared
training-role TDL-A, TDL-B, and TDL-C families with randomized delay spread and
Doppler. Randomize which physical UE receives each profile.

Do not train on the two failing benchmark episodes, their UE IDs, their exact
event positions, or the benchmark's fixed 25% replacement probability.

Keep the existing 1-to-5 dB range for the first correction. The numeric 6 and
8 dB geometry anchors are not directly comparable with TDL difficulty, and
copying them would leak benchmark choices into training. A Stage 0 report must
instead measure teacher loss by training profile and transition stratum. If a
later correction is needed, calibrate SNR only on training-role profiles so
examples span useful teacher-error levels; do not use benchmark anchors.

### Event-relative supervision

Keep five optimizer updates per generated 16-TB sequence, but select them by
receiver state rather than absolute TB index. Across each deterministic block
of training episodes, require equal counts from these strata:

1. `cold`: first active observation of a physical UE.
2. `return`: a previously observed UE becomes active after an absence.
3. `partner_change`: a UE continues while another active UE is replaced.
4. `rebuild`: Segment-Safe reports reset or current-only probation.
5. `mature`: the active set is unchanged and every active UE has valid history.

When a TB contains a partner change, calculate loss for both active UEs. This is
necessary because the continuing UE showed the largest descriptive K=2 gap.

The state labels control sample selection only. They are forbidden from the
receiver input. If an episode does not contain the requested stratum, regenerate
it from the next registered seed rather than silently substituting a different
state.

## Corrected memory construction

The current offline routing functions require every schedule row to be a
permutation of two IDs. Replace that assumption with sparse per-identity tracks.

For each training episode:

1. Extract current DMRS observations for the two scheduled positions.
2. Route each active observation to its opaque physical-UE track.
3. Apply the same `SegmentSafeDMRSGuard` transition used online.
4. Build the posterior only from accepted observations in that track's current
   segment.
5. Route the two current posteriors back to scheduled receiver order.

The pure transition kernel must be shared by training and evaluation. Separate
implementations that merely reproduce similar rules are not acceptable.

## Loss and trainable variables

Retain the existing loss and deployed variables for the first correction:

\[
L = 0.5 L_{\rm payload} + 0.5 L_{\rm matched\ K8}.
\]

- `L_payload` uses the current TB's fresh coded bits.
- `L_matched K8` uses soft logits from the sealed matched K=8 checkpoint.
- Teacher and student receive identical current payload, noise, active mask,
  current LS estimate, and received grid.
- Only the 39 variables executed by K=2 remain trainable.
- CJLR, memory filtering, and the K=8 teacher remain frozen.
- No previous payload bit or teacher hidden state crosses a TB boundary.

Keeping the loss fixed isolates the effect of correcting teacher identity,
memory semantics, and transition coverage. Intermediate-state distillation or
new loss weighting requires a later registered child experiment.

## Required Stage 0 tests

### Teacher binding

- The candidate teacher file hash equals the evaluated matched K=8 hash.
- The loaded teacher model checksum equals its sealed training checksum.
- Candidate training leaves both checksums unchanged.
- Candidate training refuses an unregistered teacher checkpoint.

### Memory equivalence

- Offline and online Segment-Safe reset masks are identical.
- History-valid masks, posteriors, and variances are identical.
- Equality holds under reordering, inactivity, replacement, re-entry, and
  multiple channel boundaries.
- No track ever receives another physical UE's pilots.

### Coverage

- Every five-stratum coverage counter is nonzero in smoke training.
- The full 500-update screen has equal stratum counts.
- At least three distinct physical IDs occur in every replacement batch.
- Numeric ID relabeling leaves neural inputs and outputs equivariant.
- Training and benchmark seed intervals remain disjoint.

### Optimization

- Every selected K=2 variable receives a finite gradient.
- Frozen variables, CJLR, and teacher checksums remain fixed.
- Exactly 500 updates occur.
- Checkpoint selection remains the last update with no benchmark validation.

## Paired screen gates

Run Benchmark v1 without changing its episodes, SNRs, scoring window, or Cold
K=8 reference.

- Core-mean K=8 gap is below 0.5 percentage points.
- Schedule-condition point gap improves by at least 1.0 point.
- Reset-V18 improves at active-set changes, confirming better current-TB K=2
  processing rather than a more permissive memory decision.
- Persistent memory gain is non-negative.
- Natural UMi, natural UMa, and topology TBLER each regress by no more than 0.5
  points.
- Candidate latency and storage remain unchanged within measurement noise.

The schedule screen contains only five current paired disagreements, so these
are development gates rather than a publication claim. No 6,000-update run is
authorized unless every gate passes.

## What this pass deliberately does not change

- PD-L8 history length.
- Segment-Safe threshold or probation semantics.
- K=2 iteration count.
- CGNN or CJLR architecture.
- Per-UE memory representation.
- Benchmark episodes or ranking.
- A UE-specific set of neural weights.

If the corrected reset path still trails K=8 after this aligned training, then
the remaining problem is genuine K=2 capacity or within-TB processing. If only
the persistent path fails after gaps, then a timestamp-aware dormant-memory
design becomes justified. Those two outcomes require different architecture
changes and must not be conflated now.

## Outcome

The aligned run passed its correctness checks but did not close the K=8 gap.
Seven of eight remaining core errors also occurred when memory was reset every
TB. A physical-distribution audit found that V2 training omitted the severe
near-far and weak-user cases responsible for most remaining failures. Further
reset-policy tuning is therefore rejected; the next work must improve
current-TB weak-user separation under randomized power imbalance.

That follow-up is implemented as the versioned
[`V3 weak-user correction`](temporal_receiver_v18_training_correction_v3.md).
