# V18 weak-user training correction

## Decision

The current failure is primarily a training-distribution failure. V18 already
has useful channel memory, but its K=2 detector was not trained on the severe
anonymous power imbalance present in the remaining errors. The architecture,
PD-L8 horizon, Segment-Safe reset rule, CJLR, and evaluator remain unchanged.

V3 is a single combined correction. It changes only data sampling,
initialization, loss weighting, and optimization. V2 remains available so its
result can be reproduced.

## Audit result

The aligned screen produced 10.12% core TBLER for V18 and 9.29% for matched
Cold K=8, a gap of 0.84 percentage points. Reset-V18 reached 10.23%, so only
0.11 points came from persistent memory on that screen.

Seven of the eight V18-only core errors also failed when memory was reset. All
five V18-only scheduling errors failed under reset as well. The error is
therefore present in current-TB K=2 processing and is not caused mainly by stale
memory.

The failing cells are sharply concentrated by received power:

| Evidence | V2 training | Remaining failures |
| --- | ---: | ---: |
| Largest simultaneous-UE power gap | 5.46 dB | 9.43--12.86 dB in 4/5 cases |
| Weakest active UE power | -4.73 dB | -6.33 to -7.42 dB in 4/5 cases |
| Errors above -5 dB UE power | Not applicable | 0% for both receivers |
| Errors below -5 dB UE power | Not covered | 72.22% V18, 58.33% K=8 |

V2 covered arrivals, returns, partner changes, resets, and mature memory. It
did not control the radio difficulty within those event classes. Event
balancing alone therefore could not teach weak-user separation.

Three optimization details compounded the mismatch:

1. Five independently selected labels sometimes pointed to the same TB. The
   nominal 500 updates contained only 439 distinct target TBs.
2. Equal UE averaging let easy strong users dilute weak-user gradients.
3. The student started from the old base checkpoint. It did not inherit the
   matched teacher's adapted first two blocks and moved only 0.28% at a fixed
   learning rate of 2e-5.

The K=2 architecture may still impose a ceiling, but that conclusion is not
identified until training covers the cases where it currently fails.

## V3 data correction

Each episode still contains 16 consecutive TBs, fresh payloads, three through
eight opaque physical UE identities, dynamic scheduling, channel changes,
gaps, hard resets, and advancing pilots.

V3 adds one independent power random stream. In 75% of episodes, the physical
UE pool receives stable anonymous attenuation values. One
UE is unattenuated, one is attenuated by a random span from 9 to 15 dB, and the
remaining UEs receive random values inside that span. The assignment is
shuffled across identities and remains fixed for the sequence.

The relative deficit multiplies the raw channel without normalizing away its
native TDL fading. The strongest and weakest assigned UEs receive opposite
half-span offsets, so their geometric-mean power stays fixed while their
separation changes. This avoids turning a near--far example into an unrealistically
low-SNR example. The other 25% of episodes retain the unmodified distribution.

Power values are training metadata only. They are never passed to StateInit,
the CGNN, PD-L8, CJLR, or the runtime adapter.

The augmentation lives in `temporal_power_training_data.py`, a training-only
subclass. The shared benchmark generator is byte-for-byte unchanged and its
frozen source checksum remains valid.
The V3 episode manifest also seals every training-source checksum.

## V3 target correction

Training still performs one update for each receiver state:

1. Cold observation.
2. Return after absence.
3. Interference-partner replacement.
4. Segment-Safe rebuild or probation.
5. Mature valid memory.

V3 solves a five-way assignment so the five labels use five distinct TBs. An
episode is regenerated when no distinct assignment exists. This removes
back-to-back duplicate updates without changing the number of optimizer steps.
Within each event class, the assignment first prefers the largest actual power
gap between simultaneously active UEs. Absolute weak-user power breaks near
ties. The 25% unmodified episodes preserve natural-power targets.

The summary records attenuation bins, actual pair-power gaps, the weakest
observed power, duplicate-target count, and accepted augmented episodes. A
full run is rejected if any declared attenuation bin is absent.

## V3 initialization and loss

Matched Cold K=8 is trained and sealed first on the exact V3 episodes. The K=2
student loads that checkpoint and copies the 39 variables shared by its
StateInit, first two CGNN blocks, and readout. Shape and value checksums must
match before optimization starts.

This warm start transfers the teacher's distribution adaptation but adds no
weights or inference work. K=2 is then trained against the same frozen K=8
checkpoint on the same episode manifest.

For UE \(u\), V3 uses

\[
w_{\rm power}(u)=1+2\,\mathrm{clip}(a_u/15,0,1),
\]

where \(a_u\) is the largest of its assigned attenuation, instantaneous
near--far deficit, and power below the nominal unit channel, in dB. Candidate
training also uses

\[
w_{\rm teach}(u)=1+\mathrm{clip}
\left(\frac{[L_{K=2}(u)-L_{K=8}(u)]_+}{0.05},0,1\right).
\]

The final stopped-gradient weight is

\[
w_u=\min(w_{\rm power}(u)w_{\rm teach}(u),4).
\]

Weights are normalized over active UEs, so they change relative attention but
not the overall loss scale. The candidate objective remains

\[
L=0.5L_{\rm payload}+0.5L_{\rm matched\ K8}.
\]

Matched K=8 uses only the power component. This keeps the candidate and
accuracy reference matched while concentrating both on weak users.

V3 replaces the fixed 2e-5 rate with a conservative schedule: 2e-5 to 1e-4
during the first 5% of updates, followed by cosine decay to 1e-5. The peak is
ten times below the original Neural RX training rate.

## Generalization safeguards

V3 does not use benchmark UE IDs, geometry, seeds, payloads, event positions,
or fixed 6/12 dB stress anchors. It trains on independent TDL-A/B/C profiles
and samples a continuous power span. Identity only selects a memory record;
all UEs share one receiver.

The unchanged benchmark uses held-out UMi/UMa geometry. A gain there cannot be
obtained by memorizing the V3 training profiles or numeric identities.

## Evaluation decision

The first run remains the 500-update screen. It must use the existing frozen
Benchmark v1 episodes and compare:

- V3 V18 with persistent memory.
- The same V3 checkpoint reset every TB.
- Training-matched V3 Cold K=8.
- Frozen Cold K=3 for latency only.

The correction passes only if the worst core K=8 gap improves without a
material regression in UMi, UMa, or topology change. The report must break out
weak-user and near-far cells. No 6,000-update run is justified until the screen
shows a real reduction on identical episodes.

If V3 improves weak-user loss but leaves the paired TBLER decisions unchanged,
the remaining limit is architectural: two CGNN exchanges and one CJLR pass do
not provide enough current-TB joint correction. The next justified change
would then be a second efficient detector-decoder feedback step, not more
channel memory or reset tuning.

## Stage 0 verification

The first implementation incorrectly created every power gap by attenuation
alone. Its smoke episode reached the intended 13.01 dB pair gap but pushed the
weakest UE to -14.40 dB, far below the -6.33 to -7.42 dB failure range. V3 now
centers the two endpoint offsets and selects targets by actual active-user
power separation. The corrected smoke episode reached a 13.07 dB gap with a
weakest power of -6.26 dB.

The corrected paired smoke passed all integrity, replay, source-checksum,
gradient, coverage, uniqueness, and warm-start checks. The complete 83-test
suite also passed.

An initialization-only audit on the same corrected episode measured mean bit
cross-entropy of 0.248995 for base K=2, 0.246228 for the first two blocks of the
trained K=8 model, and 0.243914 for the prior distilled K=2 model. Therefore,
the teacher-path copy is better than the base initialization but is not itself
evidence of K=2 compression. Its benefit after V3 teacher training remains a
screened hypothesis.

Stage 0 proves that the intended correction is implemented. It cannot prove
that TBLER improves. Only the frozen 500-update paired screen can accept or
reject the training correction.
