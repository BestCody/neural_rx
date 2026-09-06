# Temporal receiver experiment history

This file records conclusions from removed one-off experiment code. Generated
results and checkpoints remain under ignored `outputs/` directories.

| Stage | Main conclusion | Decision |
| --- | --- | --- |
| Vector and grid memory | Hidden-state reuse drifted and mixed payload state with channel state. | Rejected. |
| Pilot-domain memory | PD-L8 produced a real gain over current-only and PD-L2 channels. | Retained. |
| LAPRA and iteration shortcut | Learned residuals showed memory effects but lost accuracy or latency. | Rejected. |
| Relational and receptive-field changes | Gains were too small for their added complexity. | Rejected. |
| Distillation | Cold K=8 supervision improved the existing K=2 weights without inference modules. | Retained. |
| Cancellation and ReSIC | Oracle cancellation had headroom, but uncertain decisions erased useful signals. | Rejected. |
| JPR | Joint correction preserved uncertainty but did not meet the practical gate. | Rejected. |
| CJLR v18 | Exact coded joint refinement closed the known-comparison gap and showed strong memory gain. | Retained candidate. |
| Benchmark v1 screen | V18 memory helped UMa and re-entry but its reset detector missed the early topology boundary. | Correct reset state machine before further training. |
| Segment-Safe replay | Core TBLER fell to 9.82% versus 9.06% for Cold K=8; all five remaining schedule-only errors also failed under reset-V18. | Retain Segment-Safe inference and align distillation with UE replacement states. |
| Aligned V2 training | Core TBLER was 10.12% versus 9.29% for matched K=8; aligned and earlier Segment-Safe checkpoints made identical hard decisions on directly comparable conditions. | Do not extend this recipe without changing its physical difficulty coverage. |
| Weak-user audit | Seven of eight remaining core errors also failed with memory reset; training had no pair gap above 5.47 dB, while four of five schedule failures exceeded 9 dB. | Target invariant near-far and weak-user difficulty, not longer memory or more reset tuning. |
| V3 training correction | Added anonymous 9--15 dB episode spans, unique event targets, teacher-path warm start, and weak-user priority. | Run only the paired 500-update screen before longer training. |
| V3 paired screen | Reached 9.93% core TBLER versus 9.38% for K=8; only five differential cells remained. | Audit exact cells before longer training. |
| Spatial mismatch audit | Four of five differential cells were current-TB failures; three combined a weak UE, 9 dB deficit, and overlap above 0.43. | Correct spatial coverage and loss dilution before redesigning inference. |
| V4 training correction | Added anonymous shared-array ambiguity and a hardest-bit tail, but introduced 16 candidate failures and repaired none. | Rejected; retain V3. |
| Schedule reproducibility audit | A generator RNG change altered schedule episodes despite unchanged benchmark JSON. | Cache episode tensors and bind generator revisions for future paired comparisons. |
| V19 streamed PD-L8 | Reproduced V18 exactly while reducing mature latency below Cold K=3. | Retain the streaming implementation. |
| V19 dual-exchange CJLR | Repaired two and regressed two cells overall; repaired none and regressed one in the score window. Its p95 also narrowly exceeded Cold K=3. | Reject the second symmetric exchange. |
| V20 target rescue | It matched V18 at 9.869% TBLER, repaired no blocks, and its truth-aware oracle repaired only one of five scored gaps. | Rejected; see the [compact audit](v20_rejection_audit.json). |
| TDL-D/E diagnostic | Neural decoding saturated despite a solvable physical channel. | Replaced by geometry-aware tests. |

The repository now keeps only the accepted v18 implementation, the core
training and channel generators, and the common benchmark needed to compare
future designs. Rejected raw outputs were removed after their conclusions were
recorded here.
