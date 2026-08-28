# Temporal NRX C-RNTI Runtime Integration

This document defines the live identity/memory boundary for the temporal UE-memory receiver implemented in this repository.

## Ownership rule

The neural network does **not** infer UE identity from RF samples. The gNB owns UE identity. The stable temporal-memory key is the C-RNTI already associated with each scheduled PUSCH.

The receiver-side Python bridge is:

```text
scripts/temporal_nrx_runtime.py
```

It composes:

```text
CRNTIMemoryAdapter
    -> RuntimeUEMemoryManager
    -> TemporalUEMemoryCGNN inference
```

and performs one transaction:

```text
received_grid + ls_estimate + ordered C-RNTIs + slot
        |
        v
CRNTIMemoryAdapter.lookup(crntis, slot)
        |
        +--> prev_memory [U, d_mem]
        +--> gap_slots   [U]
        +--> valid       [U]
        |
        v
TemporalUEMemoryCGNN, deployment K=1 or K=2
        |
        +--> decoded LLRs/result
        +--> next_memory [U, d_mem]
        |
        v
CRNTIMemoryAdapter.process_result(immutable_lookup, next_memory, slot)
```

Memory is committed only after inference succeeds and the returned memory shape/value checks pass. The immutable C-RNTI order returned by `lookup` is used for commit so a receiver-position reorder cannot write state under the wrong UE.

## srsRAN PUSCH context

Patch:

```text
patches/srsran-23.10.1/0001-temporal-nrx-crnti-hook.patch
```

Target:

```text
srsRAN 23.10.1
bcf941b34faba5e14b4d614772fa45068721afdf
```

At `uplink_processor_impl::process_pusch()` the patch constructs:

```cpp
temporal_nrx_pusch_context_scope temporal_nrx_scope({
    pdu.pdu.rnti,
    pdu.pdu.slot
});

pusch_proc->process(...);
```

The scope makes the exact C-RNTI and slot available for the complete synchronous `pusch_proc->process(...)` call on that processing thread.

A downstream in-process neural receiver can query:

```cpp
const auto* context = get_current_temporal_nrx_pusch_context();
```

and obtain:

```text
context->crnti
context->slot
```

The original callback registration API remains available. The scoped accessor is preferred for synchronous receiver inference because it avoids a mutable process-wide "last C-RNTI" and preserves the identity for the duration of the actual PUSCH call.

## Runtime call contract

The live neural invocation should receive C-RNTIs in exactly the same UE/receiver-position order as the signal tensors:

```text
TemporalNRXRuntime.process(
    received_grid,
    ls_estimate,
    crntis,
    slot_index,
    active,
)
```

Internally the runtime supplies the trained model with:

```text
prev_memory  [1, U, d_mem]
memory_gap   [1, U]
memory_valid [1, U]
```

and commits:

```text
next_memory [U, d_mem]
```

under the same immutable C-RNTI keys after inference.

`TensorFlowTemporalInference` in `temporal_nrx_runtime.py` adapts the current `TemporalUEMemoryCGNN` preprocessing, K-step model call, and LLR demapping to this runtime contract. The current model returns decoded LLRs, a channel estimate, next memory, compression auxiliary loss, and reconstruction error.

## Concurrency policy

`TemporalNRXRuntime` currently serializes `lookup -> inference -> commit` with a runtime lock. This is a correctness-first policy preventing overlapping calls from updating one UE out of order. If the final production receiver supports parallel PUSCH inference, this can later be narrowed to per-UE/slot ordering.

The srsRAN context itself is thread-local, so parallel PUSCH processing threads do not overwrite each other's current C-RNTI/slot.

## Lifecycle hooks

Use these state rules:

```text
brief scheduling absence
    -> keep memory

return before expiry
    -> restore same C-RNTI memory and pass scheduling gap

absence longer than expiry_slots
    -> expire and zero row

UE release
    -> release(crnti)

handover
    -> handover(crnti)

RRC re-establishment / identity replacement
    -> reestablishment(old_crnti)
```

Freed rows are zeroed by `RuntimeUEMemoryManager` before reuse.

## Validation

The repository validates three layers independently:

1. C-RNTI identity routing and lifecycle in `test_crnti_memory_adapter.py`.
2. Full `lookup -> inference -> commit` transaction semantics in `test_temporal_nrx_runtime.py`.
3. Actual TensorFlow K=2 `TemporalUEMemoryCGNN` inference through the runtime in `test_temporal_nrx_runtime_tensorflow.py`.

The srsRAN workflow applies the patch to the exact deployed source revision, verifies the scoped C-RNTI/slot contract, and builds the complete `gnb` target.

## Remaining production boundary

The target stock srsRAN source tree does not itself contain the NVIDIA TensorFlow/TensorRT neural receiver implementation or an existing native signal adapter that converts the live PUSCH processing buffers into the `received_grid` and `ls_estimate` tensors expected by `TemporalUEMemoryCGNN`.

Therefore the identity and temporal-memory runtime path is now implemented on both sides, but a production RF deployment still needs the concrete neural-receiver signal invocation to call `TemporalNRXRuntime.process(...)` (or an equivalent native TensorRT implementation) with the live grid/LS tensors and C-RNTIs read from `get_current_temporal_nrx_pusch_context()`.

Do **not** perform TensorFlow/Python work inside the lightweight C++ notification callback. If a native neural receiver is invoked synchronously inside `pusch_proc->process(...)`, read the scoped context there and pass the C-RNTI/slot into that receiver invocation.

Do not create a second UE-identity inference mechanism inside the neural model.
