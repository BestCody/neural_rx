#!/usr/bin/env python3
"""Deprecated shared-pooler calibration utility.

This utility calibrated one Attention/CNN pooler independently of PCA capacity,
which belongs to the superseded shared-pooler protocol. The approved exhaustive
benchmark tunes the learned pooler separately for each d_mem.

Capacity-specific calibration is handled internally by
``train_temporal_ue_memory_v7_pca_capacity_tuned_pooler.py``.
"""


def main():
    raise SystemExit(
        "DEPRECATED: capacity-independent pooler calibration is disabled. "
        "Use train_temporal_ue_memory_v7_pca_capacity_tuned_pooler.py."
    )


if __name__ == "__main__":
    main()
