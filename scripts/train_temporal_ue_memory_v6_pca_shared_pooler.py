#!/usr/bin/env python3
"""Deprecated shared-pooler PCA trainer.

This trainer belongs to the superseded protocol that reuses one learned pooler
across PCA memory capacities. The approved benchmark tunes Attention/CNN for the
specific d_mem before fitting PCA.

Use ``train_temporal_ue_memory_v7_pca_capacity_tuned_pooler.py`` instead.
"""


def main():
    raise SystemExit(
        "DEPRECATED: shared learned pooling across PCA capacities is disabled. "
        "Use train_temporal_ue_memory_v7_pca_capacity_tuned_pooler.py instead."
    )


if __name__ == "__main__":
    main()
