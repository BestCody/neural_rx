#!/usr/bin/env python3
"""Deprecated shared-pooler exhaustive suite.

Do not use this entry point for research runs. It implements the superseded
protocol that reuses one learned Attention/CNN pooler across PCA memory
capacities. The approved benchmark tunes each learned pooler to its own d_mem.

Use ``run_temporal_research_suite_exhaustive_v4.py`` instead.
"""


def main():
    raise SystemExit(
        "DEPRECATED: shared-pooler PCA across capacities is not the approved "
        "benchmark. Run run_temporal_research_suite_exhaustive_v4.py instead."
    )


if __name__ == "__main__":
    main()
