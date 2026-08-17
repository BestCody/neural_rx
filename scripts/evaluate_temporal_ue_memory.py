#!/usr/bin/env python3
"""Compatibility entry point for temporal UE-memory evaluation.

The original implementation behind this filename accidentally constructed
``Parameters(training=True)`` and therefore evaluated the 4-PRB training grid.
That result is not valid research evidence. Keep the familiar filename, but
route every invocation to the corrected evaluator, which explicitly constructs
the 132-PRB ``training=False`` evaluation system.
"""

from evaluate_temporal_ue_memory_v2 import run, write


if __name__ == "__main__":
    write(run())
