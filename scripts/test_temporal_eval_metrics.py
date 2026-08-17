#!/usr/bin/env python3
"""Unit tests for temporal evaluation statistics and interpolation."""

import json

from temporal_eval_metrics import log_bler_crossing, make_snr_grid, wilson_interval


def main():
    grid = make_snr_grid(0.0, 0.8, 0.3)
    grid_ok = grid == [0.0, 0.3, 0.6, 0.8]

    # The old evaluator dropped the zero-error point and returned no crossing.
    points = [
        {
            "snr_db": 2.0,
            "bler_tb2plus": 0.12,
            "errors_tb2plus": 24,
            "blocks_tb2plus": 200,
        },
        {
            "snr_db": 2.25,
            "bler_tb2plus": 0.0,
            "errors_tb2plus": 0,
            "blocks_tb2plus": 200,
        },
    ]
    cross = log_bler_crossing(points, target=0.1)
    zero_crossing_ok = cross is not None and 2.0 < cross < 2.25

    no_cross = log_bler_crossing(
        [
            {"snr_db": 1.0, "bler_tb2plus": 0.4, "errors_tb2plus": 40, "blocks_tb2plus": 100},
            {"snr_db": 2.0, "bler_tb2plus": 0.2, "errors_tb2plus": 20, "blocks_tb2plus": 100},
        ]
    )
    no_cross_ok = no_cross is None

    ci_zero = wilson_interval(0, 100)
    ci_full = wilson_interval(100, 100)
    wilson_ok = (
        ci_zero[0] == 0.0
        and 0.0 < ci_zero[1] < 0.1
        and 0.9 < ci_full[0] < 1.0
        and ci_full[1] == 1.0
    )

    report = {
        "snr_grid_never_overshoots": grid_ok,
        "zero_error_point_still_brackets_crossing": zero_crossing_ok,
        "unbracketed_curve_returns_none": no_cross_ok,
        "wilson_interval_endpoints": wilson_ok,
        "grid": grid,
        "zero_error_crossing": cross,
    }
    report["passed"] = bool(all(v for k, v in report.items() if k not in {"grid", "zero_error_crossing"}))
    print("TEMPORAL_EVAL_METRICS_TEST=" + json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
