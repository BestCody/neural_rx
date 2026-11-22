"""The early-retransmission decisions of the paper's controller per test UE-TB, for harq.py (CPU only, no GPU).

The controller of experiments/20261118_network_decisions (CRC history and receiver evidence, with the false-alarm
budget of the rule "previous TB failed", fitted on sequences 0-15 and tested on 16-31 at the operating points) is
fitted again exactly as there, and the check at which it first flags each test UE-TB is mapped back to its sequence,
TB and receiver position. Writes inputs/harq_decisions.npz:
  operating_points_db  [condition]
  first_check          [condition, test sequence, TB, position]: -2 not scored (TB < 16 or inactive), -1 not flagged,
                       0, 1, 2 flagged at check 1, 2, 3
  failed               [condition, test sequence, TB, position]: the proposed NRX's error at the operating point
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
DECISIONS = HERE.parents[0]/"20261118_network_decisions"
sys.path.insert(0, str(DECISIONS))
import controller as c  # noqa: E402


def main():
    directory = DECISIONS/"results/evidence_pc"
    points = c.operating_points(directory)
    fit_data, test_data = c.load(directory, c.FIT, points), c.load(directory, c.TEST, points)
    budget = float(((fit_data["previous_failed"] == 1) & (fit_data["failed"] == 0)).sum()
                   / (fit_data["failed"] == 0).sum())
    first = c.step_by_step(fit_data, test_data, c.HISTORY_STEPS, budget)
    caught = float((first[test_data["failed"] == 1] >= 0).mean())
    stored = json.loads((DECISIONS/"results/controller.json").read_text())
    expected = stored["step_by_step"]["CRC history and receiver evidence, budget of the rule"]["caught"]
    if abs(caught-expected) > 1e-12:
        raise RuntimeError(f"refitted controller catches {caught}, the stored one {expected}")

    tests = list(c.TEST)
    checks = np.full((len(c.CONDITIONS), len(tests), 64, 2), -2, np.int8)
    failed = np.zeros(checks.shape, bool)
    row = 0
    for k, condition in enumerate(c.CONDITIONS):
        for j, index in enumerate(tests):
            with np.load(directory/condition/f"episode_{index:05d}.npz", allow_pickle=False) as data:
                active = data["active"][c.START:]
                point = int(np.flatnonzero(data["snr_db"] == points[condition])[0])
                errors = data["errors"][point]
            tb, position = np.nonzero(active)                  # the row order of controller.load
            rows = slice(row, row+len(tb))
            if not (np.array_equal(test_data["condition"][rows], np.full(len(tb), k))
                    and np.array_equal(test_data["episode"][rows], np.full(len(tb), index))
                    and np.array_equal(test_data["failed"][rows], errors[c.START:][tb, position].astype(float))):
                raise RuntimeError(f"rows of {condition} sequence {index} do not match")
            checks[k, j, tb+c.START, position] = first[rows]
            failed[k, j] = errors.astype(bool)
            row += len(tb)
    if row != len(first):
        raise RuntimeError("not every test UE-TB was mapped")
    out = HERE/"inputs/harq_decisions.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, operating_points_db=np.asarray([points[k] for k in c.CONDITIONS]), first_check=checks, failed=failed,
             conditions=np.asarray(c.CONDITIONS), test_sequences=np.asarray(tests), budget=np.asarray(budget))
    scored = checks >= -1
    print(json.dumps({"operating_points_db": points, "budget": budget, "caught": caught,
                      "errors": int((failed & scored).sum()),
                      "caught_at": [int(((checks == s) & failed).sum()) for s in range(3)],
                      "false_alarms": int(((checks >= 0) & ~failed & scored).sum()),
                      "scored_ue_tbs": int(scored.sum())}))


if __name__ == "__main__":
    main()
