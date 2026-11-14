"""Early-retransmission controller on the proposed NRX's logged evidence (results/evidence, from log_evidence.py).

At each check, a logistic model turns the evidence available so far into a failure score. If the score exceeds the
check's threshold, the network requests the retransmission at once, before decoding has finished; otherwise the TB
is handled as today (CRC after decoding, then HARQ). A miss therefore costs nothing beyond today's behaviour, and a
false alarm costs one unneeded retransmission.

Data: the operating point of each condition (the sweep SNR closest to the proposed NRX's 10% TBLER), TBs 17-64,
active UEs. The models and thresholds are fitted on episodes 0-15 of all three conditions and evaluated on episodes
16-31. Each threshold flags 1% of the successful TBs of the fitting episodes.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SWEEP = HERE.parent/"20261104_tbler_sweep/results"
CONDITIONS = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
START = 16
FIT, TEST = range(0, 16), range(16, 32)
FALSE_ALARM = .01
ANTENNAS = 4

# Inputs available at each point, each set extending the previous one where noted.
RECENT = ["noise", "has_memory", "recent_strength", "entries", "newest_age"]
ANY_RECEIVER = ["noise", "current_strength"]
CHECK_1 = ANY_RECEIVER+RECENT[1:]+["estimate_strength", "estimate_error"]
CHECK_2 = CHECK_1+["llr_bit_error"]
CHECK_3 = CHECK_2+["unsatisfied_checks", "failed_code_blocks", "posterior_bit_error"]
INPUT_SETS = {
    "recent observations only (known before the UE transmits)": RECENT,
    "check 1, any receiver: noise and current channel strength": ANY_RECEIVER,
    "check 1, + memory state and recent channel strength": ANY_RECEIVER+RECENT[1:],
    "check 1, + filtered estimate and its error covariance": CHECK_1,
    "check 2, + LLRs after detection": CHECK_2,
    "check 3, + parity checks after 8 BP iterations": CHECK_3,
}
STEPS = {"check 1": CHECK_1, "check 2": CHECK_2, "check 3": CHECK_3}
# The CRC history the network has anyway: whether the UE's previous TB failed, or whether it had none.
HISTORY = ["previous_failed_flag", "no_previous"]
HISTORY_STEPS = {step: HISTORY+inputs for step, inputs in STEPS.items()}


def required_snr(snr, failures, blocks, target=.1):
    """SNR at the target TBLER, as in the paper: decreasing isotonic fit, then log-TBLER interpolation."""
    from neural_rx.benchmark.plan import decreasing_isotonic, snr_at_target
    rates = np.asarray(failures, float)/blocks
    return snr_at_target(np.asarray(snr, float), decreasing_isotonic(rates, np.full(len(rates), float(blocks))), target)


def logged_curve(directory, condition, episodes=range(32), key="errors"):
    """SNR grid, failures per SNR point, and blocks of the logged episodes (TBs 17-64, active UEs)."""
    snr, failures, blocks = None, 0, 0
    for index in episodes:
        with np.load(Path(directory)/condition/f"episode_{index:05d}.npz", allow_pickle=False) as data:
            active = data["active"][START:]
            failures = failures+(data[key][:, START:] & active[None]).sum((1, 2))
            blocks += int(active.sum())
            snr = data["snr_db"]
    return snr, failures, blocks


def operating_points(directory=None):
    """The SNR point closest to the proposed NRX's 10% TBLER. Without power control, these are the sweep's points
    behind the paper (as in experiments/20261106_crc_fallback); with it, they come from the logged episodes."""
    points = {}
    if directory is None or Path(directory).name == "evidence":
        summary = json.loads((SWEEP/"summary.json").read_text())
        for condition in CONDITIONS:
            entry = summary[condition]["candidate_persistent"]
            snr = np.asarray(entry["snr_db"])
            points[condition] = float(snr[np.argmin(np.abs(snr-entry["snr_at_10pct_db"]))])
        return points
    for condition in CONDITIONS:
        snr, failures, blocks = logged_curve(directory, condition)
        points[condition] = float(snr[np.argmin(np.abs(snr-required_snr(snr, failures, blocks)))])
    return points


def logit(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.log(p/(1-p))


def load(directory, episodes, points=None):
    """Inputs, failure flags, condition and episode of every active UE-TB at the operating points."""
    points, rows = points or operating_points(directory), []
    for condition in CONDITIONS:
        for index in episodes:
            path = Path(directory)/condition/f"episode_{index:05d}.npz"
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as data:
                point = int(np.flatnonzero(data["snr_db"] == points[condition])[0])
                names = list(data["features"])
                evidence = data["evidence"][point, START:]
                active = data["active"][START:]
                failed = data["errors"][point, START:]
                noise = np.broadcast_to(data["noise_variance"][point, START:, None], active.shape)
            value = {name: evidence[..., names.index(name)][active] for name in names}
            some = value["memory_entries"] > 0
            # Whether the same UE's previous scheduled TB failed (-1 if it had none in this sequence). The network
            # knows this from the CRC before the UE transmits again.
            with np.load(path, allow_pickle=False) as data:
                all_active, ids, all_failed = data["active"], data["ue_ids"], data["errors"][point]
            previous = np.full(active.shape, -1)
            last = {}
            for tb in range(all_active.shape[0]):
                for position in np.flatnonzero(all_active[tb]):
                    ue = int(ids[tb, position])
                    if tb >= START:
                        previous[tb-START, position] = last.get(ue, -1)
                    last[ue] = int(all_failed[tb, position])
            rows.append({
                "noise": 10*np.log10(noise[active]),
                "current_strength": 10*np.log10(value["ls_power"]/(ANTENNAS*value["ls_noise"])),
                "has_memory": some.astype(float),
                "recent_strength": np.where(some, 10*np.log10(np.maximum(value["memory_newest_power"], 1e-12)
                                                              /np.maximum(ANTENNAS*value["memory_newest_noise"], 1e-12)), 0.),
                "entries": value["memory_entries"],
                "newest_age": value["memory_newest_age"],
                "estimate_strength": 10*np.log10(value["estimate_power"]/(ANTENNAS*value["ls_noise"])),
                "estimate_error": 10*np.log10(value["estimate_error"]/value["estimate_power"]),
                "llr_bit_error": logit(value["llr_bit_error"]),
                "unsatisfied_checks": np.log1p(value["unsatisfied_checks"]),
                "failed_code_blocks": value["failed_code_blocks"],
                "posterior_bit_error": logit(value["posterior_bit_error"]),
                "previous_failed": previous[active],
                "previous_failed_flag": (previous[active] == 1).astype(float),
                "no_previous": (previous[active] == -1).astype(float),
                "failed": failed[active].astype(float),
                "condition": np.full(int(active.sum()), CONDITIONS.index(condition)),
                "episode": np.full(int(active.sum()), index)})
    return {name: np.concatenate([row[name] for row in rows]) for name in rows[0]}


def fit(x, y, ridge=1.):
    """Logistic regression on standardized inputs by Newton iterations, with a small L2 penalty."""
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1.
    z = np.c_[np.ones(len(x)), (x-mean)/scale]
    w = np.zeros(z.shape[1])
    penalty = ridge*np.eye(z.shape[1])
    penalty[0, 0] = 0.
    for _ in range(100):
        p = 1/(1+np.exp(-z@w))
        step = np.linalg.solve((z*(p*(1-p))[:, None]).T@z+penalty, z.T@(p-y)+penalty@w)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return {"mean": mean, "scale": scale, "weights": w}


def score(model, x):
    return np.c_[np.ones(len(x)), (x-model["mean"])/model["scale"]]@model["weights"]


def matrix(data, inputs):
    return np.stack([data[name] for name in inputs], 1)


def threshold(scores, failed, rate):
    """Flag scores above this to flag the given share of successful TBs."""
    return float(np.quantile(scores[failed == 0], 1-rate))


def rates(flagged, failed):
    return {"caught": float(flagged[failed == 1].mean()), "false_alarm": float(flagged[failed == 0].mean())}


def single(fit_data, test_data, inputs, budget):
    """Flags of one check whose threshold flags the budget's share of successful fitting TBs."""
    model = fit(matrix(fit_data, inputs), fit_data["failed"])
    limit = threshold(score(model, matrix(fit_data, inputs)), fit_data["failed"], budget)
    return score(model, matrix(test_data, inputs)) > limit


def step_by_step(fit_data, test_data, steps, budget):
    """The first check whose score exceeds its threshold, per test UE-TB (-1 if none). All checks flag the same share
    of successful fitting TBs, chosen so that the budget's share of them is flagged by any check."""
    models = {step: fit(matrix(fit_data, inputs), fit_data["failed"]) for step, inputs in steps.items()}
    fit_scores = {step: score(models[step], matrix(fit_data, steps[step])) for step in steps}
    test_scores = {step: score(models[step], matrix(test_data, steps[step])) for step in steps}
    low, high = 0., budget
    for _ in range(60):
        share = (low+high)/2
        limits = {step: threshold(fit_scores[step], fit_data["failed"], share) for step in steps}
        union = np.any([fit_scores[step] > limits[step] for step in steps], 0)
        if union[fit_data["failed"] == 0].mean() > budget:
            high = share
        else:
            low = share
    limits = {step: threshold(fit_scores[step], fit_data["failed"], low) for step in steps}
    first = np.full(len(test_data["failed"]), -1)
    for position, step in reversed(list(enumerate(steps))):
        first[test_scores[step] > limits[step]] = position
    return first


def summary(flagged, data):
    failed = data["failed"] == 1
    first_failure = failed & (data["previous_failed"] != 1)
    return {**rates(flagged, data["failed"]),
            "caught_first_failures": float(flagged[first_failure].mean()) if first_failure.any() else None,
            "by_condition": {c: rates(flagged[data["condition"] == i], data["failed"][data["condition"] == i])
                             for i, c in enumerate(CONDITIONS)}}


def main(directory=HERE/"results/evidence", output="controller.json"):
    points = operating_points(directory)
    fit_data, test_data = load(directory, FIT, points), load(directory, TEST, points)
    everything = load(directory, range(32), points)
    failed_all = everything["failed"] == 1
    # The rule "request early if the UE's previous TB failed" and its false-alarm share on the fitting episodes,
    # which is the budget of the controllers that also use the CRC history.
    rule_budget = float(((fit_data["previous_failed"] == 1) & (fit_data["failed"] == 0)).sum()
                        /(fit_data["failed"] == 0).sum())
    report = {
        "operating_points_db": points, "false_alarm_target": FALSE_ALARM, "rule_budget": rule_budget,
        "fit": {"ue_tbs": int(len(fit_data["failed"])), "failures": int(fit_data["failed"].sum())},
        "test": {"ue_tbs": int(len(test_data["failed"])), "failures": int(test_data["failed"].sum()),
                 "first_failures": int(((test_data["failed"] == 1) & (test_data["previous_failed"] != 1)).sum())},
        "all_episodes": {"ue_tbs": int(len(failed_all)), "failures": int(failed_all.sum()),
                         "failures_after_failure": float((everything["previous_failed"][failed_all] == 1).mean()),
                         "first_failures": int((failed_all & (everything["previous_failed"] != 1)).sum()),
                         "by_condition": {c: float((everything["previous_failed"][failed_all & (everything["condition"] == i)]
                                                    == 1).mean()) for i, c in enumerate(CONDITIONS)}},
        "rule": summary(test_data["previous_failed"] == 1, test_data),
        "single_check": {name: {"inputs": inputs, **summary(single(fit_data, test_data, inputs, FALSE_ALARM), test_data)}
                         for name, inputs in INPUT_SETS.items()},
        "with_history": {f"budget {name}": {step: summary(single(fit_data, test_data, inputs, budget), test_data)
                                             for step, inputs in HISTORY_STEPS.items()}
                         for name, budget in (("1%", FALSE_ALARM), ("of the rule", rule_budget))},
        "step_by_step": {}}
    for name, steps, budget in (("receiver evidence, 1%", STEPS, FALSE_ALARM),
                                ("CRC history and receiver evidence, budget of the rule", HISTORY_STEPS, rule_budget)):
        first = step_by_step(fit_data, test_data, steps, budget)
        failed = test_data["failed"] == 1
        report["step_by_step"][name] = {
            **summary(first >= 0, test_data),
            "caught_at": {step: float((first[failed] == position).mean()) for position, step in enumerate(steps)}}
    out = HERE/"results"/output
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    def line(name, value):
        first = value["caught_first_failures"]
        print(f"  {name:62s} caught {value['caught']*100:5.1f}%  false alarms {value['false_alarm']*100:4.2f}%"
              f"  first failures {'-' if first is None else f'{first*100:5.1f}%'}")
    a = report["all_episodes"]
    print(f"operating points {points}; all episodes: {a['failures']} failures, {a['failures_after_failure']*100:.1f}% after a "
          f"failure of the same UE, {a['first_failures']} first failures")
    print(f"test: {report['test']}; budget of the rule {rule_budget*100:.2f}%")
    line("rule: previous TB failed", report["rule"])
    print("receiver evidence only, 1% budget:")
    for name, value in report["single_check"].items():
        line(name, value)
    for budget, checks in report["with_history"].items():
        print(f"CRC history and receiver evidence, {budget}:")
        for step, value in checks.items():
            line(step, value)
    for name, value in report["step_by_step"].items():
        line(f"step by step, {name}", value)
        print("      caught at " + "  ".join(f"{k} {v*100:.1f}%" for k, v in value["caught_at"].items()))
    return report


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--power-control"]:
        main(HERE/"results/evidence_pc", "controller_pc.json")
    else:
        main()
