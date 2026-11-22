"""Summary of harq.py, per condition and over all.

Both schemes are compared on the same TBs: those that fail at their first transmission and whose retransmissions fit
into the 64 slots of the sequence in both schemes (a TB near the end of a sequence is left out, as its retransmission
would fall after the last slot). For these TBs: how many each scheme decodes after one, two or three retransmissions,
the median time from the end of the TB's slot until its combined decoding finishes, and the median number of slots the
early scheme saves per TB. Over all caught errors: the slots by which the early request precedes the regular one for
decoding times of 0.25, 0.5 and 1 ms (only the request times change; whether a retransmission succeeds is taken from
the run with 0.5 ms). Also the extra transmissions of the false alarms. Writes results/harq_summary.json.
"""
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harq import SLOT_MS, T_DEC, T_DET, request_time, slots_after  # noqa: E402

CONDITIONS = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")


def finish_ms(record):
    """From the end of the TB's slot until the combined decoding that succeeds finishes."""
    return (record["decoded_at"]-record["tb"])*SLOT_MS+T_DET+T_DEC


def compare(regular, early, scored):
    common = sorted(key for key in regular if regular[key]["failed"] and not regular[key]["censored"]
                    and not early[key]["censored"])
    result = {"failed": sum(1 for r in regular.values() if r["failed"]), "compared": len(common),
              "left_out_at_sequence_end": sum(1 for r in regular.values() if r["failed"])-len(common)}
    for name, records in (("regular", regular), ("early", early)):
        chosen = [records[key] for key in common]
        decoded = [r for r in chosen if r["decoded_at"] is not None]
        result[name] = {"decoded_after_retransmissions": [sum(1 for r in decoded if len(r["slots"]) == n)
                                                          for n in (1, 2, 3)],
                        "not_decoded": len(chosen)-len(decoded),
                        "median_ms_to_decoding": statistics.median(map(finish_ms, decoded)) if decoded else None}
    both = [key for key in common if regular[key]["decoded_at"] is not None and early[key]["decoded_at"] is not None]
    result["median_slots_saved"] = (statistics.median(regular[k]["decoded_at"]-early[k]["decoded_at"] for k in both)
                                    if both else None)
    result["slots_saved_share"] = (sum(1 for k in both if early[k]["decoded_at"] < regular[k]["decoded_at"])/len(both)
                                   if both else None)
    result["early_extra_transmissions_per_100_tbs"] = (100*sum(1 for r in early.values() if not r["failed"])/scored
                                                       if scored else None)
    return result


def main():
    regular, early, scored, mismatches, report = {}, {}, {}, 0, {}
    for condition in CONDITIONS:
        regular[condition], early[condition], scored[condition] = {}, {}, 0
        for path in sorted((HERE/"results/harq"/condition).glob("episode_*.json")):
            data = json.loads(path.read_text())
            for scheme, target in (("regular", regular[condition]), ("early", early[condition])):
                for r in data["schemes"][scheme]["records"]:
                    target[(data["episode"], r["tb"], r["position"])] = r
            scored[condition] += data["scored_ue_tbs"]
            mismatches += data["mismatches"]
        report[condition] = compare(regular[condition], early[condition], scored[condition])
    merged = {s: {(c,)+k: r for c in CONDITIONS for k, r in table[c].items()} for s, table in
              (("regular", regular), ("early", early))}
    report["all"] = compare(merged["regular"], merged["early"], sum(scored.values()))
    report["all"]["decision_mismatches"] = mismatches
    caught = [r for r in merged["early"].values() if r["failed"] and r["check"] >= 0]
    report["all"]["request_slots_saved_per_caught_error"] = {
        str(t_dec): statistics.median(slots_after(request_time(-1, r["tb"], r["previous_slot"], t_dec))
                                      - slots_after(request_time(r["check"], r["tb"], r["previous_slot"], t_dec))
                                      for r in caught) if caught else None
        for t_dec in (.25, .5, 1.)}
    report["timeline"] = {"slot_ms": SLOT_MS, "detection_ms": T_DET, "decoding_ms": T_DEC}
    (HERE/"results/harq_summary.json").write_text(json.dumps(report, indent=1)+"\n")
    for name in (*CONDITIONS, "all"):
        v = report[name]
        if not v["failed"] and v["early_extra_transmissions_per_100_tbs"] is None:
            print(f"{name:27s} no results yet")
            continue
        print(f"{name:27s} failed {v['failed']:4d}, compared {v['compared']:4d} | regular decoded "
              f"{v['regular']['decoded_after_retransmissions']} not {v['regular']['not_decoded']}, median "
              f"{v['regular']['median_ms_to_decoding']} ms | early decoded {v['early']['decoded_after_retransmissions']} "
              f"not {v['early']['not_decoded']}, median {v['early']['median_ms_to_decoding']} ms | median saved "
              f"{v['median_slots_saved']} slots | extra {v['early_extra_transmissions_per_100_tbs']:.2f}/100 TBs")
    print("request slots saved per caught error:", report["all"]["request_slots_saved_per_caught_error"])


if __name__ == "__main__":
    main()
