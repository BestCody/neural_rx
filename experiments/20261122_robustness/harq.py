"""HARQ retransmissions with Chase combining for the paper's early-retransmission controller (proposed NRX, closed-loop
power control, the 16 test sequences per condition at the operating points of experiments/20261118_network_decisions).

Each sequence is decoded once as in the evaluation, keeping the NRX's LLRs of every coded bit. A TB is retransmitted
  regular  after its CRC fails;
  early    at the first check at which the controller flags it (harq_inputs.py), and after the CRC fails if it does not.
A retransmission repeats the same coded bits (redundancy version 0, which 5G NR allows; incremental redundancy would
gain more). It is rendered into a later slot of the same UE, the first slot at or after the target in which the UE is
scheduled: the sequence is generated again with the same channels, noise and schedule, with the UE's payload in that
slot replaced by the TB's payload. The memory is unchanged, as it holds pilots only. The BS adds the LLRs of all copies
and decodes their sum. A TB that still fails is retransmitted after the CRC of the combined decoding fails, in both
schemes, up to three retransmissions. A false alarm costs one transmission, which is counted but not rendered.

Timeline (30 kHz, back-to-back uplink slots of 0.5 ms; times from the end of the TB's slot): detection takes
T_DET = 0.90 ms (the measured 0.79 ms of the NRX and 0.11 ms of the channel filter), decoding T_DEC (0.5 ms), and the
first 8 of the 20 BP iterations 8/20 of it. A request at time T can use the UE's slot that starts at least one slot
(grant and UE preparation) after T, i.e. k = ceil(T/0.5)+2 slots after the TB's slot:
  CRC        T_DET+T_DEC
  check 1    the CRC of the UE's previous TB (T_DET+T_DEC minus the slots since it), at least the LS estimate and
             filter of the current slot (0.11 ms)
  check 2    T_DET
  check 3    T_DET+8/20*T_DEC
Writes results/harq/<condition>/episode_NNNNN.json.

    harq.py --condition natural_umi --first-episode 16 --last-episode 32
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SLOT_MS, T_DET, T_FILTER, T_DEC = .5, .90, .11, .5
FIRST_BP, ALL_BP = 8, 20
MAX_RETRANSMISSIONS = 3


def slots_after(t_ms):
    """Slots from the TB's slot to the first slot a request at t_ms can use."""
    return math.ceil(t_ms/SLOT_MS-1e-9)+2


def request_time(check, slot, previous_slot, t_dec=T_DEC):
    """Time of a request at a check (0, 1, 2; -1 for the CRC), in ms after the end of the TB's slot."""
    crc = T_DET+t_dec
    if check == -1:
        return crc
    if check == 0:
        return max(T_FILTER, crc-SLOT_MS*(slot-previous_slot)) if previous_slot is not None else T_FILTER
    return T_DET if check == 1 else T_DET+FIRST_BP/ALL_BP*t_dec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--first-episode", type=int, default=16)
    parser.add_argument("--last-episode", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=HERE/"results")
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf
    from neural_rx.backbone import E2E_Model, Parameters
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.candidate import checked_decoder_tensors
    from neural_rx.benchmark.frozen import make_manifest, protocol_and_plan, resolve_manifest_artifacts
    from neural_rx.benchmark.runtime import validate_episode
    from neural_rx.benchmark.scoring import payload_error_flags
    from neural_rx.receiver.benchmark_adapter import PilotHistoryBenchmarkAdapter
    sys.path.insert(0, str(HERE))
    from banks import SETTINGS_PC, ClosedLoopPowerControlGenerator, PowerControlBank

    class HarqGenerator(ClosedLoopPowerControlGenerator):
        """Reuses the sequence's channels and replaces payloads of chosen (slot, position) pairs."""

        def sample_pool_channel(self, batch_size, seq_len, channel_regime=None):
            if self.cached_channel is None:
                self.cached_channel = super().sample_pool_channel(batch_size, seq_len, channel_regime)
                self.cached_gain = self.last_power_gain_db
            self.last_power_gain_db = self.cached_gain
            return self.cached_channel

        def _sample_bits(self, shape, seed, tb):
            bits = super()._sample_bits(shape, seed, tb)
            swaps = self.substitutions.get(int(tb))
            if not swaps:
                return bits
            values = bits.numpy()
            for position, payload in swaps.items():
                values[:, position] = payload
            return tf.constant(values, dtype=bits.dtype)

    class HarqBank(PowerControlBank):
        def _make_generator(self, job, episode_index):
            generator = super()._make_generator(job, episode_index)
            generator.__class__ = HarqGenerator                   # adds methods only
            generator.cached_channel, generator.substitutions = None, {}
            return generator

        def _generator(self, job, episode_index):
            generator = super()._generator(job, episode_index)
            if getattr(generator, "cached_index", None) != int(episode_index):
                generator.cached_channel, generator.cached_index = None, int(episode_index)
            return generator

    class LLRAdapter(PilotHistoryBenchmarkAdapter):
        """PilotHistoryBenchmarkAdapter._build_execution, also returning the LLRs the decoder receives."""
        last_llr = None

        def _build_execution(self):
            buffer = self._buffer

            @tf.function(reduce_retracing=True, jit_compile=True)
            def prepare(channel, noise, ids, active, step, head, count, tables):
                history = buffer.gather(tables, head, count, ids, active, step)
                observation = self.model.observe(channel, noise)
                preparation = self.model.prepare([None, None, channel, tf.cast(active, tf.float32)],
                                                 observation, history, self.layout)
                return observation, preparation

            @tf.function(reduce_retracing=True, jit_compile=False)
            def execute(y, channel, noise, control, active, head, count):
                ids, step = tf.cast(control[:, :-1], tf.int32), control[0, -1]
                tables = tuple(v.read_value() for v in (buffer.observations, buffer.timestamps, buffer.ports))
                observation, preparation = prepare(channel, noise, ids, active, step, head, count, tables)
                inputs = self._prepare_inputs(self.receiver, y, channel, tf.cast(active, tf.float32))
                grid, _ = self.model.detect(inputs, noise, None, observation, preparation=preparation)
                pending, finite = buffer.prepare_stage(observation, head, count, ids, active)
                llr = self.grid_map.grid_to_coded(grid)
                output = self.decoder(llr)
                logits, hard, crc = checked_decoder_tensors(output)
                with tf.control_dependencies([logits, hard, crc]):
                    next_head, next_count, complete = buffer.commit(pending, finite, step)
                with tf.control_dependencies([complete]):
                    return (tf.identity(logits), tf.identity(hard), tf.identity(crc), next_head, next_count, complete,
                            tf.identity(llr))

            def run(*arguments):
                *outputs, llr = execute(*arguments)
                self.last_llr = llr.numpy()
                return tuple(outputs)
            return run

        def decode_llr(self, llr):
            _, hard, crc = checked_decoder_tensors(self.decoder(tf.constant(llr, tf.float32)))
            return np.asarray(hard.numpy()), np.asarray(crc.numpy())

    decisions = np.load(HERE/"inputs/harq_decisions.npz")
    conditions = [str(c) for c in decisions["conditions"]]
    tests = [int(t) for t in decisions["test_sequences"]]
    k = conditions.index(args.condition)
    point = float(decisions["operating_points_db"][k])

    sweep.configure_runtime()
    anchors = json.loads(sweep.ANCHORS.read_text())["conditions"]
    protocol, plan = protocol_and_plan("confirmation", sweep.PRBS, 32, anchors, protocol_path=sweep.PROTOCOL)
    parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
    bank = HarqBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf, dict(SETTINGS_PC))
    job = dict(sweep.sweep_job(plan, anchors, args.condition), snr_db=[point])
    manifest, mapping, seed = make_manifest(protocol, "candidate", sweep.CHECKPOINT)
    receiver = LLRAdapter(manifest=manifest, method="candidate_persistent", training_seed=seed,
                          artifacts=resolve_manifest_artifacts(manifest, mapping), config_name=sweep.CONFIG, tf_module=tf)

    def decode(episode):
        tbs, users = episode.active.shape
        errors, llrs = np.zeros((tbs, users), bool), []
        receiver.start_episode_batch(1)
        for tb in range(tbs):
            result = receiver.decode_tb_batch([episode.receiver_inputs[0][tb]])
            llrs.append(receiver.last_llr[0])
            errors[tb] = payload_error_flags(np.asarray(result["hard_bits"]), np.asarray(result["crc_status"]),
                                             episode.bits[tb][None], episode.active[tb][None])[0][0]
        return errors, np.stack(llrs)

    def combined_error(llr, truth):
        both = np.zeros((1, 2, llr.shape[-1]), np.float32)
        both[0, 0] = llr
        hard, crc = receiver.decode_llr(both)
        return bool(payload_error_flags(hard, crc, truth[None, None].repeat(2, 1), np.asarray([[True, False]]))[0][0, 0])

    def generate(substitutions):
        generator = bank._generator(job, index)
        generator.substitutions = substitutions
        try:
            return validate_episode(bank.generate(job, index), protocol, job)
        finally:
            generator.substitutions = {}

    output = args.output_dir/"harq"/args.condition
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.first_episode, args.last_episode):
        path = output/f"episode_{index:05d}.json"
        if path.is_file():
            continue
        started = time.perf_counter()
        j = tests.index(index)
        checks, stored_failed = decisions["first_check"][k, j], decisions["failed"][k, j]
        original = generate({})
        errors, llr = decode(original)
        scored = checks >= -1
        mismatches = int((scored & (errors != stored_failed)).sum())
        active, ids, tbs = original.active, original.ue_ids, original.active.shape[0]

        def previous_slot(tb, position):
            ue = ids[tb, position]
            earlier = [t for t in range(tb) if np.any(active[t] & (ids[t] == ue))]
            return earlier[-1] if earlier else None

        report = {"condition": args.condition, "episode": index, "snr_db": point, "mismatches": mismatches,
                  "scored_ue_tbs": int(scored.sum()), "errors": int((scored & errors).sum()), "schemes": {}}
        for scheme in ("regular", "early"):
            records = []
            for tb, position in zip(*np.nonzero(scored)):
                flagged = scheme == "early" and checks[tb, position] >= 0
                if not (errors[tb, position] or flagged):
                    continue
                check = int(checks[tb, position]) if flagged else -1
                previous = previous_slot(tb, position)
                wait = slots_after(request_time(check, tb, previous))
                records.append({"tb": int(tb), "position": int(position), "ue": int(ids[tb, position]),
                                "previous_slot": previous, "first_check": int(checks[tb, position]),
                                "failed": bool(errors[tb, position]), "check": check, "target": int(tb)+wait,
                                "slots": [], "decoded_at": None, "censored": False})
            false_alarms = sum(1 for r in records if not r["failed"])
            pending = [r for r in records if r["failed"]]
            for _ in range(MAX_RETRANSMISSIONS):
                todo = [r for r in pending if r["decoded_at"] is None and not r["censored"]]
                if not todo:
                    break
                taken, substitutions = set(), {}
                for r in sorted(todo, key=lambda r: r["target"]):
                    slot = next((t for t in range(r["target"], tbs)
                                 if np.any(active[t] & (ids[t] == r["ue"])) and (t, r["ue"]) not in taken), None)
                    if slot is None:
                        r["censored"] = True
                        continue
                    taken.add((slot, r["ue"]))
                    at = int(np.flatnonzero(active[slot] & (ids[slot] == r["ue"]))[0])
                    r["slots"].append([int(slot), at])
                    substitutions.setdefault(slot, {})[at] = original.bits[r["tb"]][r["position"]]
                if not substitutions:
                    break
                episode = generate(substitutions)
                for slot, swaps in substitutions.items():
                    for at, payload in swaps.items():
                        if not np.array_equal(episode.bits[slot][at], payload):
                            raise RuntimeError("payload of a retransmission was not replaced")
                _, retransmitted = decode(episode)
                for r in todo:
                    if r["censored"]:
                        continue
                    slot, at = r["slots"][-1]
                    r.setdefault("copies", []).append(retransmitted[slot, at])
                    total = llr[r["tb"], r["position"]]+np.sum(r["copies"], axis=0)      # Chase combining
                    if not combined_error(total, original.bits[r["tb"]][r["position"]]):
                        r["decoded_at"] = slot
                    else:
                        r["target"] = slot+slots_after(T_DET+T_DEC)
            for r in records:
                r.pop("copies", None)
            report["schemes"][scheme] = {"false_alarms": false_alarms, "records": records}
        report["seconds"] = round(time.perf_counter()-started, 1)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report))
        temporary.replace(path)
        regular, early = (report["schemes"][s]["records"] for s in ("regular", "early"))
        print(json.dumps({"condition": args.condition, "episode": index, "errors": report["errors"],
                          "mismatches": mismatches, "regular_decoded": sum(r["decoded_at"] is not None for r in regular),
                          "early_decoded": sum(r["decoded_at"] is not None for r in early if r["failed"]),
                          "seconds": report["seconds"]}), flush=True)
    if not receiver.checksums_unchanged():
        raise RuntimeError("receiver weights changed during the run")


if __name__ == "__main__":
    main()
