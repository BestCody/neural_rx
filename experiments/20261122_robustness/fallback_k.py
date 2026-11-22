"""When the receiver falls back from its memory: the number K of stored TBs the channel filter uses, per TB, UE and DMRS
symbol, on the power-control test sequences at the operating points of experiments/20261118_network_decisions.

  power_control        controlled_topology_change: the channel of the UEs changes at TB 32
  power_control_phase  natural_umi with a uniformly random phase jump at every power step (phase_jumps.py)

The decoding path is PilotHistoryBenchmarkAdapter's, unchanged; a probe runs the selection of ChannelFilter.__call__
(copied below, up to the combining gain) on the same inputs and returns, per UE and DMRS symbol, K (0: the current TB
only), whether any fit was possible (otherwise the LS estimate is used as it is), and the combining gain. Writes
results/fallback_k/<setting>/<condition>/episode_NNNNN.npz; the errors must equal the stored evaluation's at that point.

    fallback_k.py --setting power_control --condition controlled_topology_change --first-episode 0 --last-episode 16
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PC = ROOT/"experiments/20261112_early_retransmission"
POINTS = {"natural_umi": 1.5, "controlled_topology_change": 2.0, "schedule_gap_reentry": 2.5}   # controller.json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("power_control", "power_control_phase"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=16)
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
    from neural_rx.receiver.channel_filter import ChannelFilter, power, solve_positive_definite
    sys.path.insert(0, str(PC))
    from power_control import SETTINGS, PowerControlBank
    from phase_jumps import PhaseJumpBank

    class SelectionProbe(ChannelFilter):
        def select(self, observation, history, active, layout):
            """ChannelFilter.__call__ up to the combining gain, returning K, the fit flags and the gain per DMRS symbol."""
            active = tf.cast(active, tf.bool)
            valid = history.valid & active[..., None]
            users, symbols = layout.users, layout.dmrs_symbols
            entries = history.observations.shape[2]-self.fit_tbs
            batch = tf.shape(observation)[0]
            antennas = (observation.shape[-1]-1)//2
            stored = tf.reduce_sum(tf.cast(valid, tf.int32), -1)
            records = tf.concat([observation[:, :, None], tf.where(valid[..., None, None], history.observations, 0.)], 2)
            ports = tf.concat([tf.broadcast_to(tf.range(users)[None, :, None], [batch, users, 1]), history.ports], 2)
            ages = tf.concat([tf.zeros([batch, users, 1], history.age.dtype), history.age], 2)
            valid = tf.concat([active[..., None], valid], 2)
            lags, used = ages[:, :, 1:entries+1], valid[:, :, 1:entries+1]

            def members(symbol, usable):
                member = (symbol[:, :, None] == tf.range(symbols)[None, None, :, None]) & usable[:, :, None, None]
                return tf.cast(member, tf.float32)

            def per_symbol(value, symbol, usable):
                return tf.einsum('bskc,bsc->bsk', members(symbol, usable), tf.reduce_sum(value, -1))

            equations = []
            for target in range(1, self.fit_tbs+1):
                x, own, symbol, usable, found = self.design(records, ports, ages, valid, target, lags, used, layout)
                flat = tf.reshape(x, [batch, users, 1, -1, x.shape[-1]])
                member = tf.cast(tf.repeat(members(symbol, usable), antennas, -1), x.dtype)
                selected = flat*member[..., None]
                equations.append((tf.matmul(selected, flat, adjoint_a=True),
                                  tf.matmul(selected, tf.reshape(own, [batch, users, 1, -1, 1]), adjoint_a=True)[..., 0],
                                  usable, found))
            x, own, symbol, _, _ = self.design(records, ports, ages, valid, 0, lags, used, layout)
            everyone = tf.ones_like(active)
            count = per_symbol(tf.ones_like(tf.math.real(own)), symbol, everyone)
            step = x.shape[-1]//(entries+1)
            residuals = []
            for number in range(entries+1):
                width = (number+1)*step
                gram = tf.zeros([batch, users, symbols, width, width], x.dtype)
                rhs = tf.zeros([batch, users, symbols, width], x.dtype)
                fitted = tf.zeros([batch, users], tf.bool)
                for matrix, vector, usable, found in equations:
                    complete = usable & tf.reduce_all(found[..., :number], -1)
                    weight = tf.cast(complete, x.dtype)[..., None, None]
                    gram += weight[..., None]*matrix[..., :width, :width]
                    rhs += weight*vector[..., :width]
                    fitted |= complete
                fitted &= tf.reduce_all(used[..., :number], -1)
                trace = tf.math.real(tf.linalg.trace(gram))
                loading = self.ridge*trace/width+tf.cast(trace <= 0., tf.float32)
                gram += tf.cast(loading[..., None, None], gram.dtype)*tf.eye(width, dtype=gram.dtype)
                weights = solve_positive_definite(gram, rhs)
                prediction = tf.reduce_sum(x[..., :width]*tf.gather(weights, symbol, batch_dims=2)[:, :, :, None, :], -1)
                residual = tf.math.divide_no_nan(per_symbol(power(own-prediction), symbol, everyone), count)
                residuals.append(tf.where(fitted[..., None], residual, float('inf')))
            residuals = tf.stack(residuals, -1)
            best = tf.argmin(residuals, -1, output_type=tf.int32)
            residual = tf.reduce_min(residuals, -1)
            fitted = tf.cast(tf.math.is_finite(residual), tf.float32)
            noise = tf.broadcast_to(tf.maximum(observation[..., -1:], 0.), tf.shape(tf.math.real(own)))
            noise = tf.math.divide_no_nan(per_symbol(noise, symbol, everyone), count)
            residual = tf.where(fitted > 0., residual, 0.)
            gain = fitted*tf.math.divide_no_nan(noise, noise+tf.maximum(residual-noise, 1e-3*noise))
            return best, fitted, gain, stored

    class ProbeAdapter(PilotHistoryBenchmarkAdapter):
        """PilotHistoryBenchmarkAdapter._build_execution, plus the selection probe on the filter's inputs."""
        last_probe = None

        def _build_execution(self):
            buffer = self._buffer
            probe = SelectionProbe(self.model.channel_filter.fit_tbs, self.model.channel_filter.ridge)

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
                output = self.decoder(self.grid_map.grid_to_coded(grid))
                logits, hard, crc = checked_decoder_tensors(output)
                with tf.control_dependencies([logits, hard, crc]):
                    history = buffer.gather(tables, head, count, ids, active, step)       # probe only
                    selection = probe.select(observation, history, tf.cast(active, tf.float32), self.layout)
                    next_head, next_count, complete = buffer.commit(pending, finite, step)
                with tf.control_dependencies([complete]):
                    return (tf.identity(logits), tf.identity(hard), tf.identity(crc), next_head, next_count, complete,
                            *selection)

            def run(*arguments):
                *outputs, best, fitted, gain, stored = execute(*arguments)
                self.last_probe = [value.numpy() for value in (best, fitted, gain, stored)]
                return tuple(outputs)
            return run

    sweep.configure_runtime()
    anchors = json.loads(sweep.ANCHORS.read_text())["conditions"]
    protocol, plan = protocol_and_plan("confirmation", sweep.PRBS, 32, anchors, protocol_path=sweep.PROTOCOL)
    parameters = Parameters(sweep.CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=sweep.PRBS)
    Bank = PhaseJumpBank if args.setting == "power_control_phase" else PowerControlBank
    bank = Bank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf, dict(SETTINGS))
    point = POINTS[args.condition]
    job = dict(sweep.sweep_job(plan, anchors, args.condition), snr_db=[point])
    manifest, mapping, seed = make_manifest(protocol, "candidate", sweep.CHECKPOINT)
    receiver = ProbeAdapter(manifest=manifest, method="candidate_persistent", training_seed=seed,
                            artifacts=resolve_manifest_artifacts(manifest, mapping), config_name=sweep.CONFIG, tf_module=tf)
    output = args.output_dir/"fallback_k"/args.setting/args.condition
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.first_episode, args.last_episode):
        path = output/f"episode_{index:05d}.npz"
        if path.is_file():
            continue
        started = time.perf_counter()
        episode = validate_episode(bank.generate(job, index), protocol, job)
        tbs, users = episode.active.shape
        errors = np.zeros((tbs, users), bool)
        probes = []
        receiver.start_episode_batch(1)
        for tb in range(tbs):
            result = receiver.decode_tb_batch([episode.receiver_inputs[0][tb]])
            errors[tb] = payload_error_flags(np.asarray(result["hard_bits"]), np.asarray(result["crc_status"]),
                                             episode.bits[tb][None], episode.active[tb][None])[0][0]
            probes.append([value[0] for value in receiver.last_probe])
        best, fitted, gain, stored = (np.stack(values) for values in zip(*probes))
        extra = {"phase_steps": bank.last_phase_steps} if hasattr(bank, "last_phase_steps") else {}
        sweep._atomic_savez(path, k=best, fitted=fitted, gain=gain, stored_entries=stored, errors=errors,
                            active=episode.active, ue_ids=episode.ue_ids, channel_change_event=episode.channel_change_event,
                            snr_db=np.asarray(point), fingerprint=np.asarray(episode.fingerprint), **extra)
        scored = episode.active[sweep.SCORING_START:].astype(bool)
        print(json.dumps({"setting": args.setting, "condition": args.condition, "episode": index,
                          "errors": int((errors[sweep.SCORING_START:] & scored).sum()),
                          "k_zero_share": round(float((best[sweep.SCORING_START:][scored] == 0).mean()), 3),
                          "seconds": round(time.perf_counter()-started, 1)}), flush=True)


if __name__ == "__main__":
    main()
