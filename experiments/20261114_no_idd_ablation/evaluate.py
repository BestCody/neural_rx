"""Decode the TBLER test sequences with the proposed NRX without IDD, with and without its pilot memory.

The receiver is the benchmark's proposed NRX (same weights, same pilot memory, same channel filter), but its own LLRs
go straight to the standard 20-iteration LDPC decoder that every other receiver uses: no APP detector, no prior
feedback, no split of BP decoding. The memory is still written from the pilots of every TB, as in the paper.

The episodes are those of experiments/20261104_tbler_sweep (grids "sweep" and "sweep_high", no power control). Each
episode is regenerated on the stored SNR grid of the proposed NRX and must have the stored fingerprint.

    evaluate.py --gpu 0 --setting sweep --condition natural_umi
    evaluate.py --gpu 0 --setting sweep --condition natural_umi --check   # with IDD: must reproduce stored errors
"""
import argparse
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SWEEP = ROOT/"experiments/20261104_tbler_sweep/results"
METHODS = ("candidate_persistent", "candidate_reset")


def without_idd(adapter):
    """Replace the adapter's decoding graph: NRX LLRs -> 20 BP iterations, memory handling unchanged."""
    tf = adapter.tf
    from neural_rx.benchmark.candidate import checked_decoder_tensors
    from neural_rx.decoding import SoftTBInformationDecoder

    tb_decoder = adapter.receiver._tb_decoders[0]
    adapter.plain_decoder = SoftTBInformationDecoder(adapter.receiver._tb_encoders[0], tb_decoder,
                                                     int(tb_decoder._decoder._num_iter))
    adapter.bp_iterations = int(tb_decoder._decoder._num_iter)

    def build():
        buffer = adapter._buffer
        reset = adapter.method == "candidate_reset"
        model = adapter.model

        @tf.function(reduce_retracing=True, jit_compile=True)
        def prepare(channel, noise, ids, active, step, head, count, tables):
            history = buffer.gather(tables, head, count, ids, active, step)
            observation = model.observe(channel, noise)
            preparation = model.prepare([None, None, channel, tf.cast(active, tf.float32)],
                                        observation, history, adapter.layout)
            pending, finite = buffer.prepare_stage(observation, head, count, ids, active)
            return observation, preparation, pending, finite

        @tf.function(reduce_retracing=True, jit_compile=False)
        def execute(y, channel, noise, control, active, head, count):
            ids, step = tf.cast(control[:, :-1], tf.int32), control[0, -1]
            if reset:
                count = tf.zeros_like(count)
            tables = tuple(v.read_value() for v in (buffer.observations, buffer.timestamps, buffer.ports))
            observation, preparation, pending, finite = prepare(
                channel, noise, ids, active, step, head, count, tables)
            inputs = adapter._prepare_inputs(adapter.receiver, y, channel, tf.cast(active, tf.float32))
            grid, _ = model.detect(inputs, noise, None, observation, preparation=preparation)
            output = adapter.plain_decoder(adapter.grid_map.grid_to_coded(grid))
            logits, hard, crc = checked_decoder_tensors(output)
            with tf.control_dependencies([logits, hard, crc]):
                next_head, next_count, complete = buffer.commit(pending, finite, step)
            with tf.control_dependencies([complete]):
                return tf.identity(logits), tf.identity(hard), tf.identity(crc), next_head, next_count, complete

        return execute

    adapter._build_execution = build
    adapter._buffer = None  # the next start_episode_batch() builds the new graph
    adapter.start_episode()
    return adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--setting", choices=("sweep", "sweep_high"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--last-episode", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=HERE/"results/test")
    parser.add_argument("--check", action="store_true",
                        help="Keep IDD and require the stored errors of both methods (pipeline check, no output)")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.benchmark.runtime import validate_episode

    sweep.configure_runtime()
    protocol, plan, parameters, bank, anchors = sweep.sweep_context()
    base_job = sweep.sweep_job(plan, anchors, args.condition)
    receivers = {}
    for method in METHODS:
        receiver = sweep.make_receiver(method, protocol)
        receivers[method] = receiver if args.check else without_idd(receiver)
    if not args.check:
        print(json.dumps({"bp_iterations": receivers[METHODS[0]].bp_iterations}), flush=True)

    for index in range(args.first_episode, args.last_episode):
        name = f"episode_{index:05d}.npz"
        outputs = {m: args.output_dir/args.setting/args.condition/f"{m}_no_idd"/name for m in METHODS}
        if not args.check and all(path.is_file() for path in outputs.values()):
            continue
        stored = {}
        for method in METHODS:
            with np.load(SWEEP/args.setting/args.condition/method/name, allow_pickle=False) as data:
                stored[method] = {key: data[key] for key in data.files}
        snr = stored[METHODS[0]]["snr_db"]
        if not all(np.array_equal(stored[m]["snr_db"], snr) for m in METHODS):
            raise ValueError(f"{name}: stored SNR grids differ")
        job = dict(base_job, snr_db=[float(value) for value in snr])
        started = time.perf_counter()
        episode = validate_episode(bank.generate(job, index), protocol, job)
        if not all(np.array_equal(np.asarray(episode.fingerprint), stored[m]["fingerprint"]) for m in METHODS):
            raise RuntimeError(f"{args.setting}/{args.condition}/{name}: episode differs from the stored results")
        report = {"setting": args.setting, "condition": args.condition, "episode": index,
                  "generate_s": round(time.perf_counter()-started, 2)}
        for method in METHODS:
            begun = time.perf_counter()
            errors, disagreement = sweep.decode_episode(method, receivers[method], episode, None)
            report[method] = errors[:, sweep.SCORING_START:].sum((1, 2)).tolist()
            report[f"{method}_s"] = round(time.perf_counter()-begun, 2)
            if args.check:
                scored = slice(sweep.SCORING_START, None)
                differ = int((errors[:, scored] != stored[method]["errors"][:, scored]).sum())
                if differ:
                    raise RuntimeError(f"{name} {method}: {differ} error flags differ from the stored run")
                report[f"{method}_matches_stored"] = True
                continue
            keep = {key: stored[method][key] for key in ("active", "ue_ids", "snr_db", "slot_number", "fingerprint", "channel_power")
                    if key in stored[method]}
            sweep._atomic_savez(outputs[method], errors=errors, crc_disagreement=disagreement,
                                bp_iterations=np.asarray(receivers[method].bp_iterations), **keep)
        print(json.dumps(report), flush=True)
        if args.check:
            break
    if not all(receiver.checksums_unchanged() for receiver in receivers.values()):
        raise RuntimeError("receiver weights changed during the run")


if __name__ == "__main__":
    main()
