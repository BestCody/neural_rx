"""GPU time of the IDD additions per slot: error covariance, symbol-pair preparation, APP detector, and combining.

These parts run as custom CUDA kernels (native_ops) inside TensorFlow, between the two BP stages, so they cannot be
exported to TensorRT. The graph below contains only the IDD additions, given the NRX LLRs and the first BP stage of the
same slot as export.py (natural UMi, episode 0, TB 24, full memory). The TensorFlow profiler records every GPU kernel;
the GPU time per slot is the summed kernel duration divided by the number of runs. BP iterations are not included.

    LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_cupti/lib time_idd.py   # results/idd.json

The profiler needs CUPTI on the library path (TensorFlow does not find the pip-installed CUPTI by itself).
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS, WARMUP = 200, 50
GROUPS = {"covariance": ("PackUncertainty", "UncertaintyStats", "UncertaintyFeatureValues", "UncertaintyRelu",
                         "UncertaintyOutput"),
          # pilot_pair_prepare: own kernels plus the batched Cholesky factorizations and triangular solves of
          # cuSOLVER/cuBLAS (complex GEMM tiles c1688gemm) on its own stream
          "pair_preparation": ("Assemble", "Finish", "Check", "potrf", "trsm", "c1688gemm"),
          "app_detector_and_combining": ("PhysicalScores", "Correction", "InitializeCorrected")}


def kernel_times(logdir, runs):
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
    totals, lines = {}, {}
    for path in Path(logdir).rglob("*.xplane.pb"):
        space = xplane_pb2.XSpace()
        space.ParseFromString(path.read_bytes())
        for plane in space.planes:
            if not plane.name.startswith("/device:GPU:"):
                continue
            names = {key: value.name for key, value in plane.event_metadata.items()}
            for line in plane.lines:
                duration = sum(event.duration_ps for event in line.events)
                lines[line.name] = lines.get(line.name, 0)+duration/1e6/runs
                if "memcpy" in line.name.lower() and "compute" not in line.name.lower():
                    continue
                for event in line.events:
                    name = names.get(event.metadata_id, "?")
                    totals[name] = totals.get(name, 0)+event.duration_ps
    # microseconds per run
    return {name: value/1e6/runs for name, value in totals.items()}, lines


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    import tensorflow as tf
    from neural_rx.benchmark import tbler_sweep as sweep
    from neural_rx.backbone.core import prepare_cgnn_inputs
    from neural_rx.receiver.pilot_history import PilotHistory

    sweep.configure_runtime()
    protocol = sweep.sweep_context()[0]
    adapter = sweep.make_receiver("candidate_persistent", protocol)
    model, layout, decoder = adapter.model, adapter.layout, adapter.decoder
    reference = np.load(HERE/"build/reference.npz")
    feed = {key.split("/")[-1]: tf.constant(reference[key]) for key in reference.files
            if key.startswith("proposed/input/")}
    y_real, ls, active, noise = feed["rx_slot"], feed["h_hat_ls"], feed["active"], feed["noise_variance"]
    history = PilotHistory(feed["memory_observations"], feed["memory_age"], feed["memory_ports"],
                           tf.cast(feed["memory_valid"], tf.bool))
    ids = tf.constant([[0, 1]], tf.int32)
    # Rebuild the received grid in the receiver's complex layout from the real NRX input [1, S, T, 2B].
    antennas = y_real.shape[-1]//2
    y = tf.complex(y_real[..., :antennas], y_real[..., antennas:])
    y = tf.transpose(y, [0, 3, 2, 1])[:, None]
    inputs = prepare_cgnn_inputs(adapter.receiver, y, ls, active)
    assert np.array_equal(inputs[0].numpy(), y_real.numpy())
    observation = model.observe(ls, noise)
    context, channel = model.prepare(inputs, observation, history, layout)
    grid, _ = model.detect(inputs, noise, history, observation, preparation=(context, channel))
    llr = adapter.grid_map.grid_to_coded(grid)
    code_input, _, state, posterior = decoder.evidence(llr)

    @tf.function
    def idd(context, llr, posterior, code_input):
        estimate = model.channel_memory.complete(context, layout)
        physical = model.physical.begin(inputs[0], estimate, noise, inputs[3], ids, llr, layout)
        return model.physical.finish(physical, posterior, inputs[3], decoder.symbol_indices, decoder.symbol_signs,
                                     code_input, model.evidence_calibration, decoder.first.llr_limit,
                                     tf.sigmoid(model.correction_logit), model.blend_variables)

    # Same result as the receiver's own decoding graph for this slot.
    corrected = idd(context, llr, posterior, code_input)
    output = decoder.finish(corrected, state)
    full = model.decode(inputs, noise, ids, history, decoder, adapter.grid_map, observation, (context, channel))[0]
    assert np.array_equal(output.logits.numpy(), full.logits.numpy()), "IDD graph differs from the receiver"

    for _ in range(WARMUP):
        idd(context, llr, posterior, code_input).numpy()
    logdir = tempfile.mkdtemp(prefix="idd_profile_")
    tf.profiler.experimental.start(logdir)
    for _ in range(RUNS):
        idd(context, llr, posterior, code_input).numpy()
    tf.profiler.experimental.stop()
    per_kernel, per_line = kernel_times(logdir, RUNS)
    shutil.rmtree(logdir)
    groups = {group: sum(v for k, v in per_kernel.items() if any(name in k for name in names))
              for group, names in GROUPS.items()}
    grouped = set(k for k in per_kernel if any(name in k for names in GROUPS.values() for name in names))
    groups["other_tensorflow_ops"] = sum(v for k, v in per_kernel.items() if k not in grouped)
    report = {"gpu_us_per_slot": groups, "total_gpu_us_per_slot": sum(groups.values()), "runs": RUNS,
              "kernels_us_per_slot": dict(sorted(per_kernel.items(), key=lambda item: -item[1])),
              "gpu_lines_us_per_slot": per_line,
              "measure": "summed GPU kernel durations from the TensorFlow profiler, per run of the IDD additions"}
    (HERE/"results").mkdir(exist_ok=True)
    (HERE/"results/idd.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
