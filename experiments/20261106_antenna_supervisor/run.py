"""Antenna-setting supervisor for the proposed NRX, on the channels of experiments/20261104_tbler_sweep.

A UE's precoder w_k is emulated by replacing its channel H with (H w_k) w_0^H. The unchanged
transmitter (precoder w_0) then produces exactly H w_k s, so neither transmitter nor receiver
code changes. Each run is a receiver (with or without pilot memory) and a mode:
  fixed<k>    every UE uses setting k for the whole sequence (fixed0 must reproduce the sweep)
  oracle      each UE uses the setting with the highest mean channel power in its sequence
  supervised  a UE moves to the next setting after two consecutive CRC failures; its memory is cleared
Usage: python run.py <condition> <first episode> <last episode> <gpu>
"""
import json
import os
import sys
import time
from pathlib import Path

condition, first, last, gpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from neural_rx.backbone.core import expand_noise  # noqa: E402
from neural_rx.benchmark import tbler_sweep as sweep  # noqa: E402
from neural_rx.benchmark.scoring import payload_error_flags  # noqa: E402
from neural_rx.receiver.benchmark_adapter import CommittedMemory  # noqa: E402

SETTINGS = np.array([[1, 1], [1, -1], [1, 1j], [1, -1j]], np.complex64)/np.sqrt(2)
OFFSETS = tuple(float(offset) for offset in range(-14, 9))
RUNS = {**{mode: ("candidate_persistent", mode) for mode in ("fixed0", "fixed1", "fixed2", "fixed3", "oracle",
                                                            "supervised")},
        **{f"nomem_{mode}": ("candidate_reset", mode) for mode in ("fixed0", "oracle", "supervised")}}
SWITCH_AFTER = 2
TBS, USERS, POOL = 64, 2, 4
OUTPUT = Path(__file__).parent/"results"/condition


def contributions(generator, base):
    """Noiseless received signal of each UE position under each setting, plus channel power."""
    h, bits = base["h"][0], base["bits"][0]
    active = tf.cast(base["active"][0], tf.complex64)
    w0 = tf.math.conj(tf.constant(SETTINGS[0]))
    signals, power, residual = [], np.zeros((TBS, USERS, len(SETTINGS))), 0.
    for t in range(TBS):
        generator._set_slot_pilots(int(base["slot_number"][t]))
        x = generator.tx(bits[None, t])*active[None, t, :, None, None, None]
        per_position = []
        for p in range(USERS):
            mask = tf.one_hot(p, USERS, dtype=tf.complex64)[None, :, None, None, None]
            per_setting = []
            for k, w in enumerate(SETTINGS):
                effective = tf.einsum("rabsf,b->rasf", h[t][:, :, p], tf.constant(w))
                replaced = effective[:, :, None]*w0[None, None, :, None, None]
                channel = tf.stack([replaced if q == p else tf.zeros_like(replaced) for q in range(USERS)], 2)
                per_setting.append(generator.apply_channel_noiseless([x*mask, channel[None]])[0])
                power[t, p, k] = float(tf.reduce_mean(tf.abs(effective)**2))
            per_position.append(tf.stack(per_setting))
        signals.append(tf.stack(per_position))
        exact = base["y_clean"][0, t]
        residual = max(residual, float(tf.reduce_max(tf.abs(signals[t][0, 0]+signals[t][1, 0]-exact))
                                       /tf.reduce_max(tf.abs(exact))))
    return signals, power, residual


def clear_memory(adapter, row, ue):
    state = adapter._committed
    count = tf.tensor_scatter_nd_update(state.count, [[row, ue]], tf.zeros([1], state.count.dtype))
    adapter._committed = CommittedMemory(state.head, count, state.step)


def decode(mode, adapter, generator, receiver, base, signals, unit_noise, power, snr):
    points = len(snr)
    no = generator.ebno_to_no(tf.constant(snr, tf.float32)).numpy()
    active = base["active"].numpy()[0].astype(bool)
    ids = base["ue_ids"].numpy()[0].astype(int)
    slots = base["slot_number"].numpy()
    truth = base["bits"].numpy()[0]
    setting = np.zeros((points, POOL), int)
    if mode.startswith("fixed"):
        setting[:] = int(mode[-1])
    elif mode == "oracle":
        for ue in range(POOL):
            mask = ids[sweep.SCORING_START:] == ue
            if mask.any():
                setting[:, ue] = int(np.argmax(power[sweep.SCORING_START:][mask].mean(0)))
    failures = np.zeros((points, POOL), int)
    errors = np.zeros((points, TBS, USERS), bool)
    used = np.zeros((points, TBS, USERS), np.int8)
    switches = np.zeros(points, int)
    adapter.start_episode_batch(points)
    scale = tf.constant(np.sqrt(no).astype(np.complex64))[:, None, None, None, None]
    for t in range(TBS):
        generator._set_slot_pilots(int(slots[t]))
        chosen = setting[:, ids[t]]
        # Setting 0 for both positions is the original transmission; reuse it exactly.
        clean = tf.stack([base["y_clean"][0, t] if not chosen[r].any() else
                          signals[t][0, chosen[r, 0]]+signals[t][1, chosen[r, 1]] for r in range(points)])
        received = clean+unit_noise[t][None]*scale
        ls = receiver.estimate_channel(received, USERS).numpy()
        grid = received.numpy()
        inputs = [{"active": active[t], "ls_channel": ls[r], "noise_variance": np.float32(no[r]),
                   "received_grid": grid[r], "slot_number": np.asarray(slots[t], np.int32), "ue_ids": ids[t]}
                  for r in range(points)]
        output = adapter.decode_tb_batch(inputs)
        crc = np.asarray(output["crc_status"])
        errors[:, t], _ = payload_error_flags(np.asarray(output["hard_bits"]), crc,
                                              np.broadcast_to(truth[t], (points, *truth[t].shape)),
                                              np.broadcast_to(active[t], (points, USERS)))
        used[:, t] = chosen
        if mode == "supervised":
            for r in range(points):
                for p in np.flatnonzero(active[t]):
                    ue = ids[t, p]
                    failures[r, ue] = 0 if crc[r, p] else failures[r, ue]+1
                    if failures[r, ue] >= SWITCH_AFTER:
                        setting[r, ue] = (setting[r, ue]+1) % len(SETTINGS)
                        failures[r, ue] = 0
                        switches[r] += 1
                        clear_memory(adapter, r, ue)
    return errors, used, switches


def main():
    sweep.configure_runtime()
    protocol, plan, _, bank, anchors = sweep.sweep_context()
    adapters = {method: sweep.make_receiver(method, protocol) for method in ("candidate_persistent", "candidate_reset")}
    anchor = float(anchors[condition]["anchor_snr_db"])
    job = dict(sweep.sweep_job(plan, anchors, condition), snr_db=[anchor+offset for offset in OFFSETS])
    change = job["provider_settings"].get("change_tb") if job["provider"] == "shared_geometry_change" else None
    for index in range(first, last):
        assert int(job["num_tbs"]) == TBS
        targets = {run: OUTPUT/run/f"episode_{index:05d}.npz" for run in RUNS}
        if all(path.is_file() for path in targets.values()):
            continue
        started = time.perf_counter()
        generator = bank._generator(job, index)
        base = generator.sample_batch(1, TBS, job["snr_db"][0],
                                      scenario_training=job["provider"] == "shared_geometry_schedule",
                                      controlled_change_tb=change, include_clean_received=True, include_ls=False)
        scale = tf.cast(tf.sqrt(expand_noise(base["no"], base["y_clean"])), tf.complex64)
        unit_noise = tf.math.divide_no_nan(base["y"]-base["y_clean"], scale)[0]
        signals, power, residual = contributions(generator, base)
        report = {"episode": index, "generate_s": round(time.perf_counter()-started, 1),
                  "max_relative_residual": residual}
        for run, path in targets.items():
            if path.is_file():
                continue
            started = time.perf_counter()
            method, mode = RUNS[run]
            errors, used, switches = decode(mode, adapters[method], generator, bank.receiver, base, signals, unit_noise,
                                            power, job["snr_db"])
            sweep._atomic_savez(path, errors=errors, setting=used, switches=switches,
                                active=base["active"].numpy()[0].astype(bool),
                                ue_ids=base["ue_ids"].numpy()[0].astype(np.int32),
                                snr_db=np.asarray(job["snr_db"]), channel_power=power)
            report[run] = {"s": round(time.perf_counter()-started, 1),
                            "failures": int(errors[:, sweep.SCORING_START:].sum()), "switches": int(switches.sum())}
        print(json.dumps({"condition": condition, **report}), flush=True)
    if not all(adapter.checksums_unchanged() for adapter in adapters.values()):
        raise RuntimeError("receiver artifacts changed")


if __name__ == "__main__":
    main()
