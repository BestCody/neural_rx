"""TBLER-vs-SNR sweep of the memory receiver, the Cold NRX, and classical baselines.

Every receiver decodes the same channels, payloads, and unit-noise realizations at each
SNR. Episodes are generated in memory; each (condition, method, episode) result is saved
separately, so an interrupted sweep resumes and later methods reuse the same episodes.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from neural_rx.backbone import E2E_Model, Parameters
from neural_rx.backbone.core import expand_noise, make_snr_batch
from neural_rx.benchmark.baselines import (ClassicalBaselineReceiver, SYSTEMS, covariance_from_channels,
                                           normalized_covariance)
from neural_rx.benchmark.frozen import make_benchmark_adapter, make_manifest, protocol_and_plan
from neural_rx.benchmark.plan import decreasing_isotonic, snr_at_target
from neural_rx.benchmark.runtime import BenchmarkEpisode, _measure_latency, validate_episode
from neural_rx.benchmark.scenarios import TemporalBenchmarkScenarioBank
from neural_rx.benchmark.scoring import payload_error_flags
from neural_rx.paths import ROOT

CONFIG = "nrx_large.cfg"
PROTOCOL = ROOT/"config/temporal_benchmark_v1.json"
CHECKPOINT = ROOT/"models/learned_blend_joint_20261019/checkpoint_005000"
ANCHORS = ROOT/"experiments/20261019_learned_blend_joint_training_6000/inputs/anchors.json"
CORE = ("natural_umi", "natural_uma", "controlled_topology_change", "schedule_gap_reentry")
NEURAL = ("candidate_persistent", "candidate_reset", "frozen_cold_k8", "frozen_cold_k3", "frozen_cold_k2")
METHODS = NEURAL+SYSTEMS
SNR_OFFSETS_DB = tuple(float(offset) for offset in range(-6, 4))
BASELINE_CHUNK = 5
SWEEP_SEED = 90000000
COVARIANCE_SEED = 92000000
SCORING_START = 16
PRBS = 132


def configure_runtime():
    import sionna as sn
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(20260930)
    sn.config.seed = 20260930


class ChannelCapturingBank(TemporalBenchmarkScenarioBank):
    """The benchmark's episodes, also exposing the true channel and copying each TB's inputs.

    The parent's per-TB views keep the whole [TB, ...] array of every SNR alive, which is
    harmless for one SNR but exceeds host memory for a sweep. Episode fingerprints stay equal.
    """

    def generate(self, job, episode_index):
        if not 0 <= int(episode_index) < int(job["episode_count"]):
            raise ValueError("episode index is outside the job")
        generator = self._generator(job, episode_index)
        settings = job["provider_settings"]
        change = settings.get("change_tb") if job["provider"] == "shared_geometry_change" else None
        snrs = [float(value) for value in job["snr_db"]]
        base = generator.sample_batch(1, int(job["num_tbs"]), snrs[0],
                                      scenario_training=job["provider"] == "shared_geometry_schedule",
                                      controlled_change_tb=change, include_clean_received=True, include_ls=False)
        self.last_channel = base["h"]
        scale = tf.cast(tf.sqrt(expand_noise(base["no"], base["y_clean"])), tf.complex64)
        unit_noise = tf.math.divide_no_nan(base["y"]-base["y_clean"], scale)
        active = base["active"].numpy()[0].astype(bool)
        ue_ids = base["ue_ids"].numpy()[0].astype(np.int32)
        pool_change = base["channel_change_event"].numpy()[0].astype(bool)
        slots = base["slot_number"].numpy()
        receiver_inputs = []
        for snr in snrs:
            batch = make_snr_batch(base, unit_noise, snr, generator, self.receiver)
            ls, no, received = batch["ls"].numpy()[0], batch["no"].numpy()[0], batch["y"].numpy()[0]
            ids = batch["ue_ids"].numpy()[0]
            receiver_inputs.append(tuple(
                {"active": active[tb], "ls_channel": ls[tb].copy(), "noise_variance": no[tb].copy(),
                 "received_grid": received[tb].copy(), "slot_number": np.asarray(slots[tb], np.int32),
                 "ue_ids": ids[tb].copy()}
                for tb in range(int(job["num_tbs"]))))
            del batch, ls, no, received
        return BenchmarkEpisode(bits=base["bits"].numpy()[0], active=active, ue_ids=ue_ids,
                                channel_change_event=np.take_along_axis(pool_change, ue_ids, axis=1),
                                receiver_inputs=tuple(receiver_inputs))


def sweep_context(episodes=32):
    anchors = json.loads(ANCHORS.read_text())["conditions"]
    protocol, plan = protocol_and_plan("confirmation", PRBS, episodes, anchors, protocol_path=PROTOCOL)
    parameters = Parameters(CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=PRBS)
    bank = ChannelCapturingBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf)
    return protocol, plan, parameters, bank, anchors


def sweep_job(plan, anchors, condition):
    order = json.loads(PROTOCOL.read_text())["condition_order"]
    job = next(item for item in plan["jobs"] if item["condition"] == condition)
    index = order.index(condition)
    domains = json.loads(PROTOCOL.read_text())["seed_domains"]
    anchor = float(anchors[condition]["anchor_snr_db"])
    return dict(job, snr_db=[anchor+offset for offset in SNR_OFFSETS_DB],
                seed_streams={name: SWEEP_SEED+index*1000+offset for name, offset in domains.items()})


def make_receiver(method, protocol, covariance=None):
    if method in SYSTEMS:
        parameters = Parameters(CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=PRBS)
        return ClassicalBaselineReceiver(parameters, method, covariance)
    role = "candidate" if method.startswith("candidate_") else method
    manifest, mapping, seed = make_manifest(protocol, role, CHECKPOINT)
    return make_benchmark_adapter(manifest=manifest, method=method, training_seed=seed, artifact_map=mapping,
                                  config_name=CONFIG, tf_module=tf)


def decode_episode(method, receiver, episode, channel, first_tb=None, only_tbs=None):
    """Decode one episode at all of its SNR points as one batch."""
    points = len(episode.receiver_inputs)
    tbs, users = episode.active.shape
    errors = np.zeros((points, tbs, users), bool)
    disagreement = np.zeros((points, tbs, users), bool)
    stateless = method != "candidate_persistent"
    first = (SCORING_START if stateless else 0) if first_tb is None else first_tb
    if method in NEURAL:
        receiver.start_episode_batch(points)
    for tb in (range(first, tbs) if only_tbs is None else only_tbs):
        inputs = [episode.receiver_inputs[point][tb] for point in range(points)]
        truth = np.broadcast_to(episode.bits[tb], (points, *episode.bits[tb].shape))
        active = np.broadcast_to(episode.active[tb], (points, users))
        if method in SYSTEMS:
            perfect = channel is not None and method == "baseline_perf_csi_kbest"
            parts = []
            for start in range(0, points, BASELINE_CHUNK):
                chunk = inputs[start:start+BASELINE_CHUNK]
                parts.append(receiver.decode(
                    np.stack([value["received_grid"] for value in chunk]),
                    np.asarray([value["noise_variance"] for value in chunk], np.float32), active[:len(chunk)],
                    h=np.repeat(channel[:, tb], len(chunk), axis=0) if perfect else None,
                    slot_number=int(chunk[0]["slot_number"])))
            hard, crc = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        else:
            output = receiver.decode_tb_batch(inputs)
            hard, crc = output["hard_bits"], output["crc_status"]
        errors[:, tb], disagreement[:, tb] = payload_error_flags(np.asarray(hard), np.asarray(crc), truth, active)
    return errors, disagreement


def _atomic_savez(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, path)


def effective_power(channel, w):
    """Mean effective channel power per [TB, UE] after the codebook precoder."""
    h = tf.transpose(channel[0], perm=[0, 1, 3, 5, 6, 2, 4])
    h = tf.matmul(h, tf.cast(w, h.dtype)[:, None, None])
    return tf.reduce_mean(tf.abs(h)**2, axis=(1, 3, 4, 5, 6)).numpy()


def load_covariance(directory, condition):
    with np.load(Path(directory)/f"{condition}.npz", allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def run_sweep(args):
    configure_runtime()
    protocol, plan, parameters, bank, anchors = sweep_context()
    output = Path(args.output_dir)
    condition, methods = args.condition, tuple(args.methods)
    job = sweep_job(plan, anchors, condition)
    if args.snr_offsets:
        job = dict(job, snr_db=[float(anchors[condition]["anchor_snr_db"])+offset for offset in args.snr_offsets])
    covariance = load_covariance(args.covariance_dir, condition) if "baseline_lmmse_kbest" in methods else None
    receivers = {method: make_receiver(method, protocol, covariance) for method in methods}
    w = parameters.transmitters[0]._precoder._w
    for episode_index in range(args.first_episode, args.last_episode):
        pending = [m for m in methods if not (output/condition/m/f"episode_{episode_index:05d}.npz").is_file()]
        if not pending:
            continue
        started = time.perf_counter()
        episode = validate_episode(bank.generate(job, episode_index), protocol, job)
        channel = bank.last_channel
        generated = time.perf_counter()-started
        slots = np.asarray([int(tb["slot_number"]) for tb in episode.receiver_inputs[0]], np.int32)
        power = effective_power(channel, w)
        for method in pending:
            started = time.perf_counter()
            errors, disagreement = decode_episode(method, receivers[method], episode,
                                                  channel if method == "baseline_perf_csi_kbest" else None)
            _atomic_savez(output/condition/method/f"episode_{episode_index:05d}.npz",
                          errors=errors, crc_disagreement=disagreement, active=episode.active,
                          ue_ids=episode.ue_ids, snr_db=np.asarray(job["snr_db"]), slot_number=slots,
                          channel_power=power,
                          fingerprint=np.asarray(episode.fingerprint))
            print(json.dumps({"condition": condition, "episode": episode_index, "method": method,
                              "generate_s": round(generated, 2), "decode_s": round(time.perf_counter()-started, 2),
                              "errors": errors[:, SCORING_START:].sum((1, 2)).tolist()}), flush=True)
    for method, receiver in receivers.items():
        if method in NEURAL and not receiver.checksums_unchanged():
            raise RuntimeError(f"{method} artifacts changed during the sweep")


def run_covariance(args):
    """Condition-specific covariance from independently seeded channel draws."""
    configure_runtime()
    protocol, plan, parameters, bank, anchors = sweep_context()
    output = Path(args.covariance_dir)
    output.mkdir(parents=True, exist_ok=True)
    keep = (parameters.transmitters[0]._resource_grid.fft_size)//6
    w = tf.cast(parameters.transmitters[0]._precoder._w, tf.complex64)
    for condition in args.conditions:
        job = sweep_job(plan, anchors, condition)
        job = dict(job, seed_streams={k: v-SWEEP_SEED+COVARIANCE_SEED for k, v in job["seed_streams"].items()})
        strata = len(job["provider_settings"].get("doppler_strata_hz", [0]))
        total, count = None, 0
        for draw in range(args.draws):
            generator = bank._generator(job, draw % strata)
            tf.random.set_seed(COVARIANCE_SEED+draw)
            h = generator.sample_pool_channel(args.batch_size, 1)[:, 0, 0]
            h = tf.transpose(h, perm=[0, 2, 4, 5, 1, 3])
            effective = tf.matmul(h, tf.reshape(w[0], [1, 1, 1, 1, 2, 1]))[..., 0]
            effective = tf.transpose(effective, perm=[0, 1, 4, 2, 3])[..., :keep]
            values = covariance_from_channels(effective)
            total = values if total is None else {name: total[name]+values[name] for name in values}
            count += 1
        result = normalized_covariance(total, count)
        np.savez(output/f"{condition}.npz", **result)
        print(json.dumps({"condition": condition, "power": result["power"], "topologies": count*args.batch_size,
                          "time_diag": np.real(np.diag(result["time"]))[:3].round(4).tolist(),
                          "space_diag": np.real(np.diag(result["space"])).round(4).tolist()}), flush=True)


def run_reproduce(args):
    """Check that the sweep path reproduces sealed confirmation episodes and decisions."""
    configure_runtime()
    protocol, plan, _, bank, _ = sweep_context(episodes=max(args.episodes, 1))
    with np.load(args.sealed, allow_pickle=False) as data:
        sealed = {name: data[name] for name in data.files}
    order = json.loads(PROTOCOL.read_text())["condition_order"]
    receivers = {m: make_receiver(m, protocol) for m in args.methods}
    report = {}
    for condition in args.conditions:
        job = next(item for item in plan["jobs"] if item["condition"] == condition)
        index = order.index(condition)
        same_episode, same_blocks, compared = 0, 0, 0
        for episode_index in range(args.episodes):
            episode = validate_episode(bank.generate(job, episode_index), protocol, job)
            expected = sealed["episode_fingerprint"][index, episode_index]
            same_episode += bool(np.array_equal(episode.fingerprint, expected))
            active = episode.active[SCORING_START:]
            for method, receiver in receivers.items():
                errors, _ = decode_episode(method, receiver, episode, None)
                old = sealed[f"errors_{method}"][index, 0, episode_index, SCORING_START:]
                new = errors[0, SCORING_START:]
                same_blocks += int(((old == new) | ~active).sum())
                compared += int(active.size)
        report[condition] = {"episodes": args.episodes, "fingerprints_equal": same_episode,
                             "blocks_equal": same_blocks, "blocks": compared}
        print(json.dumps({condition: report[condition]}), flush=True)
    return report


def run_doubletdl(args):
    """Compare the restored baselines with upstream's stored Double-TDL curves."""
    from sionna.utils import BinarySource, ebnodb2no
    configure_runtime()
    parameters = Parameters(CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=PRBS)
    tx = parameters.transmitters[0]
    source, batch = BinarySource(), args.batch_size
    covariance = load_covariance(args.covariance_dir, "natural_umi")
    covariance["space"] = covariance["space"]/covariance["power"]  # upstream's unit-power UMi statistics
    results = {}
    for system in args.systems:
        receiver = ClassicalBaselineReceiver(parameters, system, covariance)
        results[system] = {}
        for snr in args.snr_db:
            failures, blocks, started = 0, 0, time.perf_counter()
            no = ebnodb2no(float(snr), tx._num_bits_per_symbol, tx._target_coderate, tx._resource_grid)
            for _ in range(args.batches):
                bits = source([batch, parameters.max_num_tx, tx._tb_size])
                y, h = parameters.channel([tx(bits), no])
                hard, crc = receiver.decode(y, tf.fill([batch], no), tf.ones([batch, 2], tf.bool),
                                            h=h if system == "baseline_perf_csi_kbest" else None)
                failures += int(np.any(hard != bits.numpy(), axis=-1).sum())
                blocks += batch*2
            results[system][str(snr)] = {"failures": failures, "blocks": blocks, "tbler": failures/blocks,
                                         "seconds": round(time.perf_counter()-started, 1)}
            print(json.dumps({"system": system, "snr_db": snr, **results[system][str(snr)]}), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=1))


def run_latency(args):
    """Median per-TB decode latency on the benchmark's timer and inputs."""
    configure_runtime()
    protocol, plan = protocol_and_plan("latency", PRBS, protocol_path=PROTOCOL)
    parameters = Parameters(CONFIG, training=False, num_tx_eval=2, n_size_bwp_override=PRBS)
    bank = TemporalBenchmarkScenarioBank(parameters, E2E_Model(parameters, training=False, mcs_arr_eval_idx=0), tf)
    adapters = {method: make_receiver(method, protocol) for method in args.methods}
    timings = _measure_latency(protocol, plan, bank, adapters, tuple(args.methods))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), **timings)
    results = {m: {"median_ms": float(np.median(timings[f"latency_ms_{m}"])),
                   "samples": len(timings[f"latency_ms_{m}"])} for m in args.methods}
    output.write_text(json.dumps(results, indent=1))
    print(json.dumps(results), flush=True)


def _summarize_directory(results_dir):
    """Failure counts per [condition, method, SNR] of one sweep, checking receiver pairing."""
    counts = {}
    for condition_dir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        fingerprints = {}
        for method_dir in sorted(p for p in condition_dir.iterdir() if p.is_dir()):
            failures, blocks, snr, episodes, disagree = None, 0, None, 0, 0
            for path in sorted(method_dir.glob("episode_*.npz")):
                with np.load(path, allow_pickle=False) as data:
                    active = data["active"][SCORING_START:]
                    new = (data["errors"][:, SCORING_START:] & active[None]).sum((1, 2))
                    failures = new if failures is None else failures+new
                    blocks += int(active.sum())
                    disagree += int(data["crc_disagreement"][:, SCORING_START:].sum())
                    if snr is None:
                        snr = data["snr_db"]
                    elif not np.array_equal(snr, data["snr_db"]):
                        raise ValueError("SNR grids differ between episodes")
                    value = data["fingerprint"].tobytes()
                    if fingerprints.setdefault(path.name, value) != value:
                        raise ValueError(f"{condition_dir.name}/{path.name} was decoded on different episodes")
                    episodes += 1
            if failures is not None:
                counts.setdefault(condition_dir.name, {})[method_dir.name] = {
                    "failures": dict(zip(snr.tolist(), failures.tolist(), strict=True)), "blocks": blocks,
                    "episodes": episodes, "crc_disagreements": disagree}
    return counts


def summarize(results_dirs, target=.1):
    """Merge sweeps into TBLER curves and SNR@TBLER=10%; shared SNR points must agree."""
    results_dirs = [results_dirs] if isinstance(results_dirs, (str, Path)) else list(results_dirs)
    merged, overlaps = {}, []
    for directory in results_dirs:
        for condition, methods in _summarize_directory(directory).items():
            for method, value in methods.items():
                entry = merged.setdefault(condition, {}).setdefault(method, None)
                if entry is None:
                    merged[condition][method] = value
                    continue
                if entry["blocks"] != value["blocks"] or entry["episodes"] != value["episodes"]:
                    raise ValueError(f"{condition}/{method} sweeps cover different blocks")
                for snr, failures in value["failures"].items():
                    if snr in entry["failures"]:
                        overlaps.append({"condition": condition, "method": method, "snr_db": snr,
                                         "first": entry["failures"][snr], "second": failures})
                    else:
                        entry["failures"][snr] = failures
                entry["crc_disagreements"] += value["crc_disagreements"]
    summary = {"overlap_points": overlaps}
    for condition, methods in merged.items():
        summary[condition] = {}
        for method, value in methods.items():
            snr = np.asarray(sorted(value["failures"]))
            failures = np.asarray([value["failures"][point] for point in snr])
            rates = failures/value["blocks"]
            fitted = decreasing_isotonic(rates, np.full(len(rates), float(value["blocks"])))
            summary[condition][method] = {"snr_db": snr.tolist(), "tbler": rates.tolist(),
                                          "failures": failures.tolist(), "blocks": value["blocks"],
                                          "episodes": value["episodes"],
                                          "crc_disagreements": value["crc_disagreements"],
                                          "snr_at_10pct_db": snr_at_target(snr, fitted, target)}
    return summary
