#!/usr/bin/env python3
"""Generate temporally correlated Neural RX TB sequences with stable UE identity.

Each physical UE gets one continuous TDL trajectory over the full sequence.
A scheduler then maps a subset of those physical UEs into the receiver's current
input positions at each TB.  Payload bits and AWGN are freshly sampled per TB.

This makes schedules such as

    TB1: [A, B]
    TB2: [B, C]
    TB3: [A, C]

physically meaningful: UE B's channel remains UE B's channel even when B moves
from input position 1 to input position 0.  Returned ``ue_ids`` are the stable
keys used by the external temporal-memory manager.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="nrx_large.cfg")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--ebno-db", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--ue-pool-size", type=int, default=4)
    p.add_argument("--dynamic-scheduling", action="store_true")
    p.add_argument("--schedule-switch-prob", type=float, default=0.65)
    p.add_argument("--schedule-reorder-prob", type=float, default=0.50)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(ARGS.gpu))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import sionna as sn
from sionna.channel import ApplyOFDMChannel, cir_to_ofdm_channel, subcarrier_frequencies
from sionna.channel.tr38901 import TDL
from sionna.utils import ebnodb2no

sys.path.insert(0, "..")
from utils import Parameters, E2E_Model
from utils.channel_models import ue_correlation_matrix, gnb_correlation_matrix


NUM_TX = 2
NUM_RX_ANT = 4


class TemporalTrainingDataGenerator:
    """Generate [batch, time, ...] TBs with persistent physical UE identities."""

    def __init__(
        self,
        sys_parameters,
        e2e_model,
        ue_pool_size=NUM_TX,
        dynamic_scheduling=False,
        schedule_switch_prob=0.65,
        schedule_reorder_prob=0.50,
    ):
        self.p = sys_parameters
        self.tx = e2e_model._transmitters[0]
        self.source = e2e_model._source
        self.rx = e2e_model._receiver
        self.rg = self.tx.resource_grid
        self.freqs = subcarrier_frequencies(
            self.rg.fft_size, self.rg.subcarrier_spacing)
        self.apply_channel = ApplyOFDMChannel()

        self.num_scheduled_tx = int(sys_parameters.max_num_tx)
        self.ue_pool_size = int(ue_pool_size)
        self.dynamic_scheduling = bool(dynamic_scheduling)
        self.schedule_switch_prob = float(schedule_switch_prob)
        self.schedule_reorder_prob = float(schedule_reorder_prob)

        if self.ue_pool_size < self.num_scheduled_tx:
            raise ValueError(
                "ue_pool_size must be >= the receiver's max_num_tx "
                f"({self.num_scheduled_tx})"
            )
        for name, value in [
            ("schedule_switch_prob", self.schedule_switch_prob),
            ("schedule_reorder_prob", self.schedule_reorder_prob),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        # Preserve the original DoubleTDL-low channel family.  For pools larger
        # than two physical UEs, independent TDL realizations alternate between
        # the same B100/400-Hz and C300/100-Hz profiles used by the two-UE setup.
        fc = sys_parameters.carrier_frequency
        tx_corr = ue_correlation_matrix(sys_parameters.num_antenna_ports, 0)
        rx_corr = gnb_correlation_matrix(NUM_RX_ANT, 0)
        profiles = [
            ("B100", 100e-9, 400),
            ("C300", 300e-9, 100),
        ]
        self.tdls = []
        self.ue_profiles = []
        for ue_id in range(self.ue_pool_size):
            model, delay_spread, doppler_hz = profiles[ue_id % len(profiles)]
            self.tdls.append(
                TDL(
                    model,
                    delay_spread,
                    fc,
                    max_speed=doppler_hz * sn.SPEED_OF_LIGHT / fc,
                    num_tx_ant=sys_parameters.num_antenna_ports,
                    num_rx_ant=NUM_RX_ANT,
                    rx_corr_mat=rx_corr,
                    tx_corr_mat=tx_corr,
                )
            )
            self.ue_profiles.append({
                "ue_id": ue_id,
                "model": model,
                "delay_spread_s": delay_spread,
                "doppler_hz": doppler_hz,
            })

    def sample_schedule(self, batch_size, seq_len):
        """Return stable physical UE IDs for each receiver position.

        Shape is [batch, time, max_num_tx].  Within one TB, IDs are unique.
        Dynamic scheduling changes at most one user at a time and optionally
        reorders positions, making identity routing observable during training.
        """
        batch_size = int(batch_size)
        seq_len = int(seq_len)
        n = self.num_scheduled_tx
        pool = self.ue_pool_size

        if not self.dynamic_scheduling or pool == n:
            fixed = np.arange(n, dtype=np.int32)
            schedule = np.tile(fixed[None, None, :], [batch_size, seq_len, 1])
            if self.dynamic_scheduling and n > 1:
                # Even with no spare UEs, reordering still tests that identity
                # follows the UE rather than the current input position.
                for b in range(batch_size):
                    for t in range(1, seq_len):
                        if np.random.random() < self.schedule_reorder_prob:
                            schedule[b, t] = np.random.permutation(schedule[b, t])
            return tf.convert_to_tensor(schedule, tf.int32)

        schedule = np.empty((batch_size, seq_len, n), dtype=np.int32)
        universe = np.arange(pool, dtype=np.int32)

        for b in range(batch_size):
            current = np.random.choice(universe, size=n, replace=False)
            schedule[b, 0] = current

            for t in range(1, seq_len):
                current = current.copy()

                if np.random.random() < self.schedule_switch_prob:
                    # Replace one scheduled UE with one currently absent UE.
                    replace_pos = int(np.random.randint(0, n))
                    absent = np.setdiff1d(universe, current, assume_unique=False)
                    if len(absent):
                        current[replace_pos] = np.random.choice(absent)

                if n > 1 and np.random.random() < self.schedule_reorder_prob:
                    current = np.random.permutation(current)

                if len(np.unique(current)) != n:
                    raise RuntimeError("Scheduler generated duplicate physical UE IDs")
                schedule[b, t] = current

        return tf.convert_to_tensor(schedule, tf.int32)

    def sample_pool_channel(self, batch_size, seq_len):
        """Draw a continuous TDL trajectory for every physical UE in the pool.

        Returns
        -------
        tf.Tensor
            [batch, time, num_rx, num_rx_ant, ue_pool, num_tx_ant,
             num_ofdm_symbols, fft_size]
        """
        num_sym = self.rg.num_ofdm_symbols
        sampling_frequency = 1.0 / self.rg.ofdm_symbol_duration

        h_users = []
        for tdl in self.tdls:
            # One call spans the entire sequence for this physical UE.
            a, tau = tdl(batch_size, seq_len * num_sym, sampling_frequency)
            h_user = cir_to_ofdm_channel(
                self.freqs, a, tau, normalize=False)
            h_users.append(h_user)

        # The TDL user axis is axis=3 in Neural RX's channel convention.
        h_pool = tf.concat(h_users, axis=3)
        slots = tf.split(h_pool, seq_len, axis=5)
        return tf.stack(slots, axis=1)

    def gather_scheduled_channel(self, h_pool, ue_ids):
        """Map physical-UE channel trajectories into current receiver positions."""
        seq_len = ue_ids.shape[1]
        if seq_len is None:
            raise ValueError("seq_len must be statically known for sequence training")

        gathered = []
        for t in range(seq_len):
            # h_pool[:,t] shape:
            # [B, num_rx, num_rx_ant, ue_pool, num_tx_ant, sym, fft]
            # batch_dims=1 applies each batch element's own schedule.
            h_t = tf.gather(
                h_pool[:, t],
                ue_ids[:, t],
                axis=3,
                batch_dims=1,
            )
            gathered.append(h_t)
        return tf.stack(gathered, axis=1)

    def sample_sequence_channel(self, batch_size, seq_len, ue_ids=None):
        """Backward-compatible helper returning channels in scheduled positions."""
        if ue_ids is None:
            ue_ids = self.sample_schedule(batch_size, seq_len)
        h_pool = self.sample_pool_channel(batch_size, seq_len)
        return self.gather_scheduled_channel(h_pool, ue_ids)

    def ebno_to_no(self, ebno_db):
        return ebnodb2no(
            ebno_db,
            self.tx._num_bits_per_symbol,
            self.tx._target_coderate,
            self.tx.resource_grid,
        )

    def sample_batch(
        self,
        batch_size,
        seq_len,
        ebno_db,
        ue_ids=None,
        include_pool_channel=False,
    ):
        """Generate a full temporal training batch.

        ``ue_ids`` may be supplied by a correctness test to force a known
        schedule.  Otherwise it is sampled according to the configured scheduler.
        """
        if ue_ids is None:
            ue_ids = self.sample_schedule(batch_size, seq_len)
        else:
            ue_ids = tf.cast(ue_ids, tf.int32)
            expected = [int(batch_size), int(seq_len), self.num_scheduled_tx]
            if list(ue_ids.shape) != expected:
                raise ValueError(
                    f"ue_ids must have shape {expected}, got {list(ue_ids.shape)}"
                )
            tf.debugging.assert_greater_equal(ue_ids, 0)
            tf.debugging.assert_less(ue_ids, self.ue_pool_size)

        h_pool = self.sample_pool_channel(batch_size, seq_len)
        h = self.gather_scheduled_channel(h_pool, ue_ids)
        no = self.ebno_to_no(ebno_db)

        bits = []
        received = []
        ls_estimates = []
        active = []

        for t in range(seq_len):
            h_t = h[:, t]
            b_t = self.source([
                batch_size,
                self.p.max_num_tx,
                self.tx._tb_size,
            ])
            x_t = self.tx(b_t)
            # Fresh independent AWGN is drawn on every TB.
            y_t = self.apply_channel([x_t, h_t, no])
            ls_t = self.rx.estimate_channel(y_t, self.p.max_num_tx)
            active_t = tf.ones(
                [batch_size, self.p.max_num_tx], dtype=tf.float32)

            bits.append(b_t)
            received.append(y_t)
            ls_estimates.append(ls_t)
            active.append(active_t)

        result = {
            "bits": tf.stack(bits, axis=1),
            "y": tf.stack(received, axis=1),
            "ls": tf.stack(ls_estimates, axis=1),
            "h": h,
            "active": tf.stack(active, axis=1),
            "ue_ids": ue_ids,
            "no": no,
        }
        if include_pool_channel:
            result["h_pool"] = h_pool
        return result


def _complex_corr(a, b):
    a = np.asarray(a).reshape(a.shape[0], -1)
    b = np.asarray(b).reshape(b.shape[0], -1)
    num = np.abs(np.sum(np.conj(a) * b, axis=1))
    den = np.sqrt(
        np.sum(np.abs(a) ** 2, axis=1) *
        np.sum(np.abs(b) ** 2, axis=1)
    )
    return float(np.mean(num / np.maximum(den, 1e-12)))


def validate_batch(batch, dynamic_scheduling=False):
    # For dynamic schedules, validate temporal correlation on the physical pool,
    # not on input positions whose UE identities may change.
    channel_key = "h_pool" if "h_pool" in batch else "h"
    h = batch[channel_key].numpy()
    bits = batch["bits"].numpy()
    ue_ids = batch["ue_ids"].numpy()
    seq_len = h.shape[1]

    adjacent = [_complex_corr(h[:, t], h[:, t + 1])
                for t in range(seq_len - 1)]
    lag2 = [_complex_corr(h[:, t], h[:, t + 2])
            for t in range(seq_len - 2)] if seq_len > 2 else []

    bit_disagreement = [
        float(np.mean(bits[:, t] != bits[:, t + 1]))
        for t in range(seq_len - 1)
    ]

    schedule_changes = [
        float(np.mean(np.any(ue_ids[:, t] != ue_ids[:, t + 1], axis=-1)))
        for t in range(seq_len - 1)
    ]
    unique_per_tb = all(
        np.all([
            len(np.unique(ue_ids[b, t])) == ue_ids.shape[-1]
            for b in range(ue_ids.shape[0])
        ])
        for t in range(ue_ids.shape[1])
    )

    checks = {
        "nearby_channels_related": bool(np.mean(adjacent) > 0.5),
        "payloads_independent": bool(0.45 < np.mean(bit_disagreement) < 0.55),
        "unique_physical_ues_per_tb": bool(unique_per_tb),
    }
    if dynamic_scheduling and seq_len > 1:
        checks["schedule_changes_present"] = bool(np.mean(schedule_changes) > 0.0)

    summary = {
        "bits_shape": list(batch["bits"].shape),
        "y_shape": list(batch["y"].shape),
        "ls_shape": list(batch["ls"].shape),
        "h_shape": list(batch["h"].shape),
        "ue_ids_shape": list(batch["ue_ids"].shape),
        "ue_ids_example": ue_ids[0].tolist(),
        "active_shape": list(batch["active"].shape),
        "adjacent_channel_corr_mean": float(np.mean(adjacent)),
        "adjacent_channel_corr": adjacent,
        "lag2_channel_corr_mean": float(np.mean(lag2)) if lag2 else None,
        "first_last_channel_corr": _complex_corr(h[:, 0], h[:, -1]),
        "adjacent_payload_bit_disagreement_mean": float(np.mean(bit_disagreement)),
        "adjacent_payload_bit_disagreement": bit_disagreement,
        "schedule_change_fraction_mean": (
            float(np.mean(schedule_changes)) if schedule_changes else 0.0
        ),
        "schedule_change_fraction": schedule_changes,
        "checks": checks,
    }
    return summary


def main():
    np.random.seed(ARGS.seed)
    tf.random.set_seed(ARGS.seed)
    try:
        sn.config.seed = ARGS.seed
    except Exception:
        pass

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    p = Parameters(
        ARGS.config,
        training=False,
        num_tx_eval=NUM_TX,
        system="nrx",
    )
    e2e = E2E_Model(p, training=False, mcs_arr_eval_idx=0)
    generator = TemporalTrainingDataGenerator(
        p,
        e2e,
        ue_pool_size=ARGS.ue_pool_size,
        dynamic_scheduling=ARGS.dynamic_scheduling,
        schedule_switch_prob=ARGS.schedule_switch_prob,
        schedule_reorder_prob=ARGS.schedule_reorder_prob,
    )
    batch = generator.sample_batch(
        batch_size=ARGS.batch_size,
        seq_len=ARGS.seq_len,
        ebno_db=ARGS.ebno_db,
        include_pool_channel=True,
    )

    summary = validate_batch(
        batch, dynamic_scheduling=ARGS.dynamic_scheduling)
    summary["ue_pool_size"] = ARGS.ue_pool_size
    summary["ue_profiles"] = generator.ue_profiles
    print("TEMPORAL_DATA_SUMMARY=" + json.dumps(summary, indent=2))

    if ARGS.output_json:
        path = Path(ARGS.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")

    if ARGS.validate:
        if not all(summary["checks"].values()):
            raise SystemExit("Temporal data validation failed")
        print("VALIDATION_PASSED")


if __name__ == "__main__":
    main()
