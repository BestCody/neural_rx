#!/usr/bin/env python3
"""Generate correlated NRX training TB sequences over a continuous TDL channel.

This module keeps the normal NRX ingredients (PUSCH transmitter, LS estimate,
noise model, tensor shapes) but changes the sampling unit from one independent
TB to a short ordered sequence of TBs for the same physical UEs.

The payload bits and AWGN are newly sampled for every TB. The channel is drawn
once over the full sequence time axis and then split into consecutive slots, so
nearby TBs are naturally correlated through Doppler evolution.

Run from neural_rx/scripts, e.g.:
    python temporal_training_data.py --validate --batch-size 16 --seq-len 8
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
    """Generate batches shaped as [batch, time, ...] with correlated channels."""

    def __init__(self, sys_parameters, e2e_model):
        self.p = sys_parameters
        self.tx = e2e_model._transmitters[0]
        self.source = e2e_model._source
        self.rx = e2e_model._receiver
        self.rg = self.tx.resource_grid
        self.freqs = subcarrier_frequencies(
            self.rg.fft_size, self.rg.subcarrier_spacing)
        self.apply_channel = ApplyOFDMChannel()

        # Match Neural RX's DoubleTDLlow two-user evaluation channel.
        fc = sys_parameters.carrier_frequency
        tx_corr = ue_correlation_matrix(sys_parameters.num_antenna_ports, 0)
        rx_corr = gnb_correlation_matrix(NUM_RX_ANT, 0)
        self.tdls = [
            TDL(
                "B100",
                100e-9,
                fc,
                max_speed=400 * sn.SPEED_OF_LIGHT / fc,
                num_tx_ant=sys_parameters.num_antenna_ports,
                num_rx_ant=NUM_RX_ANT,
                rx_corr_mat=rx_corr,
                tx_corr_mat=tx_corr,
            ),
            TDL(
                "C300",
                300e-9,
                fc,
                max_speed=100 * sn.SPEED_OF_LIGHT / fc,
                num_tx_ant=sys_parameters.num_antenna_ports,
                num_rx_ant=NUM_RX_ANT,
                rx_corr_mat=rx_corr,
                tx_corr_mat=tx_corr,
            ),
        ]

    def sample_sequence_channel(self, batch_size, seq_len):
        """Draw one continuous channel trajectory and split it into TB slots.

        Returns
        -------
        h_seq : tf.Tensor
            Shape [batch, time, num_rx, num_rx_ant, num_tx, num_tx_ant,
                   num_ofdm_symbols, fft_size].
        """
        num_sym = self.rg.num_ofdm_symbols
        sampling_frequency = 1.0 / self.rg.ofdm_symbol_duration

        h_users = []
        for tdl in self.tdls:
            # One TDL call spans the *entire* sequence. This is the key change
            # versus independently generating a new channel for every TB.
            a, tau = tdl(batch_size, seq_len * num_sym, sampling_frequency)
            h_user = cir_to_ofdm_channel(
                self.freqs, a, tau, normalize=False)
            h_users.append(h_user)

        h = tf.concat(h_users, axis=3)
        # Original time axis is seq_len*num_sym. Split it into adjacent slots,
        # then put time after batch: [B,T,...].
        slots = tf.split(h, seq_len, axis=5)
        return tf.stack(slots, axis=1)

    def ebno_to_no(self, ebno_db):
        return ebnodb2no(
            ebno_db,
            self.tx._num_bits_per_symbol,
            self.tx._target_coderate,
            self.tx.resource_grid,
        )

    def sample_batch(self, batch_size, seq_len, ebno_db):
        """Generate a full sequence batch for temporal NRX training.

        Every TB gets new random payload bits and a fresh AWGN draw. The same
        two UEs persist at the same UE indices across the sequence, while their
        TDL channel evolves continuously in time.

        Returns a dictionary whose tensors all use [batch, time, ...].
        """
        h = self.sample_sequence_channel(batch_size, seq_len)
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
            # ApplyOFDMChannel adds a fresh independent AWGN realization here.
            y_t = self.apply_channel([x_t, h_t, no])
            ls_t = self.rx.estimate_channel(y_t, self.p.max_num_tx)
            active_t = tf.ones(
                [batch_size, self.p.max_num_tx], dtype=tf.float32)

            bits.append(b_t)
            received.append(y_t)
            ls_estimates.append(ls_t)
            active.append(active_t)

        return {
            "bits": tf.stack(bits, axis=1),
            "y": tf.stack(received, axis=1),
            "ls": tf.stack(ls_estimates, axis=1),
            "h": h,
            "active": tf.stack(active, axis=1),
            "no": no,
        }


def _complex_corr(a, b):
    a = np.asarray(a).reshape(a.shape[0], -1)
    b = np.asarray(b).reshape(b.shape[0], -1)
    num = np.abs(np.sum(np.conj(a) * b, axis=1))
    den = np.sqrt(
        np.sum(np.abs(a) ** 2, axis=1) *
        np.sum(np.abs(b) ** 2, axis=1)
    )
    return float(np.mean(num / np.maximum(den, 1e-12)))


def validate_batch(batch):
    h = batch["h"].numpy()
    bits = batch["bits"].numpy()
    seq_len = h.shape[1]

    adjacent = [_complex_corr(h[:, t], h[:, t + 1])
                for t in range(seq_len - 1)]
    lag2 = [_complex_corr(h[:, t], h[:, t + 2])
            for t in range(seq_len - 2)] if seq_len > 2 else []

    # Independent Bernoulli payloads should disagree about half of the time.
    bit_disagreement = [
        float(np.mean(bits[:, t] != bits[:, t + 1]))
        for t in range(seq_len - 1)
    ]

    summary = {
        "bits_shape": list(batch["bits"].shape),
        "y_shape": list(batch["y"].shape),
        "ls_shape": list(batch["ls"].shape),
        "h_shape": list(batch["h"].shape),
        "active_shape": list(batch["active"].shape),
        "adjacent_channel_corr_mean": float(np.mean(adjacent)),
        "adjacent_channel_corr": adjacent,
        "lag2_channel_corr_mean": float(np.mean(lag2)) if lag2 else None,
        "first_last_channel_corr": _complex_corr(h[:, 0], h[:, -1]),
        "adjacent_payload_bit_disagreement_mean": float(np.mean(bit_disagreement)),
        "adjacent_payload_bit_disagreement": bit_disagreement,
        "checks": {
            "nearby_channels_related": bool(np.mean(adjacent) > 0.5),
            "payloads_independent": bool(0.45 < np.mean(bit_disagreement) < 0.55),
        },
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
    generator = TemporalTrainingDataGenerator(p, e2e)
    batch = generator.sample_batch(
        batch_size=ARGS.batch_size,
        seq_len=ARGS.seq_len,
        ebno_db=ARGS.ebno_db,
    )

    summary = validate_batch(batch)
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
