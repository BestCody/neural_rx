from __future__ import annotations

import tensorflow as tf
from sionna.utils import flatten_last_dims


def demap_llr(ofdm, llr_grid, num_tx, mcs_idx=0):
    llr = tf.cast(llr_grid, tf.float32)
    llr = tf.transpose(llr, [0, 1, 3, 2, 4])
    llr = tf.expand_dims(llr, axis=1)
    llr = ofdm._rg_demapper(llr)
    llr = llr[:, :num_tx]
    llr = flatten_last_dims(llr, 2)
    if ofdm._layer_demappers is None:
        return tf.squeeze(llr, axis=-2)
    return ofdm._layer_demappers[mcs_idx](llr)


def received_to_grid(received) -> tf.Tensor:
    received = tf.convert_to_tensor(received)
    if received.shape.rank != 5 or received.shape[1] != 1:
        raise ValueError("received tensor must have shape [B,1,R,S,F]")
    return tf.transpose(tf.squeeze(received, axis=1), [0, 3, 2, 1])


def exact_contributions(physical_channel, transmitted) -> tf.Tensor:
    physical_channel = tf.convert_to_tensor(physical_channel)
    transmitted = tf.cast(transmitted, physical_channel.dtype)
    if physical_channel.shape.rank != 7:
        raise ValueError("physical channel must have rank seven")
    if transmitted.shape.rank != 5:
        raise ValueError("transmitted grid must have rank five")
    return tf.einsum(
        "bqruasf,buasf->bufsr", physical_channel, transmitted
    )

