from __future__ import annotations

from typing import NamedTuple
import tensorflow as tf


EPS = 1.1920928955078125e-7


def power(value):
    return tf.square(tf.math.real(value))+tf.square(tf.math.imag(value))


def complex_channel(value):
    real, imag = tf.split(value, 2, -1)
    return tf.complex(real, imag)


class ChannelEstimate(NamedTuple):
    coefficients: tf.Tensor
    covariance: tf.Tensor

    @property
    def variance(self):
        return tf.math.real(tf.linalg.diag_part(self.covariance))


class FilterContext(NamedTuple):
    coefficients: tf.Tensor
    beta: tf.Tensor
    weights: tf.Tensor
    innovation: tf.Tensor
    h0: tf.Tensor
    q0: tf.Tensor
    qj: tf.Tensor
    p0: tf.Tensor
    reference: tf.Tensor
    floor: tf.Tensor
    active: tf.Tensor


class ChannelMemory(tf.keras.layers.Layer):
    def __init__(self, filter_hidden=16, error_hidden=16, **kwargs):
        super().__init__(**kwargs)
        if any(type(n) is not int or n < 1 for n in (filter_hidden, error_hidden)):
            raise ValueError("Channel filter widths must be positive integers")
        self.filter = tf.keras.Sequential([
            tf.keras.layers.Dense(filter_hidden, activation="relu"),
            tf.keras.layers.Dense(7, kernel_initializer="zeros")], name="pilot_filter")
        self.error = tf.keras.Sequential([
            tf.keras.layers.Dense(error_hidden, activation="relu"),
            tf.keras.layers.Dense(3, kernel_initializer="zeros")], name="channel_error")

    def build(self, input_shape):
        self.filter.build((None, 13))
        self.error.build((None, 9))
        super().build(input_shape)

    def call(self, observation, history, active, layout, defer_uncertainty=False):
        active = tf.cast(active, tf.bool)
        valid = history.valid & active[..., None]
        ports = tf.broadcast_to(tf.range(layout.users)[None, :, None], [tf.shape(observation)[0], layout.users, 1])
        records = tf.concat([observation[:, :, None], tf.where(valid[..., None, None], history.observations, 0.)], 2)
        ports = tf.concat([ports, history.ports], 2)
        ages = tf.concat([tf.zeros_like(history.age[..., :1]), history.age], 2)
        valid = tf.concat([active[..., None], valid], 2)
        old, local_valid, delta = layout.neighbors(records, ports)
        central = tf.reduce_all(delta == 0., -1)
        is_current = tf.range(tf.shape(records)[2])[None, None, :, None, None] == 0
        allowed = valid[..., None, None] & local_valid & ~(is_current & central)
        old = tf.where(allowed[..., None], old, 0.)
        def flatten(value):
            order = [0, 1, 3, 2, 4]+list(range(5, value.shape.rank))
            value = tf.transpose(value, order)
            return tf.reshape(value, [tf.shape(value)[0], layout.users, layout.count, -1, *value.shape[5:]])
        old, allowed = flatten(old), flatten(allowed)
        h0, hj = complex_channel(observation[..., :-1]), complex_channel(old[..., :-1])
        q0, qj = tf.maximum(observation[..., -1], 0.), tf.maximum(old[..., -1], 0.)
        p0, pj = tf.reduce_mean(power(h0), -1), tf.reduce_mean(power(hj), -1)
        reference = tf.maximum(p0+q0, 1e-30)
        floor = EPS*reference
        correlation = tf.reduce_mean(tf.math.conj(h0[..., None, :])*hj, -1)
        correlation /= tf.cast(tf.sqrt(tf.maximum(p0[..., None], floor[..., None])*
                                      tf.maximum(pj, floor[..., None])), tf.complex64)
        innovation = tf.where(allowed[..., None], hj-h0[..., None, :], 0.)
        disagreement = tf.reduce_mean(power(innovation), -1)
        age_ms = tf.cast(ages, tf.float32)*layout.tb_seconds*1000.
        coordinates = tf.concat([tf.broadcast_to(tf.math.log1p(age_ms)[..., None, None, None], tf.shape(delta[..., :1])),
                                 delta[..., :1]-age_ms[..., None, None, None], delta[..., 1:]], -1)
        coordinates = flatten(coordinates)
        features = tf.stack([tf.math.real(correlation), tf.math.imag(correlation),
            tf.math.log(tf.maximum(pj+qj, floor[..., None])/reference[..., None]),
            tf.broadcast_to((q0/reference)[..., None], tf.shape(qj)),
            qj/tf.maximum(pj+qj, floor[..., None]),
            disagreement/tf.maximum(reference[..., None]+pj+qj, floor[..., None])], -1)
        geometry = tf.broadcast_to(layout.geometry_features[None, ..., None, :],
                                   tf.concat([tf.shape(features)[:-1], [4]], 0))
        scores = self.filter(tf.concat([features, coordinates, geometry], -1))
        logits = tf.where(allowed, scores[..., 0], -1e9)
        weights = tf.nn.softmax(tf.concat([logits, tf.zeros_like(logits[..., :1])], -1), -1)[..., :-1]
        beta = tf.cast(weights[..., None], tf.complex64)*tf.complex(tf.tanh(scores[..., 1::2]), tf.tanh(scores[..., 2::2]))
        coefficients = tf.einsum('bucjk,bucjr->bucrk', beta, innovation)
        coefficients += h0[..., None]*tf.constant([1., 0., 0.], tf.complex64)
        context = FilterContext(coefficients, beta, weights, innovation, h0, q0, qj, p0, reference, floor, active)
        if defer_uncertainty:
            return context
        covariance = self.uncertainty(context, layout)
        return ChannelEstimate(tf.where(active[..., None, None, None], coefficients, 0.), covariance)

    def uncertainty(self, context, layout, parameters=None):
        coefficients, beta, weights, innovation, h0, q0, qj, p0, reference, floor, active = context
        d0 = tf.constant([1., 0., 0.], tf.complex64)-tf.reduce_sum(beta, -2)
        factor = tf.cast(layout.r_layout, tf.complex64)
        nominal = q0*tf.reduce_sum(power(tf.einsum('ucik,buck->buci', factor, d0)), -1)
        nominal += tf.reduce_sum(qj*tf.reduce_sum(power(tf.einsum('ucik,bucjk->bucji', factor, beta)), -1), -1)
        observed = tf.einsum('bucj,bucjr,bucjs->bucrs', tf.cast(weights, tf.complex64), innovation, tf.math.conj(innovation))
        antennas = h0.shape[-1]
        observed += tf.cast((nominal+floor)[..., None, None], tf.complex64)*tf.eye(antennas, dtype=tf.complex64)
        diagonal = tf.math.real(tf.linalg.diag_part(observed))
        inv = tf.math.rsqrt(diagonal)
        correlation_matrix = observed*tf.cast(inv[..., :, None]*inv[..., None, :], tf.complex64)
        normalized = observed/tf.cast(reference[..., None, None], tf.complex64)
        diagonal_normalized = tf.math.real(tf.linalg.diag_part(normalized))
        total_energy = tf.reduce_sum(power(normalized), [-1, -2])
        off_energy = tf.maximum(total_energy-tf.reduce_sum(tf.square(diagonal_normalized), -1), 0.)
        off_fraction = off_energy/tf.maximum(total_energy, EPS*EPS)
        shape = tf.shape(power(h0))
        broadcast = lambda value: tf.broadcast_to(value[..., None], shape)
        slope = tf.reduce_sum(power(coefficients[..., 1:]), -1)
        effective = tf.math.divide_no_nan(tf.square(tf.reduce_sum(weights, -1)), tf.reduce_sum(tf.square(weights), -1))
        features = tf.stack([broadcast(p0/reference), broadcast(nominal/reference),
            broadcast(q0/reference), power(h0)/reference[..., None],
            (diagonal-nominal[..., None]-floor[..., None])/reference[..., None],
            slope/reference[..., None], broadcast(tf.reduce_sum(weights, -1)),
            broadcast(tf.math.log1p(effective)), broadcast(off_fraction)], -1)
        if parameters is None:
            calibration = self.error(features)
        else:
            kernel, bias, output_kernel, output_bias = parameters
            calibration = tf.matmul(tf.nn.relu(tf.matmul(features, kernel)+bias), output_kernel)+output_bias
        positive = tf.nn.softplus(calibration[..., :2])
        variance = positive[..., 0]*nominal[..., None]+positive[..., 1]*reference[..., None]+floor[..., None]
        rho = tf.sigmoid(tf.reduce_mean(calibration[..., 2], -1))
        identity = tf.eye(antennas, dtype=tf.complex64)
        matrix = tf.cast(rho[..., None, None], tf.complex64)*correlation_matrix+tf.cast((1.-rho)[..., None, None], tf.complex64)*identity
        scale = tf.sqrt(variance)
        covariance = matrix*tf.cast(scale[..., :, None]*scale[..., None, :], tf.complex64)
        return tf.where(active[..., None, None, None], covariance, 0.)

    def complete(self, context, layout):
        from .native_ops import load
        native = load()
        parameters = tuple(self.error.weights)
        @tf.custom_gradient
        def operation(*values):
            current, parameters = FilterContext(*values[:11]), values[11:]
            covariance = native.pilot_channel_uncertainty(*current, tf.constant(layout.r_layout), *parameters,
                allow_tf32=tf.config.experimental.tensor_float_32_execution_enabled())
            def gradient(incoming):
                learned = (*current[:3], *parameters)
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(learned)
                    result = self.uncertainty(current, layout, parameters)
                grads = tape.gradient(result, learned, output_gradients=incoming,
                                      unconnected_gradients=tf.UnconnectedGradients.ZERO)
                return (*grads[:3], *([None]*8), *grads[3:])
            return covariance, gradient
        covariance = operation(*context, *parameters)
        return ChannelEstimate(tf.where(context.active[..., None, None, None], context.coefficients, 0.), covariance)
