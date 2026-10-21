from __future__ import annotations

import numpy as np
import tensorflow as tf

from .channel_filter import EPS, power, complex_channel
from .native_ops import load


def select_partners(channel, variance, noise, active, identities):
    users = channel.shape[-1]
    reference = tf.maximum(tf.reduce_mean(tf.reduce_sum(power(channel), -1), -1)+noise[:, None], 1e-30)
    diagonal = tf.maximum(noise[:, None, None]+tf.reduce_sum(variance, -1), EPS*reference[..., None])
    whitened = channel/tf.cast(tf.sqrt(diagonal[..., None]), channel.dtype)
    gram = tf.linalg.matmul(whitened, whitened, adjoint_a=True)
    eligible = (tf.cast(active[:, None, :, None], tf.bool)
                & tf.cast(active[:, None, None, :], tf.bool) & ~tf.eye(users, dtype=tf.bool))
    strength = tf.where(eligible, power(gram), -1.)
    tied = eligible & (strength == tf.reduce_max(strength, -1, keepdims=True))
    ids = tf.broadcast_to(identities[:, None, None, :], tf.shape(strength))
    choice = tf.argmin(tf.where(tied, ids, tf.int32.max), -1, output_type=tf.int32)
    return tf.where(tf.reduce_any(eligible, -1), choice, -1)


class DecoderCorrection:
    def __init__(self, mapper, bits):
        labels = ((np.arange(2**bits)[:, None] >> np.arange(bits-1, -1, -1)) & 1).astype(np.int32)
        points = np.asarray(mapper(tf.constant(labels, tf.float32))).reshape(-1)
        if len(points) != len(labels) or len(np.unique(points)) != len(points):
            raise ValueError('Symbol labels must map one-to-one onto the constellation')
        self.symbol_energy = float(np.mean(np.abs(points)**2))
        if not np.isfinite(self.symbol_energy) or self.symbol_energy <= 0:
            raise ValueError('Constellation energy must be positive')
        points = points/np.sqrt(self.symbol_energy)
        self.symbols = tf.constant(points, tf.complex64)
        with tf.device(self.symbols.device):
            self.labels = tf.Variable(labels, trainable=False, dtype=tf.int64)
        self.bits = bits
        energies = np.abs(points.astype(np.complex128))**2
        levels, indices = [], []
        for value in energies:
            match = next((i for i, level in enumerate(levels) if np.isclose(value, level, rtol=4*EPS, atol=0.)), None)
            if match is None:
                match = len(levels); levels.append(float(value))
            indices.append(match)
        self.energies = np.asarray(levels, np.float32)
        self.energy_levels = tf.constant(self.energies)
        with tf.device(self.symbols.device):
            self.energy_indices = tf.Variable(indices, trainable=False, dtype=tf.int64)
        self.energy = tf.constant(np.abs(points)**2, tf.float32)
        self.native = load()
        self._tables = {}
        self._bit_tables = {}

    def _matrix_and_rhs(self, coefficients, covariance, noise, background):
        antennas = coefficients.shape[-2]
        eye = tf.eye(antennas, dtype=tf.complex64)
        ea, eb = np.meshgrid(self.energies, self.energies, indexing='ij')
        ca, cb = covariance[:, :, 0], covariance[:, :, 1]
        reference = tf.maximum(tf.reduce_mean(tf.reduce_sum(power(coefficients[..., 0]), 2), -1)+noise[:, None], 1e-30)
        floor = 8*antennas*EPS*reference
        diagonal = tf.maximum(noise[:, None], floor)
        matrix = tf.cast(self.symbol_energy, tf.complex64)*(
            tf.cast(ea.reshape(-1)[None, None, :, None, None], tf.complex64)*ca[:, :, None]+
            tf.cast(eb.reshape(-1)[None, None, :, None, None], tf.complex64)*cb[:, :, None])
        matrix += tf.cast(diagonal[..., None, None, None], tf.complex64)*eye
        if background.shape.num_elements() != 0:
            matrix += background[:, :, None]
        theta = tf.transpose(coefficients, [0, 1, 3, 2, 4])
        theta = tf.reshape(theta, [tf.shape(theta)[0], tf.shape(theta)[1], antennas, 6])
        rhs = tf.cast(np.sqrt(self.symbol_energy), tf.complex64)*theta[:, :, None]
        batch_shape = tf.broadcast_dynamic_shape(tf.shape(matrix)[:-2], tf.shape(rhs)[:-2])
        rhs = tf.broadcast_to(rhs, tf.concat([batch_shape, tf.shape(rhs)[-2:]], 0))
        return matrix, rhs

    def prepare(self, coefficients, covariance, noise, background):
        background = tf.zeros([0], tf.complex64) if background is None else background
        @tf.custom_gradient
        def operation(coefficients, covariance, noise, background):
            factor, white, logdet = self.native.pilot_pair_prepare(
                coefficients, covariance, noise, background, self.energy_levels, symbol_energy=self.symbol_energy)
            def gradient(gfactor, gwhite, glogdet):
                grhs = tf.linalg.triangular_solve(factor, gwhite, adjoint=True)
                gfactor -= tf.linalg.band_part(tf.linalg.matmul(grhs, white, adjoint_b=True), -1, 0)
                gfactor += tf.linalg.diag(tf.cast(2*glogdet[..., None]/tf.math.real(tf.linalg.diag_part(factor)), tf.complex64))
                identity = tf.eye(tf.shape(factor)[-1], batch_shape=tf.shape(factor)[:-2], dtype=factor.dtype)
                inverse = tf.linalg.triangular_solve(factor, identity)
                middle = tf.linalg.matmul(factor, gfactor, adjoint_a=True)
                middle = tf.linalg.band_part(tf.linalg.set_diag(middle, .5*tf.linalg.diag_part(middle)), -1, 0)
                gmatrix = tf.linalg.matmul(tf.linalg.matmul(inverse, middle, adjoint_a=True), inverse)
                gmatrix = .5*(gmatrix+tf.linalg.adjoint(gmatrix))
                values = [coefficients, covariance, noise, background]
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(values)
                    matrix, rhs = self._matrix_and_rhs(*values)
                return tape.gradient((matrix, rhs), values, output_gradients=(gmatrix, grhs),
                                     unconnected_gradients=tf.UnconnectedGradients.ZERO)
            return (factor, white, logdet), gradient
        return operation(coefficients, covariance, noise, background)

    def tables(self, layout, device):
        key = (id(layout), device)
        if key not in self._tables:
            with tf.init_scope(), tf.device(device):
                users, n = layout.users, len(layout.code_positions)
                positions = np.full((len(layout.cell_positions), max(map(len, layout.cell_positions))), -1, np.int32)
                for cell, rows in enumerate(layout.cell_positions):
                    positions[cell, :len(rows)] = rows
                values = (layout.code_positions, layout.code_cells, positions, np.arange(n, dtype=np.int32),
                          np.arange(n, dtype=np.int32)[:, None])
                indices = tuple(tf.Variable(x, trainable=False, dtype=tf.int64) for x in values)
                phi = layout.code_phi
                if users == 1:
                    phi = np.concatenate([phi, np.zeros_like(phi)], 1)
                self._tables[key] = (layout, (*indices, tf.constant(phi, tf.float32)))
        return self._tables[key][1]

    def score(self, received, prepared, phi, cells, enabled, single_user=False):
        return tuple(tf.stop_gradient(value) for value in self.native.pilot_pair_scores(
            received, *prepared, phi, cells, self.symbols, self.energy_indices,
            self.labels, enabled, single_user=single_user))

    def begin(self, received, estimate, noise, active, identities, llr, layout):
        batch, users = tf.shape(llr)[0], layout.users
        n, antennas = len(layout.code_positions), received.shape[-1]//2
        code_positions, cells, positions, pair_cells, pair_positions, phi = self.tables(layout, self.symbols.device)
        with tf.control_dependencies([llr]):
            y = tf.gather(tf.reshape(complex_channel(received), [batch, -1, antennas]), code_positions, axis=1)
            theta, covariance = layout.joint(estimate.coefficients), layout.joint(estimate.covariance)
            theta = tf.where(tf.cast(active[:, None, :, None, None], tf.bool), theta, 0.)
            covariance = tf.where(tf.cast(active[:, None, :, None, None], tf.bool), covariance, 0.)
            noise = tf.where(tf.reduce_any(tf.cast(active, tf.bool), -1), noise, 1.)
            if users <= 2:
                wanted = tf.broadcast_to(tf.cast(active[..., None], tf.bool), [batch, users, n])
                if users == 1:
                    theta = tf.concat([theta, tf.zeros_like(theta)], 2)
                    covariance = tf.concat([covariance, tf.zeros_like(covariance)], 2)
                    wanted = tf.concat([wanted, tf.zeros_like(wanted)], 1)
                prepared = self.prepare(theta, covariance, noise, None)
                enabled = tf.ones(tf.shape(theta)[:2], tf.bool)
                scores = self.score(y, prepared, phi, cells, enabled, users == 1)
                return [(0, 0 if users == 1 else 1, y, prepared, enabled, wanted, phi, cells, positions, scores)]
            h = tf.transpose(theta[..., 0], [0, 1, 3, 2])*tf.cast(np.sqrt(self.symbol_energy), tf.complex64)
            v = tf.transpose(tf.math.real(tf.linalg.diag_part(covariance)), [0, 1, 3, 2])*self.symbol_energy
            selected = select_partners(h, v, noise, active, identities)
            partners = tf.gather(selected, cells, axis=1)
            theta = tf.gather(theta, cells, axis=1)
            covariance = tf.gather(covariance, cells, axis=1)
            h = tf.reduce_sum(theta*tf.cast(phi[None, ..., None, :], tf.complex64), -1)*tf.cast(np.sqrt(self.symbol_energy), tf.complex64)
            logits = tf.transpose(tf.reshape(llr, [batch, users, n, self.bits]), [0, 2, 1, 3])
            probability = tf.nn.softmax(tf.einsum('bnuq,mq->bnum', logits, tf.cast(self.labels, tf.float32)), -1)
            mean = tf.reduce_sum(tf.cast(probability, tf.complex64)*self.symbols, -1)
            variance = tf.reduce_sum(probability*power(self.symbols-mean[..., None]), -1)
            energy = tf.reduce_sum(probability*self.energy, -1)
            pairs = []
            for u in range(users):
                for v in range(u, users):
                    left = (partners[..., u] == (-1 if u == v else v)) & tf.cast(active[:, u, None], tf.bool)
                    right = tf.zeros_like(left) if u == v else (partners[..., v] == u) & tf.cast(active[:, v, None], tf.bool)
                    enabled = left | right
                    pair_phi = tf.stack([phi[:, u], tf.zeros_like(phi[:, u]) if u == v else phi[:, v]], 1)
                    def process():
                        mask = active[:, None, :]*(1.-tf.one_hot(u, users))
                        if u != v:
                            mask *= 1.-tf.one_hot(v, users)
                        background_h = h*tf.cast(mask[..., None], tf.complex64)
                        residual = y-tf.reduce_sum(background_h*mean[..., None], 2)
                        background = tf.einsum('bnur,bnus,bnu->bnrs', background_h, tf.math.conj(background_h), tf.cast(variance, tf.complex64))
                        background += tf.reduce_sum(covariance*tf.cast((self.symbol_energy*mask*energy)[..., None, None], tf.complex64), 2)
                        pair_theta = tf.stack([theta[:, :, u], tf.zeros_like(theta[:, :, u]) if u == v else theta[:, :, v]], 2)
                        pair_cov = tf.stack([covariance[:, :, u], tf.zeros_like(covariance[:, :, u]) if u == v else covariance[:, :, v]], 2)
                        prepared = self.prepare(pair_theta, pair_cov, noise, background)
                        scores = self.score(residual, prepared, pair_phi, pair_cells, enabled, u == v)
                        return residual, prepared, scores
                    groups = len(self.energies)**2
                    def empty():
                        m = 2**self.bits
                        return (tf.zeros([0, n, antennas], tf.complex64),
                                (tf.zeros([0, n, groups, antennas, antennas], tf.complex64),
                                 tf.zeros([0, n, groups, antennas, 6], tf.complex64), tf.zeros([0, n, groups])),
                                (tf.zeros([0, n, m*(1 if u == v else m)+self.bits*(1 if u == v else 2)]),
                                 tf.zeros([0, n])))
                    residual, prepared, scores = tf.cond(tf.reduce_any(enabled), process, empty)
                    pairs.append((u, v, residual, prepared, enabled, tf.stack([left, right], 1),
                                  pair_phi, pair_cells, pair_positions, scores))
            return pairs

    def finish(self, pairs, evidence, active, bit_indices, bit_signs, effective_input, calibration, llr_limit,
               strength, blend):
        key = bit_indices.ref()
        if key not in self._bit_tables:
            with tf.init_scope(), tf.device(self.symbols.device):
                self._bit_tables[key] = tf.Variable(tf.cast(bit_indices, tf.int64), trainable=False)
        bit_indices = self._bit_tables[key]
        users = active.shape[-1]
        fused = users <= 2
        if not fused:
            evidence = (evidence-effective_input)*active[..., None, None]
        result = tf.zeros_like(evidence)
        for u, v, received, prepared, enabled, wanted, phi, cells, positions, scores in pairs:
            def correct():
                if users == 2:
                    e, base, indices, signs, activity = evidence, effective_input, bit_indices, bit_signs, active
                else:
                    e = tf.stack([evidence[:, u], tf.zeros_like(evidence[:, u]) if u == v else evidence[:, v]], 1)
                    base = tf.stack([effective_input[:, u], tf.zeros_like(effective_input[:, u]) if u == v
                                     else effective_input[:, v]], 1)
                    indices, signs, activity = (tf.gather(bit_indices, [u, v]), tf.gather(bit_signs, [u, v]),
                                                tf.gather(active, [u, v], axis=1))
                delta = self.native.pilot_pair_correction(received, *prepared, e, phi, cells, positions,
                    self.symbols, self.energy_indices, self.labels, enabled, indices, signs, wanted, base, calibration,
                    strength, activity, *scores, *blend,
                    single_user=(u == v), llr_limit=llr_limit, apply_update=fused)
                if users <= 2:
                    return delta[:, :users]
                return (delta[:, 0, None]*tf.one_hot(u, users)[None, :, None, None]+
                        delta[:, 1, None]*tf.one_hot(v, users)[None, :, None, None])
            contribution = correct() if users <= 2 else tf.cond(tf.reduce_any(enabled), correct, lambda: tf.zeros_like(result))
            result = contribution if users <= 2 else result+contribution
        if fused:
            return result
        result *= active[..., None, None]
        return effective_input+result

    def __call__(self, received, estimate, noise, active, identities, llr, evidence, layout, bit_indices, bit_signs,
                 effective_input, calibration, llr_limit, strength, blend):
        return self.finish(self.begin(received, estimate, noise, active, identities, llr, layout), evidence, active,
                           bit_indices, bit_signs, effective_input, calibration, llr_limit, strength, blend)
