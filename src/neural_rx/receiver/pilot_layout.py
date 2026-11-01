from __future__ import annotations

import hashlib

import numpy as np
import tensorflow as tf


class PilotLayout:
    def __init__(self, estimator, resource_grid, neighbor_radius):
        mask = np.asarray(estimator._pilot_pattern.mask, bool)
        pilots = np.asarray(estimator._pilot_pattern.pilots)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("Pilot history supports one layer per scheduled UE")
        if type(estimator._interpol).__name__ != "NearestNeighborInterpolator":
            raise ValueError("Pilot packing requires the supplied nearest-neighbor LS grid")
        self.users, _, self.symbols, self.frequency = mask.shape
        if self.frequency != resource_grid.fft_size:
            raise ValueError("Pilot history requires an unpunctured allocation")
        self.key = (self.users, self.frequency, self.symbols)
        self.tb_seconds = float(resource_grid.ofdm_symbol_duration)*self.symbols
        symbol_seconds = float(resource_grid.ofdm_symbol_duration)
        spacing = float(resource_grid.subcarrier_spacing)
        n = 2*int(estimator._num_cdm_groups_without_data)
        dmrs_length = int(estimator._dmrs_length)
        per_time = int(estimator._num_pilots_per_dmrs_sym)
        if per_time % n:
            raise ValueError("Pilot count does not fit the estimator's CDM groups")
        raw = np.arange(pilots.shape[-1])
        groups = (raw//per_time//dmrs_length)*(per_time//n)+(raw % per_time)//n
        nn = np.asarray(estimator._interpol._gather_ind, np.int32)[:, 0]
        packs, maps, coordinates, factors = [], [], [], []
        for user in range(self.users):
            positions = np.argwhere(mask[user, 0])
            used = np.abs(pilots[user, 0]) > 0
            unique = np.unique(groups[used])
            index = np.full(len(groups), -1, np.int32)
            coordinate, factor = [], []
            for i, group in enumerate(unique):
                support = np.flatnonzero((groups == group) & used)
                if len(support) != 2*dmrs_length:
                    raise ValueError("Unexpected pilot support for Sionna CDM averaging")
                index[(groups == group) & used] = i
                coordinate.append(positions[support].mean(0))
                factor.append(np.sum(1./np.abs(pilots[user, 0, support])**2)/len(support)**2)
            grid_map = index[nn[user]].T.reshape(-1)
            if np.any(grid_map < 0):
                raise ValueError("Interpolation references an unused pilot")
            first = np.full(len(unique), -1, np.int32)
            for i, group in enumerate(grid_map):
                if first[group] == -1:
                    first[group] = i
            if np.any(first < 0):
                raise ValueError("A distinct pilot estimate is absent from the supplied LS grid")
            packs.append(first)
            maps.append(grid_map)
            coordinates.append(coordinate)
            factors.append(factor)
        if len({len(x) for x in packs}) != 1:
            raise ValueError("Scheduled pilot layouts must have equal observation counts")
        self.pack_indices = np.asarray(packs, np.int32)
        self.grid_indices = np.asarray(maps, np.int32)
        self.coordinates = np.asarray(coordinates, np.float32)
        self.noise_factors = np.asarray(factors, np.float32)
        self.count = self.pack_indices.shape[1]
        self.data_mask = np.transpose(np.asarray(resource_grid.build_type_grid())[:, 0] == 0, (0, 2, 1))
        if not np.all(self.data_mask == self.data_mask[:1]):
            raise ValueError("Joint separation requires common data positions for all scheduled UEs")
        joint, inverse = np.unique(self.grid_indices.T, axis=0, return_inverse=True)
        self.joint_indices = joint.T.astype(np.int32)
        self.joint_grid = inverse.astype(np.int32)
        self.data_counts = np.stack([np.bincount(row, weights=mask.reshape(-1), minlength=self.count)
            for row, mask in zip(self.grid_indices, self.data_mask)]).astype(np.float32)
        self.group_sizes = np.stack([np.bincount(row, minlength=self.count)
                                     for row in self.grid_indices]).astype(np.int32)
        if np.any(self.group_sizes == 0):
            raise ValueError("A pilot observation has no assigned grid positions")
        offsets = np.arange(self.group_sizes.max(), dtype=np.int32)
        starts = np.cumsum(self.group_sizes, axis=1)-self.group_sizes
        positions = starts[..., None]+offsets
        self.group_valid = offsets < self.group_sizes[..., None]
        order = np.argsort(self.grid_indices, axis=1, kind="stable")
        self.group_indices = np.take_along_axis(order[:, :, None],
            np.minimum(positions, order.shape[1]-1), axis=1).astype(np.int32)
        self._local_geometry(symbol_seconds, spacing)
        if type(neighbor_radius) is not int or neighbor_radius < 0:
            raise ValueError("The pilot neighborhood radius must be a nonnegative integer")
        self.neighbor_radius = neighbor_radius
        times = [np.unique(c[:, 0]) for c in self.coordinates]
        if len({len(t) for t in times}) != 1:
            raise ValueError("Scheduled pilot layouts must have equal DMRS time counts")
        neighbors = len(times[0])*(2*neighbor_radius+1)
        shape = (self.users, self.users, self.count, neighbors)
        self.neighbor_indices = np.zeros(shape, np.int32)
        self.neighbor_valid = np.zeros(shape, bool)
        self.neighbor_delta = np.zeros((*shape, 2), np.float32)
        offsets = np.arange(-neighbor_radius, neighbor_radius+1)
        for current in range(self.users):
            for old in range(self.users):
                for i, position in enumerate(self.coordinates[current]):
                    start = 0
                    for time in times[old]:
                        members = np.flatnonzero(self.coordinates[old, :, 0] == time)
                        order = np.argsort(self.coordinates[old, members, 1])
                        members = members[order]
                        nearest = np.argmin(np.abs(self.coordinates[old, members, 1]-position[1]))
                        proposed = nearest+offsets
                        stop = start+len(offsets)
                        selected = members[np.clip(proposed, 0, len(members)-1)]
                        self.neighbor_indices[current, old, i, start:stop] = selected
                        self.neighbor_valid[current, old, i, start:stop] = (proposed >= 0) & (proposed < len(members))
                        delta = self.coordinates[old, selected]-position
                        delta *= np.array([symbol_seconds*1000., spacing/1e6], np.float32)
                        self.neighbor_delta[current, old, i, start:stop] = delta
                        start = stop
        digest = hashlib.sha256()
        for array in (mask, self.pack_indices, self.grid_indices, self.coordinates, self.noise_factors):
            digest.update(array.tobytes())
        digest.update(np.asarray([self.tb_seconds, spacing, neighbor_radius]).tobytes())
        self.fingerprint = digest.hexdigest()

    def pack(self, channel, noise):
        channel = tf.convert_to_tensor(channel, tf.float32)
        if tuple(channel.shape[1:4]) != self.key:
            raise ValueError("The channel grid differs from the registered pilot layout")
        flat = tf.reshape(channel, [tf.shape(channel)[0], self.users, -1, channel.shape[-1]])
        observations = tf.gather(tf.transpose(flat, [1, 0, 2, 3]), self.pack_indices, axis=2, batch_dims=1)
        observations = tf.transpose(observations, [1, 0, 2, 3])
        variance = tf.reshape(tf.cast(noise, tf.float32), [-1, 1, 1])*self.noise_factors[None]
        return tf.concat([observations, variance[..., None]], -1)

    def expand(self, records):
        @tf.custom_gradient
        def apply(records):
            value = tf.gather(tf.transpose(records, [1, 0, 2, 3]), self.grid_indices, axis=2, batch_dims=1)
            value = tf.transpose(value, [1, 0, 2, 3])
            value = tf.reshape(value, [tf.shape(value)[0], self.users, self.frequency, self.symbols, records.shape[-1]])

            def gradient(incoming):
                return tf.reduce_sum(self._groups(incoming), axis=-2)

            return value, gradient

        return apply(records)

    def _groups(self, state):
        flat = tf.reshape(state, [tf.shape(state)[0], self.users, -1, state.shape[-1]])
        groups = tf.gather(tf.transpose(flat, [1, 0, 2, 3]), self.group_indices,
                           axis=2, batch_dims=1)
        groups = tf.transpose(groups, [1, 0, 2, 3, 4])
        return tf.where(self.group_valid[None, ..., None], groups, tf.zeros_like(groups))

    def channel_targets(self, channel):
        from .channel_filter import complex_channel
        groups = complex_channel(self._groups(channel))
        groups = tf.where(self.target_valid[None, ..., None], groups, 0.)
        q = tf.cast(self.q_layout, groups.dtype)
        coefficients = tf.einsum('bucnr,ucnk->bucrk', groups, q)
        residual = groups-tf.einsum('bucrk,ucnk->bucnr', coefficients, q)
        count = tf.cast(self.data_counts[None, ..., None, None], groups.dtype)
        z = tf.math.divide_no_nan(coefficients, tf.sqrt(count))
        covariance = tf.math.divide_no_nan(tf.einsum('bucnr,bucns->bucrs', residual, tf.math.conj(residual)), count)
        z = tf.reshape(z, [tf.shape(z)[0], self.users, self.count, 3*(channel.shape[-1]//2)])
        diagonal = tf.math.real(tf.linalg.diag_part(covariance))
        off = tf.gather(tf.reshape(covariance, [tf.shape(covariance)[0], self.users, self.count, -1]),
                        np.flatnonzero(np.triu(np.ones((channel.shape[-1]//2,)*2, bool), 1)), axis=-1)
        return tf.concat([tf.math.real(z), tf.math.imag(z), diagonal, tf.math.real(off), tf.math.imag(off)], -1)

    def _local_geometry(self, symbol_seconds, spacing):
        positions = np.stack(np.meshgrid(np.arange(self.frequency), np.arange(self.symbols), indexing='ij'), -1)
        positions = positions[..., ::-1].reshape(-1, 2).astype(np.float64)
        self.centers = np.zeros((self.users, self.count, 2), np.float32)
        self.scales = np.zeros_like(self.centers)
        phi = np.ones((self.users, self.frequency*self.symbols, 3), np.float64)
        q = np.zeros((*self.group_indices.shape, 3), np.float64)
        r = np.zeros((self.users, self.count, 3, 3), np.float64)
        self.target_valid = np.zeros_like(self.group_valid)
        for u in range(self.users):
            for c in range(self.count):
                assigned = self.group_indices[u, c, self.group_valid[u, c]]
                data = self.data_mask[u].reshape(-1)[assigned]
                fitting = assigned[data] if np.any(data) else assigned
                center = positions[fitting].mean(0)
                scale = np.max(np.abs(positions[fitting]-center), axis=0)
                self.centers[u, c], self.scales[u, c] = center, scale
                phi[u, assigned, 1:] = np.divide(positions[assigned]-center, scale,
                    out=np.zeros((len(assigned), 2)), where=scale > 0)
                self.target_valid[u, c, :len(assigned)] = data
                _, runtime_r = np.linalg.qr(phi[u, fitting]/np.sqrt(len(fitting)), mode='reduced')
                r[u, c, :runtime_r.shape[0]] = runtime_r
                if np.any(data):
                    thin_q, thin_r = np.linalg.qr(phi[u, assigned[data]]/np.sqrt(data.sum()), mode='reduced')
                    q[u, c, np.flatnonzero(data), :thin_q.shape[1]] = thin_q
                    r[u, c, :thin_r.shape[0]] = thin_r
        self.phi = phi.astype(np.float32)
        self.q_layout, self.r_layout = q.astype(np.float32), r.astype(np.float32)
        self.geometry_features = np.concatenate([
            (self.centers-self.coordinates)*np.array([symbol_seconds*1000, spacing/1e6]),
            self.scales*np.array([symbol_seconds*1000, spacing/1e6])], -1).astype(np.float32)
        tf_positions = np.flatnonzero(self.data_mask[0].T.reshape(-1))
        self.code_positions = ((tf_positions % self.frequency)*self.symbols+tf_positions//self.frequency).astype(np.int32)
        self.code_cells = self.joint_grid[self.code_positions]
        self.code_phi = self.phi[:, self.code_positions].transpose(1, 0, 2).copy()
        self.cell_positions = [np.flatnonzero(self.code_cells == c).astype(np.int32)
                               for c in range(self.joint_indices.shape[1])]

    def expand_channel(self, coefficients):
        shape = tf.shape(coefficients)
        flat = tf.reshape(coefficients, [shape[0], self.users, self.count, coefficients.shape[-2]*3])
        expanded = self.expand(flat)
        expanded = tf.reshape(expanded, [shape[0], self.users, self.frequency, self.symbols,
                                        coefficients.shape[-2], 3])
        phi = tf.constant(self.phi.reshape(self.users, self.frequency, self.symbols, 3), coefficients.dtype)
        return tf.reduce_sum(expanded*phi[None, ..., None, :], -1)

    def joint(self, records):
        rank = records.shape.rank
        values = tf.gather(tf.transpose(records, [1, 0, 2, *range(3, rank)]), self.joint_indices,
                           axis=2, batch_dims=1)
        return tf.transpose(values, [1, 2, 0, *range(3, rank)])

    def neighbors(self, records, ports):
        current = tf.broadcast_to(tf.range(self.users)[None, :, None], tf.shape(ports))
        addresses = tf.stack([current, ports], -1)
        indices = tf.gather_nd(self.neighbor_indices, addresses)
        selected = tf.gather(records, indices, axis=3, batch_dims=3)
        valid = tf.gather_nd(self.neighbor_valid, addresses)
        delta = tf.gather_nd(self.neighbor_delta, addresses)
        return selected, valid, delta
