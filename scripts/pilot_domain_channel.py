#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class PilotLayout:

    num_tx: int
    num_dmrs_symbols: int
    pilots_per_dmrs_symbol: int
    valid_indices: tuple[tuple[tuple[int, ...], ...], ...]
    dmrs_symbol_indices: tuple[int, ...]

    @property
    def valid_pilots_per_symbol(self) -> int:
        return len(self.valid_indices[0][0])

    @property
    def total_pilot_symbols(self) -> int:
        return self.num_dmrs_symbols * self.pilots_per_dmrs_symbol

    def frequency_indices(self, tx: int, phase: int) -> np.ndarray:
        return np.asarray(self.valid_indices[tx][phase], dtype=np.int64)

    def fold_ids(self) -> np.ndarray:
        result = np.empty(
            [
                self.num_tx,
                self.num_dmrs_symbols,
                self.valid_pilots_per_symbol,
            ],
            dtype=np.int8,
        )
        for tx in range(self.num_tx):
            for phase in range(self.num_dmrs_symbols):
                frequencies = self.frequency_indices(tx, phase)
                result[tx, phase] = (frequencies // 12) % 2
        return result

def pilot_layout(estimator, num_tx: int) -> PilotLayout:
    num_tx = int(num_tx)
    num_dmrs = int(estimator._num_dmrs_syms)
    per_dmrs = int(estimator._num_pilots_per_dmrs_sym)
    pilots = np.asarray(estimator._pilot_pattern.pilots.numpy())
    if pilots.shape[0] < num_tx or pilots.shape[1] != 1:
        raise ValueError("unsupported pilot tensor geometry")
    pilots = pilots[:num_tx, 0].reshape(num_tx, num_dmrs, per_dmrs)
    valid = []
    counts = set()
    for tx in range(num_tx):
        phases = []
        for phase in range(num_dmrs):
            indices = tuple(
                int(value)
                for value in np.flatnonzero(np.abs(pilots[tx, phase]) > 0.0)
            )
            phases.append(indices)
            counts.add(len(indices))
        valid.append(tuple(phases))
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError("valid pilot counts must be uniform and positive")

    mask = np.asarray(estimator._pilot_pattern.mask.numpy())[:num_tx, 0]
    symbol_sets = []
    for tx in range(num_tx):
        symbol_sets.append(tuple(
            int(value)
            for value in np.flatnonzero(np.any(mask[tx] > 0, axis=-1))
        ))
    if len(set(symbol_sets)) != 1 or len(symbol_sets[0]) != num_dmrs:
        raise ValueError("DMRS symbol positions differ across transmitters")
    return PilotLayout(
        num_tx=num_tx,
        num_dmrs_symbols=num_dmrs,
        pilots_per_dmrs_symbol=per_dmrs,
        valid_indices=tuple(valid),
        dmrs_symbol_indices=symbol_sets[0],
    )


def extract_pilot_statistics(
    estimator,
    y,
    no,
    num_tx: int,
    tf,
) -> tuple[np.ndarray, np.ndarray, PilotLayout]:
    from sionna.utils import flatten_last_dims

    layout = pilot_layout(estimator, num_tx)
    y_effective = estimator._removed_nulled_scs(y)
    y_flat = flatten_last_dims(y_effective)
    y_pilots = tf.gather(y_flat, estimator._pilot_ind, axis=-1)
    full_estimate, full_error = estimator.estimate_at_pilot_locations(
        y_pilots, no
    )
    full_estimate = np.asarray(full_estimate.numpy())
    full_error = np.broadcast_to(
        np.asarray(full_error.numpy()), full_estimate.shape
    )
    batch = full_estimate.shape[0]
    num_rx_ant = full_estimate.shape[2]
    valid_count = layout.valid_pilots_per_symbol
    compact = np.empty(
        [
            batch,
            layout.num_tx,
            layout.num_dmrs_symbols,
            valid_count,
            num_rx_ant,
        ],
        dtype=full_estimate.dtype,
    )
    error = np.empty(compact.shape, dtype=full_error.dtype)
    for tx in range(layout.num_tx):
        source_estimate = full_estimate[:, 0, :, tx, 0]
        source_error = full_error[:, 0, :, tx, 0]
        for phase in range(layout.num_dmrs_symbols):
            offset = phase * layout.pilots_per_dmrs_symbol
            indices = offset + layout.frequency_indices(tx, phase)
            compact[:, tx, phase] = np.take(
                source_estimate, indices, axis=-1
            ).transpose(0, 2, 1)
            error[:, tx, phase] = np.take(
                source_error, indices, axis=-1
            ).transpose(0, 2, 1)
    if not np.all(np.isfinite(compact)):
        raise FloatingPointError("pilot estimates are not finite")
    if not np.all(np.isfinite(error)) or not np.all(error > 0.0):
        raise FloatingPointError("pilot error variances must be positive")
    return compact, error, layout


def extract_raw_pilot_estimates(
    estimator,
    y,
    num_tx: int,
    tf,
) -> tuple[np.ndarray, PilotLayout]:
    from sionna.utils import flatten_last_dims

    layout = pilot_layout(estimator, num_tx)
    y_effective = estimator._removed_nulled_scs(y)
    y_flat = flatten_last_dims(y_effective)
    y_pilots = tf.gather(y_flat, estimator._pilot_ind, axis=-1)
    pilots = tf.cast(estimator._pilot_pattern.pilots, y_pilots.dtype)
    pilots = pilots[None, None, None]
    raw = tf.math.divide_no_nan(y_pilots, pilots)
    raw = np.asarray(raw.numpy())
    batch = raw.shape[0]
    num_rx_ant = raw.shape[2]
    valid_count = layout.valid_pilots_per_symbol
    compact = np.empty(
        [
            batch,
            layout.num_tx,
            layout.num_dmrs_symbols,
            valid_count,
            num_rx_ant,
        ],
        dtype=raw.dtype,
    )
    for tx in range(layout.num_tx):
        source = raw[:, 0, :, tx, 0]
        for phase in range(layout.num_dmrs_symbols):
            offset = phase * layout.pilots_per_dmrs_symbol
            indices = offset + layout.frequency_indices(tx, phase)
            compact[:, tx, phase] = np.take(
                source, indices, axis=-1
            ).transpose(0, 2, 1)
    if not np.all(np.isfinite(compact)):
        raise FloatingPointError("raw pilot estimates are not finite")
    return compact, layout


def cdm_average_raw_pilots(raw_pilots: np.ndarray) -> np.ndarray:
    raw_pilots = np.asarray(raw_pilots)
    if raw_pilots.ndim != 5 or raw_pilots.shape[3] % 2:
        raise ValueError("raw pilots require an even valid-pilot dimension")
    shape = raw_pilots.shape
    pairs = raw_pilots.reshape(
        shape[0], shape[1], shape[2], shape[3] // 2, 2, shape[4]
    )
    averaged = np.mean(pairs, axis=4, keepdims=True)
    result = np.repeat(averaged, 2, axis=4).reshape(shape)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("CDM-averaged pilots are not finite")
    return result.astype(raw_pilots.dtype, copy=False)


def estimate_cdm_noise_variance(
    raw_pilots: np.ndarray,
) -> tuple[np.ndarray, dict]:
    raw_pilots = np.asarray(raw_pilots)
    if raw_pilots.ndim != 5 or raw_pilots.shape[3] % 2:
        raise ValueError("raw pilots require an even valid-pilot dimension")
    shape = raw_pilots.shape
    pairs = raw_pilots.reshape(
        shape[0], shape[1], shape[2], shape[3] // 2, 2, shape[4]
    )
    difference = pairs[..., 0, :] - pairs[..., 1, :]
    estimate = np.mean(np.square(np.abs(difference)), axis=3) / 4.0
    if not np.all(np.isfinite(estimate)) or np.any(estimate <= 0.0):
        raise FloatingPointError("estimated pilot variance must be positive")
    broadcast = np.broadcast_to(
        estimate[:, :, :, None, :], raw_pilots.shape
    ).copy()
    return broadcast.astype(np.float32), {
        "pair_count_per_phase": int(shape[3] // 2),
        "minimum_variance": float(np.min(estimate)),
        "maximum_variance": float(np.max(estimate)),
        "mean_variance": float(np.mean(estimate)),
    }


def interpolate_pilot_statistics(
    estimator,
    compact: np.ndarray,
    error: np.ndarray,
    layout: PilotLayout,
    tf,
) -> tuple[np.ndarray, np.ndarray]:
    compact = np.asarray(compact)
    error = np.asarray(error)
    if compact.shape != error.shape or compact.ndim != 5:
        raise ValueError("compact pilot and error tensors must match")
    expected = (
        layout.num_tx,
        layout.num_dmrs_symbols,
        layout.valid_pilots_per_symbol,
    )
    if compact.shape[1:4] != expected:
        raise ValueError("compact pilot tensor does not match its layout")
    if not np.all(error >= 0.0):
        raise ValueError("pilot error variance cannot be negative")

    batch, _, _, _, num_rx_ant = compact.shape
    full_shape = [
        batch,
        1,
        num_rx_ant,
        layout.num_tx,
        1,
        layout.total_pilot_symbols,
    ]
    full_estimate = np.zeros(full_shape, dtype=compact.dtype)
    full_error = np.zeros(full_shape, dtype=error.dtype)
    for tx in range(layout.num_tx):
        for phase in range(layout.num_dmrs_symbols):
            offset = phase * layout.pilots_per_dmrs_symbol
            indices = offset + layout.frequency_indices(tx, phase)
            estimate_view = full_estimate[:, 0, :, tx, 0]
            error_view = full_error[:, 0, :, tx, 0]
            estimate_view[:, :, indices] = compact[:, tx, phase].transpose(
                0, 2, 1
            )
            error_view[:, :, indices] = error[:, tx, phase].transpose(
                0, 2, 1
            )

    grid, grid_error = estimator._interpol(
        tf.convert_to_tensor(full_estimate),
        tf.convert_to_tensor(full_error),
    )
    grid_error = tf.maximum(grid_error, tf.cast(0, grid_error.dtype))
    grid = np.asarray(grid.numpy())[:, 0, :, :layout.num_tx, 0]
    grid_error = np.asarray(grid_error.numpy())[
        :, 0, :, :layout.num_tx, 0
    ]
    grid = grid.transpose(0, 2, 4, 3, 1)
    grid_error = grid_error.transpose(0, 2, 4, 3, 1)
    return grid, grid_error


def interpolate_pilot_sequence(
    estimator,
    compact: np.ndarray,
    error: np.ndarray,
    layout: PilotLayout,
    tf,
    subcarrier_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    compact = np.asarray(compact)
    error = np.asarray(error)
    if compact.shape != error.shape or compact.ndim != 6:
        raise ValueError("pilot sequences must have matching six-rank shapes")
    if subcarrier_stride < 1:
        raise ValueError("subcarrier stride must be positive")
    grid_episodes = []
    error_episodes = []
    for episode in range(compact.shape[0]):
        grid, grid_error = interpolate_pilot_statistics(
            estimator,
            compact[episode],
            error[episode],
            layout,
            tf,
        )
        grid_episodes.append(grid[:, :, ::subcarrier_stride])
        error_episodes.append(grid_error[:, :, ::subcarrier_stride])
    return np.stack(grid_episodes), np.stack(error_episodes)


def _toeplitz_projection(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0]
    result = np.empty_like(matrix)
    for lag in range(size):
        upper = np.diag(matrix, k=lag)
        lower = np.conj(np.diag(matrix, k=-lag))
        value = np.mean(np.concatenate([upper, lower]))
        rows = np.arange(size - lag)
        result[rows, rows + lag] = value
        result[rows + lag, rows] = np.conj(value)
    result[np.diag_indices(size)] = np.real(np.diag(result))
    return result


def project_toeplitz_psd(
    matrix: np.ndarray,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid projection controls")
    current = 0.5 * (matrix + matrix.conj().T)
    relative_change = np.inf
    for iteration in range(1, max_iterations + 1):
        toeplitz = _toeplitz_projection(current)
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (toeplitz + toeplitz.conj().T)
        )
        clipped = np.maximum(eigenvalues, 0.0)
        updated = (eigenvectors * clipped) @ eigenvectors.conj().T
        denominator = max(float(np.linalg.norm(current)), EPSILON)
        relative_change = float(
            np.linalg.norm(updated - current) / denominator
        )
        current = 0.5 * (updated + updated.conj().T)
        if relative_change < tolerance:
            break
    else:
        raise FloatingPointError("Toeplitz PSD projection did not converge")
    final_toeplitz = _toeplitz_projection(current)
    toeplitz_residual = float(
        np.linalg.norm(current - final_toeplitz)
        / max(float(np.linalg.norm(current)), EPSILON)
    )
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(current)))
    return current, {
        "iterations": iteration,
        "relative_change": relative_change,
        "toeplitz_residual": toeplitz_residual,
        "minimum_eigenvalue": minimum_eigenvalue,
    }


def pilot_domain_wiener_estimate(
    observation: np.ndarray,
    error_variance: np.ndarray,
    max_history: int = 8,
    fold_ids: np.ndarray | None = None,
    history_source: np.ndarray | None = None,
    history_error_variance: np.ndarray | None = None,
    projection_tolerance: float = 1e-8,
    projection_max_iterations: int = 100,
    capture_weight_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    observation = np.asarray(observation)
    error_variance = np.asarray(error_variance)
    history_source = (
        observation if history_source is None else np.asarray(history_source)
    )
    history_error_variance = (
        error_variance
        if history_error_variance is None
        else np.asarray(history_error_variance)
    )
    if observation.shape != error_variance.shape or observation.ndim != 6:
        raise ValueError("pilot observation and error shapes must match")
    if history_source.shape != observation.shape:
        raise ValueError("history source must match pilot observations")
    if history_error_variance.shape != observation.shape:
        raise ValueError("history error source must match observations")
    if max_history < 1 or max_history > observation.shape[1]:
        raise ValueError("invalid pilot history length")
    if not np.all(error_variance > 0.0):
        raise ValueError("pilot error variances must be positive")

    episodes, num_tbs, num_users, num_phases, num_pilots, _ = (
        observation.shape
    )
    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids)
        if fold_ids.shape != (num_users, num_phases, num_pilots):
            raise ValueError("fold IDs do not match pilot geometry")
        if set(np.unique(fold_ids).tolist()) != {0, 1}:
            raise ValueError("cross fitting requires folds zero and one")

    result = np.empty_like(observation)
    posterior = np.empty_like(error_variance)
    max_weight = 0.0
    min_uncertainty = np.inf
    max_uncertainty = 0.0
    min_eigenvalue = np.inf
    max_condition = 0.0
    max_projection_iterations = 0
    max_toeplitz_residual = 0.0
    roundoff_clamps = 0
    total_updates = 0
    mature_weight_sum = np.zeros(max_history, dtype=np.complex128)
    mature_weight_count = 0
    weight_history = None
    weight_history_count = None
    if capture_weight_history:
        weight_history = np.zeros(
            [episodes, num_tbs, num_users, num_phases, max_history],
            dtype=np.complex128,
        )
        weight_history_count = np.zeros(
            [episodes, num_tbs, num_users, num_phases],
            dtype=np.int8,
        )

    for episode in range(episodes):
        for tb in range(num_tbs):
            history = min(tb + 1, max_history)
            for user in range(num_users):
                for phase in range(num_phases):
                    snapshots = [observation[episode, tb, user, phase]]
                    variances = [error_variance[episode, tb, user, phase]]
                    for lag in range(1, history):
                        snapshots.append(
                            history_source[
                                episode, tb - lag, user, phase
                            ]
                        )
                        variances.append(
                            history_error_variance[
                                episode, tb - lag, user, phase
                            ]
                        )

                    if fold_ids is None:
                        fold_pairs = [(np.ones(num_pilots, bool),) * 2]
                    else:
                        phase_folds = fold_ids[user, phase]
                        fold_pairs = [
                            (phase_folds == 1, phase_folds == 0),
                            (phase_folds == 0, phase_folds == 1),
                        ]

                    for source_mask, target_mask in fold_pairs:
                        source = np.stack(
                            [value[source_mask] for value in snapshots],
                            axis=-1,
                        ).reshape(-1, history)
                        target = np.stack(
                            [value[target_mask] for value in snapshots],
                            axis=-1,
                        )
                        if source.shape[0] < history:
                            raise ValueError("insufficient pilot covariance samples")
                        covariance = source.conj().T @ source / source.shape[0]
                        noise = np.asarray([
                            float(np.mean(value[source_mask]))
                            for value in variances
                        ])
                        channel_raw = covariance - np.diag(noise)
                        channel_covariance, projection = project_toeplitz_psd(
                            channel_raw,
                            tolerance=projection_tolerance,
                            max_iterations=projection_max_iterations,
                        )
                        average_power = max(
                            float(np.trace(channel_covariance).real) / history,
                            EPSILON,
                        )
                        diagonal_floor = (
                            np.finfo(np.float64).eps * average_power
                        )
                        observation_covariance = (
                            channel_covariance
                            + np.diag(noise)
                            + diagonal_floor * np.eye(history)
                        )
                        cross = channel_covariance[:, 0]
                        coefficients = np.linalg.solve(
                            observation_covariance, cross
                        )
                        if weight_history is not None:
                            weight_history[
                                episode, tb, user, phase, :history
                            ] += coefficients
                            weight_history_count[
                                episode, tb, user, phase
                            ] += 1
                        estimate = np.tensordot(
                            target, coefficients, axes=([-1], [0])
                        )
                        result[
                            episode, tb, user, phase, target_mask
                        ] = estimate
                        raw_uncertainty = float(
                            channel_covariance[0, 0].real
                            - np.vdot(cross, coefficients).real
                        )
                        tolerance = 1e-10 * max(
                            float(channel_covariance[0, 0].real), 1.0
                        )
                        if raw_uncertainty < -tolerance:
                            raise FloatingPointError(
                                "pilot posterior uncertainty is negative"
                            )
                        if raw_uncertainty < 0.0:
                            raw_uncertainty = 0.0
                            roundoff_clamps += 1
                        posterior[
                            episode, tb, user, phase, target_mask
                        ] = raw_uncertainty

                        eigenvalues = np.linalg.eigvalsh(
                            observation_covariance
                        )
                        max_weight = max(
                            max_weight, float(np.max(np.abs(coefficients)))
                        )
                        min_uncertainty = min(
                            min_uncertainty, raw_uncertainty
                        )
                        max_uncertainty = max(
                            max_uncertainty, raw_uncertainty
                        )
                        min_eigenvalue = min(
                            min_eigenvalue, float(eigenvalues[0])
                        )
                        max_condition = max(
                            max_condition,
                            float(eigenvalues[-1] / eigenvalues[0]),
                        )
                        max_projection_iterations = max(
                            max_projection_iterations,
                            int(projection["iterations"]),
                        )
                        max_toeplitz_residual = max(
                            max_toeplitz_residual,
                            float(projection["toeplitz_residual"]),
                        )
                        if history == max_history:
                            mature_weight_sum += coefficients
                            mature_weight_count += 1
                        total_updates += 1

    arrays = [result, posterior]
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("pilot-domain output is not finite")
    if np.any(posterior < 0.0):
        raise FloatingPointError("pilot posterior variance is negative")
    mean_weights = (
        mature_weight_sum / mature_weight_count
        if mature_weight_count
        else mature_weight_sum
    )
    diagnostics = {
        "cross_fitted": fold_ids is not None,
        "max_abs_weight": max_weight,
        "min_uncertainty": float(min_uncertainty),
        "max_uncertainty": max_uncertainty,
        "min_observation_covariance_eigenvalue": min_eigenvalue,
        "max_condition_number": max_condition,
        "max_projection_iterations": max_projection_iterations,
        "max_toeplitz_residual": max_toeplitz_residual,
        "roundoff_clamps": roundoff_clamps,
        "total_updates": total_updates,
        "mean_mature_weights": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in mean_weights
        ],
    }
    if weight_history is not None:
        counts = np.maximum(weight_history_count, 1)[..., None]
        weight_history = weight_history / counts
        if not np.all(np.isfinite(weight_history)):
            raise FloatingPointError("weight history is not finite")
        diagnostics["weight_history"] = weight_history
    return result, posterior, diagnostics
