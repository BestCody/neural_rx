#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from pilot_domain_channel import (
    EPSILON,
    PilotLayout,
    interpolate_pilot_statistics,
    pilot_layout,
)


def _validate_cached_layout(estimator, layout: PilotLayout) -> None:
    pilots = np.asarray(estimator._pilot_pattern.pilots.numpy())
    expected_shape = (
        layout.num_tx,
        layout.num_dmrs_symbols,
        layout.pilots_per_dmrs_symbol,
    )
    if pilots.shape[0] < layout.num_tx or pilots.shape[1] != 1:
        raise ValueError("cached pilot tensor geometry changed")
    pilots = pilots[:layout.num_tx, 0].reshape(expected_shape)
    for tx in range(layout.num_tx):
        for phase in range(layout.num_dmrs_symbols):
            active = np.flatnonzero(np.abs(pilots[tx, phase]) > 0.0)
            if not np.array_equal(
                active, layout.frequency_indices(tx, phase)
            ):
                raise ValueError("cached pilot frequency layout changed")
    mask = np.asarray(estimator._pilot_pattern.mask.numpy())[
        :layout.num_tx, 0
    ]
    for tx in range(layout.num_tx):
        symbols = tuple(
            int(value)
            for value in np.flatnonzero(np.any(mask[tx] > 0, axis=-1))
        )
        if symbols != layout.dmrs_symbol_indices:
            raise ValueError("cached pilot symbol layout changed")


def extract_raw_pilot_estimates_cached(
    estimator,
    y,
    num_tx: int,
    tf,
    layout: PilotLayout | None = None,
) -> tuple[np.ndarray, PilotLayout]:
    from sionna.utils import flatten_last_dims

    if layout is None:
        layout = pilot_layout(estimator, num_tx)
    elif int(layout.num_tx) != int(num_tx):
        raise ValueError("cached pilot layout transmitter count changed")
    elif (
        int(estimator._num_dmrs_syms) != layout.num_dmrs_symbols
        or int(estimator._num_pilots_per_dmrs_sym)
        != layout.pilots_per_dmrs_symbol
    ):
        raise ValueError("cached pilot layout geometry changed")
    else:
        _validate_cached_layout(estimator, layout)
    y_effective = estimator._removed_nulled_scs(y)
    y_flat = flatten_last_dims(y_effective)
    y_pilots = tf.gather(y_flat, estimator._pilot_ind, axis=-1)
    pilots = tf.cast(estimator._pilot_pattern.pilots, y_pilots.dtype)
    raw = tf.math.divide_no_nan(y_pilots, pilots[None, None, None])
    raw = np.asarray(raw.numpy())
    batch = raw.shape[0]
    num_rx_ant = raw.shape[2]
    compact = np.empty(
        [
            batch,
            layout.num_tx,
            layout.num_dmrs_symbols,
            layout.valid_pilots_per_symbol,
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


def interpolate_pilot_bundle(
    estimator,
    posterior: np.ndarray,
    posterior_error: np.ndarray,
    observation: np.ndarray,
    observation_error: np.ndarray,
    layout: PilotLayout,
    tf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    posterior = np.asarray(posterior)
    posterior_error = np.asarray(posterior_error)
    observation = np.asarray(observation)
    observation_error = np.asarray(observation_error)
    if posterior.shape != posterior_error.shape:
        raise ValueError("posterior pilot tensors must match")
    if observation.shape != observation_error.shape:
        raise ValueError("observation pilot tensors must match")
    if posterior.shape != observation.shape or posterior.ndim != 5:
        raise ValueError("pilot bundle tensors must share rank-five shape")
    batch = int(posterior.shape[0])
    grid, error = interpolate_pilot_statistics(
        estimator,
        np.concatenate([posterior, observation], axis=0),
        np.concatenate([posterior_error, observation_error], axis=0),
        layout,
        tf,
    )
    return grid[:batch], error[:batch], error[batch:]


def _toeplitz_projection_batch(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("batched covariance matrices must be square")
    size = int(matrix.shape[-1])
    result = np.empty_like(matrix)
    for lag in range(size):
        upper = np.diagonal(matrix, offset=lag, axis1=-2, axis2=-1)
        lower = np.conj(
            np.diagonal(matrix, offset=-lag, axis1=-2, axis2=-1)
        )
        value = np.mean(np.concatenate([upper, lower], axis=-1), axis=-1)
        rows = np.arange(size - lag)
        result[:, rows, rows + lag] = value[:, None]
        result[:, rows + lag, rows] = np.conj(value[:, None])
    diagonal = np.arange(size)
    result[:, diagonal, diagonal] = np.real(
        result[:, diagonal, diagonal]
    )
    return result


def project_toeplitz_psd_batch(
    matrix: np.ndarray,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("batched covariance matrices must be square")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid projection controls")
    current = 0.5 * (matrix + np.swapaxes(matrix.conj(), -1, -2))
    count = int(current.shape[0])
    active = np.ones(count, bool)
    iterations = np.zeros(count, np.int32)
    relative_change = np.full(count, np.inf, np.float64)
    for iteration in range(1, max_iterations + 1):
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break
        selected = current[indices]
        toeplitz = _toeplitz_projection_batch(selected)
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (toeplitz + np.swapaxes(toeplitz.conj(), -1, -2))
        )
        clipped = np.maximum(eigenvalues, 0.0)
        updated = np.matmul(
            eigenvectors * clipped[:, None, :],
            np.swapaxes(eigenvectors.conj(), -1, -2),
        )
        denominator = np.maximum(
            np.linalg.norm(selected, axis=(-2, -1)), EPSILON
        )
        change = np.linalg.norm(
            updated - selected, axis=(-2, -1)
        ) / denominator
        current[indices] = 0.5 * (
            updated + np.swapaxes(updated.conj(), -1, -2)
        )
        relative_change[indices] = change
        converged = change < tolerance
        if np.any(converged):
            finished = indices[converged]
            iterations[finished] = iteration
            active[finished] = False
    if np.any(active):
        raise FloatingPointError("Toeplitz PSD projection did not converge")
    final_toeplitz = _toeplitz_projection_batch(current)
    denominator = np.maximum(
        np.linalg.norm(current, axis=(-2, -1)), EPSILON
    )
    residual = np.linalg.norm(
        current - final_toeplitz, axis=(-2, -1)
    ) / denominator
    minimum_eigenvalue = np.min(np.linalg.eigvalsh(current), axis=-1)
    return current, {
        "iterations": iterations,
        "relative_change": relative_change,
        "toeplitz_residual": residual,
        "minimum_eigenvalue": minimum_eigenvalue,
    }


def pilot_domain_wiener_latest(
    observation: np.ndarray,
    error_variance: np.ndarray,
    max_history: int = 8,
    projection_tolerance: float = 1e-8,
    projection_max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict]:
    observation = np.asarray(observation)
    error_variance = np.asarray(error_variance)
    if observation.shape != error_variance.shape or observation.ndim != 6:
        raise ValueError("pilot observation and error shapes must match")
    if max_history < 1:
        raise ValueError("invalid pilot history length")
    if not np.all(error_variance > 0.0):
        raise ValueError("pilot error variances must be positive")
    if observation.shape[1] > max_history:
        observation = observation[:, -max_history:]
        error_variance = error_variance[:, -max_history:]

    episodes, history, users, phases, pilots, antennas = observation.shape
    newest_first = observation[:, ::-1]
    error_newest_first = error_variance[:, ::-1]
    samples = np.ascontiguousarray(np.transpose(
        newest_first, [0, 2, 3, 4, 5, 1]
    ).reshape(-1, pilots * antennas, history))
    sample_error = np.ascontiguousarray(np.transpose(
        error_newest_first, [0, 2, 3, 4, 5, 1]
    ).reshape(-1, pilots * antennas, history))
    covariance = np.stack([
        value.conj().T @ value / float(pilots * antennas)
        for value in samples
    ])
    noise = np.asarray([
        [float(np.mean(value[:, lag])) for lag in range(history)]
        for value in sample_error
    ])
    identity = np.eye(history)[None]
    channel_raw = covariance - identity * noise[:, None, :]
    channel_covariance, projection = project_toeplitz_psd_batch(
        channel_raw,
        tolerance=projection_tolerance,
        max_iterations=projection_max_iterations,
    )
    average_power = np.maximum(
        np.trace(channel_covariance, axis1=-2, axis2=-1).real
        / float(history),
        EPSILON,
    )
    diagonal_floor = np.finfo(np.float64).eps * average_power
    observation_covariance = (
        channel_covariance
        + identity * noise[:, None, :]
        + identity * diagonal_floor[:, None, None]
    )
    cross = channel_covariance[:, :, 0]
    coefficients = np.linalg.solve(observation_covariance, cross)
    target = np.ascontiguousarray(np.transpose(
        newest_first, [0, 2, 3, 4, 5, 1]
    ).reshape(-1, pilots, antennas, history))
    estimate = np.stack([
        np.tensordot(value, weight, axes=([-1], [0]))
        for value, weight in zip(target, coefficients)
    ])
    raw_uncertainty = np.asarray([
        covariance[0, 0].real - np.vdot(vector, weight).real
        for covariance, vector, weight in zip(
            channel_covariance, cross, coefficients
        )
    ])
    uncertainty_tolerance = 1e-10 * np.maximum(
        channel_covariance[:, 0, 0].real, 1.0
    )
    if np.any(raw_uncertainty < -uncertainty_tolerance):
        raise FloatingPointError("pilot posterior uncertainty is negative")
    roundoff = raw_uncertainty < 0.0
    raw_uncertainty = np.maximum(raw_uncertainty, 0.0)
    posterior = np.broadcast_to(
        raw_uncertainty[:, None, None], estimate.shape
    ).copy()
    output_shape = (episodes, users, phases, pilots, antennas)
    estimate = estimate.reshape(output_shape).astype(
        observation.dtype, copy=False
    )
    posterior = posterior.reshape(output_shape).astype(
        error_variance.dtype, copy=False
    )
    eigenvalues = np.linalg.eigvalsh(observation_covariance)
    diagnostics = {
        "max_abs_weight": float(np.max(np.abs(coefficients))),
        "min_uncertainty": float(np.min(raw_uncertainty)),
        "max_uncertainty": float(np.max(raw_uncertainty)),
        "min_observation_covariance_eigenvalue": float(
            np.min(eigenvalues)
        ),
        "max_condition_number": float(
            np.max(eigenvalues[:, -1] / eigenvalues[:, 0])
        ),
        "max_projection_iterations": int(
            np.max(projection["iterations"])
        ),
        "max_toeplitz_residual": float(
            np.max(projection["toeplitz_residual"])
        ),
        "roundoff_clamps": int(np.count_nonzero(roundoff)),
        "total_updates": int(episodes * users * phases),
    }
    arrays = (estimate, posterior)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("pilot-domain output is not finite")
    return estimate, posterior, diagnostics
