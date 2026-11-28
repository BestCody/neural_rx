from __future__ import annotations

import tensorflow as tf


def power(value):
    return tf.square(tf.math.real(value))+tf.square(tf.math.imag(value))


def complex_channel(value):
    real, imag = tf.split(value, 2, -1)
    return tf.complex(real, imag)


def solve_positive_definite(matrix, rhs):
    """Solution of matrix @ x = rhs for Hermitian positive definite matrices [..., n, n] and rhs [..., n], by
    Gauss-Jordan elimination. Such matrices need no pivoting, and plain tensor operations keep the filter independent
    of the GPU's solver library, which allocates memory of its own and fails when TensorFlow holds all of it."""
    size = matrix.shape[-1]
    system = tf.concat([matrix, rhs[..., None]], -1)
    for k in range(size):
        row = system[..., k, :]/system[..., k, k, None]
        unit = tf.one_hot(k, size, dtype=system.dtype)
        system += (unit-system[..., :, k])[..., :, None]*row[..., None, :]      # clear column k, set row k
    return system[..., -1]


class ChannelFilter:
    """Channel filter over the current pilots and the pilot memory, with weights fitted per UE on the memory itself.

    The memory holds the LS estimates of a UE's previous TBs. The filter predicts the LS estimate of every pilot cell
    of the current TB from the other estimates around it: those of the neighbouring cells in the same TB and those of
    the UE's newest entries. Its complex weights, one set per UE and DMRS symbol, are fitted by least squares on the
    memory: each of the `fit_tbs` newest entries is predicted in the same way from its neighbouring cells and from the
    entries that are as much older than it as the current TB's entries are older than the current TB. The noise of a
    predicted estimate is independent of these inputs, so the weights are those that predict the true channel; no
    channel statistics and no training are needed, and the fit only uses TBs that were received earlier.

    How many entries still describe the current channel is decided on the current TB: the fit is made for the newest
    0, 1, ... entries (0: the neighbouring cells of the same TB only), each on the stored TBs that have all of these
    entries, and the filter uses the number whose prediction is closest to the current LS estimates. After a channel
    change or a pause it thus leaves the outdated entries out. The prediction and the cell's own LS estimate are then
    combined by their error variances; the prediction's is that mean squared difference minus the LS noise variance.

    Returns the coefficients of a first-order channel model per pilot cell, [batch, users, cells, antennas, 3], whose
    slopes are differences of the refined estimates of neighbouring cells. A UE without a valid entry keeps its LS
    estimate; unscheduled UEs get zeros.
    """

    def __init__(self, fit_tbs=3, ridge=1e-3):
        if type(fit_tbs) is not int or fit_tbs < 1:
            raise ValueError("The number of fitted TBs must be a positive integer")
        self.fit_tbs, self.ridge = fit_tbs, float(ridge)

    def design(self, records, ports, ages, valid, target, lags, used, layout):
        """Inputs and target for record `target`: the estimates around each of its pilot cells in the record itself and
        in the records that are older than it by `lags` [batch, users, entries], as [batch, users, cells, antennas,
        inputs] (zero where missing); its own estimates [batch, users, cells, antennas]; the DMRS symbol of its cells;
        whether the record is valid; and which of the older records exist [batch, users, entries]."""
        wanted = ages[:, :, target, None]+lags
        match = (valid[:, :, None, 1:] & used[..., None] & (ages[:, :, None, 1:] == wanted[..., None]))
        found = tf.reduce_any(match, -1)
        index = 1+tf.argmax(tf.cast(match, tf.int32), -1, output_type=tf.int32)      # ages of valid entries differ
        # the record itself first; its shape must not come from index, which is empty when no entry is used (K = 0)
        own = tf.fill(tf.concat([tf.shape(index)[:-1], [1]], 0), target)
        index = tf.concat([own, index], -1)
        current, usable = ports[:, :, target], valid[:, :, target]
        present = tf.concat([usable[..., None], found & usable[..., None]], -1)
        old, local_valid, delta = layout.neighbors(tf.gather(records, index, axis=2, batch_dims=2),
                                                   tf.gather(ports, index, batch_dims=2), current)
        central = tf.reduce_all(delta == 0., -1) & (tf.range(index.shape[-1])[None, None, :, None, None] == 0)
        allowed = present[..., None, None] & local_valid & ~central
        x = tf.where(allowed[..., None], complex_channel(old[..., :-1]), 0.)    # [b, u, records, cells, neighbours, a]
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
        x = tf.reshape(x, [tf.shape(x)[0], layout.users, layout.count, x.shape[3], index.shape[-1]*old.shape[4]])
        own = complex_channel(records[:, :, target, :, :-1])
        return x, own, tf.gather(tf.constant(layout.cell_symbol), current), usable, present[..., 1:]

    def __call__(self, observation, history, active, layout):
        active = tf.cast(active, tf.bool)
        valid = history.valid & active[..., None]
        users, symbols = layout.users, layout.dmrs_symbols
        entries = history.observations.shape[2]-self.fit_tbs
        if entries < 0:
            raise ValueError("The memory holds fewer TBs than the fit uses")
        batch = tf.shape(observation)[0]
        antennas = (observation.shape[-1]-1)//2
        records = tf.concat([observation[:, :, None], tf.where(valid[..., None, None], history.observations, 0.)], 2)
        ports = tf.concat([tf.broadcast_to(tf.range(users)[None, :, None], [batch, users, 1]), history.ports], 2)
        ages = tf.concat([tf.zeros([batch, users, 1], history.age.dtype), history.age], 2)
        valid = tf.concat([active[..., None], valid], 2)
        lags, used = ages[:, :, 1:entries+1], valid[:, :, 1:entries+1]          # the current TB's newest entries

        def members(symbol, usable):
            """1 for the cells of each DMRS symbol of a usable record, [batch, users, symbols, cells]."""
            member = (symbol[:, :, None] == tf.range(symbols)[None, None, :, None]) & usable[:, :, None, None]
            return tf.cast(member, tf.float32)

        def per_symbol(value, symbol, usable):
            """Sum over the cells of each DMRS symbol and the antennas: [b, u, cells, a] -> [b, u, symbols]."""
            return tf.einsum('bskc,bsc->bsk', members(symbol, usable), tf.reduce_sum(value, -1))

        # normal equations of each of the newest entries, per UE and DMRS symbol
        equations = []
        for target in range(1, self.fit_tbs+1):
            x, own, symbol, usable, found = self.design(records, ports, ages, valid, target, lags, used, layout)
            flat = tf.reshape(x, [batch, users, 1, -1, x.shape[-1]])             # [b, u, 1, cells*antennas, inputs]
            member = tf.cast(tf.repeat(members(symbol, usable), antennas, -1), x.dtype)
            selected = flat*member[..., None]                                    # [b, u, symbols, cells*antennas, inputs]
            equations.append((tf.matmul(selected, flat, adjoint_a=True),
                              tf.matmul(selected, tf.reshape(own, [batch, users, 1, -1, 1]), adjoint_a=True)[..., 0],
                              usable, found))

        # the current TB, and one prediction per number of entries used
        x, own, symbol, _, _ = self.design(records, ports, ages, valid, 0, lags, used, layout)
        everyone = tf.ones_like(active)
        count = per_symbol(tf.ones_like(tf.math.real(own)), symbol, everyone)
        size, step = x.shape[-1], x.shape[-1]//(entries+1)
        predictions, residuals = [], []
        for number in range(entries+1):
            width = (number+1)*step
            gram = tf.zeros([batch, users, symbols, width, width], x.dtype)
            rhs = tf.zeros([batch, users, symbols, width], x.dtype)
            fitted = tf.zeros([batch, users], tf.bool)
            for matrix, vector, usable, found in equations:                      # entries that have all of these lags
                complete = usable & tf.reduce_all(found[..., :number], -1)
                weight = tf.cast(complete, x.dtype)[..., None, None]
                gram += weight[..., None]*matrix[..., :width, :width]
                rhs += weight*vector[..., :width]
                fitted |= complete
            fitted &= tf.reduce_all(used[..., :number], -1)                      # and the current TB has these entries
            trace = tf.math.real(tf.linalg.trace(gram))
            loading = self.ridge*trace/width+tf.cast(trace <= 0., tf.float32)    # identity where there is nothing to fit
            gram += tf.cast(loading[..., None, None], gram.dtype)*tf.eye(width, dtype=gram.dtype)
            weights = solve_positive_definite(gram, rhs)                         # [b, u, symbols, inputs used]
            prediction = tf.reduce_sum(x[..., :width]*tf.gather(weights, symbol, batch_dims=2)[:, :, :, None, :], -1)
            residual = tf.math.divide_no_nan(per_symbol(power(own-prediction), symbol, everyone), count)
            predictions.append(prediction)
            residuals.append(tf.where(fitted[..., None], residual, float('inf')))
        residuals = tf.stack(residuals, -1)                                      # [b, u, symbols, numbers]
        best = tf.argmin(residuals, -1, output_type=tf.int32)                    # the fewest entries on a tie
        residual = tf.reduce_min(residuals, -1)
        fitted = tf.cast(tf.math.is_finite(residual), tf.float32)                # per UE and DMRS symbol
        chosen = tf.gather(tf.one_hot(best, entries+1, dtype=own.dtype), symbol, batch_dims=2)   # [b, u, cells, numbers]
        prediction = tf.reduce_sum(tf.stack(predictions, -1)*chosen[:, :, :, None, :], -1)

        noise = tf.broadcast_to(tf.maximum(observation[..., -1:], 0.), tf.shape(tf.math.real(own)))
        noise = tf.math.divide_no_nan(per_symbol(noise, symbol, everyone), count)
        residual = tf.where(fitted > 0., residual, 0.)
        gain = fitted*tf.math.divide_no_nan(noise, noise+tf.maximum(residual-noise, 1e-3*noise))
        gain = tf.gather(gain, symbol, batch_dims=2)                             # [b, u, cells]
        refined = own+tf.cast(gain[..., None], own.dtype)*(prediction-own)

        # first-order model per cell
        at = lambda column: tf.gather(refined, tf.constant(layout.slope_cells[..., column])[None]
                                      +tf.zeros_like(symbol), axis=2, batch_dims=2)
        sloped = tf.reduce_min(fitted, -1)                                       # every DMRS symbol has a fit
        factors = tf.constant(layout.slope_factors)[None]*sloped[..., None, None]   # [b, u, cells, 4]
        factor = lambda column: tf.cast(factors[..., column, None], refined.dtype)
        in_time, in_frequency = at(0)-refined, at(2)-at(1)
        coefficients = tf.stack([refined+in_time*factor(0)+in_frequency*factor(1),
                                 in_time*factor(2), in_frequency*factor(3)], -1)
        return tf.where(active[..., None, None, None], coefficients, 0.)
