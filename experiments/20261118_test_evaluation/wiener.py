"""Classical LMMSE (Wiener) channel estimation over several slots, from known channel statistics.

The estimator uses what the proposed NRX's pilot memory holds, i.e. the LS estimates at the pilots of the current TB
and of earlier TBs of the UE, but no fitted weights: its filters follow from the second-order statistics of the
channel, as in the LMMSE baseline of the sweep (src/neural_rx/benchmark/baselines.py), whose covariance matrices over
the antennas and over the subcarriers of a block of PRBs it shares. Like that baseline (Sionna's LMMSEInterpolator),
it filters one dimension after the other and passes the error variance on:

  antennas     every LS estimate over the BS antennas;
  frequency    every DMRS symbol of every TB over the subcarriers of its block;
  time         every subcarrier over the DMRS symbols of the current TB and of the stored TBs, to all OFDM symbols of
               the current slot. This step needs the correlation of the channel over several slots (statistics.py).

Between two steps, the LMMSE estimate h_hat with error variance e and prior variance r is scaled by r/(r-e), which
makes it the channel plus an error of variance r e/(r-e) that the next step treats as noise. With `history=False`,
only the current TB is used (a single-slot LMMSE estimator, for comparison with the sweep's baseline).

    estimator = SlotWiener(tf, layout, covariance, time_correlation, blocks)
    h_hat, err_var = estimator(observation, history, active)     # [batch, users, antennas, symbols, subcarriers]

`observation` and `history` are those of the NRX's adapter (PilotLayout.pack, PilotMemoryBuffer.gather).
"""
import numpy as np


def solve(tf, matrix, rhs):
    """Solution of matrix @ x = rhs for Hermitian positive definite matrices [..., n, n] and right-hand sides
    [..., n, k] by Gauss-Jordan elimination: plain tensor operations, as solve_positive_definite of the receiver,
    because the GPU's solver library fails when TensorFlow holds all of the GPU's memory."""
    size = matrix.shape[-1]
    system = tf.concat([matrix, rhs], -1)
    for k in range(size):
        row = system[..., k, :]/system[..., k, k, None]
        unit = tf.one_hot(k, size, dtype=system.dtype)
        system += (unit-system[..., :, k])[..., :, None]*row[..., None, :]      # clear column k, set row k
    return system[..., size:]


class SlotWiener:
    def __init__(self, tf, layout, covariance, time_correlation, blocks, history=True):
        self.tf, self.history = tf, history
        users, cells, self.symbols, frequency = layout.users, layout.count, layout.symbols, layout.frequency
        coordinates = np.asarray(layout.coordinates, np.float64)           # [port, cell, (symbol, subcarrier)]
        space = np.asarray(covariance["space"], np.complex128)
        block = np.asarray(covariance["frequency"], np.complex128)         # one block of subcarriers
        width = frequency//blocks
        if frequency % blocks or block.shape != (width, width):
            raise ValueError("The frequency covariance does not fit the blocks of the grid")
        dmrs = np.unique(coordinates[..., 0])
        per_block = cells//(len(dmrs)*blocks)
        if cells != len(dmrs)*blocks*per_block:
            raise ValueError("The pilot cells do not divide into DMRS symbols and blocks")
        order = np.lexsort((coordinates[..., 1], coordinates[..., 0]))     # per port: by DMRS symbol, then subcarrier
        shape = (users, len(dmrs), blocks, per_block)
        symbol = np.take_along_axis(coordinates[..., 0], order, 1).reshape(shape)
        local = np.take_along_axis(coordinates[..., 1], order, 1).reshape(shape)-(np.arange(blocks)*width)[:, None]
        if not (np.all(symbol == dmrs[None, :, None, None]) and np.all(local == local[:, :1, :1])):
            raise ValueError("The pilot cells are not the same in every DMRS symbol and block")
        local = local[:, 0, 0]                                             # [port, cell of a block]
        # A cell between two subcarriers observes their mean.
        pick = np.zeros((users, per_block, width))
        low = np.floor(local).astype(int)
        for port in range(users):
            pick[port, np.arange(per_block), low[port]] = 1.-(local[port]-low[port])
            high = local[port] > low[port]
            pick[port, np.arange(per_block)[high], low[port][high]+1] = (local[port]-low[port])[high]
        observed = np.einsum("pif,fg,pjg->pij", pick, block, pick)         # covariance of a block's cells
        mu, vectors = np.linalg.eigh(observed)
        loads, basis = np.linalg.eigh(space)
        complex_ = lambda value: tf.constant(value, tf.complex64)
        real = lambda value: tf.constant(value, tf.float32)
        self.order = tf.constant(order, tf.int32)
        self.mu, self.vectors = real(np.maximum(mu, 0.)), complex_(vectors)
        self.spread = complex_(np.einsum("fg,pjg,pjk->pfk", block, pick, vectors))    # block covariance x eigenvectors
        self.loads, self.basis = real(np.maximum(loads, 0.)), complex_(basis)
        self.share = real(np.abs(basis)**2)                                # [antenna, eigenvector]
        self.prior = real(np.real(np.diag(space)))                         # [antenna]
        self.block_power = real(np.real(np.diag(block)))                   # [subcarrier of a block]
        self.dmrs, self.dmrs_count, self.blocks, self.per_block, self.width = real(dmrs), len(dmrs), blocks, per_block, width
        self.antennas = space.shape[0]
        correlation = np.asarray(time_correlation, np.complex128)          # lags 0, 1, ... OFDM symbols
        self.reach = len(correlation)-1
        self.correlation = complex_(np.concatenate([np.conj(correlation[:0:-1]), correlation]))

    def correlate(self, lag):
        """Time correlation E[h(t+lag) h(t)^*] at any lag in OFDM symbols (linear between the stored lags)."""
        tf = self.tf
        lag = tf.clip_by_value(lag, -float(self.reach), float(self.reach))
        low = tf.floor(lag)
        index = tf.cast(low, tf.int32)+self.reach
        part = tf.cast(lag-low, tf.complex64)
        return ((1.-part)*tf.gather(self.correlation, index)
                +part*tf.gather(self.correlation, tf.minimum(index+1, 2*self.reach)))

    def __call__(self, observation, history, active):
        tf, antennas = self.tf, self.antennas
        cast = lambda value: tf.cast(value, tf.complex64)
        active = tf.cast(active, tf.bool)
        shape = tf.shape(observation)
        batch, users = shape[0], shape[1]
        own_port = tf.broadcast_to(tf.range(users, dtype=tf.int32)[None], [batch, users])
        stored = history.valid & active[..., None] if self.history else tf.zeros_like(history.valid)
        records = tf.concat([observation[:, :, None], history.observations], 2)    # [batch, users, records, cells, .]
        ports = tf.concat([own_port[..., None], history.ports], 2)
        ages = tf.concat([tf.zeros_like(own_port)[..., None], history.age], 2)
        valid = tf.concat([active[..., None], stored], 2)
        count = records.shape[2]
        estimate = tf.complex(records[..., :antennas], records[..., antennas:2*antennas])
        noise = tf.reduce_mean(tf.maximum(records[..., -1], 0.), -1)               # [batch, users, records]

        # antennas
        loads = self.loads[None, None, None]                                       # [1, 1, 1, eigenvector]
        total = loads+noise[..., None]
        parts = tf.einsum("ai,burca->burci", tf.math.conj(self.basis), estimate)
        parts *= cast(tf.math.divide_no_nan(loads, total))[:, :, :, None]
        estimate = tf.einsum("ai,burci->burca", self.basis, parts)
        error = tf.einsum("ai,buri->bura", self.share, tf.math.divide_no_nan(loads*noise[..., None], total))
        scale = self.prior/tf.maximum(self.prior-error, 1e-6*self.prior)           # [batch, users, records, antenna]
        estimate, noise = estimate*cast(scale)[:, :, :, None], error*scale

        # frequency, per DMRS symbol and block
        estimate = tf.gather(estimate, tf.gather(self.order, ports), axis=3, batch_dims=3)
        estimate = tf.reshape(estimate, [batch, users, count, -1, self.blocks, self.per_block, antennas])
        vectors, spread = tf.gather(self.vectors, ports), tf.gather(self.spread, ports)
        mu = tf.gather(self.mu, ports)                                             # [batch, users, records, cell]
        parts = tf.einsum("burij,burmkia->burmkja", tf.math.conj(vectors), estimate)
        inverse = tf.math.divide_no_nan(1., self.prior[:, None]*mu[:, :, :, None]+noise[..., None])
        parts *= cast(tf.transpose(self.prior[:, None]*inverse, [0, 1, 2, 4, 3]))[:, :, :, None, None]
        estimate = tf.einsum("burfj,burmkja->burmkfa", spread, parts)              # [.., symbol, block, subcarrier, a]
        prior = self.block_power[:, None]*self.prior[None]                         # [subcarrier of a block, antenna]
        error = prior-self.prior**2*tf.einsum("burfj,buraj->burfa", tf.abs(spread)**2, inverse)
        scale = prior/tf.maximum(prior-error, 1e-6*prior)
        estimate *= cast(scale)[:, :, :, None, None]
        noise = tf.tile(tf.maximum(error, 0.)*scale, [1, 1, 1, self.blocks, 1])    # [batch, users, records, f, a]
        prior = tf.tile(prior, [self.blocks, 1])                                   # [subcarrier, antenna]
        estimate = tf.reshape(estimate, [batch, users, count, -1, self.blocks*self.width, antennas])

        # time: the DMRS symbols of all valid records, to every OFDM symbol of the current slot
        inputs = count*self.dmrs_count
        moment = self.dmrs[None, None, None]-float(self.symbols)*tf.cast(ages, tf.float32)[..., None]
        moment = tf.reshape(moment, [batch, users, inputs])
        among = self.correlate(moment[..., :, None]-moment[..., None, :])          # [batch, users, n, n]
        toward = self.correlate(tf.range(self.symbols, dtype=tf.float32)[None, None, :, None]-moment[:, :, None])
        estimate = tf.reshape(estimate, [batch, users, inputs, self.blocks*self.width, antennas])
        estimate = tf.transpose(estimate, [0, 1, 3, 4, 2])                         # [batch, users, f, a, n]
        usable = tf.reshape(tf.repeat(valid, self.dmrs_count, axis=2), [batch, users, 1, 1, inputs])
        noise = tf.transpose(tf.repeat(noise, self.dmrs_count, axis=2), [0, 1, 3, 4, 2])
        noise = tf.where(usable, noise, 1e6*prior[None, None, :, :, None])         # left out: no information
        system = cast(prior)[None, None, :, :, None, None]*among[:, :, None, None]+tf.linalg.diag(cast(noise))
        target = tf.math.conj(tf.transpose(toward, [0, 1, 3, 2]))[:, :, None, None]        # [batch, users, 1, 1, n, s]
        weights = tf.math.conj(solve(tf, system, tf.broadcast_to(target, tf.concat(
            [tf.shape(system)[:-1], [self.symbols]], 0))))                         # [batch, users, f, a, n, s]
        channel = cast(prior)[None, None, :, :, None]*tf.einsum("bufans,bufan->bufas", weights, estimate)
        explained = tf.math.real(tf.einsum("bufans,busn->bufas", weights, tf.math.conj(toward)))
        error = prior[None, None, :, :, None]*(1.-prior[None, None, :, :, None]*explained)
        keep = active[..., None, None, None]
        channel = tf.where(keep, tf.transpose(channel, [0, 1, 3, 4, 2]), tf.zeros([], tf.complex64))
        error = tf.where(keep, tf.transpose(tf.maximum(error, 0.), [0, 1, 3, 4, 2]), 0.)
        return channel, error
