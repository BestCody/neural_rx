from __future__ import annotations

import tensorflow as tf

from .channel_filter import ChannelFilter
from .pilot_history import PilotHistory


DIMENSIONS = ('history_tbs', 'fit_tbs', 'neighbor_radius', 'state_features')


def architecture_settings(settings):
    architecture = settings['architecture']
    expected = {'cgnn_rounds': 2, 'state_features': 56, 'memory': 'pilot_filter'}
    if any(architecture.get(k) != v for k, v in expected.items()):
        raise ValueError('Expected the current pilot-history receiver')
    return {k: architecture[k] for k in DIMENSIONS}


class PilotHistoryReceiver(tf.keras.Model):
    """Two CGNN iterations whose channel input is the LS estimate refined by a filter over the current pilots and a
    memory of the LS estimates of earlier TBs. The filter's weights are fitted per UE on the memory (ChannelFilter), so
    the receiver's only trained weights are those of the CGNN."""

    def __init__(self, base_cgnn, layout, history_tbs=7, fit_tbs=3, neighbor_radius=1, state_features=56, **kwargs):
        super().__init__(**kwargs)
        if int(base_cgnn._num_it) != 2 or type(history_tbs) is not int or history_tbs < fit_tbs:
            raise ValueError('Pilot history requires K=2 and a memory of at least the fitted TBs')
        if base_cgnn._num_mcss_supported != 1 or base_cgnn._s_init[0]._output_conv.filters != state_features:
            raise ValueError('The memory architecture requires the configured single-MCS backbone width')
        self.base = base_cgnn
        self.history_tbs, self.neighbor_radius = history_tbs, neighbor_radius
        self.channel_filter = ChannelFilter(fit_tbs)
        self.layouts = {}
        self.register_layout(layout)
        self._preparation_functions = self._no_dependency({})

    def register_layout(self, layout):
        if layout.neighbor_radius != self.neighbor_radius:
            raise ValueError('Pilot lookup and model neighborhood radii differ')
        old = self.layouts.get(layout.key)
        if old is not None and old.fingerprint != layout.fingerprint:
            raise ValueError('Pilot layout changed; construct a receiver with empty history')
        self.layouts[layout.key] = layout

    def layout_for(self, channel):
        return self.layouts[tuple(channel.shape[1:4])]

    def optimization_variables(self):
        variables = list(self.base._s_init[0].trainable_variables)
        for layer in self.base._iterations[:2]:
            variables += layer.trainable_variables
        variables += self.base._readout_llrs[0].trainable_variables
        variables += self.base._readout_chest.trainable_variables
        return variables

    def observe(self, channel, noise_variance):
        return self.layout_for(channel).pack(channel, noise_variance)

    def empty_history(self, observation):
        shape = tf.shape(observation)
        records = tf.zeros([shape[0], shape[1], self.history_tbs, shape[2], shape[3]], observation.dtype)
        metadata = tf.zeros([shape[0], shape[1], self.history_tbs], tf.int32)
        return PilotHistory(records, metadata, metadata, tf.zeros_like(metadata, tf.bool))

    def prepare(self, inputs, observation, history, layout):
        if layout.key not in self._preparation_functions:
            @tf.function(jit_compile=True, reduce_retracing=True)
            def prepare(observation, history, active):
                observation = tf.stop_gradient(observation)
                history = history._replace(observations=tf.stop_gradient(history.observations))
                coefficients = self.channel_filter(observation, history, active, layout)
                channel = layout.expand_channel(coefficients)
                return coefficients, tf.concat([tf.math.real(channel), tf.math.imag(channel)], -1)
            self._preparation_functions[layout.key] = prepare
        return self._preparation_functions[layout.key](observation, history, inputs[3])

    def iterate(self, inputs):
        """The two CGNN iterations on prepared inputs: LLRs on the grid and the final state."""
        state = self.base.initialize_state(inputs)
        for iteration in self.base._iterations[:2]:
            state = iteration([state, inputs[1], inputs[3]])
        return self.base._readout_llrs[0](state), state

    def detect_state(self, inputs, noise, history, observation=None, *, preparation=None):
        layout = self.layout_for(inputs[2])
        observation = self.observe(inputs[2], noise) if observation is None else observation
        coefficients, channel = (self.prepare(inputs, observation, history, layout) if preparation is None
                                 else preparation)
        grid, state = self.iterate([inputs[0], inputs[1], channel, inputs[3], inputs[4]])
        return grid, coefficients, state

    def detect(self, inputs, noise, history, observation=None, *, preparation=None):
        return self.detect_state(inputs, noise, history, observation, preparation=preparation)[:2]

    def channel_estimate(self, state):
        """The network's channel estimate from its final state, [batch, users, frequency, symbols, 2 antennas], on
        the scale of its inputs (the received signal normalized to unit power, see input_scale)."""
        return self.base._readout_chest(state)

    def call(self, inputs, noise_variance, history=None, training=None):
        observation = self.observe(inputs[2], noise_variance)
        history = self.empty_history(observation) if history is None else history
        return self.detect(inputs, noise_variance, history, observation)[0], observation
