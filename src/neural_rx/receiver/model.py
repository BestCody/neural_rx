from __future__ import annotations

import tensorflow as tf

from .channel_filter import ChannelMemory
from .pilot_history import PilotHistory
from .decoder_correction import DecoderCorrection


DIMENSIONS = ('history_tbs', 'filter_hidden', 'error_hidden', 'neighbor_radius', 'state_features', 'blend_hidden')


def architecture_settings(settings):
    architecture = settings['architecture']
    expected = {'cgnn_rounds': 2, 'state_features': 56, 'memory': 'pilot_filter_learned_blend',
                'blend_features': 'corrected_endpoints_residual_fit_v1'}
    if any(architecture.get(k) != v for k, v in expected.items()):
        raise ValueError('Expected the current pilot-history receiver with learned decoder blending')
    total, first = architecture['ldpc_iterations'], architecture['ldpc_first_iterations']
    if type(total) is not int or type(first) is not int or not 0 < first < total:
        raise ValueError('The decoder split must use exactly the declared total iteration budget')
    if type(architecture['blend_hidden']) is not int or architecture['blend_hidden'] < 1:
        raise ValueError('Blend width must be a positive integer')
    return {k: architecture[k] for k in DIMENSIONS}


class PilotHistoryReceiver(tf.keras.Model):
    def __init__(self, base_cgnn, layout, mapper, history_tbs=7, neighbor_radius=1,
                 filter_hidden=16, error_hidden=16, state_features=56, blend_hidden=8, **kwargs):
        super().__init__(**kwargs)
        if int(base_cgnn._num_it) != 2 or type(history_tbs) is not int or history_tbs < 1:
            raise ValueError('Pilot history requires K=2 and a positive history length')
        if base_cgnn._num_mcss_supported != 1 or base_cgnn._s_init[0]._output_conv.filters != state_features:
            raise ValueError('The memory architecture requires the configured single-MCS backbone width')
        self.base = base_cgnn
        self.history_tbs, self.neighbor_radius = history_tbs, neighbor_radius
        self.physical = DecoderCorrection(mapper, int(base_cgnn._num_bits_per_symbol[0]))
        self.channel_memory = ChannelMemory(filter_hidden, error_hidden, name='channel_memory')
        self.correction_logit = self.add_weight('correction_logit', shape=(), initializer='zeros')
        self.evidence_calibration = self.add_weight('evidence_calibration', shape=(7,), initializer='zeros')
        self.blend_kernel = self.add_weight('blend_kernel', shape=(8, blend_hidden), initializer='glorot_uniform')
        self.blend_bias = self.add_weight('blend_bias', shape=(blend_hidden,), initializer='zeros')
        self.blend_output = self.add_weight('blend_output', shape=(blend_hidden,), initializer='zeros')
        self.blend_output_bias = self.add_weight('blend_output_bias', shape=(), initializer='zeros')
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
        backbone = list(self.base._s_init[0].trainable_variables)
        for layer in self.base._iterations[:2]:
            backbone += layer.trainable_variables
        backbone += self.base._readout_llrs[0].trainable_variables
        return [self.correction_logit, self.evidence_calibration, *self.blend_variables]+self.channel_memory.trainable_variables, backbone

    @property
    def blend_variables(self):
        return (self.blend_kernel, self.blend_bias, self.blend_output, self.blend_output_bias)

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
                context = self.channel_memory(observation, history, active, layout=layout, defer_uncertainty=True)
                coefficients = tf.where(context.active[..., None, None, None], context.coefficients, 0.)
                channel = layout.expand_channel(coefficients)
                return context, tf.concat([tf.math.real(channel), tf.math.imag(channel)], -1)
            self._preparation_functions[layout.key] = prepare
        return self._preparation_functions[layout.key](observation, history, inputs[3])

    def detect(self, inputs, noise, history, observation=None, *, preparation=None):
        layout = self.layout_for(inputs[2])
        observation = self.observe(inputs[2], noise) if observation is None else observation
        context, channel = self.prepare(inputs, observation, history, layout) if preparation is None else preparation
        refined_inputs = [inputs[0], inputs[1], channel, inputs[3], inputs[4]]
        state = self.base.initialize_state(refined_inputs)
        for iteration in self.base._iterations[:2]:
            state = iteration([state, inputs[1], inputs[3]])
        grid = self.base._readout_llrs[0](state)
        return grid, context

    def decode(self, inputs, noise, ue_ids, history, decoder, mapping, observation=None, preparation=None):
        observation = self.observe(inputs[2], noise) if observation is None else observation
        grid, context = self.detect(inputs, noise, history, observation, preparation=preparation)
        llr = mapping.grid_to_coded(grid)
        with tf.control_dependencies([llr]):
            estimate = self.channel_memory.complete(context, self.layout_for(inputs[2]))
        physical = self.physical.begin(inputs[0], estimate, noise, inputs[3], ue_ids, llr, self.layout_for(inputs[2]))
        code_input, _, state, posterior = decoder.evidence(llr)
        corrected = self.physical.finish(physical, posterior, inputs[3], decoder.symbol_indices, decoder.symbol_signs,
                                         code_input, self.evidence_calibration, decoder.first.llr_limit,
                                         tf.sigmoid(self.correction_logit), self.blend_variables)
        return (decoder.finish(corrected, state), llr, decoder.first.from_code_blocks(corrected),
                decoder.first.from_code_blocks(posterior), observation, estimate)

    def call(self, inputs, noise_variance, history=None, training=None):
        observation = self.observe(inputs[2], noise_variance)
        history = self.empty_history(observation) if history is None else history
        return self.detect(inputs, noise_variance, history, observation)[0], observation
