import tensorflow as tf
import numpy as np
import lunzi.nn as nn


class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, blocks, activation, squeeze=False, weight_initializer=None, output_activation=None):
        super().__init__()

        self._blocks = blocks

        with self.scope:
            kwargs = {}
            if weight_initializer is not None:
                kwargs['weight_initializer'] = weight_initializer
            layers = []
            for in_features, out_features in zip(blocks[:-1], blocks[1:]):
                if layers:
                    layers.append(activation())
                layers.append(nn.Linear(in_features, out_features, **kwargs))
            if squeeze:
                layers.append(nn.Squeeze(axis=-1))
            if output_activation:
                layers.append(output_activation())
            self._modules = {i: module for i, module in enumerate(layers)}

        self._squeeze = squeeze
        self._activation = activation
        self._built = False

    @property
    def blocks(self):
        return self._blocks

    def build(self):
        if not self._built:
            self._built = True
            self.op_inputs = tf.placeholder(tf.float32, [None, self._blocks[0]])
            self.op_outputs = self.forward(self.op_inputs)

    def forward(self, *inputs):
        return super().forward(inputs[0] if len(inputs) == 1 else tf.concat(inputs, axis=-1))

    def fast(self, *inputs):
        return super().fast(np.concatenate(inputs, axis=-1))

    def copy(self):
        return MultiLayerPerceptron(self._blocks, self._activation, self._squeeze)

    def extra_repr(self):
        return f'activation = {self._activation}, blocks = {self._blocks}, squeeze = {self._squeeze}'
