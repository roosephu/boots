from typing import List
import tensorflow as tf
import lunzi as lz
from lunzi.typing import *
import lunzi.nn as nn


class MLPVFunction(nn.Module, BaseVFunction):
    def __init__(self, dim_state: int, hidden_sizes: List[int], normalizer: nn.Module = None):
        super().__init__()

        self.mlp = lz.MultiLayerPerceptron((dim_state, *hidden_sizes, 1), nn.ReLU, squeeze=True)
        self.normalizer = normalizer
        self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state])
        self.op_values = self.forward(self.op_states)

    def forward(self, states):
        if self.normalizer is not None:
            states = self.normalizer(states)
        return self.mlp(states)

    @nn.make_method(fetch='values')
    def get_values(self, states): pass

    def copy(self):
        return MLPVFunction(self.mlp.blocks[0], self.mlp.blocks[1:-1], self.normalizer)
