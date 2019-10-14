from typing import List
import tensorflow as tf
from lunzi import nn, rl, MultiLayerPerceptron


class MLPQFunction(nn.Module, rl.BaseQFunction):
    def __init__(self, dim_state: int, dim_action: int, hidden_states: List[int], normalizer: nn.Module = None):
        super().__init__()
        self._dim_state = dim_state
        self._dim_action = dim_action
        self._normalizer = normalizer

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, [None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, [None, dim_action])
            self.mlp = MultiLayerPerceptron(
                (dim_state + dim_action, *hidden_states, 1), activation=nn.ReLU, squeeze=True)

        self.op_Q = self.forward(self.op_states, self.op_actions)

    def forward(self, states, actions):
        if self._normalizer:
            states = self._normalizer(states)
        return self.mlp.forward(states, actions)

    @nn.make_method(fetch='Q')
    def get_q(self, states, actions): pass

    def copy(self):
        return MLPQFunction(self._dim_state, self._dim_action, self.mlp.blocks[1:-1], self._normalizer)
