import lunzi.nn as nn
import lunzi as lz
import tensorflow as tf


class MinQFunction(nn.Module, lz.rl.BaseQFunction):
    def __init__(self, dim_state, dim_action, qfns):
        super().__init__()
        self.qfns = nn.ModuleList(qfns)
        self.dim_state = dim_state
        self.dim_action = dim_action

    def build(self):
        with self.scope:
            self.op_states = tf.placeholder(tf.float32, [None, self.dim_state])
            self.op_actions = tf.placeholder(tf.float32, [None, self.dim_action])

        self.op_q = self.forward(self.op_states, self.op_actions)
        return self

    def forward(self, states, actions):
        return tf.reduce_min([qfn(states, actions) for qfn in self.qfns], axis=0)

    @nn.make_method(fetch='q')
    def get_q(self, states, actions): pass
