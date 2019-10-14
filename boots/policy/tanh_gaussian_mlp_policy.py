from typing import List
import tensorflow as tf
import numpy as np
import lunzi as lz
import lunzi.rl as rl
import lunzi.nn as nn
from boots import TanhNormal
from boots.normalizer import Normalizer


class TanhGaussianMLPPolicy(nn.Module, rl.BasePolicy):
    MIN_LOG = -20
    MAX_LOG = 2
    op_states: lz.Tensor

    def __init__(self, dim_state: int, dim_action: int, hidden_sizes: List[int], normalizer: Normalizer = None):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes
        self.normalizer = normalizer
        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state], name='states')
            self.op_actions_ = tf.placeholder(tf.float32, shape=[None, dim_action], name='actions')

            self.net = lz.MultiLayerPerceptron([dim_state, *self.hidden_sizes, dim_action * 2], activation=nn.ReLU)

        self.distribution = self(self.op_states)
        self.op_actions = self.distribution.sample()

    def forward(self, states):
        if self.normalizer is not None:
            states = self.normalizer(states)
        outputs = self.net(states)
        actions_mean, actions_log_std = outputs[:, :self.dim_action], outputs[:, self.dim_action:]
        actions_log_std = tf.clip_by_value(actions_log_std, TanhGaussianMLPPolicy.MIN_LOG, TanhGaussianMLPPolicy.MAX_LOG)
        distribution = TanhNormal(actions_mean, tf.exp(actions_log_std))
        return distribution

    def get_actions(self, states):
        return self.eval('actions', states=states)

    def copy(self):
        return TanhGaussianMLPPolicy(self.dim_state, self.dim_action, self.hidden_sizes, self.normalizer)


class DetTanhPolicy(nn.Module, lz.rl.BasePolicy):
    def __init__(self, policy: TanhGaussianMLPPolicy):
        super().__init__()
        self.policy = policy
        self.op_states = policy.op_states
        self.op_actions = policy.distribution.mean()

    @nn.make_method(fetch='actions')
    def get_actions(self, states): pass

    def forward(self, states):
        return tf.tanh(self.policy(states).mean())
