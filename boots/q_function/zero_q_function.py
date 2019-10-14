import lunzi as lz
import lunzi.nn as nn
import numpy as np
import tensorflow as tf


class ZeroQFunction(nn.Module, lz.rl.BaseQFunction):
    def forward(self, states, actions):
        n = tf.shape(states)[0]
        return tf.zeros(n)

    def get_q(self, states, actions):
        n = len(states)
        return np.zeros(n)
