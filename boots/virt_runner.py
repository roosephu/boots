import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import lunzi as lz
import lunzi.nn as nn


class VirtualRunner(nn.Module):
    def __init__(self, dim_state, model, policy, horizon, dtype):
        super().__init__()
        self.model = model
        self.policy = policy
        self.dim_state = dim_state
        self.horizon = horizon
        self.dtype = dtype

    def build(self):
        self.op_init_states = tf.placeholder(tf.float32, [None, self.dim_state])
        self.op_states, self.op_actions, self.op_next_states, self.op_rewards, self.op_dones = \
            self.forward(self.op_init_states, self.horizon)
        return self

    def forward(self, init_states, horizon):
        from collections import namedtuple
        Step = namedtuple('Step', 'state action next_state reward done')
        steps = []
        n_envs = tf.shape(init_states)[0]

        states = self.op_init_states
        for i in range(horizon):
            actions = self.policy(states)
            if isinstance(actions, tfp.distributions.Distribution):
                actions = actions.sample()
            next_states, rewards, dones = self.model(states, actions)
            steps.append(Step(states, actions, next_states, rewards, dones))
            states = next_states

        states, actions, next_states, rewards, dones = [tf.concat(x, axis=0) for x in zip(*steps)]
        return states, actions, next_states, rewards, dones

    def run(self, init_states):
        states, actions, next_states, rewards, dones = \
            self.eval('states actions next_states rewards dones', init_states=init_states)
        timeouts = np.zeros_like(dones)
        return lz.Dataset.fromdict(state=states, action=actions, next_state=next_states, reward=rewards,
                                   done=dones, timeout=timeouts)


__all__ = ['VirtualRunner']
