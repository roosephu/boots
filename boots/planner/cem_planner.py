import numpy as np

import lunzi as lz
import lunzi.nn as nn
from boots.policy.tanh_gaussian_mlp_policy import TanhGaussianMLPPolicy
import tensorflow as tf
from .reparametrize import reparametrize


class FLAGS(lz.BaseFLAGS):
    n_candidates = 500
    h_plan = 4
    h_sim = 0
    horizon = 4

    n_cem_iters = 5
    n_elites = 50
    alpha = 0.5

    init_std = 1.0

    @classmethod
    def finalize(cls):
        cls.horizon = cls.h_plan + cls.h_sim


class CEMPlanner(nn.Module, lz.rl.BasePolicy):
    FLAGS = FLAGS

    @FLAGS.inject
    def __init__(self, dim_state, dim_action, model, policy: TanhGaussianMLPPolicy, qfn,
                 *, init_std, horizon, n_candidates, h_plan, h_sim, n_cem_iters, n_elites, alpha):
        super().__init__()
        self.policy = policy
        self.model = model
        self.qfn = qfn

        self.dim_state = dim_state
        self.dim_action = dim_action

        self.n_steps = 0
        self.init_std = init_std
        self.horizon = h_plan + h_sim
        self.h_plan = h_plan
        self.h_sim = h_sim
        self.n_cem_iters = n_cem_iters
        self.n_elites = n_elites
        self.alpha = alpha
        self.n_candidates = n_candidates

        self.reset()

    def build(self):
        self.op_states = tf.placeholder(tf.float32, [None, self.dim_state])
        self.op_xis = tf.placeholder(tf.float32, [self.horizon, None, self.dim_action])
        self.op_best_xi = tf.placeholder(tf.float32, [None, self.dim_action])
        self.op_actions = reparametrize(self.policy(self.op_states), self.op_best_xi)

        states = self.op_states
        from collections import namedtuple
        Step = namedtuple('Step', 'reward done')

        steps = []
        returns = 0
        for i in range(self.horizon):
            actions = reparametrize(self.policy(states), self.op_xis[i])
            next_states, rewards, dones = self.model(states, actions)
            steps.append(Step(rewards, dones))
            states = next_states
            if i == self.horizon - 1:
                returns = self.qfn(states, actions)

        for t in reversed(range(self.horizon - 1)):
            returns = (1. - tf.cast(steps[t].done, tf.float32)) * returns + steps[t].reward

        self.op_returns = returns
        return self

    def forward(self, states, actions):
        pass

    @FLAGS.inject
    def get_actions(self, states, *, _log):
        self.n_steps += 1
        n = len(states)
        assert n == 1
        repeated_states = np.repeat(states, self.n_candidates, axis=0)

        best_xi = np.zeros([1, self.dim_action])
        for _ in range(self.n_cem_iters):
            candidates = np.random.randn(self.h_plan + self.h_sim, self.n_candidates, self.dim_action)
            candidates[:self.h_plan] = candidates[:self.h_plan] * self.sigma.reshape((-1, 1, self.dim_action)) + \
                self.mu.reshape((-1, 1, self.dim_action))

            returns = self.eval('returns', states=repeated_states, xis=candidates)
            indices = np.argsort(returns)
            top_k = indices[-self.n_elites:]
            self.mu = (1 - self.alpha) * self.mu + self.alpha * np.mean(candidates[:self.h_plan, top_k, :], axis=1)
            self.sigma = (1 - self.alpha) * self.sigma + self.alpha * np.std(candidates[:self.h_plan, top_k, :], axis=1)

            best_xi = candidates[0, [indices[-1]]]
        actions = self.eval('actions', states=states, best_xi=best_xi)
        if self.n_steps % 10000 == 0:
            _log.info(f'step {self.n_steps}, first sigma mu: {np.round(self.mu[0], 3)}, {np.round(self.sigma[0], 3)}, '
                      f'actions = {np.round(actions, 3)}')
        return actions

    def reset(self, indices=None):
        self.mu = np.zeros([self.h_plan, self.dim_action])
        self.sigma = np.ones([self.h_plan, self.dim_action])

    def step(self):
        if self.h_plan != 0:
            self.mu = np.vstack([self.mu[1:, :], np.zeros((1, self.dim_action))])
            self.sigma = np.vstack([self.sigma[1:, :], np.ones([1, self.dim_action])])
