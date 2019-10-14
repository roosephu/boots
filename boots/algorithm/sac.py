from typing import List

import numpy as np
import tensorflow as tf
import lunzi as lz
from lunzi import nn


class FLAGS(lz.BaseFLAGS):
    lr = 3.e-4
    tau = 5e-3
    alpha = None   # set to None or 0.0 to enable auto alpha tuning
    gamma = 0.99
    batch_size = 256
    target_update = 1
    target_entropy = None  # None == -dim_action

    min_pool = 1000
    n_exploration_steps = 10_000
    n_grad_iters = 1


class SoftActorCritic(nn.Module):
    FLAGS = FLAGS

    @FLAGS.inject
    def __init__(self, qfns: List[lz.rl.BaseNNQFunction], policy: lz.rl.BaseNNPolicy, dim_state: int, dim_action: int,
                 *, alpha):
        super().__init__()
        self.qfns = qfns
        self.qfns_target = [qfn.copy() for qfn in qfns]
        self.policy = policy
        self.dim_action = dim_action

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, [None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, [None, dim_action])
            self.op_rewards = tf.placeholder(tf.float32, [None])
            self.op_next_states = tf.placeholder(tf.float32, [None, dim_state])
            self.op_dones = tf.placeholder(tf.float32, [None])

        if alpha:
            self.auto_entropy = False
            self.op_alpha = tf.constant(alpha, dtype=tf.float32)
        else:
            self.auto_entropy = True
            self.log_alpha = nn.Parameter(0.0, name='alpha', dtype=tf.float32)
            self.op_alpha = tf.exp(self.log_alpha)

        self.op_qfn_losses, self.op_train_qfn = self.train_qfn(
            self.op_states, self.op_actions, self.op_rewards, self.op_next_states, self.op_dones)
        with tf.control_dependencies([self.op_train_qfn]):
            self.op_train_policy, self.op_train_alpha = self.train_policy(self.op_states)
        self.op_update_targets = self.update_targets()

        self._n_updates = 0

    def init(self):
        for qfn, qfn_target in zip(self.qfns, self.qfns_target):
            qfn_target.load_state_dict(qfn.state_dict())

    def forward(self):
        pass

    @FLAGS.inject
    def train_qfn(self, states, actions, rewards, next_states, dones, *, gamma, lr):
        next_actions, log_prob_actions = self.policy(next_states).sample_with_log_prob()
        next_qf = tf.reduce_min([qfn_target(next_states, next_actions) for qfn_target in self.qfns_target], axis=0)

        qf_ = tf.stop_gradient(rewards + (1 - dones) * gamma * (next_qf - self.op_alpha * log_prob_actions))
        optimizer = tf.train.AdamOptimizer(lr)
        qfn_losses = [tf.reduce_mean(tf.pow(qfn(states, actions) - qf_, 2)) * 0.5 for qfn in self.qfns]
        op_train = tf.group(
            *[optimizer.minimize(qf_loss, var_list=qfn.parameters()) for qfn, qf_loss in zip(self.qfns, qfn_losses)])

        return tf.stack(qfn_losses), op_train

    @FLAGS.inject
    def train_policy(self, states, *, lr, target_entropy):
        distribution: tf.distributions.Distribution = self.policy(states)
        actions, log_prob_actions = distribution.sample_with_log_prob()
        min_qf = tf.reduce_min([qfn(states, actions) for qfn in self.qfns], axis=0)
        self.op_dist_std = tf.reduce_mean(tf.exp(tf.reduce_mean(tf.log(distribution.stddev()), axis=1)))
        policy_loss = tf.reduce_mean(self.op_alpha * log_prob_actions - min_qf)

        train_policy = tf.train.AdamOptimizer(lr).minimize(policy_loss, var_list=self.policy.parameters())

        if self.auto_entropy:
            if target_entropy is None:
                target_entropy = -self.dim_action
            alpha_loss = -self.log_alpha * tf.stop_gradient(tf.reduce_mean(log_prob_actions) + target_entropy)
            train_alpha = tf.train.AdamOptimizer(lr).minimize(alpha_loss, var_list=[self.log_alpha])
        else:
            train_alpha = tf.no_op()

        return train_policy, train_alpha

    @FLAGS.inject
    def update_targets(self, *, tau):
        ops = []
        for qfn, qfn_target in zip(self.qfns, self.qfns_target):
            for p, q in zip(qfn.parameters(), qfn_target.parameters()):
                ops.append(tf.assign(q, p * tau + q * (1 - tau)))

        return tf.group(*ops)

    @FLAGS.inject
    def train(self, samples: lz.Dataset, *, target_update, _log):
        self._n_updates += 1
        qfn_losses, dist_std, alpha, *_ = self.eval(
            'qfn_losses dist_std alpha train_qfn train_policy train_alpha',
            states=samples.state, actions=samples.action, next_states=samples.next_state, rewards=samples.reward,
            dones=samples.done & ~samples['timeout'])
        if self._n_updates % target_update == 0:
            self.eval('update_targets')
        if self._n_updates % 1000 == 0:
            _log.info(f'[SAC] # {self._n_updates:d}: alpha = {alpha:.3f}, Q loss = {qfn_losses.round(3)}, '
                      f'dist_std = {dist_std:.3f}')
        for p in self.policy.parameters():
            p.invalidate()

    def reset(self):
        self._n_updates = 0
