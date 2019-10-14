import numpy as np
import tensorflow as tf
import lunzi as lz
from lunzi import nn, Tensor
from boots.normalizer import Normalizers
from boots.utils.average_meter import AverageMeter
from boots.partial_envs import FLAGS as EnvFLAGS


class FLAGS(lz.BaseFLAGS):
    env = EnvFLAGS

    class arch(lz.BaseFLAGS):
        n_blocks = 2
        n_units = 200

    loss = 'L2'  # possibly L1, L2, MSE, G
    lr = 1e-3
    weight_decay = 1e-5
    batch_size = 64
    max_grad_norm = 1000.
    n_dev_iters = 1000
    n_pretrain_iters = 10000


class FixupResBlock(nn.Module):
    def __init__(self, x, n_total_blocks):
        super().__init__()
        std = np.sqrt(2. / x / n_total_blocks)
        self.bias1a = nn.Parameter(tf.zeros(1), name='bias1a')
        self.fc1 = nn.Linear(x, x, bias=False, weight_initializer=tf.initializers.random_normal(0, stddev=std))
        self.bias1b = nn.Parameter(tf.zeros(1), name='bias1b')
        self.relu = nn.ReLU()
        self.bias2a = nn.Parameter(tf.zeros(1), name='bias2a')
        self.fc2 = nn.Linear(x, x, bias=False, weight_initializer=tf.initializers.zeros())
        self.scale = nn.Parameter(tf.ones(1), name='scale')
        self.bias2b = nn.Parameter(tf.zeros(1), name='bias2b')

    def forward(self, x):
        y = x
        y = self.fc1(y + self.bias1a)
        y = self.relu(y + self.bias1b)
        y = self.fc2(y + self.bias2a)
        y = y * self.scale + self.bias2b
        return self.relu(x + y)


def make_blocks(block, x, n_blocks, n_total_blocks):
    return nn.Sequential(*[block(x, n_total_blocks) for _ in range(n_blocks)])


class DeterministicModel(nn.Module):
    FLAGS = FLAGS

    @FLAGS.inject
    def __init__(self, dim_state: int, dim_action: int, normalizers: Normalizers, *, arch: FLAGS.arch):
        super().__init__()
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1e-5)

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.op_states = tf.placeholder(tf.float32, shape=[None, self.dim_state], name='states')
        self.op_actions = tf.placeholder(tf.float32, shape=[None, self.dim_action], name='actions')
        self.mlp = nn.Sequential(
            nn.Linear(dim_state + dim_action, arch.n_units, weight_initializer=initializer),
            nn.ReLU(),
            make_blocks(FixupResBlock, arch.n_units, arch.n_blocks, arch.n_blocks),
            nn.Linear(arch.n_units, dim_state, weight_initializer=initializer),
        )

        self.normalizers = normalizers
        self.build()

    def build(self):
        # self.op_next_states, self.op_rewards, self.op_dones = self.forward(self.op_states, self.op_actions)
        self.op_next_states, _, _ = self.forward(self.op_states, self.op_actions)

    def get_rewards(self, states, actions, next_states):
        if FLAGS.env.id == 'ModelBasedHumanoid-v2':
            actions = 0.4 * actions   # TODO: this is ad-hoc hack

            pos_before = states[:, -1]
            pos_after = next_states[:, -1]
            alive_bonus = 5.0
            lin_vel_cost = 1.25 * (pos_after - pos_before) / 0.015
            quad_ctrl_cost = 0.1 * tf.reduce_sum(tf.math.square(actions), axis=-1)
            quad_impact_cost = .5e-6 * tf.reduce_sum(tf.math.square(states[:, 292:376]), axis=-1)  # cfrc_ext
            quad_impact_cost = tf.clip_by_value(quad_impact_cost, 0, 10)
            rewards = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
            height = next_states[..., 0]
            dones = tf.logical_or(tf.less(height, 1.0), tf.greater(height, 2.0))
            return rewards, dones
        elif FLAGS.env.id == 'ModelBasedHalfCheetah-v2':
            xposbefore = states[:, 0]
            xposafter = next_states[:, 0]
            reward_ctrl = - 0.1 * tf.reduce_sum(tf.math.square(actions), axis=-1)
            reward_run = (xposafter - xposbefore) / 0.05
            reward = reward_ctrl + reward_run
            return reward, tf.zeros_like(reward, dtype=np.bool)
        elif FLAGS.env.id == 'ModelBasedWalker2d-v2':
            posbefore = states[:, 0]
            posafter = next_states[:, 0]
            height = next_states[:, 1]
            ang = next_states[:, 2]
            alive_bonus = 1.0
            reward = ((posafter - posbefore) / 0.008)
            reward += alive_bonus
            # reward -= 1e-3 * tf.square(actions).reduce_sum(axis=-1)
            dones = tf.logical_not(tf.logical_or(tf.greater(height, 0.8), tf.less(height, 2.0)))
            # done = ~((height > 0.8) & (height < 2.0) & (ang > -1.0) & (ang < 1.0))
            return reward, dones
        elif FLAGS.env.id == 'ModelBasedAnt-v2':
            xposbefore = states[:, -1]
            xposafter = next_states[:, -1]
            forward_reward = (xposafter - xposbefore) / 0.05

            ctrl_cost = .5 * tf.reduce_sum(tf.square(actions), axis=-1)
            contact_cost = 0.5 * 1e-3 * tf.reduce_sum(
                tf.square(tf.clip_by_value(next_states[:, 27:111], -1, 1)), axis=-1)
            survive_reward = 1.0
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward
            dones = tf.logical_not(tf.logical_and(tf.greater(next_states[:, 0], 0.2), tf.less(next_states[:, 0], 1.0)))
            # done = ~((next_states[:, 0] >= 0.2) & (next_states[:, 0] <= 1.0))
            return reward, dones
        return None, None
        # assert 0

    def forward(self, states, actions):
        # assert actions.shape[-1] == self.dim_action
        inputs = tf.concat([self.normalizers.state(states), tf.clip_by_value(actions, -1., 1.)], axis=1)

        normalized_diffs = self.mlp.forward(inputs)
        next_states = states + self.normalizers.diff(normalized_diffs, inverse=True)

        rewards, dones = self.get_rewards(states, actions, next_states)
        return next_states, rewards, dones

    @nn.make_method(fetch='next_states')
    def get_next_states(self, states, actions): pass


class DeterministicModelTrainer(nn.Module):
    @FLAGS.inject
    def __init__(self, model: DeterministicModel, dim_state: int, dim_action: int, normalizers, datasets,
                 *, loss, batch_size):
        super().__init__()

        self.model = model
        self.iterators = {
            'train': datasets['train'].iterator(batch_size, n_epochs=-1),
            'dev': datasets['dev'].sample_iterator(1024),
        }
        self._n_updates = 0
        criterion_map = {
            'L1': nn.L1Loss(),
            'L2': nn.L2Loss(),
            'MSE': nn.MSELoss(),
            # 'G': DescLoss(vfn, normalizers, dim_state),
        }
        self.normalizers = normalizers
        self.criterion = criterion_map[loss]

        with self.scope:
            self.op_states = self.model.op_states
            self.op_actions = self.model.op_actions
            self.op_next_states = self.model.op_next_states
            self.op_next_states_ = tf.placeholder(tf.float32, shape=[None, dim_state])

        self.build()
        self.train_loss_meter = AverageMeter()

    @FLAGS.inject
    def build(self, lr: float, weight_decay: float, max_grad_norm: float):
        next_states = self.model.op_next_states
        diffs = next_states - self.op_next_states_
        weighted_diffs = diffs / tf.maximum(self.normalizers.diff.op_std, 1e-6)
        self.op_loss = self.criterion(weighted_diffs, 0)

        loss = tf.reduce_mean(self.op_loss)

        optimizer = tf.train.AdamOptimizer(lr)
        params = self.model.parameters()
        regularization = weight_decay * tf.add_n([tf.nn.l2_loss(t) for t in params], name='regularization')

        grads_and_vars = optimizer.compute_gradients(loss + regularization, var_list=params)
        print([var.shape.as_list() for grad, var in grads_and_vars])
        clip_grads, op_grad_norm = tf.clip_by_global_norm([grad for grad, _ in grads_and_vars], max_grad_norm)
        clip_grads_and_vars = [(grad, var) for grad, (_, var) in zip(clip_grads, grads_and_vars)]
        self.op_train = optimizer.apply_gradients(clip_grads_and_vars)
        self.op_grad_norm = op_grad_norm

    def forward(self):
        pass

    @nn.make_method(fetch='loss')
    def get_loss(self, states, next_states_, actions) -> np.ndarray: pass

    @FLAGS.inject
    def step(self, *, n_dev_iters, _log, _writer):
        self._n_updates += 1

        samples = next(self.iterators['train'])
        _, train_loss, grad_norm = self.get_loss(
            samples['state'], samples['next_state'], samples['action'], fetch='train loss grad_norm')
        self.train_loss_meter.update(train_loss.mean())
        # ideally, we should define an Optimizer class, which takes parameters as inputs.
        # The `update` method of `Optimizer` will invalidate all parameters during updates.
        for param in self.model.parameters():
            param.invalidate()

        if self._n_updates % n_dev_iters == 0:
            train_loss = self.train_loss_meter.get()
            self.train_loss_meter.reset()
            samples = next(self.iterators['dev'])
            dev_loss = self.get_loss(
                samples['state'], samples['next_state'], samples['action'])
            dev_loss = dev_loss.mean()
            _log.info('# Iter %3d: Loss = [train = %.3f, dev = %.3f], grad_norm = %.6f',
                      self._n_updates, train_loss, dev_loss, grad_norm)
            _writer.add_scalar('model/loss/dev', dev_loss)
            _writer.add_scalar('model/loss/train', train_loss)
            _writer.add_scalar('model/grad_norm', grad_norm)

    @FLAGS.inject
    def pretrain(self, *, n_pretrain_iters):
        for _ in range(n_pretrain_iters):
            self.step()
