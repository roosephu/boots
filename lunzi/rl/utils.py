import numpy as np

import gym

from . import BaseVFunction
from ..dataset import Dataset


def compute_advantage(samples: Dataset, gamma: float = 1., lambda_: float = 1., vfn: BaseVFunction = None, n_envs=1):
    assert lambda_ == 1. or vfn is not None, "vfn shouldn't be None if lambda != 1."

    n_steps = len(samples) // n_envs
    samples = samples.reshape((n_steps, n_envs))
    use_next_vf = ~samples['done']
    if 'timeout' in samples.dtype.names:
        use_next_adv = ~(samples['done'] | samples['timeout'])
    else:
        use_next_adv = ~samples['done']

    if lambda_ != 1.:
        next_values = vfn.get_values(samples[-1]['next_state'])
        values = vfn.get_values(samples.reshape(-1)['state']).reshape(n_steps, n_envs)
    else:
        next_values = np.zeros(n_envs)
        values = np.zeros((n_steps, n_envs))
    advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
    last_gae_lambda = 0.

    for t in reversed(range(n_steps)):
        delta = samples[t]['reward'] + gamma * next_values * use_next_vf[t] - values[t]
        advantages[t] = last_gae_lambda = delta + gamma * lambda_ * last_gae_lambda * use_next_adv[t]
        next_values = values[t]
    return advantages.reshape(-1), values.reshape(-1)


def gen_dtype(env: gym.Env, fields: str, dtype='f8'):
    dtypes = {
        'state': ('state', dtype, env.observation_space.shape),
        'action': ('action', dtype, env.action_space.shape),
        'next_state': ('next_state', dtype, env.observation_space.shape),
        'reward': ('reward', dtype),
        'done': ('done', 'bool'),
        'timeout': ('timeout', 'bool'),
        'return_': ('return_', dtype),
        'advantage': ('advantage', dtype),
    }
    return [dtypes[field] for field in fields.split(' ')]
