from typing import Union
import numpy as np
import gym
from .env import BaseBatchedEnv
from .policy import BasePolicy
from ..dataset import Dataset
from .utils import gen_dtype


class Runner(object):
    _states: np.ndarray  # [np.float]
    _n_steps: np.ndarray
    _returns: np.ndarray

    def __init__(self, env: Union[BaseBatchedEnv, gym.Wrapper], max_steps: int):
        self.env = env
        self.n_envs = env.n_envs
        self.max_steps = max_steps
        self._dtype = gen_dtype(env, 'state action next_state reward done timeout')

        self.reset()

    def reset(self):
        self.set_state(self.env.reset(), set_env_state=False)

    def set_state(self, states: np.ndarray, set_env_state=True):
        self._states = states.copy()
        if set_env_state:
            self.env.set_state(states)
        self._n_steps = np.zeros(self.n_envs, 'i4')
        self._returns = np.zeros(self.n_envs, 'f8')

    def get_state(self):
        return self._states.copy()

    def run(self, policy: BasePolicy, n_samples: int):
        ep_infos = []
        n_steps = n_samples // self.n_envs
        assert n_steps * self.n_envs == n_samples
        dataset = Dataset(self._dtype, n_samples).reshape(n_steps, self.n_envs)

        for T in range(n_steps):
            actions = policy.get_actions(self._states)

            next_states, rewards, dones, infos = self.env.step(actions)
            if hasattr(policy, 'step'):
                policy.step()
            dones = dones.astype(bool)
            self._returns += rewards
            self._n_steps += 1
            timeouts = self._n_steps == self.max_steps

            steps = [self._states.copy(), actions, next_states.copy(), rewards, dones, timeouts]
            dataset[T] = np.rec.fromarrays(steps, dtype=self._dtype)

            indices = np.where(dones | timeouts)[0]
            if len(indices) > 0:
                if hasattr(policy, 'reset'):
                    policy.reset(indices)
                next_states = next_states.copy()
                next_states[indices] = self.env.partial_reset(indices)
                for index in indices:
                    infos[index]['episode'] = {'return': self._returns[index], 'extra': infos[index]}
                self._n_steps[indices] = 0
                self._returns[indices] = 0.

            self._states = next_states.copy()
            ep_infos.extend([info['episode'] for info in infos if 'episode' in info])

        return dataset.reshape(-1), ep_infos

