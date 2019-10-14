import numpy as np
import abc
import gym


class BaseBatchedEnv(gym.Env, abc.ABC):
    # thought about using `@property @abc.abstractmethod` here but we don't need explicit `@property` function here.
    n_envs: int

    @abc.abstractmethod
    def step(self, actions):
        pass

    def reset(self):
        return self.partial_reset(range(self.n_envs))

    @abc.abstractmethod
    def partial_reset(self, indices):
        pass


class BaseModelBasedEnv(gym.Env, abc.ABC):
    @abc.abstractmethod
    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        raise NotImplementedError

    def seed(self, seed: int = None):
        pass

    def verify(self):
        raise NotImplementedError

