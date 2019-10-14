import numpy as np
from gym.spaces import Box
from boots.model.deterministic_model import DeterministicModel
from boots.envs import BaseBatchedEnv, BaseModelBasedEnv


class VirtualEnv(BaseBatchedEnv):
    _states: np.ndarray

    def __init__(self, model: DeterministicModel, env: BaseModelBasedEnv, n_envs: int, opt_model=False):
        super().__init__()
        self.n_envs = n_envs
        self.observation_space = env.observation_space  # ???

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]
        if opt_model:
            self.action_space = Box(low=np.r_[env.action_space.low, np.zeros(dim_state) - 1.],
                                    high=np.r_[env.action_space.high, np.zeros(dim_state) + 1.],
                                    dtype=np.float32)
        else:
            self.action_space = env.action_space

        self._opt_model = opt_model
        self._model = model
        self._env = env
        assert np.allclose(self.action_space.low, -1.) and np.allclose(self.action_space.high, 1.)

        self._states = np.zeros((self.n_envs, dim_state), dtype=np.float32)

    def step(self, actions):
        if self._opt_model:
            actions = actions[..., :self._env.action_space.shape[0]]

        # next_states = self._model.eval('next_states', states=self._states, actions=actions)
        next_states = self._model.get_next_states(self._states, actions)
        rewards, dones = self._env.mb_step(self._states, actions, next_states)
        dones = dones & False

        self._states = next_states
        return self._states.copy(), rewards, dones, [{} for _ in range(self.n_envs)]

    def reset(self):
        return self.partial_reset(range(self.n_envs))

    def partial_reset(self, indices):
        """
            It can be a little bit tricky here. For any invertible
            transformation, given the transformed states, we can easily
            invert them to original state space. However, when the
            transformation is costly and may not be invertible, it's
            more difficult to compute the reward since in our
            transformed dynamics, the reward function may not be well
            defined. The best thing I can do is to use the inversion
            to approximate the reward.
        """

        initial_states = np.array([self._env.reset() for _ in indices])

        self._states = self._states.copy()
        self._states[indices] = initial_states

        return initial_states.copy()

    def set_state(self, states):
        self._states = states.copy()

    def render(self, mode='human'):
        pass
